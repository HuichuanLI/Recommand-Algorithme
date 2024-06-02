# -*- coding: utf-8 -*-
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import dataset, dataloader


# region 定义网络结构

class SparseEmbedding(nn.Module):
    def __init__(self, sparse_field_nums, embed_dim):
        super(SparseEmbedding, self).__init__()
        self.sparse_num_field = len(sparse_field_nums)  # 稀疏(离散的分类)特征的数量
        self.embed_layer = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=embed_dim)
        self.offsets = np.asarray((0, *np.cumsum(sparse_field_nums)[:-1]), dtype=np.int32)

    def forward(self, x):
        """
        前向过程
        :param x: 输入的原始系数特征id [N,sparse_num_field]
        :return: [N,sparse_num_field,embed_dim]
        """
        x = x + x.new_tensor(self.offsets)
        z = self.embed_layer(x)  # [N,sparse_num_field,embed_dim]
        return z


class DenseEmbedding(nn.Module):
    def __init__(self, dense_num_field, embed_dim):
        super(DenseEmbedding, self).__init__()
        self.dense_num_field = dense_num_field  # 稠密(连续)特征的数量
        self.dense_w = nn.Parameter(torch.empty(1, self.dense_num_field, embed_dim))
        nn.init.normal_(self.dense_w)

    def forward(self, x):
        """
        前向过程
        :param x: 原始输入的稠密特征向量x [N,dense_num_field]
        :return: [N,dense_num_field,embed_dim]
        """
        x = x.view(-1, self.dense_num_field, 1)  # [N,dense_num_field] -> [N,dense_num_field,1]
        # [1,dense_num_field,embed_dim] * [N,dense_num_field,1] -> [N,dense_num_field,embed_dim]
        z = self.dense_w * x
        return z


class MultilayerPerceptron(nn.Module):
    def __init__(self, sparse_field_nums, dense_num_field, embed_dim, units):
        super(MultilayerPerceptron, self).__init__()
        self.sparse_embedding = SparseEmbedding(sparse_field_nums, embed_dim)
        self.dense_embedding = DenseEmbedding(dense_num_field, embed_dim)
        _layers = []
        in_features = (len(sparse_field_nums) + dense_num_field) * embed_dim  # 输入的特征维度大小 = 特征数量 * embed_dim
        last_idx = len(units) - 1
        for idx, unit in enumerate(units):
            _layers.append(nn.Linear(in_features=in_features, out_features=unit))
            if idx != last_idx:
                # 除了最后一层，均添加一个激活函数
                _layers.append(nn.LeakyReLU())
            in_features = unit
        self.mlp = nn.Sequential(*_layers)

    def forward(self, sparse_x, dense_x):
        sparse_v = self.sparse_embedding(sparse_x)  # [N,sparse_num_fields,E]
        dense_v = self.dense_embedding(dense_x)  # [N,dense_num_fields,E]
        v = torch.cat([sparse_v, dense_v], dim=1)  # [N,num_fields,E]
        v_size = v.size()
        v = v.view([-1, v_size[1] * v_size[2]])  # [N,num_fields,E] -> [N,num_fields*E]
        v = self.mlp(v)
        return v


class UserSideDSSM(nn.Module):
    def __init__(self, sparse_field_nums, dense_num_field, embed_dim, user_mlp_units):
        super(UserSideDSSM, self).__init__()
        if user_mlp_units is None:
            user_mlp_units = [128, 64, 32, 8]
        self.user_mlp = MultilayerPerceptron(
            sparse_field_nums=sparse_field_nums,
            dense_num_field=dense_num_field,
            embed_dim=embed_dim,
            units=user_mlp_units
        )

    def forward(self, user_sparse, user_dense):
        v = self.user_mlp(user_sparse, user_dense)
        v = F.normalize(v, p=2, dim=1)  # 向量做一个L2-norm的处理
        return v


class SpuSideDSSM(nn.Module):
    def __init__(self, sparse_field_nums, dense_num_field, embed_dim, spu_mlp_units):
        super(SpuSideDSSM, self).__init__()
        if spu_mlp_units is None:
            spu_mlp_units = [128, 64, 8]
        self.spu_mlp = MultilayerPerceptron(
            sparse_field_nums=sparse_field_nums,
            dense_num_field=dense_num_field,
            embed_dim=embed_dim,
            units=spu_mlp_units
        )

    def forward(self, spu_sparse, spu_dense):
        v = self.spu_mlp(spu_sparse, spu_dense)
        v = F.normalize(v, p=2, dim=1)  # 向量做一个L2-norm的处理
        return v


class DSSM(nn.Module):
    def __init__(self,
                 user_sparse_field_nums, user_dense_num_field,
                 spu_sparse_field_nums, spu_dense_num_field,
                 embed_dim, user_mlp_units=None, spu_mlp_units=None
                 ):
        super(DSSM, self).__init__()

        self.register_buffer('user_sparse_field_nums', torch.tensor(user_sparse_field_nums))
        self.register_buffer('user_dense_num_field', torch.tensor(user_dense_num_field))
        self.register_buffer('spu_sparse_field_nums', torch.tensor(spu_sparse_field_nums))
        self.register_buffer('spu_dense_num_field', torch.tensor(spu_dense_num_field))
        self.register_buffer('embed_dim', torch.tensor(embed_dim))

        self.user = UserSideDSSM(user_sparse_field_nums, user_dense_num_field, embed_dim, user_mlp_units)
        self.spu = SpuSideDSSM(spu_sparse_field_nums, spu_dense_num_field, embed_dim, spu_mlp_units)

    def forward(self, user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x):
        """
        :param user_sparse_x: 稀疏特征属性, [batch_size, user_sparse_num_field], int32类型的id
        :param user_dense_x: 稠密特征属性, [batch_size, user_dense_num_field], float类型的具体特征值
        :param spu_sparse_x: 稀疏特征属性, [batch_size, spu_sparse_num_field], int32类型的id
        :param spu_dense_x: 稠密特征属性, [batch_size, spu_dense_num_field], float类型的具体特征值
        :return:
        """
        # 1. 获取用户向量
        user_vector = self.user(user_sparse_x, user_dense_x)  # [batch_size, embed_dim]
        # 2. 获取商品向量
        spu_vector = self.spu(spu_sparse_x, spu_dense_x)  # [batch_size, embed_dim]
        # 3. 计算用户向量和商品向量之间的相似度（余弦相似度 --> 前面先做了一个norm处理 --> 等价于两个向量的内积）
        # [batch_size, embed_dim] * [batch_size, embed_dim] -> [batch_size]
        score = (user_vector * spu_vector).sum(dim=1)
        return score


# endregion

# region 测试代码


def t0():
    dssm_net = DSSM(
        user_sparse_field_nums=[1000, 2000, 355, 140, 250],
        user_dense_num_field=10,
        spu_sparse_field_nums=[5222, 352, 1000],
        spu_dense_num_field=23,
        embed_dim=5
    )
    batch_size = 2
    user_sparse_x = torch.randint(100, (batch_size, 5))
    user_dense_x = torch.randn(batch_size, 10)
    spu_sparse_x = torch.randint(100, (batch_size, 3))
    spu_dense_x = torch.randn(batch_size, 23)
    r = dssm_net(user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x)
    print(r)

    user_vector = dssm_net.user(user_sparse_x, user_dense_x)
    print(user_vector)

    spu_vector = dssm_net.spu(spu_sparse_x, spu_dense_x)
    print(spu_vector)


# endregion

# region 训练相关代码

class SimpleMapping(object):
    """
    定义的一个简单的映射表
    """

    def __init__(self, path):
        self.mapping = {}
        with open(path, 'r', encoding='utf-8') as reader:
            for line in reader:
                arr = line.strip().split("\t")
                self.mapping[arr[0]] = int(arr[1])

    def get(self, key) -> int:
        return self.mapping.get(str(key), self.mapping['unk'])

    def size(self) -> int:
        return len(self.mapping)


class SelfDataset(dataset.Dataset):
    def __init__(self, root_dir):
        super(SelfDataset, self).__init__()
        # 1. 加载mapping对象
        user_id_mapping = SimpleMapping(os.path.join(root_dir, "dict", "user_id.dict"))
        age_mapping = SimpleMapping(os.path.join(root_dir, "dict", "age.dict"))
        gender_mapping = SimpleMapping(os.path.join(root_dir, "dict", "gender.dict"))
        occupation_mapping = SimpleMapping(os.path.join(root_dir, "dict", "occupation.dict"))
        location_mapping = SimpleMapping(os.path.join(root_dir, "dict", "location.dict"))
        movie_genre_mapping = SimpleMapping(os.path.join(root_dir, "dict", "movie_genre.dict"))
        movie_id_mapping = SimpleMapping(os.path.join(root_dir, "dict", "movie_id.dict"))

        # 2. 特征定义
        _user_sparse_column = [
            'user_id', 'age', 'gender', 'occupation', 'location', 'max_rating_genre', 'max_rete_items_genre'
        ]
        _spu_sparse_column = [
            'movie_id',
            'unknown', 'action', 'adventure', 'animation',
            'children', 'comedy', 'crime', 'documentary',
            'drama', 'fantasy', 'film_noir', 'horror',
            'musical', 'mystery', 'romance', 'sci_fi',
            'thriller', 'war', 'western'
        ]
        _user_dense_column = [
            'action_mean_rating', 'adventure_mean_rating', 'animation_mean_rating', 'children_mean_rating',
            'comedy_mean_rating', 'crime_mean_rating', 'documentary_mean_rating', 'drama_mean_rating',
            'fantasy_mean_rating', 'film_noir_mean_rating', 'horror_mean_rating', 'musical_mean_rating',
            'mystery_mean_rating', 'romance_mean_rating', 'sci_fi_mean_rating', 'thriller_mean_rating',
            'unknown_mean_rating', 'war_mean_rating', 'western_mean_rating', 'user_mean_rating'
        ]
        _spu_dense_column = [
            'movie_mean_rating', 'm_mean_rating', 'f_mean_rating'
        ]

        # 3. 加载数据
        df = pd.read_csv(os.path.join(root_dir, "feature_fm.csv"), low_memory=False)

        # 4. 数据拆分转换
        user_sparse_field_dims = []
        spu_sparse_field_dims = []
        df['user_id'] = df.user_id.apply(lambda t: user_id_mapping.get(t))
        user_sparse_field_dims.append(user_id_mapping.size())
        df['age'] = df.age.apply(lambda t: age_mapping.get(t))
        user_sparse_field_dims.append(age_mapping.size())
        df['gender'] = df.gender.apply(lambda t: gender_mapping.get(t))
        user_sparse_field_dims.append(gender_mapping.size())
        df['occupation'] = df.occupation.apply(lambda t: occupation_mapping.get(t))
        user_sparse_field_dims.append(occupation_mapping.size())
        df['location'] = df.location.apply(lambda t: location_mapping.get(t))
        user_sparse_field_dims.append(location_mapping.size())
        df['max_rating_genre'] = df.max_rating_genre.apply(lambda t: movie_genre_mapping.get(t))
        user_sparse_field_dims.append(movie_genre_mapping.size())
        df['max_rete_items_genre'] = df.max_rete_items_genre.apply(lambda t: movie_genre_mapping.get(t))
        user_sparse_field_dims.append(movie_genre_mapping.size())
        df['movie_id'] = df.movie_id.apply(lambda t: movie_id_mapping.get(t))
        spu_sparse_field_dims.append(movie_id_mapping.size())
        for i in range(19):
            spu_sparse_field_dims.append(2)

        # 5. 最终数据拆分
        self.user_sparse_df = np.asarray(df[_user_sparse_column])
        self.spu_sparse_df = np.asarray(df[_spu_sparse_column])
        self.user_dense_df = np.asarray(df[_user_dense_column])
        self.spu_dense_df = np.asarray(df[_spu_dense_column])
        # NOTE: 将评分≥3的认为是类别1，其它认为是类别0
        self.target_df = (np.asarray(df['rating']) >= 3).astype(np.int32)
        self.user_sparse_field_dims = user_sparse_field_dims
        self.spu_sparse_field_dims = spu_sparse_field_dims
        self.user_dense_num_fields = len(_user_dense_column)
        self.spu_dense_num_fields = len(_spu_dense_column)

    def __getitem__(self, item):
        return self.user_sparse_df[item], self.user_dense_df[item], \
               self.spu_sparse_df[item], self.spu_dense_df[item], \
               self.target_df[item]

    def __len__(self):
        return len(self.target_df)


# noinspection DuplicatedCode
def training(root_dir, output_dir):
    embed_dim = 8
    total_epoch = 2
    batch_size = 8

    # 1. 数据集的构建
    train_dateset = SelfDataset(root_dir)
    train_loader = dataloader.DataLoader(
        dataset=train_dateset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 2. 模型对象的构建
    net = DSSM(
        user_sparse_field_nums=train_dateset.user_sparse_field_dims,
        user_dense_num_field=train_dateset.user_dense_num_fields,
        spu_sparse_field_nums=train_dateset.spu_sparse_field_dims,
        spu_dense_num_field=train_dateset.spu_dense_num_fields,
        embed_dim=embed_dim,
        user_mlp_units=[128, 64, 32, 8],
        spu_mlp_units=[512, 128, 64, 32, 8]
    )
    print("=" * 100)
    print(net)
    # 一般情况下，FM模型用来训练CTR或者CVR预估的模型，也就是实际标签一般是0/1
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=0.001)

    # 3. 迭代训练
    for epoch in range(total_epoch):
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            # 1. 还原
            user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x, labels = batch

            # 2. 前向过程
            outputs = net(user_sparse_x.long(), user_dense_x.float(), spu_sparse_x.long(), spu_dense_x.float())
            loss = loss_fn(outputs, labels.float())

            # 3. 反向过程
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training!")

    # 4. 持久化
    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹
    torch.save(net, os.path.join(output_dir, "model.pkl"))
    # 将所有dict的需要进行copy
    shutil.rmtree(os.path.join(output_dir, 'dict'), ignore_errors=True)
    shutil.copytree(
        src=os.path.join(root_dir, 'dict'),
        dst=os.path.join(output_dir, 'dict')
    )
    # 保存额外的信息
    now = datetime.now()
    model_version = f'dssm_{now.strftime("%Y%m%d_%H%M%S")}'
    extra_files = {
        'model_version': model_version
    }
    json.dump(dict(extra_files), open(os.path.join(output_dir, 'info.json'), 'w', encoding='utf-8'))


# endregion

# region 用户侧向量子模型、商品侧向量子模型拆分

# noinspection DuplicatedCode

def export(model_dir):
    """
    加载训练好的原始模型文件，转换为静态结构，并提取用户侧向量子模型、商品侧向量子模型转换静态结构保存
    :param model_dir: 文件夹路径
    :return:
    """
    # 1. 模型恢复
    net: DSSM = torch.load(os.path.join(model_dir, 'model.pkl'), map_location='cpu')
    net.eval().cpu()
    _extra_files = json.load(open(os.path.join(model_dir, 'info.json'), 'r', encoding='utf-8'))

    user_sparse_field_nums = net.user_sparse_field_nums
    user_dense_num_field = net.user_dense_num_field
    spu_sparse_field_nums = net.spu_sparse_field_nums
    spu_dense_num_field = net.spu_dense_num_field

    batch_size = 2
    user_sparse_x = torch.randint(1, (batch_size, user_sparse_field_nums.shape[0]))
    user_dense_x = torch.randn(batch_size, user_dense_num_field.item())
    spu_sparse_x = torch.randint(1, (batch_size, spu_sparse_field_nums.shape[0]))
    spu_dense_x = torch.randn(batch_size, spu_dense_num_field.item())

    # 2. 整个模型转换为静态结构 --> 用于排序阶段
    jit_net = torch.jit.trace(
        net,
        example_inputs=(user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x)
    )
    torch.jit.save(jit_net, os.path.join(model_dir, 'model.pt'), _extra_files)

    # 3. 用户侧向量子模型
    user_side = net.user
    user_jit_net = torch.jit.trace(
        user_side,
        example_inputs=(user_sparse_x, user_dense_x)
    )
    torch.jit.save(user_jit_net, os.path.join(model_dir, 'user_model.pt'), _extra_files)

    # 4. 物品侧向量子模型
    print("-" * 100)
    spu_side = net.spu
    spu_jit_net = torch.jit.trace(
        spu_side,
        example_inputs=(spu_sparse_x, spu_dense_x)
    )
    torch.jit.save(spu_jit_net, os.path.join(model_dir, 'spu_model.pt'), _extra_files)


# endregion

# region 物品特征向量矩阵构建


class SpuSelfDataset(dataset.Dataset):
    def __init__(self, root_dir):
        super(SpuSelfDataset, self).__init__()
        # 1. 加载mapping对象
        movie_id_mapping = SimpleMapping(os.path.join(root_dir, "dict", "movie_id.dict"))

        _spu_sparse_column = [
            'movie_id',
            'unknown', 'action', 'adventure', 'animation',
            'children', 'comedy', 'crime', 'documentary',
            'drama', 'fantasy', 'film_noir', 'horror',
            'musical', 'mystery', 'romance', 'sci_fi',
            'thriller', 'war', 'western'
        ]
        _spu_dense_column = [
            'movie_mean_rating', 'm_mean_rating', 'f_mean_rating'
        ]
        _all_column = []
        _all_column.extend(_spu_sparse_column)
        _all_column.extend(_spu_dense_column)
        df = pd.read_csv(os.path.join(root_dir, "feature_fm.csv"), low_memory=False)
        df = df[_all_column]
        df = df.drop_duplicates(['movie_id'])  # 按照movie_id去重
        # 计算映射mapping(序号id --> 实际类别id)
        movie_ids = dict(zip(range(len(df)), np.asarray(df['movie_id'])))
        spu_sparse_field_dims = []
        df['movie_id'] = df.movie_id.apply(lambda t: movie_id_mapping.get(t))
        spu_sparse_field_dims.append(movie_id_mapping.size())
        for i in range(19):
            spu_sparse_field_dims.append(2)

        # 属性定义
        self.spu_sparse_df = np.asarray(df[_spu_sparse_column])
        self.spu_dense_df = np.asarray(df[_spu_dense_column])
        self.spu_sparse_field_dims = spu_sparse_field_dims
        self.spu_dense_num_fields = len(_spu_dense_column)
        self.movie_ids = movie_ids

    def __getitem__(self, index):
        return self.spu_sparse_df[index], self.spu_dense_df[index]

    def __len__(self):
        return len(self.spu_sparse_df)


def process_spu_embedding(root_dir, model_dir):
    # 1. 加载所有数据
    all_dataset = SpuSelfDataset(root_dir)
    all_loader = dataloader.DataLoader(
        dataset=all_dataset,
        batch_size=16,
        shuffle=False,  # shuffle必须为False，不能打乱顺序
        num_workers=0
    )

    # 2. 恢复模型
    extra_files = {
        'model_version': ''
    }
    spu_net = torch.jit.load(os.path.join(model_dir, "spu_model.pt"), map_location='cpu', _extra_files=extra_files)
    spu_net.eval()
    print(extra_files)

    # 3. 遍历所有数据，获得对应的向量
    spu_embedding = None
    with torch.no_grad():
        for i, batch in enumerate(all_loader, 0):
            # 1. 还原
            spu_sparse_x, spu_dense_x = batch

            # 2. 前向过程
            outputs = spu_net(spu_sparse_x.long(), spu_dense_x.float()).numpy()  # [batch_size, embed_dim+1]

            # 3. 合并所有
            if spu_embedding is None:
                spu_embedding = outputs
            else:
                spu_embedding = np.vstack([spu_embedding, outputs])

    # 4. spu特征向量矩阵保存(保存到本地文件夹中)
    print(spu_embedding.shape)
    np.savez_compressed(
        os.path.join(model_dir, 'spu_embedding.npz'),
        spu_embedding=spu_embedding,  # embedding信息
        id_mapping=np.asarray(list(all_dataset.movie_ids.items()))  # id映射
    )


# endregion

# region 模型上传

def upload(model_dir):
    """
    将本地文件夹中的内容上传到服务器上
    :param model_dir: 本地待上传的文件夹路径
    :return:
    """
    base_url = "http://127.0.0.1:5051"
    # base_url = "http://121.40.96.93:9999"
    name = 'dssm'
    sess = requests.session()

    # 1. version信息恢复
    extra_files = json.load(open(os.path.join(model_dir, 'info.json'), 'r', encoding='utf-8'))
    model_version = extra_files['model_version']

    # 删除文件夹
    sess.get(f"{base_url}/deleter", params={"version": model_version, "name": name})

    # 2. 上传文件
    def upload_file(_f, pname=None, fname=None, sub_dir_names=None):
        data = {
            "version": model_version,
            "name": pname or name
        }
        if fname is not None:
            data['filename'] = fname
        if sub_dir_names is not None:
            data['sub_dir_names'] = sub_dir_names
        res1 = sess.post(
            url=f"{base_url}/uploader",
            data=data,
            files={
                "file": open(_f, 'rb')
            }
        )
        if res1.status_code == 200:
            _data = res1.json()
            if _data['code'] != 200:
                raise ValueError(f"上传文件失败，异常信息为:{_data['msg']}")
            else:
                print(f"上传成功，version:'{_data['version']}'，filename:'{_data['filename']}'")
        else:
            raise ValueError("网络异常!")

    def upload(_f, pname=None, fname=None, sub_dir_names=None):
        if os.path.isfile(_f):
            upload_file(_f, pname, fname, sub_dir_names)
        else:
            cur_dir_name = os.path.basename(_f)
            fname = fname or cur_dir_name  # 选择外部给定的名称，或者当前自身的名称
            if sub_dir_names is None:
                sub_dir_names = f"{fname}"
            else:
                sub_dir_names = f"{sub_dir_names},{fname}"
            # 子文件的处理
            for _name in os.listdir(_f):
                upload(
                    _f=os.path.join(_f, _name),
                    pname=pname,
                    fname=None,  # 子文件无法重命名
                    sub_dir_names=sub_dir_names
                )

    upload(_f=os.path.join(model_dir, "model.pt"))
    upload(_f=os.path.join(model_dir, "spu_model.pt"))
    upload(_f=os.path.join(model_dir, "user_model.pt"))
    upload(_f=os.path.join(model_dir, "spu_embedding.npz"))
    upload(_f=os.path.join(model_dir, "dict"))
    upload(_f=os.path.join(model_dir, "info.json"))


# endregion

# region 模型部署：用户侧向量子模型部署、商品侧向量子模型部署、商品向量faiss所有构建....

def deploy(model_dir):
    base_url = "http://127.0.0.1:5051"
    # base_url = "http://121.40.96.93:9999"
    name = 'dssm'  # 当前必须为dssm
    sess = requests.session()
    extra_files = json.load(open(os.path.join(model_dir, 'info.json'), 'r', encoding='utf-8'))
    model_version = extra_files['model_version']

    # dssm索引上线
    r = sess.get(
        url=f"{base_url}/faiss/build",
        params={"version": model_version, "name": name}
    )
    if r.status_code == 200:
        print(r.json())
    else:
        print(f"调用服务器异常:{r.status_code}")


# endregion

if __name__ == '__main__':
    t0()
