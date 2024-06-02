# -*- coding: utf-8 -*-
"""
FM实现召回和排序
"""
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import json

# region 模型结构
from torch.utils.data import dataloader, dataset


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


class SideFMVectorBaseModule(nn.Module):
    def __init__(self, sparse_field_nums, dense_num_field, embed_dim, is_spu_side=False):
        super(SideFMVectorBaseModule, self).__init__()
        self.is_spu_side = is_spu_side
        # 1阶部分
        self.sparse_linear = SparseEmbedding(sparse_field_nums, embed_dim=1)
        self.dense_linear = DenseEmbedding(dense_num_field, embed_dim=1)
        # 2阶部分
        self.sparse_second_order = SparseEmbedding(sparse_field_nums, embed_dim=embed_dim)
        self.dense_second_order = DenseEmbedding(dense_num_field, embed_dim=embed_dim)

    def internal_forward(self, sparse_x, dense_x):
        # 1阶部分
        # [N,sparse_num_field] -> [N,sparse_num_field,1] -> [N]
        v1_sparse = self.sparse_linear(sparse_x).squeeze(-1).sum(dim=1)
        # [N,dense_num_field] -> [N,dense_num_field,1] -> [N]
        v1_dense = self.dense_linear(dense_x).squeeze(-1).sum(dim=1)
        v1 = v1_sparse + v1_dense  # [N]

        # 2阶部分
        # [N,sparse_num_field] -> [N,sparse_num_field,embed_dim]
        v2_sparse = self.sparse_second_order(sparse_x)
        # [N,dense_num_field] -> [N,dense_num_field,embed_dim]
        v2_dense = self.dense_second_order(dense_x)
        # num_field = sparse_num_field + dense_num_field
        # 合并 [N,num_field,embed_dim]
        v2 = torch.concat([v2_sparse, v2_dense], dim=1)

        return v1, v2

    def forward(self, sparse_x, dense_x):
        return self.internal_forward(sparse_x, dense_x)

    def get_vectors(self, sparse_x, dense_x):
        """
        返回样本对应的向量 --> 可用于召回阶段
        :param sparse_x: [N,sparse_num_field] 稀疏特征
        :param dense_x: [N,dense_num_field] 稠密特征
        :return: [N,embed_dim+1]
        """
        # v1:[N];    针对每个样本存在一个一维的置信度值
        # v2:[N,num_field,embed_dim] 针对每个样本都存在num_field个的embed_dim维度的特征向量;
        # num_field=sparse_num_field+dense_num_field
        v1, v2 = self.internal_forward(sparse_x, dense_x)

        # 向量直接将所有field的向量累计即可
        v = v2.sum(dim=1)  # [N,num_field,embed_dim] -> [N,embed_dim]

        # 将1阶部分的值，加入到最终向量的尾部
        v1 = v1.view(-1, 1)  # [N] -> [N,1]
        if self.is_spu_side:
            print("当前是物品侧向量子模型....")
            # 商品侧自身和自身的二阶部分的置信度
            square_sum = v2.sum(dim=1).pow(2).sum(dim=1)
            # [N,num_field,embed_dim] -> [N,num_field,embed_dim] -> [N,embed_dim] -> [N]
            sum_square = v2.pow(2).sum(dim=1).sum(dim=1)
            v2_v1 = 0.5 * (square_sum - sum_square)  # [N]
            v2_v1 = v2_v1.view(-1, 1)
            v1 = v1 + v2_v1
        else:
            # 当前是用户侧子模型，直接填充1即可；所以重置v1=1
            v1 = torch.ones_like(v1)
        v = torch.concat([v, v1], dim=1)  # [N,embed_dim] concat [N,1] -> [N,embed_dim+1]

        return v


# noinspection DuplicatedCode
class FM(nn.Module):
    def __init__(self, user_sparse_field_nums, user_dense_num_field, spu_sparse_field_nums, spu_dense_num_field,
                 embed_dim):
        """
        :param user_sparse_field_nums: 所有用户稀疏特征对应的每个特征的类别数目，eg: [10000,3,100,50]
        :param user_dense_num_field: 稠密特征的数量
        :param spu_sparse_field_nums: 所有商品稀疏特征对应的每个特征的类别数目，eg: [5000,300]
        :param spu_dense_num_field: 稠密特征的数量
        :param embed_dim: 二阶特征部分，映射的向量维度大小
        """
        super(FM, self).__init__()
        self.register_buffer('user_sparse_field_nums', torch.tensor(user_sparse_field_nums))
        self.register_buffer('user_dense_num_field', torch.tensor(user_dense_num_field))
        self.register_buffer('spu_sparse_field_nums', torch.tensor(spu_sparse_field_nums))
        self.register_buffer('spu_dense_num_field', torch.tensor(spu_dense_num_field))
        self.register_buffer('embed_dim', torch.tensor(embed_dim))
        # 0阶部分
        self.bias = nn.Parameter(torch.zeros(1))
        # 用户侧子模型
        self.user_side = SideFMVectorBaseModule(
            user_sparse_field_nums, user_dense_num_field, embed_dim, is_spu_side=False
        )
        # 商品侧子模型
        self.spu_side = SideFMVectorBaseModule(
            spu_sparse_field_nums, spu_dense_num_field, embed_dim, is_spu_side=True
        )

    def forward(self, user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x):
        # 1. 提取用户侧的相关信息
        # v1_user:[N];    v2_user:[N,user_num_field,embed_dim]
        v1_user, v2_user = self.user_side(user_sparse_x, user_dense_x)
        # 2. 提取物品侧的相关信息
        # v1_spu:[N];    v2_spu:[N,spu_num_field,embed_dim]
        v1_spu, v2_spu = self.spu_side(spu_sparse_x, spu_dense_x)

        # 1阶部分
        v1 = v1_spu + v1_user

        # 2阶部分
        # 合并 [N,num_field,embed_dim]
        v2 = torch.concat([v2_user, v2_spu], dim=1)
        # 快速计算
        # [N,num_field,embed_dim] --> [N,embed_dim] --> [N,embed_dim] --> [N]
        square_sum = v2.sum(dim=1).pow(2).sum(dim=1)
        # [N,num_field,embed_dim] -> [N,num_field,embed_dim] -> [N,embed_dim] -> [N]
        sum_square = v2.pow(2).sum(dim=1).sum(dim=1)
        v2 = 0.5 * (square_sum - sum_square)

        # 合并0、1、2三个部分的置信度
        z = self.bias + v1 + v2
        return z


def t0():
    fmnet = FM(
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
    r = fmnet(user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x)
    print(r)

    user_vector = fmnet.user_side.get_vectors(user_sparse_x, user_dense_x)
    print(user_vector)

    spu_vector = fmnet.spu_side.get_vectors(spu_sparse_x, spu_dense_x)
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
        self.target_df = np.asarray(df['rating'])
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
    net = FM(
        user_sparse_field_nums=train_dateset.user_sparse_field_dims,
        user_dense_num_field=train_dateset.user_dense_num_fields,
        spu_sparse_field_nums=train_dateset.spu_sparse_field_dims,
        spu_dense_num_field=train_dateset.spu_dense_num_fields,
        embed_dim=embed_dim
    )
    print("=" * 100)
    print(net)
    # 一般情况下，FM模型用来训练CTR或者CVR预估的模型，也就是实际标签一般是0/1
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.SmoothL1Loss()  # NOTE: 这是因为训练数据利用的是评分数据，所以采用回归损失
    optimizer = optim.SGD(params=net.parameters(), lr=0.001)

    # 3. 迭代训练
    for epoch in range(total_epoch):
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            # 1. 还原
            user_sparse_x, user_dense_x, spu_sparse_x, spu_dense_x, labels = batch

            # 2. 前向过程
            outputs = net(user_sparse_x.long(), user_dense_x.float(), spu_sparse_x.long(), spu_dense_x.float())
            loss = loss_fn(outputs, labels.long())

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
    # shutil.copytree(
    #     src=os.path.join(root_dir, 'dict'),
    #     dst=os.path.join(output_dir, 'dict')
    # )


# endregion

# region 用户侧向量子模型、商品侧向量子模型拆分

def export(model_dir):
    """
    加载训练好的原始模型文件，转换为静态结构，并提取用户侧向量子模型、商品侧向量子模型转换静态结构保存
    :param model_dir: 文件夹路径
    :return:
    """
    # 1. 模型恢复
    net: FM = torch.load(os.path.join(model_dir, 'model.pkl'), map_location='cpu')
    net.eval().cpu()

    user_sparse_field_nums = net.user_sparse_field_nums
    user_dense_num_field = net.user_dense_num_field
    spu_sparse_field_nums = net.spu_sparse_field_nums
    spu_dense_num_field = net.spu_dense_num_field
    embed_dim = net.embed_dim

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
    torch.jit.save(jit_net, os.path.join(model_dir, 'model.pt'))

    # 3. 用户侧向量子模型
    user_side = net.user_side
    print(user_side.is_spu_side)
    user_side.forward = user_side.get_vectors  # 底层里面forward实际执行的是get_vectors方法
    user_jit_net = torch.jit.trace(
        user_side,
        example_inputs=(user_sparse_x, user_dense_x)
    )
    torch.jit.save(user_jit_net, os.path.join(model_dir, 'user_model.pt'))

    # 4. 物品侧向量子模型
    print("-" * 100)
    spu_side = net.spu_side
    print(spu_side.is_spu_side)
    spu_side.forward = spu_side.get_vectors  # 底层里面forward实际执行的是get_vectors方法
    spu_jit_net = torch.jit.trace(
        spu_side,
        example_inputs=(spu_sparse_x, spu_dense_x)
    )
    torch.jit.save(spu_jit_net, os.path.join(model_dir, 'spu_model.pt'))


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
        shuffle=False,
        num_workers=0
    )

    # 2. 恢复模型
    spu_net = torch.jit.load(os.path.join(model_dir, "spu_model.pt"), map_location='cpu')
    spu_net.eval()

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

    # 4. spu特征向量矩阵保存
    print(spu_embedding.shape)
    np.save(os.path.join(model_dir, 'spu_embedding.npy'), spu_embedding)
    np.savetxt(os.path.join(model_dir, 'spu_embedding.txt'), spu_embedding)
    np.savez_compressed(os.path.join(model_dir, 'spu_embedding.npz'), spu_embedding=spu_embedding)


# endregion


def upload(model_dir):
    """
    将本地文件夹中的内容上传到服务器上
    :param model_dir: 本地待上传的文件夹路径
    :return:
    """
    base_url = "http://127.0.0.1:5051"
    name = 'fm'  # 当前必须为fm
    sess = requests.session()

    # 1. version信息恢复
    extra_files = {
        'model_version': ''
    }
    spu_net = torch.jit.load(os.path.join(model_dir, "spu_model.pt"), map_location='cpu', _extra_files=extra_files)
    spu_net.eval()
    for k in extra_files:
        extra_files[k] = str(extra_files[k], encoding='utf-8')
    model_version = "fm_20240121_155900"
    extra_files['model_version'] = model_version
    print(extra_files)
    json.dump(dict(extra_files), open(os.path.join(model_dir, 'info.json'), 'w', encoding='utf-8'))
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

    #


# endregion

# region 模型部署：用户侧向量子模型部署、商品侧向量子模型部署、商品向量faiss所有构建....

def deploy(model_dir):
    base_url = "http://127.0.0.1:5051"
    name = 'fm'  # 当前必须为fm
    sess = requests.session()
    extra_files = json.load(open(os.path.join(model_dir, 'info.json'), 'r', encoding='utf-8'))
    model_version = extra_files['model_version']

    # fm索引上线
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
    pass

