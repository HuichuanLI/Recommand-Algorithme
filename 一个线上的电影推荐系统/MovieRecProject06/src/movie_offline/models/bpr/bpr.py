# -*- coding: utf-8 -*-
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dask.bytes.tests.test_http import requests
from torch import optim

from torch.utils.data import dataset, dataloader


# region 模型结构构造

class BPRNetwork(nn.Module):
    def __init__(self, user_number, spu_number, embedd_dim, add_score_loss=True, add_bias=True):
        super(BPRNetwork, self).__init__()
        self.add_score_loss = add_score_loss
        self.add_bias = add_bias

        if add_bias:
            self.m = nn.Parameter(torch.zeros((1,)))
            self.bu = nn.Parameter(torch.empty(user_number, ))
            self.bi = nn.Parameter(torch.empty(spu_number, ))
            nn.init.normal_(self.bu)
            nn.init.normal_(self.bi)
        else:
            self.register_buffer('m', None)
            self.register_buffer('bu', None)
            self.register_buffer('bi', None)
        self.p = nn.Parameter(torch.empty(user_number, embedd_dim))
        self.q = nn.Parameter(torch.empty(spu_number, embedd_dim))

        # 注册一些属性信息
        self.register_buffer('user_number', torch.tensor(user_number))
        self.register_buffer('embedd_dim', torch.tensor(embedd_dim))
        self.register_buffer('spu_number', torch.tensor(spu_number))

        nn.init.normal_(self.p)
        nn.init.normal_(self.q)

    def forward(self, u, i, j, ui, uj):
        """
        计算用户u对应商品i、商品j的评分，并且构造损失值返回
        :param u: 用户id [N]
        :param i: 商品id [N]
        :param j: 商品id [N]
        :param ui: 用户u对商品i的实际评分 [
        :param uj: 用户u对商品j的实际评分 [NN]]
        :return:
        """
        # 1. 获取用户u对商品i的预测评分
        rui = self.forward_with_score(u, i)
        # 2. 获取用户u对商品j的预测评分
        ruj = self.forward_with_score(u, j)
        # 3. 计算用户u在商品i、j上的评分差值
        r_uij = rui - ruj
        # 4. 构建BPR损失，由于BPR模型的建模思路就是用户u对商品i的实际评分高于用户u对商品j的评分 ---> 希望r_uij越大越好
        los_bpr = -F.logsigmoid(r_uij).mean()
        if self.add_score_loss:
            # 5. 构建一个矩阵分解的损失: 要求预测评分和实际评分要尽可能的接近 --> 希望rui和ui、ruj和uj更加的接近
            los_score = 0.5 * (F.smooth_l1_loss(rui, ui) + F.smooth_l1_loss(ruj, uj))
            # 6. 将两个损失合并成一个损失 --> 如何高效合并呢？
            bpr_alpha = 0.7
            los_value = bpr_alpha * los_bpr + (1.0 - bpr_alpha) * los_score
        else:
            los_value = los_bpr
        return los_value

    def forward_with_score(self, u, i):
        """
        计算u对商品i的评分
        :param u: 用户id [N]
        :param i: 商品id [N]
        :return: 评分 [N]
        """
        if self.m is not None:
            rui = self.m + self.bu[u] + self.bi[i]
        else:
            rui = 0.0
        pu = self.p[u, :]  # [N,embedd_dim]
        qi = self.q[i, :]  # [N,embedd_dim]
        rui = rui + (pu * qi).sum(dim=1)  # [N,embedd_dim] * [N,embedd_dim] -> [N]
        return rui


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
        self.user_id_mapping = SimpleMapping(os.path.join(root_dir, "bpr_dict", "user_id.dict"))
        self.spu_id_mapping = SimpleMapping(os.path.join(root_dir, "bpr_dict", "spu_id.dict"))

        # 2. 加载数据
        df = pd.read_csv(os.path.join(root_dir, "feature_bpr2.csv"), sep="\t", low_memory=False, header=None)
        # df = df.loc[:100000]  # 快速进行训练
        df.columns = ['u', 'i', 'j', 'ui', 'uj']

        # 3. 最终数据拆分
        self.udf = np.asarray(df['u'])
        self.idf = np.asarray(df['i'])
        self.jdf = np.asarray(df['j'])
        self.uidf = np.asarray(df['ui'])
        self.ujdf = np.asarray(df['uj'])

    def __getitem__(self, item):
        return self.udf[item], \
               self.idf[item], self.jdf[item], \
               self.uidf[item], self.ujdf[item]

    def __len__(self):
        return len(self.udf)


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
    net = BPRNetwork(
        user_number=train_dateset.user_id_mapping.size(),
        spu_number=train_dateset.spu_id_mapping.size(),
        embedd_dim=embed_dim
    )
    print("=" * 100)
    print(net)
    optimizer = optim.SGD(params=net.parameters(), lr=0.001)

    # 3. 迭代训练
    for epoch in range(total_epoch):
        running_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            # 1. 还原
            ux, ix, jx, uix, yjx = batch

            # 2. 前向过程
            loss = net(ux.long(), ix.long(), jx.long(), uix.float(), yjx.float())

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
    shutil.copytree(
        src=os.path.join(root_dir, 'bpr_dict'),
        dst=os.path.join(output_dir, 'bpr_dict')
    )


# endregion

# region 静态化转换 --> 为了上线部署

def export(model_dir):
    """
    加载训练好的原始模型文件，转换为静态结构，并提取用户侧向量子模型、商品侧向量子模型转换静态结构保存
    :param model_dir: 文件夹路径
    :return:
    """
    # 1. 模型恢复
    net: BPRNetwork = torch.load(os.path.join(model_dir, 'model.pkl'), map_location='cpu')
    net.eval().cpu()

    batch_size = 2
    u = torch.randint(5, (batch_size,))
    i = torch.randint(5, (batch_size,))

    now = datetime.now()
    model_version = f'bpr_{now.strftime("%Y%m%d_%H%M%S")}'
    _extra_files = {
        'model_version': model_version
    }
    # 2. 整个模型转换为静态结构 --> 用于排序阶段
    net.forward = net.forward_with_score
    jit_net = torch.jit.trace(
        net,
        example_inputs=(u, i)
    )
    torch.jit.save(jit_net, os.path.join(model_dir, 'model.pt'), _extra_files)
    json.dump(_extra_files, open(os.path.join(model_dir, 'info.json'), 'w', encoding='utf-8'))


# endregion

# region 模型上传

def upload(model_dir):
    """
    将本地文件夹中的内容上传到服务器上
    :param model_dir: 本地待上传的文件夹路径
    :return:
    """
    base_url = "http://127.0.0.1:5051"
    base_url = "http://121.40.96.93:9999"
    name = 'bpr'  # 当前必须为fm
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
    upload(_f=os.path.join(model_dir, "bpr_dict"))
    upload(_f=os.path.join(model_dir, "info.json"))

    #

# endregion
