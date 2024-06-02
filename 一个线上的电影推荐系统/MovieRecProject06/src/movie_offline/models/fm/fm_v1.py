# -*- coding: utf-8 -*-
"""
实现一些FM基本结构
"""

import numpy as np
import torch
import torch.nn as nn


class FMV0(nn.Module):
    def __init__(self, sparse_field_nums, k=5):
        """
        :param sparse_field_nums: 所有稀疏特征对应的每个特征的类别数目，eg: [10000,3,100,50,1000000,5000,300]
        """
        super(FMV0, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        # 一阶部分
        self.embedding_linear = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=1)
        # 二阶部分
        self.embedding_second_order = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=k)
        self.offsets = np.asarray((0, *np.cumsum(sparse_field_nums)[:-1]), dtype=np.int32)

    def forward(self,
                user_id, user_gender, user_occupation, user_address,
                product_id, product_store_id, product_category_id, label=None
                ):
        """
        前向过程
        :param user_id: [N] 用户id， int
        :param user_gender:  [N] 用户性别， int
        :param user_occupation:  [N] 用户职业id， int
        :param user_address:  [N] 用户地址id， int
        :param product_id:  [N] 商品id， int
        :param product_store_id:  [N] 商品店铺id， int
        :param product_category_id:  [N] 商品品类id， int
        :param label:  [N] 真实的标签，None表示没有给定，1表示user点击了product，0表示未点击
        :return:
        """
        # 0. 特征合并
        x = torch.stack([
            user_id, user_gender, user_occupation, user_address,
            product_id, product_store_id, product_category_id
        ], dim=-1)  # [N,7]
        x = x + x.new_tensor(self.offsets)
        # 1. 1阶部分的计算
        linear_feature = self.embedding_linear(x)  # [N,7,1]
        linear_feature = linear_feature.squeeze(-1).sum(dim=1)  # [N,7,1] -> [N,7] -> [N]
        # 2. 2阶部分的计算
        second_order_features = self.embedding_second_order(x)  # [N,7,5]
        a = second_order_features.sum(dim=1).pow(2).sum(dim=1)  # [N,7,5] -> [N,5] -> [N,5] -> [N]
        b = second_order_features.pow(2).sum(dim=1).sum(dim=1)  # [N,7,5] -> [N,7,5] -> [N,5] -> [N]
        second_order_feature = 0.5 * (a - b)
        return self.bias + linear_feature + second_order_feature


class FMV1(nn.Module):
    def __init__(self, sparse_field_nums, k=5):
        """
        :param sparse_field_nums: 所有稀疏特征对应的每个特征的类别数目，eg: [10000,3,100,50,1000000,5000,300]
        """
        super(FMV1, self).__init__()
        self.sparse_num_field = len(sparse_field_nums)  # 稀疏特征的数量
        self.bias = nn.Parameter(torch.zeros(1))
        # 一阶部分
        self.embedding_linear = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=1)
        # 二阶部分
        self.embedding_second_order = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=k)
        self.offsets = np.asarray((0, *np.cumsum(sparse_field_nums)[:-1]), dtype=np.int32)

    def forward(self,
                sparse_x, label=None
                ):
        """
        前向过程
        :param sparse_x: [N, sparse_num_field] 稀疏特征，[N,sparse_num_field]
        :param label:  [N] 真实的标签，None表示没有给定，1表示user点击了product，0表示未点击
        :return:
        """
        # 0. 特征合并
        x = sparse_x  # [N,sparse_num_field]
        x = x + x.new_tensor(self.offsets)
        # 1. 1阶部分的计算
        linear_feature = self.embedding_linear(x)  # [N,sparse_num_field,1]
        # [N,sparse_num_field,1] -> [N,sparse_num_field] -> [N]
        linear_feature = linear_feature.squeeze(-1).sum(dim=1)
        # 2. 2阶部分的计算
        second_order_features = self.embedding_second_order(x)  # [N,sparse_num_field,5]
        # [N,sparse_num_field,5] -> [N,5] -> [N,5] -> [N]
        a = second_order_features.sum(dim=1).pow(2).sum(dim=1)
        # [N,sparse_num_field,5] -> [N,sparse_num_field,5] -> [N,5] -> [N]
        b = second_order_features.pow(2).sum(dim=1).sum(dim=1)
        second_order_feature = 0.5 * (a - b)
        return self.bias + linear_feature + second_order_feature


class FMV2(nn.Module):
    def __init__(self, sparse_field_nums, dense_num_field, k=5):
        """
        :param sparse_field_nums: 所有稀疏特征对应的每个特征的类别数目，eg: [10000,3,100,50,1000000,5000,300]
        :param dense_num_field: 稠密特征的数量
        """
        super(FMV2, self).__init__()
        self.sparse_num_field = len(sparse_field_nums)  # 稀疏(离散的分类)特征的数量
        self.dense_num_field = dense_num_field  # 稠密(连续)特征的数量
        self.num_field = self.sparse_num_field + self.dense_num_field  # 总特征的数量
        self.bias = nn.Parameter(torch.zeros(1))
        # 一阶部分
        self.embedding_linear = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=1)
        self.dense_linear_w = nn.Parameter(torch.empty(1, self.dense_num_field, 1))
        # 二阶部分
        self.embedding_second_order = nn.Embedding(num_embeddings=sum(sparse_field_nums), embedding_dim=k)
        self.dense_second_order_w = nn.Parameter(torch.empty(1, self.dense_num_field, k))
        self.offsets = np.asarray((0, *np.cumsum(sparse_field_nums)[:-1]), dtype=np.int32)

        nn.init.normal_(self.dense_linear_w)
        nn.init.normal_(self.dense_second_order_w)

    def forward(self,
                sparse_x, dense_x, label=None
                ):
        """
        前向过程
        :param sparse_x: [N, sparse_num_field] 稀疏特征
        :param dense_x: [N, dense_num_field] 稠密特征
        :param label:  [N] 真实的标签，None表示没有给定，1表示user点击了product，0表示未点击
        :return:
        """
        # 0. 特征合并
        x = sparse_x  # [N,sparse_num_field]
        x = x + x.new_tensor(self.offsets)
        # 1. 1阶部分的计算
        sparse_linear_feature = self.embedding_linear(x)  # [N,sparse_num_field,1]
        # [1,dense_num_field, 1] * [N,dense_num_field,1] -> [N,dense_num_field,1]
        dense_linear_feature = self.dense_linear_w * dense_x.view(-1, self.dense_num_field, 1)  # [N,dense_num_field,1]
        # concat([[N,sparse_num_field,1], [N,dense_num_field,1]]) -> [N,num_field,1]
        linear_feature = torch.concat([sparse_linear_feature, dense_linear_feature], dim=1)
        # [N,sparse_num_field,1] -> [N,sparse_num_field] -> [N]
        linear_feature = linear_feature.squeeze(-1).sum(dim=1)

        # 2. 2阶部分的计算
        sparse_second_order_features = self.embedding_second_order(x)  # [N,sparse_num_field,5]
        # [1,dense_num_field, 5] * [N,dense_num_field,1] -> [N,dense_num_field,5]
        dense_second_order_features = self.dense_second_order_w * dense_x.view(-1, self.dense_num_field,
                                                                               1)  # [N,dense_num_field,5]
        # concat([[N,sparse_num_field,5], [N,dense_num_field,5]]) -> [N,num_field,5]
        second_order_features = torch.concat([sparse_second_order_features, dense_second_order_features], dim=1)
        # [N,num_field,5] -> [N,5] -> [N,5] -> [N]
        a = second_order_features.sum(dim=1).pow(2).sum(dim=1)
        # [N,num_field,5] -> [N,num_field,5] -> [N,5] -> [N]
        b = second_order_features.pow(2).sum(dim=1).sum(dim=1)
        second_order_feature = 0.5 * (a - b)

        return self.bias + linear_feature + second_order_feature


def t0():
    N = 4
    sparse_field_nums = [10000, 3, 100, 50, 1000000, 5000, 300]
    fm_net = FMV0(sparse_field_nums=sparse_field_nums)

    user_id = torch.randint(sparse_field_nums[0], (N,))
    user_gender = torch.randint(sparse_field_nums[1], (N,))
    user_occupation = torch.randint(sparse_field_nums[2], (N,))
    user_address = torch.randint(sparse_field_nums[3], (N,))
    product_id = torch.randint(sparse_field_nums[4], (N,))
    product_store_id = torch.randint(sparse_field_nums[5], (N,))
    product_category_id = torch.randint(sparse_field_nums[6], (N,))

    r = fm_net(
        user_id, user_gender, user_occupation, user_address,
        product_id, product_store_id, product_category_id
    )
    print(r)
    print(r.shape)


def t1():
    N = 4
    sparse_field_nums = [10000, 3, 100, 50, 1000000, 5000, 300]
    fm_net = FMV1(sparse_field_nums=sparse_field_nums)

    user_id = torch.randint(sparse_field_nums[0], (N,))
    user_gender = torch.randint(sparse_field_nums[1], (N,))
    user_occupation = torch.randint(sparse_field_nums[2], (N,))
    user_address = torch.randint(sparse_field_nums[3], (N,))
    product_id = torch.randint(sparse_field_nums[4], (N,))
    product_store_id = torch.randint(sparse_field_nums[5], (N,))
    product_category_id = torch.randint(sparse_field_nums[6], (N,))
    sparse_x = torch.stack([
        user_id, user_gender, user_occupation, user_address,
        product_id, product_store_id, product_category_id
    ], dim=-1)

    r = fm_net(sparse_x)
    print(r)
    print(r.shape)


def t2():
    N = 4
    sparse_field_nums = [10000, 3, 100, 50, 1000000, 5000, 300]
    dense_num_field = 3
    fm_net = FMV2(sparse_field_nums=sparse_field_nums, dense_num_field=dense_num_field)

    sparse_x = torch.randint(3, (N, len(sparse_field_nums)))
    dense_x = torch.randn((N, dense_num_field))
    r = fm_net(sparse_x, dense_x)
    print(r)
    print(r.shape)


if __name__ == '__main__':
    t2()
