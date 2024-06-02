# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def t0():
    # 考虑机器学习中，离散数据的学习方式
    user_id = 5  # 假设总用户数:10
    user_gender = 1  # 假设总性别数:3

    # 首先onehot处理
    user_id_onehot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    user_gender_onehot = [0, 1, 0]

    # 合并成一个特征
    x = user_id_onehot + user_gender_onehot
    user_id_w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    user_gender_w = [1.1, 1.2, 1.3]
    w = user_id_w + user_gender_w  # 假定的一个初始的参数w

    # 线性计算
    # 0*0.1 + 0*0.2 + 0*0.3 + 0*0.4 + 0*0.5 + 1*0.6 + 0*0.7 + 0*0.8 + 0*0.9 + 0*1.0 +
    # 0*1.1 + 1*1.2 + 0*1.3
    y = np.asarray(x) * np.asarray(w)
    print(y)
    y = np.sum(y)
    print(y)

    # 直接从各个属性的w中获取index对应位置的值
    user_id_y = user_id_w[user_id]
    user_gender_y = user_gender_w[user_gender]
    print(user_id_y, user_gender_y)
    print(user_id_y + user_gender_y)


def t1():
    user_embed = nn.Embedding(
        num_embeddings=10, embedding_dim=1,
        _weight=torch.reshape(torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]), (-1, 1))
    )
    user_embed2 = nn.Embedding(
        num_embeddings=10, embedding_dim=5
    )
    user_id = torch.tensor(5)  # 假设总用户数:10
    user_id_y = user_embed(user_id)
    print(user_id_y)
    print(user_embed2(user_id))



if __name__ == '__main__':
    t1()
