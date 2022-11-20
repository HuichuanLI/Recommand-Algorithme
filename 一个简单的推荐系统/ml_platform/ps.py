# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/11/10|16:35
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class Singleton(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._instance[cls]


# 定义一个参数服务k-v机构map{hashcode, embedding}
class PS(metaclass=Singleton):
    def __init__(self, embedding_dim):
        np.random.seed(2020)
        self.params_server = dict()
        self.dim = embedding_dim
        print("ps inited...")

    def pull(self, keys):
        values = []
        # 这里传进来的数据是[batch, feature_len]
        for k in keys:
            tmp = []
            for arr in k:
                value = self.params_server.get(arr, None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.params_server[arr] = value
                tmp.append(value)
            values.append(tmp)

        return np.asarray(values, dtype='float32')

    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.params_server[keys[i][j]] = values[i][j]

    def delete(self, keys):
        for k in keys:
            self.params_server.pop(k)

    def save(self, path):
        print("总共包含keys: ", len(self.params_server))
        writer = open(path, "w")
        for k, v in self.params_server.items():
            writer.write(
                str(k) + "\t" + ",".join(['%.8f' % _ for _ in v]) + "\n")
        writer.close()


if __name__ == '__main__':
    # 测试下ps的各个功能

    ps_local = PS(8)
    keys = [123, 234]
    # 从参数服务pull keys,如果参数服务中有这个key就直接取出，若没有就随机初始取出
    res = ps_local.pull(keys)
    print(ps_local.params_server)
    print(res)

    # 经过模型迭代更新后，传入参数服务器中
    gradient = 10
    res = res - 0.01 * gradient
    ps_local.push(keys, res)
    print(ps_local.params_server)

    # 经过上述多轮的pull参数，然后梯度更新后，获得最终的key对应的向量embedding
    # 保存向量，该向量用于召回
    # path = "F:\\6_class\\first_project\\feature_embedding"
    # ps_local.save(path)
