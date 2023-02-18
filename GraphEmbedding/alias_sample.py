# -*- coding: utf-8 -*-
# @Time    : 2021/10/24 5:10 PM
# @Author  : zhangchaoyang
# @File    : alias_sample.py

import numpy as np
import math


class AliasSample(object):
    def __init__(self, ratio_list):
        '''
        :param ratio_list: 各事件的概率（概率之和需要等于1）
        '''
        if math.fabs(math.fsum(ratio_list) - 1) > 1e-5:
            raise Exception("各事件概率之和不等于1")
        n = len(ratio_list)  # 事件的个数
        self.accept, self.alias = [0] * n, [0] * n
        small, large = [], []
        # 各概率乘以n
        ratio_list = np.array(ratio_list) * n

        # 概率小于1的放入small栈，大于1的放入large栈
        for i, prob in enumerate(ratio_list):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)
        # 分别从2个栈顶取出一个large和一个small，small不足1的部分由large来填充，large分割出去一部分后如果自身比1还小则进入small栈，如果比1还大再放回large栈
        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            self.accept[small_idx] = ratio_list[small_idx]
            self.alias[small_idx] = large_idx
            ratio_list[large_idx] -= (1 - ratio_list[small_idx])
            if ratio_list[large_idx] < 1.0:
                small.append(large_idx)
            elif ratio_list[large_idx] > 1.0:
                large.append(large_idx)
        # 理论上2个栈会同时清空，但是要考虑计算机精度损失带来的误差
        # while large:
        #     large_idx = large.pop()
        #     self.accept[large_idx] = 1.0
        # while small:
        #     small_idx = small.pop()
        #     self.accept[small_idx] = 1.0

    def sample(self):
        '''
        采样，返回事件的索引
        '''
        n = len(self.accept)
        i = int(np.random.random() * n)
        r = np.random.random()
        if r < self.accept[i]:
            return i
        else:
            return self.alias[i]

    def sample_i(self, i):
        r = np.random.random()
        if r < self.accept[i]:
            return i
        else:
            return self.alias[i]


if __name__ == "__main__":
    ratio_list = [1 / 2, 1 / 3, 1 / 12, 1 / 12]
    sampler = AliasSample(ratio_list)
    LOOP = 1000000
    cnt_list = [0] * len(ratio_list)
    for i in range(LOOP):
        cnt_list[sampler.sample()] += 1
    for i, ele in enumerate(cnt_list):
        print(i, ele / LOOP, ratio_list[i])
