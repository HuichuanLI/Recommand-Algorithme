# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 5:17 PM
# @Author  : zhangchaoyang
# @File    : confidence.py

import math

# 置信度到z值的对应关系
z_conf = {0.9: 1.64, 0.95: 1.96, 0.99: 2.58}


def var_bernoulli(total, positive):
    '''伯努力分布的样本方差'''
    mean = positive / total
    return (positive * (1 - mean) ** 2 + (total - positive) * mean ** 2) / (total - 1)


def significant_difference(total1, positive1, total2, positive2, confidence):
    z = z_conf.get(confidence)
    if not z:
        raise Exception("confidence must be 0.9, 0.95 or 0.99")
    mean1 = positive1 / total1
    mean2 = positive2 / total2
    var1 = var_bernoulli(total1, positive1)
    var2 = var_bernoulli(total2, positive2)
    bias = z * math.sqrt(var1 / total1 + var2 / total2)
    floor = (mean1 - mean2) - bias
    ceil = (mean1 - mean2) + bias
    print(floor, ceil)
    if ceil > 0 and floor > 0:
        return "1显著胜出"
    elif ceil < 0 and floor < 0:
        return "2显著胜出"
    else:
        return "无法得到显著结论"


if __name__ == "__main__":
    print(significant_difference(99458, 10045, 100134, 9282, 0.95))
    print(significant_difference(99705, 10268, 99458, 10045, 0.95))
