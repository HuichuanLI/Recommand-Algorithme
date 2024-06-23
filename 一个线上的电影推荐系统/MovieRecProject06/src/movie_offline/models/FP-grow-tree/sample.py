# -*- coding:utf-8 –*-

import FP_Grow_tree

sample = [
    ['milk', 'eggs', 'bread', 'chips'],
    ['eggs', 'popcorn', 'chips', 'beer'],
    ['eggs', 'bread', 'chips'],
    ['milk', 'eggs', 'bread', 'popcorn', 'chips', 'beer'],
    ['milk', 'bread', 'beer'],
    ['eggs', 'bread', 'beer'],
    ['milk', 'bread', 'chips'],
    ['milk', 'eggs', 'bread', 'butter', 'chips'],
    ['milk', 'eggs', 'butter', 'chips']
]
sample1 = [
    [u'牛奶', u'鸡蛋', u'面包', u'薯片'],
    [u'鸡蛋', u'爆米花', u'薯片', u'啤酒'],
    [u'鸡蛋', u'面包', u'薯片'],
    [u'牛奶', u'鸡蛋', u'面包', u'爆米花', u'薯片', u'啤酒'],
    [u'牛奶', u'面包', u'啤酒'],
    [u'鸡蛋', u'面包', u'啤酒'],
    [u'牛奶', u'面包', u'薯片'],
    [u'牛奶', u'鸡蛋', u'面包', u'黄油', u'薯片'],
    [u'牛奶', u'鸡蛋', u'黄油', u'薯片']
]
# print(sample1)
##参数说明 sample为事务数据集 []为递归过程中的基,support为最小支持度
support = 3
ff = FP_Grow_tree.FP_Grow_tree(sample1, [], support)
##打印频繁集
ff.printfrequent()
