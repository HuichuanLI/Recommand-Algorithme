# -*- coding:utf-8 –*-
__author__ = 'Dodd'

import fptree
import tree


class FP_Grow_tree:
    def __init__(self, datas, a, support):
        self.f = fptree.fptree(datas, support)
        self.f.fp_tree()  # 项头表的构建
        self.f.tree = tree.tree(self.f.getRootTree(), self.f.headtable, support, a)
        # print(a)
        self.f.tree.FP_growth(self.f.headnode, self.f.headtable)
        pass

    def printfrequent(self):
        self.f.tree.printfrequent()
        pass
