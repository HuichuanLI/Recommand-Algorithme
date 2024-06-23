# -*- coding:utf-8 –*-
__author__ = 'Dodd'

import unit
import FP_Grow_tree


class tree:
    frequent = []

    def __init__(self, headnode, headtable, support, a):
        self.a = a
        self.headnode = headnode
        self.headtable = headtable
        self.support = support
        # print('ji:')
        # print(a)
        # tree.printTree(self.headnode)
        # tree.printheadtable(self.headtable)
        pass

    def printfrequent(self):
        y = sorted(tree.frequent, key=lambda x: x[1], reverse=True)
        for x in y:
            print(x)
            pass
        print(len(y))

    def FP_growth(self, headnode, headtable):
        a = self.a
        if tree.checkTreeOneWay(headnode):
            add = unit.generateCombination(headtable, a, self.support)
            if len(add) > 0:
                tree.frequent += add
            # print('frequent')
            # print(tree.frequent)
            pass
        else:
            for item in headtable:
                # datas为条件模式基
                datas = unit.generateSubset(headtable, item, self.a, tree.frequent)
                if datas:
                    # print(item)
                    if item:
                        x = a[:]
                        x.append(item)
                        f = FP_Grow_tree.FP_Grow_tree(datas, x, self.support)
                        # print('----------------ddddd-')
                        # print(f.f.pretable)
                        for jix in f.f.pretable:
                            xx = a[:]
                            xx.append(item)
                            xx.append(jix[0])
                            tree.frequent.append((",".join(str(i) for i in xx), jix[1]))
                            pass
                pass
            pass
        pass

    def checkTreeOneWay(nodex):
        nodesx = nodex
        # print(nodesx)
        while nodesx:
            # print(nodesx)
            if len(nodesx.child) > 1:
                return False
            if len(nodesx.child) > 0:
                nodesx = nodesx.child[0]
            if len(nodesx.child) == 0:
                break
            nodesx = nodesx.child[0]
        return True

    def printTree(node):
        if len(node.child) != 0:
            print(node.name + str(node.count) + 'p   ' + node.parent.name if node.parent else 'not')
            for nodes in node.child:
                tree.printTree(nodes)
                pass
        else:
            print(node.name + str(node.count) + 'p   ' + node.parent.name if node.parent else 'not')
        print('--------------')

    def printheadtable(headtable):
        print(headtable)
        for x in headtable:
            print(headtable[x])
            y = headtable[x]
            i = 0
            print(x)
            while y.next:
                y = y.next
                print(y)
                i += 1
                pass
            print(i)
            pass
