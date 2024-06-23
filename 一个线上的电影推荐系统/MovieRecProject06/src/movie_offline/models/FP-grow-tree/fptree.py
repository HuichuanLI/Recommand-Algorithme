# -*- coding:utf-8 –*-
__author__ = 'Dodd'

import node
import tree
from operator import itemgetter, attrgetter


class fptree:
    def __init__(self, datas, support):
        self.pretable = {}
        self.datas = datas
        self.support = support
        self.headnode = node.node('null', None)
        self.tree = None
        self.headtable = {}

    def fp_tree(self):
        self.pretable = self.getpretable()
        self.pretable = [i for i in self.pretable if i[1] >= self.support]

    # print(self.pretable)
    def getpretable(self):
        pretable = {}
        for t in self.datas:
            for item in t:
                pretable.setdefault(item, 0);
                pretable[item] += 1
        return sorted(pretable.items(), key=itemgetter(1, 0), reverse=True)

    def getRootTree(self):
        nowheadtable = {}
        for t in self.datas:
            headnode = self.headnode
            for item in self.pretable:
                if item[0] in t:
                    thenode = headnode.findchildnode(item[0])
                    if not thenode:
                        thenode = node.node(item[0], headnode)
                        headnode.child.append(thenode)
                        self.headtable.setdefault(item[0], thenode)
                        if item[0] in nowheadtable.keys():
                            nowheadtable[item[0]].next = thenode
                            nowheadtable[item[0]] = thenode
                        nowheadtable.setdefault(item[0], thenode)
                    thenode.count += 1
                    headnode = thenode
        # print(headnode)
        # print('fds')
        return self.headnode
