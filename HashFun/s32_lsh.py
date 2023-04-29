import torch
import hashlib
import collections


# 取md5 hash
def getMd5Hash(band):
    hashobj = hashlib.md5()
    hashobj.update(band.encode())
    hashValue = hashobj.hexdigest()
    return hashValue


class LSH():

    def __init__(self, w, rows, bands):
        self.w = w
        self.r = rows
        self.b = bands

    # 得到签名矩阵
    def getSigMatrics(self, x):
        '''
        :param x: 输入的向量 [ batch_size, dim ]
        :return: 签名矩阵 [ 签名次数(rows * bands), batch_size ]
        '''
        n = self.r * self.b
        # 直接生成一个签名次数 * 向量维度的矩阵
        v = torch.rand((n, x.shape[1]))
        # 生成偏置项
        bias = torch.rand(n, 1) * self.w
        # 一步生成签名矩阵
        sm = (torch.matmul(v, x.T) + bias) // self.w
        return sm

    # 切分哈希桶
    def getHashBuket(self, sigMatrix, r):
        hashBuckets = {}
        begin = 0
        end = r
        b_index = 1  # 为了防止跨band匹配

        while end <= sigMatrix.shape[0]:
            for colNum in range(sigMatrix.shape[1]):
                # 将rows个签名与band index字符串合并后 取 md5 哈希
                band = str(sigMatrix[begin: end, colNum]) + str(b_index)
                hashValue = getMd5Hash(band)
                if hashValue not in hashBuckets:
                    hashBuckets[hashValue] = [colNum]
                elif colNum not in hashBuckets[hashValue]:
                    # 哈希值相同的分在同一个哈希桶内
                    hashBuckets[hashValue].append(colNum)
            begin += r
            end += r
            b_index += 1
        return hashBuckets

    # 去重及去除子集
    def __deleteCopy(self, group, copy, g1):
        for g2 in group:
            if g1 != g2:
                if set(g1) - set(g2) == set():
                    copy.remove(g1)
                    return

    # 将相似item聚类起来
    def sepGroup(self, hashBuket):
        group = set()
        for v in hashBuket.values():
            group.add(tuple(v))
        copy = group.copy()
        for g1 in group:
            self.__deleteCopy(group, copy, g1)
        return copy

    # 传入用户数量与聚类的分组得到最终给每个用户召回的物品集
    def doRecall(self, group, u_number):
        recall_dict = collections.defaultdict(set)
        # 得到用户索引集
        us = set(range(u_number))
        for i in group:
            i = set(i)
            ius = i & us
            if len(ius) > 0:  # 如分组中有用户索引则进行处理
                for u in ius:
                    # 给每个用户记录召回的物品索引
                    recall_dict[u] |= (i - ius)
        return recall_dict

    def getRecalls(self, u, x):
        # 将用户与物品向量拼起来
        ux = torch.cat([u, x], dim=0)
        # 将拼起来的向量一同得到签名矩阵
        sm = self.getSigMatrics(ux)
        # 根据签名矩阵进行哈希分桶
        hb = self.getHashBuket(sm, self.r)
        # 将相似向量聚类起来
        group = self.sepGroup(hb)
        # 得到用户数量
        u_number = u.shape[0]
        # 传入用户数量与聚类的分组得到最终给每个用户召回的物品集
        recall_dict = self.doRecall(group, u_number)
        return recall_dict


if __name__ == '__main__':
    x = [[8, 7, 6, 4, 8, 9],
         [7, 8, 5, 8, 9, 7],
         [3, 2, 0, 1, 2, 3],
         [3, 3, 2, 3, 3, 3],
         [21, 21, 22, 99, 2, 12],
         [1, 1, 1, 0, 1, 0],
         [1, 1, 1, 1, 1, 0]]
    u = torch.FloatTensor([[1, 1, 1, 1, 1, 0],
                           [3, 3, 2, 3, 3, 3]])
    x = torch.FloatTensor(x)
    lsh = LSH(4, 3, 6)

    recall_dict = lsh.getRecalls(u, x)

    # 注意用户与物品的向量拼接后，向量的索引位置用户在前，物品在后。
    print(recall_dict)
