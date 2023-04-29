import numpy as np
import random
import hashlib


# 取md5 hash
def getMd5Hash(band):
    hashobj = hashlib.md5()
    hashobj.update(band.encode())
    hashValue = hashobj.hexdigest()
    return hashValue


# 得到hash字典
def getHashBuket(sigMatrix, r):
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


# 得到签名矩阵
def getSigMatricx(input_matrix, n):
    result = []
    for i in range(n):
        sig = doSig(input_matrix)
        result.append(sig)
    return np.array(result)


# 一次签名
def doSig(inputMatrix):
    '''
    :param inputMatrix: 传入共现矩阵
    :return: 一次置换得到的签名
    '''
    # 生成一个行index组成的列表
    seqSet = [i for i in range(inputMatrix.shape[0])]
    # 生成一个长度为列数,值为-1的列表
    result = [-1 for i in range(inputMatrix.shape[1])]
    count = 0
    while len(seqSet) > 0:
        randomSeq = random.choice(seqSet)  # 随机选择一个序号
        for i in range(inputMatrix.shape[1]):  # 遍历所有数据在那一行的值
            # 如果那一行的值为1，且result列表中对应位置的值仍为-1。（意为还没赋过值）
            if inputMatrix[randomSeq][i] != 0 and result[i] == -1:
                # 则将那一行的序号赋值给result列表中对应的位置
                result[i] = randomSeq
                count += 1
        # 当count数量等于数据长度后说明result中的值均不为-1，意味着均赋过值了，所以跳出循环。
        if count == inputMatrix.shape[1]:
            break
        # 一轮下来result列表没收集出足够的数值则继续循环，但不会再选择刚那一行。
        seqSet.remove(randomSeq)

    return result


# 去重及去除子集
def __deleteCopy(group, copy, g1):
    for g2 in group:
        if g1 != g2:
            if set(g1) - set(g2) == set():
                copy.remove(g1)
                return


# 将相似item聚类起来
def sepGroup(hashBuket):
    group = set()
    for v in hashBuket.values():
        group.add(tuple(v))

    copy = group.copy()

    for g1 in group:
        __deleteCopy(group, copy, g1)

    return copy


def minhash(dataset, b, r):
    inputMatrix = np.array(dataset).T  # 将dataset转置一下
    sigMatrix = getSigMatricx(inputMatrix, b * r)  # 得到签名矩阵
    hashBuket = getHashBuket(sigMatrix, r)  # 得到hash字典
    groups = sepGroup(hashBuket)  # 将相似item聚类起来
    return groups


if __name__ == '__main__':
    documents = [[1, 1, 0, 1, 1, 1],
                 [1, 1, 0, 1, 1, 1],
                 [0, 0, 1, 1, 1, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 1, 0],
                 [1, 1, 1, 1, 1, 0]]

    groups = minhash(documents, b=20, r=5)
    print(groups)
