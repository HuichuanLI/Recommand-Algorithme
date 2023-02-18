import numpy as np


# CN相似度 (common neighbors)
def CN(set1, set2):
    return len(set1 & set2)


# Jaccard相似度
def Jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


# 两个向量间的cos相似度
def cos4vector(v1, v2):
    return (np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 两个集合间的cos相似度
def cos4set(set1, set2):
    return len(set1 & set2) / (len(set1) * len(set2)) ** 0.5


# 两个向量间的pearson相似度
def pearson(v1, v2):
    v1_mean = np.mean(v1)
    v2_mean = np.mean(v2)
    return (np.dot(v1 - v1_mean, v2 - v2_mean)) / \
        (np.linalg.norm(v1 - v1_mean) *
         np.linalg.norm(v2 - v2_mean))


# 两个向量间的pearson相似度
def pearsonSimple(v1, v2):
    v1 -= np.mean(v1)
    v2 -= np.mean(v2)
    return cos4vector(v1, v2)  # 调用余弦相似度函数


if __name__ == '__main__':
    a = {1, 2, 3}
    b = {2, 3, 4}

    a = [1, 3, 2]
    b = [8, 9, 1]

    # a = [ 1, 2, 3 ]
    # b = [ 8, 1, 9 ]

    print(cos4vector(a, b))
    print(pearson(a, b))
    print(pearsonSimple(a, b))
    # #print(pearsonSimple(b, c))
    # #print(pearsonSimple(a, c))
