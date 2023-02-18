import numpy as np


def svd(data, k):
    u, i, v = np.linalg.svd(data)  # numpy里的SVD函数
    u = u[:, 0:k]
    i = np.diag(i[0:k])
    v = v[0:k, :]
    return u, i, v


def predictSingle(u_index, i_index, u, i, v):
    return u[u_index].dot(i).dot(v.T[i_index].T)


def play():
    k = 3
    # 假设用户物品共现矩阵如下
    data = np.mat([[1, 2, 3, 1, 1],
                   [1, 3, 2.1, 1, 2],
                   [3, 1, 1, 2, 1],
                   [1, 2, 3, 3, 1]])
    u, i, v = svd(data, k)
    print(u.dot(i).dot(v))

    print(predictSingle(2, 1, u, i, v))


'''
[[0.81676129 2.19400953 2.66841606 1.18997743 1.29544867]
 [1.11125826 2.88220195 3.20133002 0.88465015 1.82061047]
 [2.99284782 1.00757258 0.98705761 2.0074152  1.01153196]
 [1.04537234 1.95196066 3.0821046  2.95295906 0.92684298]]
'''

if __name__ == '__main__':
    play()
