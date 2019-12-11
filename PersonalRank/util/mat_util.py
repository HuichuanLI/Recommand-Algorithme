from scipy.sparse import coo_matrix

import numpy as np
import util.read as read


def graph_to_m(graph):
    """
    Args:
        graph:user item graph
    Return:
        a coo_matrix, sparse mat M
        a list, total user item point
        a dict, map all the point to row index
    """

    # 定点矩阵
    vertex = graph.keys()
    # 顶点位置
    address_dict = {}
    vertex = np.array(vertex).tolist()

    for index, v in enumerate(vertex):
        address_dict[v] = index
    row = []
    col = []
    data = []
    total_len = len(vertex)

    # 对于每个行向量 其实对于出度就是M的值
    for element_i in graph:
        weight = round(1 / len(graph[element_i]), 3)
        row_index = address_dict[element_i]
        for element_j in graph[element_i]:
            col_index = address_dict[element_j]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    m = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    # vertex全部的顶点
    # address_dict 顶点对于矩阵的位置
    # m 为稀疏矩阵
    return m, vertex, address_dict


def mat_all_point(m_mat, vertex, alpha):
    """
    get E-alpha*m_mat.T
    Args:
        m_mat:
        vertex: total item and user point
        alpha: the prob for random walking 随机游走的概率
    Return:
        a sparse
    """

    total_len = len(vertex)
    row = []
    col = []
    data = []
    # 处理稀疏矩阵的方法
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    eye_t = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    # tocsr 加速计算
    return eye_t.tocsr() - alpha * m_mat.tocsr().transpose()


if __name__ == "__main__":
    graph = read.get_graph_from_data("../data/log.txt")
    m, vertex, address_dict = graph_to_m(graph)
    print(mat_all_point(m, vertex, 0.8).todense())

    print(m.todense())
