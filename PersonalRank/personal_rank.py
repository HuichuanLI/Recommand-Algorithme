import sys

# sys.path.append("../util")
import util.read as read
import util.mat_util as mat_util
from scipy.sparse.linalg import gmres
import numpy as np


def personal_rank(graph, root, alpha, iter_num, recom_num=10):
    """
    Args
        graph: user item graph
        root: the  fixed user for which to recom
        alpha: the prob to go to random walk
        iter_num:iteration num
        recom_num: recom item num
    Return:
        a dict, key itemid, value pr
    """

    rank = {}
    rank = {point: 0 for point in graph}
    rank[root] = 1
    recom_result = {}
    # 迭代次数
    for iter_index in range(iter_num):
        tmp_rank = {point: 0 for point in graph}
        for out_point, out_dict in graph.items():
            for inner_point, value in graph[out_point].items():
                tmp_rank[inner_point] += round(alpha * rank[out_point] / len(out_dict), 4)
            tmp_rank[root] += round(1 - alpha, 4)
        if tmp_rank == rank:
            print("out" + str(iter_index))
            break
        rank = tmp_rank
    right_num = 0
    for zuhe in sorted(rank.items(), key=lambda x: x[1], reverse=True):
        point, pr_score = zuhe[0], zuhe[1]
        if len(point.split('_')) < 2:
            continue
        if point in graph[root]:
            continue

        recom_result[point] = round(pr_score, 4)
        right_num += 1
        if right_num > recom_num:
            break
    return recom_result


def get_one_user_recom():
    """
    give one fix_user recom result
    """
    user = "1"
    alpha = 0.8
    graph = read.get_graph_from_data("./data/ratings.txt")
    iter_num = 100
    recom_result = personal_rank(graph, user, alpha, iter_num, 100)
    item_info = read.get_item_info("./data/movies.txt")

    for itemid in graph[user]:
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])
    print("result---")
    for itemid in recom_result:
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])
        print(recom_result[itemid])
    return recom_result


def personal_rank_mat(graph, root, alpha, recom_num=10):
    """
    Args:
        graph:user item graph
        root:the fix user to recom
        alpha:the prob to random walk
        recom_num:recom item num
    Return:
        a dict, key: itemid, value: pr score
    A*r = r0
    """
    m, vertex, address_dict = mat_util.graph_to_m(graph)
    vertex = np.array([v for v in vertex])
    if root not in address_dict:
        return {}
    score_dict = {}
    recom_dict = {}
    mat_all = mat_util.mat_all_point(m, vertex, alpha)
    index = address_dict[root]
    initial_list = [[0] for row in range(len(vertex))]
    initial_list[index] = [1]
    r_zero = np.array(initial_list)
    res = gmres(mat_all, r_zero, tol=1e-8)[0]
    for index in range(len(res)):
        point = vertex[index]
        if len(point.strip().split("_")) < 2:
            continue
        if point in graph[root]:
            continue
        score_dict[point] = round(res[index], 3)
    for zuhe in sorted(score_dict.items(), key=lambda v: v[1], reverse=True)[:recom_num]:
        point, score = zuhe[0], zuhe[1]
        recom_dict[point] = score
    return recom_dict


def get_one_user_by_mat():
    """
    give one fix user by mat
    """
    user = "1"
    alpha = 0.8
    graph = read.get_graph_from_data("./data/ratings.txt")
    recom_result = personal_rank_mat(graph, user, alpha, 100)
    return recom_result


if __name__ == "__main__":
    # recom_result_base = get_one_user_recom()
    recom_result_base = get_one_user_recom()
    recom_result_mat = get_one_user_by_mat()
    num = 0
    for ele in recom_result_base:
        if ele in recom_result_mat:
            num += 1
    print(num)
