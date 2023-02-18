import dataloader
from data_set import filepaths as fp
import s34_userCF_01label as userCF
from data_set import filepaths as fp
from tqdm import tqdm  # 进度条的库
import basic_sim as b_sim
import collections


def getSet(triples):
    # 已物品为索引，喜欢物品的用户集
    item_users = collections.defaultdict(set)
    user_items = collections.defaultdict(set)
    for u, i, r in triples:
        if r == 1:
            item_users[i].add(u)
            user_items[u].add(i)
    return item_users, user_items


# 得到基于相似物品的推荐列表
def get_recomedations_by_itemCF(item_sims, user_o_set):
    '''
    :param item_sims: 物品的近邻集:{样本1:[近邻1,近邻2，近邻3]}
    :param user_o_set: 用户的原本喜欢的物品集合:{用户1:{物品1,物品2，物品3}}
    :return: 每个用户的推荐列表{用户1:[物品1，物品2，物品3]}
    '''
    recomedations = collections.defaultdict(set)
    for u in user_o_set:
        for item in user_o_set[u]:
            # 将自己喜欢物品的近邻物品与自己观看过的视频去重后推荐给自己
            if item in item_sims:
                recomedations[u] |= set(item_sims[item]) - user_o_set[u]
    return recomedations


# 得到基于ItemCF的推荐列表
def trainItemCF(item_users, sim_method, user_items, k=5):
    item_sims = userCF.knn4set(item_users, k, sim_method)
    recomedations = get_recomedations_by_itemCF(item_sims, user_items)
    return recomedations


if __name__ == '__main__':
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    item_users, user_items = getSet(train_set)
    recomedations_by_itemCF = trainItemCF(item_users, b_sim.cos4set, user_items, k=5)
    print(recomedations_by_itemCF)
