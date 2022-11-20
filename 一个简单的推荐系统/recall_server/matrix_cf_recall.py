# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/1|上午 11:29
# @Moto   : Knowledge comes from decomposition

from tqdm import tqdm
from sklearn import neighbors
import numpy as np
import pandas as pd
import warnings
import redis
import traceback
import json

warnings.filterwarnings('ignore')


def save_redis(items, db=1):
    redis_url = 'redis://127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()


# 读取文件
def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")]
            dic[int(tmp[0])] = embedding
    return dic


# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embedding_sim(item_emb_file, cut_off=20):
    """todo: 思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章，
         只不过由于文章数量太多，这里用了faiss进行加速
    """
    # 文章索引与文章id的字典映射

    item_embedding = read_embedding_file(item_emb_file)
    item_idx_2_rawid_dict = {}
    item_emb_np = []
    for i, (k, v) in enumerate(item_embedding.items()):
        item_idx_2_rawid_dict[i] = k
        item_emb_np.append(v)

    item_emb_np = np.asarray(item_emb_np)

    item_emb_np = item_emb_np / np.linalg.norm(
        item_emb_np, axis=1, keepdims=True)

    # 建立faiss/BallTree索引
    print("start build tree ... ")
    item_tree = neighbors.BallTree(item_emb_np, leaf_size=40)
    print("build tree end")
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_tree.query(item_emb_np, cut_off)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_emb_sim_dict = {}
    for target_idx, sim_value_list, rele_idx_list in tqdm(
            zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        sim_tmp = {}
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            sim_tmp[rele_raw_id] = sim_value
        item_emb_sim_dict[target_raw_id] = sorted(
            sim_tmp.items(), key=lambda _: _[1], reverse=True)[:cut_off]

    # 保存i2i相似度矩阵
    print("start saved ...")
    item_simi_tuple = [(_, json.dumps(v)) for _, v in item_emb_sim_dict.items()]
    save_redis(item_simi_tuple, db=3)
    print("saved end")


if __name__ == '__main__':
    data_path = "../data/"
    item_emb_file = data_path + 'matrixcf_articles_emb.csv'
    embedding_sim(item_emb_file, 20)
