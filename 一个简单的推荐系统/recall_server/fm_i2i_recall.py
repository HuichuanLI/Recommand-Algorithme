# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/1|下午 07:08
# @Moto   : Knowledge comes from decomposition

import numpy as np
import pandas as pd
from sklearn import neighbors
import redis
import traceback
import json
from tqdm import tqdm


def bkdr2hash64(str01):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str01:
        hash = hash * seed + ord(s)
    return hash & mask60


# 读取文件
def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")][:-1]
            dic[tmp[0]] = embedding
    return dic


def get_hash2id(data_path):
    ds = pd.read_csv(data_path)
    ds = ds[['user_id', 'article_id', 'environment', 'region']]
    users = list(ds['user_id'].unique())
    items = list(ds['article_id'].unique())
    environment = list(ds['environment'].unique())
    region = list(ds['region'].unique())

    users_dict = {str(bkdr2hash64("user_id=" + str(u))): int(u) for u in users}
    items_dict = {
        str(bkdr2hash64("article_id=" + str(i))): int(i) for i in items}
    environment_dict = {
        str(bkdr2hash64("environment=" + str(i))): int(i) for i in environment}
    region_dict = {
        str(bkdr2hash64("region=" + str(i))): int(i) for i in region}
    return users_dict, items_dict, environment_dict, region_dict


def split_user_item(embedding_file, train_file):
    user_dict, item_dict, env_dict, region_dict = get_hash2id(train_file)
    embedding_dict = read_embedding_file(embedding_file)

    item_embedding = {}
    user_embedding = {}

    for k, v in embedding_dict.items():
        m_id = item_dict.get(k, None)
        if m_id is not None:
            item_embedding[m_id] = v
        u_id = user_dict.get(k, None)
        if u_id is not None:
            user_embedding["user_id=" + str(u_id)] = v
        env_ = env_dict.get(k, None)
        if env_ is not None:
            user_embedding["env=" + str(env_)] = v
        region_ = region_dict.get(k, None)
        if region_ is not None:
            user_embedding["region=" + str(region_)] = v
    print('item_embedding size: ', len(item_embedding))
    print('user_embedding size: ', len(user_embedding))
    return item_embedding, user_embedding


def save_redis(items, db=1):
    redis_url = 'redis://127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()


# 用于i2i模式
def embedding_sim(item_embedding, cut_off=20):
    # 向量进行正则化
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
    save_redis(item_simi_tuple, db=6)
    print("saved end")


def write_embeding(emb, file):
    wfile = open(file, 'w')
    for k, v in emb.items():
        wfile.write(str(k) + "\t" + ','.join([str(_) for _ in v]) + '\n')


if __name__ == '__main__':
    data_path = "../data/"
    embedding_file = data_path + "saved_dnn_embedding"
    train_file = data_path + "click_log.csv"
    item_embedding, user_embedding = split_user_item(embedding_file, train_file)
    write_embeding(item_embedding, data_path + "fm_articles_emb")
    write_embeding(user_embedding, data_path + "fm_user_emb")
    embedding_sim(item_embedding, 20)
