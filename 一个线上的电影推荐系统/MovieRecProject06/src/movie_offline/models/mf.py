# -*- coding: utf-8 -*-
"""
MF的离线模型的训练以及推荐结果的产生
"""
import heapq
import os
from datetime import datetime

import numpy as np
import pandas as pd
from surprise import Reader, Dataset, KNNBaseline, accuracy, SVD
from surprise.model_selection import GridSearchCV
from tqdm import tqdm

from ..config import global_config
from ..utils.mysql_util import DB
from ..utils.redis_util import RedisUtil


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
def download_data(save_path):
    """
    从原始数据源(MySQL、Hive、HBase等)下载模型需要的数据，并保存到磁盘对应文件中
    :param save_path: 对应的文件路径
    :return:
    """
    # 创建输出文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 获取数据库里面的所有数据
    datas = DB.query_sql(
        sql="select user_id,movie_id,rating from user_movie_rating"
    )

    # 构建一个DataFrame
    df = pd.DataFrame(datas)
    df.columns = ['user_id', 'item_id', 'rating']
    df.to_csv(save_path, sep="\t", header=False, index=False)

    return save_path


def get_dataset(path):
    reader = Reader(
        line_format="user item rating",
        sep="\t",
        rating_scale=(1, 5),
        skip_lines=0,
    )
    data = Dataset.load_from_file(path, reader)
    return data


def _user_predict_item_to_redis(trainset, algo):
    top_k = 100
    eval_predictions = []
    # with open('mf.txt', 'w', encoding="utf-8") as writer:
    #     for iuid in tqdm(range(trainset.n_users)):
    #         uid = trainset.to_raw_uid(iuid)  # 将模型内部用户id转换为实际用户id
    #         user_item_rating_list = []
    #         user_view_iiid_list = {t[0]: t[1] for t in trainset.ur[iuid]}  # 当前用户已经评论过的商品内部id列表
    #         for iiid in range(trainset.n_items):
    #             iid = trainset.to_raw_iid(iiid)  # 将模型内部物品id转换为实际物品id
    #             if iiid in user_view_iiid_list:
    #                 y_ = algo.predict(uid, iid, r_ui=user_view_iiid_list[iiid])
    #                 eval_predictions.append(y_)  # 用来进行模型评估
    #             else:
    #                 rating = algo.predict(uid=uid, iid=iid).est  # 预测当前用户uid对当前物品iid的评分
    #                 user_item_rating_list.append((rating, iid))  # 临时保存当前用户对iid的评分rating
    #
    #         # 排序
    #         user_item_rating_list = heapq.nlargest(top_k, user_item_rating_list, key=lambda t: t[0])
    #         if len(user_item_rating_list) == 0:
    #             continue
    #
    #         # 输出
    #         rec_items = ",".join(map(lambda t: f"{t[1]}:{t[0]:.3f}", user_item_rating_list))
    #         writer.writelines(f"{uid}\t{rec_items}\n")

    with RedisUtil.get_redis() as redis:
        for iuid in tqdm(range(trainset.n_users)):
            uid = trainset.to_raw_uid(iuid)  # 将模型内部用户id转换为实际用户id
            user_item_rating_list = []
            user_view_iiid_list = {t[0]: t[1] for t in trainset.ur[iuid]}  # 当前用户已经评论过的商品内部id列表
            for iiid in range(trainset.n_items):
                iid = trainset.to_raw_iid(iiid)  # 将模型内部物品id转换为实际物品id
                if iiid in user_view_iiid_list:
                    y_ = algo.predict(uid, iid, r_ui=user_view_iiid_list[iiid])
                    eval_predictions.append(y_)  # 用来进行模型评估
                else:
                    rating = algo.predict(uid=uid, iid=iid).est  # 预测当前用户uid对当前物品iid的评分
                    user_item_rating_list.append((rating, iid))  # 临时保存当前用户对iid的评分rating

            # 排序
            user_item_rating_list = heapq.nlargest(top_k, user_item_rating_list, key=lambda t: t[0])
            if len(user_item_rating_list) == 0:
                continue

            # 输出
            rec_items = ",".join(map(lambda t: f"{t[1]}:{t[0]:.3f}", user_item_rating_list))
            redis.hset(
                name=f'rec:recall:u2i:{uid}',
                key='mf',
                value=rec_items
            )
            redis.expire(f'rec:recall:u2i:{uid}', 48 * 60 * 60)

    # 评估
    print(f"fcp:{accuracy.fcp(eval_predictions)}")
    print(f"mae:{accuracy.mae(eval_predictions)}")
    print(f"rmse:{accuracy.rmse(eval_predictions)}")


def _item_sim_item_to_redis(trainset, algo):
    # 针对商品获取相似商品列表，并保存到redis中
    with RedisUtil.get_redis() as redis:
        min_sim = 0.4
        top_k = 100
        tmp_records = []

        def save_to_redis():
            if len(tmp_records) == 0:
                return
            _expire = 2 * 24 * 60 * 60
            _pipe = redis.pipeline()
            for _name, _value in tmp_records:
                _pipe.hset(_name, 'mf', _value)
                _pipe.expire(_name, _expire)
            _pipe.execute()

        def calc_sim(v1, v2):
            a = 0.0
            b = 0.0
            c = 0.0
            for v1, v2 in zip(v1, v2):
                a += v1 * v2  # 分子
                b += v1 ** 2  # 分母1
                c += v2 ** 2  # 分母2
            return a / (np.sqrt(b) * np.sqrt(c))

        # 1. 获取商品特征向量矩阵
        item_features = algo.qi
        # 2. 遍历所有的物品，根据物品与物品的相似度，获取最匹配的k个商品保存
        for iiid1 in tqdm(range(trainset.n_items)):
            iid_sim_list = []  # 临时保存当前商品满足条件的其它相似商品列表
            for iiid2 in range(trainset.n_items):
                if iiid1 == iiid2:
                    continue
                iiid_sim = calc_sim(item_features[iiid1], item_features[iiid2])
                if iiid_sim < min_sim:
                    continue
                iid_sim_list.append((iiid_sim, trainset.to_raw_iid(iiid2)))  # 保存相似度和实际的商品id

            # 排序
            iid_sim_list = heapq.nlargest(top_k, iid_sim_list, key=lambda t: t[0])
            if len(iid_sim_list) == 0:
                continue

            # 存储
            iid = trainset.to_raw_iid(iiid1)
            sim_itemms = ",".join(map(lambda t: f"{t[1]}:{t[0]:.3f}", iid_sim_list))
            tmp_records.append((f'rec:recall:i2i:{iid}', sim_itemms))

            # 当临时列表的数目超过一定限制的时候，进行提交
            if len(tmp_records) > 500:
                save_to_redis()
                tmp_records = []

        # 3. 最后再保存一次
        save_to_redis()


def training(path):
    """
    基于给定路径训练对应的UserCF模型，并将训练结果保存到redis中
    :param path: 数据路径
    :return:
    """
    # 1. 加载数据
    dataset = get_dataset(path)
    # 数据转换（外部id转内部id）
    trainset = dataset.build_full_trainset()

    # 3. 模型创建及训练
    algo = SVD(
        n_factors=10,
        n_epochs=30,
        biased=True,
        lr_all=0.005,
        reg_all=0.02
    )
    algo.fit(trainset)

    # 基于协同过滤算法(UserCF、ItemCF、SVD)获取每个用户的推荐列表，
    #         并将每个用户的推荐列表保存到文件中，一行一个用户的推荐列表
    #         如果用户对商品有评分，那么该商品不应该作为推荐结果
    print("=" * 100)
    print("UserCF模型构建完成，开始针对所有用户产生推荐结果:")
    _user_predict_item_to_redis(trainset, algo)
    _item_sim_item_to_redis(trainset, algo)


def timed_scheduling():
    """
    定时调度，当需要更新模型的时候，直接调用这个方法即可
    :return:
    """
    now = datetime.now()
    data_path = os.path.join(
        global_config.model_root_dir, "tmp", "mf", now.strftime("%Y%m%d_%H%M%S"), "training.txt"
    )
    data_path = os.path.abspath(data_path)
    print(f"开始下载数据:{data_path}")
    download_data(data_path)
    print("数据下载完成，开始模型训练....")
    training(data_path)
    print("模型训练完成!")
