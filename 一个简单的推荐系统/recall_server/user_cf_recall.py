# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/1|下午 03:02
# @Moto   : Knowledge comes from decomposition

import math
from tqdm import tqdm
import pandas as pd
import redis
import traceback
import json
from collections import defaultdict


def save_redis(items, db=1):
    redis_url = 'redis://127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items.items():
            pool.set(item[0], json.dumps(item[1]))
    except:
        traceback.print_exc()


# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['timestamp']))

    click_df = click_df.sort_values('timestamp')
    item_user_time_df = click_df.groupby('article_id')[
        'user_id', 'timestamp'].apply(lambda x: make_user_time_pair(x)) \
        .reset_index().rename(columns={0: 'user_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['article_id'],
                                   item_user_time_df['user_time_list']))
    return item_user_time_dict


# def get_user_activate_degree_dict(click_df):
#     all_click_df_ = click_df.groupby('user_id')[
#         'article_id'].count().reset_index()
#
#     # 用户活跃度归一化
#     mm = MinMaxScaler()
#     all_click_df_['article_id'] = mm.fit_transform(
#         all_click_df_[['article_id']])
#     user_activate_degree_dict = dict(
#         zip(all_click_df_['user_id'], all_click_df_['article_id']))
#
#     return user_activate_degree_dict


def user_cf_sim(
        item_user_time_dict, pool, cut_off=20):
    """todo: 用户与用户之间的相似性矩阵计算
        用户活跃度，两个用户活跃度的均值越大权重越大
    """
    # 定义一个缓存信息的缓存区
    user_info = {}
    # 计算用户的相似度
    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善

                user_u_info = user_info.get(u, None)
                if user_u_info is None:
                    user_u_info = json.loads(pool.get(str(u)))
                    user_info[u] = user_u_info
                user_v_info = user_info.get(v, None)
                if user_v_info is None:
                    user_v_info = json.loads(pool.get(str(v)))
                    user_info[v] = user_v_info

                activate_weight = 0.1 * 0.5 * (len(user_v_info['hists']) +
                                               len(user_u_info['hists']))
                u2u_sim[u][v] += activate_weight / math.log(
                    len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        tmp = {}
        for v, wij in related_users.items():
            tmp[v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])
        u2u_sim_[u] = sorted(
            tmp.items(), key=lambda _: _[1], reverse=True)[:cut_off]

    # 将得到的相似性矩阵保存到本地
    save_redis(u2u_sim_, db=5)

    return u2u_sim_


def main():
    # 定义
    data_path = "../data/"
    # 读取数据
    click_df = pd.read_csv(data_path + '/click_log.csv')
    print("user history gen ...")
    item_user_time_dict = get_item_user_time_dict(click_df)
    print("user history end")

    # user_activate_degree_dict = get_user_activate_degree_dict(click_df)
    # print("user_activate_degree_dict size: ", len(user_activate_degree_dict))
    redis_url = 'redis://127.0.0.1:6379/1'
    pool = redis.from_url(redis_url)
    print("get iu2u matrix ...")
    user_cf_sim(item_user_time_dict, pool, cut_off=20)
    print("get u2u matrix end")


if __name__ == '__main__':
    main()
