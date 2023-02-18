# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/1|下午 02:05
# @Moto   : Knowledge comes from decomposition


import math
from tqdm import tqdm
import redis
import traceback
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import json

warnings.filterwarnings('ignore')


def save_redis(items, db=1):
    redis_url = 'redis://127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items.items():
            pool.set(item[0], json.dumps(item[1]))
    except:
        traceback.print_exc()


# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('timestamp')

    def make_item_time_pair(df):
        return list(zip(df['article_id'], df['timestamp']))

    user_item_time_df = click_df.groupby('user_id')[
        'article_id', 'timestamp'].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(
        zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


def item_cf_sim(user_item_time_dict, pool, cut_off=20):
    """ todo: 文章与文章之间的相似性矩阵计算
          点击时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
          两篇文章的类别的权重，其中类别相同权重大
    """
    # 定义一个缓存item信息的缓存区
    item_info = {}
    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                # 点击时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
                click_time_weight = np.exp(
                    0.7 ** np.abs(i_click_time - j_click_time))

                # 两篇文章的类别的权重，其中类别相同权重大
                item_i_info = item_info.get(i, None)
                if item_i_info is None:
                    item_i_info = json.loads(pool.get(str(i)))
                    item_info[i] = item_i_info
                item_j_info = item_info.get(j, None)
                if item_j_info is None:
                    item_j_info = json.loads(pool.get(str(j)))
                    item_info[j] = item_j_info

                type_weight = 1.0 if item_i_info['category_id'] == item_j_info[
                    'category_id'] else 0.7

                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += \
                    click_time_weight \
                    * type_weight \
                    / math.log(len(item_time_list) + 1)

    print("item_info get nums: ", len(item_info))
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        tmp = {}
        for j, wij in related_items.items():
            tmp[j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
        i2i_sim_[i] = sorted(
            tmp.items(), key=lambda _: _[1], reverse=True)[:cut_off]

    # 将得到的相似性矩阵保存到本地
    save_redis(i2i_sim_, db=4)


def main():
    # 定义
    data_path = "../data/"
    click_df = pd.read_csv(data_path + '/click_log.csv')

    print("user history gen ...")
    user_item_time_dict = get_user_item_time(click_df)
    print("user history end")

    redis_url = 'redis://127.0.0.1:6379/2'
    pool = redis.from_url(redis_url)
    print("get i2i matrix ...")
    item_cf_sim(user_item_time_dict, pool, cut_off=200)
    print("get i2i matrix end")


if __name__ == '__main__':
    main()
