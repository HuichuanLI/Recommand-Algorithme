# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/12/31|下午 05:45
# @Moto   : Knowledge comes from decomposition

import redis
import pandas as pd
import traceback
import json


def save_redis(items, db=1):
    redis_url = 'redis://127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()


def get_user_feature():
    ds = pd.read_csv("../data/click_log.csv")
    print(ds)

    click_df = ds.sort_values('timestamp')
    # {199999: (4,13)}
    user_environment_region_dict = {}
    for info in zip(click_df['user_id'],
                    click_df['environment'], click_df['region']):
        user_environment_region_dict[info[0]] = (info[1], info[2])

    def make_item_time_pair(df):
        return list(zip(df['article_id'], df['timestamp']))

    # {199999, [(160417, 1507029570190), (5408, 1507029571478), (50823, 1507029601478)]}
    user_item_time_df = click_df.groupby('user_id')[
        'article_id', 'timestamp'].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(
        zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    user_feature = []
    for user, item_time_dict in user_item_time_dict.items():
        info = user_environment_region_dict[user]
        tmp = (str(user), json.dumps({
            'user_id': user,  # 199999
            'hists': item_time_dict,  # [(160417, 1507029570190), (5408, 1507029571478), (50823, 1507029601478)]
            'environment': info[0],  # 4
            'region': info[1],  # 13
        }))
        user_feature.append(tmp)

    save_redis(user_feature, 1)


def get_item_feature():
    ds = pd.read_csv("../data/articles.csv")
    ds = ds.to_dict(orient='records')
    item_feature = []
    # (1336, {'article_id': 1336, 'category_id': 1, 'created_at_ts': 1474660333000, 'words_count': 223})
    for d in ds:
        item_feature.append((d['article_id'], json.dumps(d)))

    save_redis(item_feature, 2)


if __name__ == '__main__':
    print('gen user feature ...')
    get_user_feature()
    print('gen item feature ...')
    get_item_feature()
