# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/2/5|下午 10:18
# @Moto   : Knowledge comes from decomposition

import os
import redis
import pandas as pd
import traceback
import json


def save_redis(items, db=1):
    redis_url = 'redis://:123456@127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()


def ext_feature():
    names = ['goods_id', 'user_id', 'user_session', 'user_type', 'user_cate1',
             'user_cate2', 'user_cate3', 'user_env', 'label']

    raw_data_path = "../data2/rawdata"
    files_path = []
    for file in os.listdir(raw_data_path):
        files_path.append(os.path.join(raw_data_path, file))

    print("all file: ", files_path)

    files_path.sort()
    ds_list = []
    for f in files_path:
        ds_list.append(pd.read_csv(f, header=None, names=names))

    ds = pd.concat(ds_list, axis=0, ignore_index=True)
    ds = ds.astype(str)
    print(ds.head(5))

    # 抽取用户特征
    user_feature_dict = {}
    for info in zip(ds['user_id'], ds['user_session'],
                    ds['user_type'], ds['user_cate1'],
                    ds['user_cate2'], ds['user_cate3'], ds['user_env']):
        user_feature_dict[info[0]] = json.dumps({
            "user_id": info[0],
            "user_session": info[1],
            "user_type": info[2],
            "user_cate1": info[3],
            "user_cate2": info[4],
            "user_cate3": info[5],
            "user_env": info[6],
        })
    save_redis(user_feature_dict.items(), 1)

    # 抽取物品特征
    item_feature_dict = {}
    for info in zip(ds['goods_id']):
        item_feature_dict[info[0]] = json.dumps({
            "goods_id": info[0],
        })
    save_redis(item_feature_dict.items(), 2)


if __name__ == '__main__':
    print('gen user and item feature ...')
    ext_feature()
