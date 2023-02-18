# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/2/24|上午 09:47
# @Moto   : Knowledge comes from decomposition


# 曝光量，点击量，已经点击率
import pandas as pd
import numpy as np
import redis
import traceback

ds = pd.read_csv(
    "../data3/raw/train/behaviors.tsv",
    names=['index_id', 'user_id', 'timestamp', 'hist', 'doc_id'], sep='\t')

ds = ds[['doc_id']]
print(ds.head())

doc_show_count = {}
doc_click_count = {}

for item in ds['doc_id'].values:
    tmp_iter = item.split()
    for tmp in tmp_iter:
        item, behavior = tmp.split('-')
        doc_click_count.setdefault(item, 0)
        doc_show_count.setdefault(item, 0)
        if behavior == '1':
            doc_click_count[item] += 1
        doc_show_count[item] += 1

item_show_click_dic = []
for doc, show in doc_show_count.items():
    click = doc_click_count.get(doc, 0)
    item_show_click_dic.append(
        {
            "doc": doc,
            "show": show,
            "click": click,
        }
    )

item_show_click = pd.DataFrame(item_show_click_dic)
print(item_show_click.describe())

# show
item_show_click = item_show_click[item_show_click['show'] > 288]
print(len(item_show_click))

# click
# 方法一，基于点击数进行倒排
#####归一化函数#####
reg = lambda x: x / np.max(x)
item_show_click['click_reg'] = item_show_click[['click']].apply(reg)
print(item_show_click.head())

item_click_count = {}
for d in item_show_click[['doc', 'click_reg']].values:
    item_click_count[d[0]] = d[1]

# 方法二，基于点击数和点击率的加权求和进行倒排
item_show_click['ctr'] = item_show_click['click'] / item_show_click['show']
print(item_show_click.head(30))

w1 = 0.3
w2 = 0.7
item_show_click['ctr_click'] = w1 * item_show_click['click_reg'] + w2 * \
                               item_show_click['ctr']
print(item_show_click.head(30))

item_ctr_click_count = {}
for d in item_show_click[['doc', 'ctr_click']].values:
    item_click_count[d[0]] = d[1]


def save_redis(items, db=1):
    redis_url = 'redis://:123456@127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items.items():
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()


save_redis(item_click_count, db=11)
save_redis(item_ctr_click_count, db=12)
