# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import dssm
import tensorflow as tf

# load index2id
lines = open("../data/index")

index2id = {}
for line in lines:
    line = line.strip().split("\t")
    index2id[int(line[1])] = line[0]

# load model
mcf = dssm.DSSM()
mcf.load_model()
test_ds = mcf.init_dataset("../data/test", is_train=False)
user_emb_dic = {}
item_emb_dic = {}
try:
    test_ds = iter(test_ds)
    while True:
        ds = next(test_ds)
        ds.pop("ctr")
        user_index = ds['user'].numpy()
        item_index = ds['item'].numpy()
        res = mcf.infer(ds)
        user_emb = res['user_emb'].numpy()
        item_emb = res['item_emb'].numpy()

        for k, v in zip(user_index, user_emb):
            user_emb_dic[index2id[k[0]]] = v
        for k, v in zip(item_index, item_emb):
            item_emb_dic[index2id[k[0]]] = v
except (StopIteration, tf.errors.OutOfRangeError):
    print("The dataset iterator is exhausted")

# show
for k, v in user_emb_dic.items():
    print(k, v)
    break

# vector server
from sklearn.neighbors import BallTree
import numpy as np
item_id = []
item_emb_numpy = []
for k, v in item_emb_dic.items():
    item_id.append(k)
    item_emb_numpy.append(v)

tree = BallTree(np.array(item_emb_numpy), leaf_size=10)
dist, ind = tree.query([item_emb_numpy[0]], k=20)
print(ind)
# write in redis and file
import redis
pool = redis.ConnectionPool(host='127.0.0.1', port='6379', db=3)
r = redis.Redis(connection_pool=pool)

recall_result_tofile = open("../data/recall.result", "w")
for k, v in user_emb_dic.items():
    dist, ind = tree.query([v], k=20)
    res = str(k) + "\t" + ",".join([str(item_id[i]) for i in ind[0]])
    r.set(str(k), ",".join([str(item_id[i]) for i in ind[0]]), nx=True)
    recall_result_tofile.write(res + "\n")
