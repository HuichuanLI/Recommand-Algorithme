# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import double_tower
import tensorflow as tf


class online_recall():

    def __init__(self):
        # load index2id
        lines = open("../data/index")

        index2id = {}
        for line in lines:
            line = line.strip().split("\t")
            index2id[int(line[1])] = line[0]

        # load model
        self.mcf = double_tower.DoubleTower()
        self.mcf.load_model_item()
        self.mcf.load_model_user()
        test_ds = self.mcf.init_dataset("../data/test", is_train=False)
        item_emb_dic = {}
        try:
            test_ds = iter(test_ds)
            while True:
                ds = next(test_ds)
                ds.pop("ctr")
                ds.pop("user")
                ds.pop("utag1")
                ds.pop("utag2")
                item_index = ds['item'].numpy()
                res = self.mcf.infer_item(ds)
                item_emb = res.numpy()
                for k, v in zip(item_index, item_emb):
                    item_emb_dic[index2id[k[0]]] = v
        except (StopIteration, tf.errors.OutOfRangeError):
            print("The dataset iterator is exhausted")

        # show
        for k, v in item_emb_dic.items():
            print(k, v)
            break

        # vector server
        from sklearn.neighbors import BallTree
        import numpy as np
        self.item_id = []
        item_emb_numpy = []
        for k, v in item_emb_dic.items():
            self.item_id.append(k)
            item_emb_numpy.append(v)

        self.tree = BallTree(np.array(item_emb_numpy), leaf_size=10)
        dist, ind = self.tree.query([item_emb_numpy[0]], k=20)
        print(ind)

    def recall(self, user_feature_dic):
        res = self.mcf.infer_user(user_feature_dic)
        recall_res = []
        for re in res:
            dist, ind = self.tree.query([re], k=20)
            recall_res.append([str(self.item_id[i]) for i in ind[0]])
        return recall_res


class predict(object):

    def __init__(self):

        # index
        self.id2index = {}
        lines = open("../data/index")
        for line in lines:
            line = line.strip().split("\t")
            self.id2index[line[0]] = int(line[1])

        # user feature
        self.user_feature_dic = {}
        lines = open("../data/user_feature.dat", encoding="utf8")
        for line in lines:
            line = line.strip().split(",")
            self.user_feature_dic[line[0]] = [line[1], line[2]]

        # item feaure
        self.item_feature_dic = {}
        lines = open("../data/item_feature.dat", encoding="utf8")
        for line in lines:
            line = line.strip().split(",")
            self.item_feature_dic[line[0]] = [line[1], line[2], line[3]]
        # load model
        self.m = online_recall()

    def recall(self, user):
        X = {"user": [], "utag1": [], "utag2": []}
        uf = self.user_feature_dic.get(user, ["utag1", "utag2"])
        X['utag1'].append([self.id2index[uf[0]]])
        X['utag2'].append([self.id2index[uf[1]]])
        X['user'].append([self.id2index[user]])
        return self.m.recall(X)


# 测试用户 10000386113374244550

# server
from flask import Flask, request
import json

app = Flask(__name__)
pred = predict()


# curl http://127.0.0.1:5000/predict -d '{"user_id":"10000386113374244550","type":"recall"}'
@app.route("/predict", methods=["POST"])
def infer():

    return_dict = {}
    if request.get_data() is None:
        return_dict["errcode"] = 1
        return_dict["errdesc"] = "data is None"

    data = json.loads(request.get_data())
    user_id = data.get("user_id", None)
    type = data.get("type", "recall")
    if user_id is not None:
        if type == "recall":
            res = pred.recall(user_id)
            print(res)
            return json.dumps(res)
        


if __name__ == "__main__":
    app.run(debug=True)
