# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import matrixcf
import redis
# load id2index
lines = open("../data/index")


class predict(object):

    def __init__(self):

        self.id2index = {}
        for line in lines:
            line = line.strip().split("\t")
            self.id2index[line[0]] = int(line[1])

        # load model
        self.mcf = matrixcf.MCF()
        self.mcf.load_model()

        # conn
        pool = redis.ConnectionPool(host='127.0.0.1', port='6379', db=0)
        self.r = redis.Redis(connection_pool=pool)

    def recall(self, user):
        recall_result01 = self.r.get(user)

        return recall_result01.decode().split(",")

    def sort(self, user, recall_result):
        X = {"user": [], "item": []}
        for item in recall_result:
            X['user'].append([self.id2index[user]])
            X['item'].append([self.id2index[item]])
        res = self.mcf.infer(X)
        res = res['pred'].numpy().tolist()
        result = {}
        for i in range(len(X["item"])):
            result[recall_result[i]] = res[i]
        return result


# 测试用户 60414487256349587

# server
from flask import Flask, request
import json

app = Flask(__name__)
pred = predict()

# curl http://127.0.0.1:5000/predict -d '{"user_id":"60414487256349587","type":"recall"}'
# curl http://127.0.0.1:5000/predict -d '{"user_id":"60414487256349587","type":"sorted"}'
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
        if type == "sorted":
            res = pred.recall(user_id)
            res = pred.sort(user_id, res)
            print(res)
            return json.dumps(res)


if __name__ == "__main__":
    app.run(debug=True)
