# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import project5.code.mmoe as mmoe
# load id2index
import requests
import json


class predict(object):

    def __init__(self):

        # index
        self.id2index = {}
        lines = open("../data/index")
        for line in lines:
            line = line.strip().split("\t")
            self.id2index[line[0]] = int(line[1])

        # load model
        self.dcnModel = mmoe.DCN()
        self.dcnModel.load_model()

    #
    def split_float(self, x):
        if x < 2:
            return 0
        if x < 4:
            return 1
        if x < 6:
            return 2
        if x < 8:
            return 3
        if x < 10:
            return 4
        return 5

    def sort(self, user, recall_result):
        X = {
            "uid": [],
            "utag1": [],
            "utag2": [],
            "utag3": [],
            "utag4": [],
            "iid": [],
            "itag1": [],
            "itag2": [],
            "itag3": [],
            "itag4": [],
        }

        # step1 获取特征
        URL = "http://127.0.0.1:1001/feature"
        headers = {'Content-Type': 'application/json'}
        user_data = json.dumps({"uid": user})
        userf = requests.post(URL, data=user_data, headers=headers)
        userf = json.loads(userf.text)

        item_data = json.dumps({"iids": recall_result})
        itemf = requests.post(URL, data=item_data, headers=headers)
        itemf = json.loads(itemf.text)

        # step2 特征处理
        for res_json in itemf:
            for k, v in userf.items():
                vTmp = k + "=" + str(v)
                tmp = self.id2index.get(vTmp, 0)

                X[k].append([tmp])
            for k, v in res_json.items():
                if k == "itag4":
                    v = self.split_float(v)
                vTmp = k + "=" + str(v)
                tmp = self.id2index.get(vTmp, 0)
                X[k].append([tmp])

        # step3 预估
        res = self.dcnModel.infer(X)
        res = res.numpy().tolist()
        result = {}
        for i in range(len(X["iid"])):
            result[recall_result[i]] = res[i]
        return result


# 测试用户 10000386113374244550

# server
from flask import Flask, request
import json

app = Flask(__name__)
pred = predict()


# curl http://127.0.0.1:1002/predict -d '{"uid":"1","iids":["1", "2", "3"]}'
@app.route("/predict", methods=["POST"])
def infer():

    return_dict = {}
    if request.get_data() is None:
        return_dict["errcode"] = 1
        return_dict["errdesc"] = "data is None"

    data = json.loads(request.get_data())
    uid = data.get("uid", None)
    iids = data.get("iids", None)
    if uid is not None and iids is not None:
        res = pred.sort(uid, iids)
        print(res)
        return json.dumps(res)


if __name__ == "__main__":
    app.run(debug=True, port=1002)
