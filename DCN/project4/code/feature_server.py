# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

# server
from flask import Flask, request
import json

app = Flask(__name__)


class feature_server(object):

    def __init__(self):
        # user feature
        self.user_feature_dic = {}
        lines = open("../data/user_feature.dat", encoding="utf8")
        for line in lines:
            data = json.loads(line.strip())
            self.user_feature_dic[data["uid"]] = data

        # item feaure
        self.item_feature_dic = {}
        lines = open("../data/ad_feature.dat", encoding="utf8")
        for line in lines:
            data = json.loads(line.strip())
            self.item_feature_dic[data["iid"]] = data

    def get_user_feature(self, uid):
        return self.user_feature_dic.get(uid, None)

    def get_item_feature(self, iid):
        return self.item_feature_dic.get(iid, None)


app = Flask(__name__)
server = feature_server()

# curl http://127.0.0.1:1001/feature -d '{"uid":"1"}'
# curl http://127.0.0.1:1001/feature -d '{"iids":["1", "2"]}'
@app.route("/feature", methods=["POST"])
def infer():

    return_dict = {}
    if request.get_data() is None:
        return_dict["errcode"] = 1
        return_dict["errdesc"] = "data is None"

    data = json.loads(request.get_data())
    uid = data.get("uid", None)
    if uid is not None:
        res = server.get_user_feature(uid)
        return json.dumps(res)

    iids = data.get("iids", None)
    if iids is not None:
        res = []
        for i in iids:
            res.append(server.get_item_feature(i))
        return json.dumps(res)
    return "None"

if __name__ == "__main__":
    app.run(debug=True, port=1001)
