# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/7/9|10:25
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dcn import *
import json
import requests
# slots = 18
pb_path = "../data7/saved_model"
feature_embeddings = "../data1/saved_dnn_embedding"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_embed():
    hashcode_dict = {}

    with open(feature_embeddings, "r") as lines:
        for line in lines:
            tmp = line.strip().split("\t")
            if len(tmp) == 2:
                vec = [float(_) for _ in tmp[1].split(",")]
                hashcode_dict[tmp[0]] = vec

    return hashcode_dict


def batch_predict(line_arr):
    """
    input_data : 按 slotid的顺序组装好，以 \t 分割特征，多值用"," 分割
    curl -d '{"instances": [{"x1": [1]}]}' -X POST http://localhost:9001/v1/models/lr:predict
    """
    embed_dict = load_embed()
    dim = config['embedding_dim']
    input_size = config['feature_len'] * config['embedding_dim']

    url = "http://172.30.31.8:8501/v1/models/dcn:predict"

    init_arr = [0.0] * input_size
    batch = []
    result = []
    for line in line_arr:
        tmp_arr = init_arr.copy()
        for i, f in enumerate(line):
            tmp = embed_dict.get(f, None)
            if tmp is not None:
                tmp_arr[i * dim:(i + 1) * dim] = tmp
        batch.append({"input0": tmp_arr})
        if len(batch) > 20:
            instances = {"instances": batch}

            json_response = requests.post(url, data=json.dumps(instances))
            res = json.loads(json_response.text)
            for p in res['predictions']:
                result.append(p[0])
            batch = []
    return result


def main():
    lines = open("../data7/rawdata/test.csv.hash")
    line_arr = []
    for line in lines:
        line_arr.append(line.strip().split(","))
    print(len(line_arr))
    result = batch_predict(line_arr)

    wfile = open("../data7/rawdata/test.csv.hash.result_tf.csv", "w")
    format_line = open("../data7/rawdata/test.csv")
    for i, line in enumerate(format_line):
        if i == 0:
            wfile.write("id,isClick\n")
            continue
        wfile.write(line.strip().split(",")[0] + "," + str(result[i-1]) + "\n")
        if i > 1000:
            break
    wfile.close()


if __name__ == '__main__':
    main()
