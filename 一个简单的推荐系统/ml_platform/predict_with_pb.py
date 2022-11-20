# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/7/9|10:25
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from mmoe import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import numpy as np

# slots = 18
pb_path = "../data6/saved_model"
feature_embeddings = "../data6/saved_dnn_embedding"


def load_embed():
    hashcode_dict = {}

    with open(feature_embeddings, "r") as lines:
        for line in lines:
            tmp = line.strip().split("\t")
            if len(tmp) == 2:
                vec = [float(_) for _ in tmp[1].split(",")]
                hashcode_dict[tmp[0]] = vec

    return hashcode_dict


def load_model():
    sess = tf.Session()
    # tf.saved_model.loader.load(sess, ['serve'], model_file)
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING],
                                                pb_path)

    signature = meta_graph_def.signature_def
    # get tensor name
    in_tensor_name = \
        signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
            'input0'].name
    out_tensor_name = \
        signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
            'output0'].name
    # get tensor
    in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
    out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)
    return sess, in_tensor, out_tensor
    #


def batch_predict(line_arr):
    """
    input_data : 按 slotid的顺序组装好，以 \t 分割特征，多值用"," 分割

    """
    embed_dict = load_embed()
    sess, in_tensor, out_tensor = load_model()
    print(in_tensor)
    print(out_tensor)
    print("====load success====")
    dim = config['embedding_dim']
    input_size = config['feature_len'] * config['embedding_dim']

    init_arr = [0.0] * input_size
    batch = []
    result = []
    for line in line_arr:
        tmp_arr = init_arr.copy()
        for i, f in enumerate(line):
            tmp = embed_dict.get(f, None)
            if tmp is not None:
                tmp_arr[i * dim:(i + 1) * dim] = tmp
        batch.append(tmp_arr)
        if len(batch) > 20:

            prediction = sess.run(out_tensor, feed_dict={in_tensor: np.array(batch)})
            for p in prediction:
                result.append(p)
            batch = []
            # print("==2==", prediction)
    prediction = sess.run(out_tensor, feed_dict={in_tensor: np.array(batch)})
    for p in prediction:
        result.append(p)

    return result


def main():
    lines = open("../data6/180.hash")
    line_arr = []
    for line in lines:
        line_arr.append(line.strip().split(","))

    result = batch_predict(line_arr)
    wfile = open("../data6/180.hash.result.csv", "w")
    wfile.write("id,isClick,isShop\n")
    for i, line in enumerate(result):
        wfile.write(str(i) + "," + str(result[i]) + "\n")
        if i > 100:
            break
    wfile.close()

if __name__ == '__main__':
    main()
