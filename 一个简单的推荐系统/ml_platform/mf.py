# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/11/10|9:59
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = {
    "batch": 32,
    "embedding_dim": 8,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "learning_rate": 0.01,
    "feature_len": 2,
    "label_len": 1,

    "train_file": "../data0/rawdata/train",
    "test_file": "../data0/rawdata/val",
    "saved_embedding": "../data0/feature_embedding",
    "max_steps": 10000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "mf",
    "saved_pb": "../data0/saved_model",

    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}

def mf_fn(inputs, is_test):
    # 取特征和y值，feature为：user_id 和 movie_id
    embed_layer = inputs["feature_embedding"]  # [batch, 2, embedding_dim]
    embed_layer = tf.reshape(embed_layer, shape=[-1, 2, config["embedding_dim"]])
    label = inputs["label"]  # [batch, 1]
    # 切分数据，获得user_id的embedding 和 movie_id的embedding
    embed_layer = tf.split(embed_layer, num_or_size_splits=2, axis=1)  # [batch, embedding_dim] * 2
    user_id_embedding = tf.reshape(embed_layer[0], shape=[-1, config["embedding_dim"]])  # [batch, embedding_dim]
    movie_id_embedding = tf.reshape(embed_layer[1], shape=[-1, config["embedding_dim"]])  # [batch, embedding_dim]
    # 根据公式进行乘积并求和
    out_ = tf.reduce_mean(
        user_id_embedding * movie_id_embedding, axis=1)  # [batch]
    # 设定预估部分
    out_tmp = tf.sigmoid(out_)  # batch
    if is_test:
        tf.add_to_collections("input_tensor", embed_layer)
        tf.add_to_collections("output_tensor", out_tmp)

    # 损失函数loss
    label_ = tf.reshape(label, [-1])  # [batch]
    loss_ = tf.reduce_sum(tf.square(label_ - out_))  # 1

    out_dic = {
        "loss": loss_,
        "ground_truth": label_,
        "prediction": out_
    }

    return out_dic


# 定义整个图结构，并给出梯度更新方式
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope("net_graph", reuse=is_test):
        # 初始模型图
        net_out_dic = mf_fn(inputs, is_test)

        loss = net_out_dic["loss"]
        result["out"] = net_out_dic

        if is_test:
            return result

        # SGD
        emb_grad = tf.gradients(
            loss, [inputs["feature_embedding"]], name="feature_embedding")[0]

        result["feature_new_embedding"] = \
            inputs["feature_embedding"] - config["learning_rate"] * emb_grad

        result["feature_embedding"] = inputs["feature_embedding"]
        result["feature"] = inputs["feature"]
        return result
