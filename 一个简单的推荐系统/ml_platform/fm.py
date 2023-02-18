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
    "feature_len": 4 ,
    "embedding_dim": 17,
    "label_len": 1,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 16,
    "learning_rate": 0.01,

    "train_file": "../data/train",
    "test_file": "../data/val",
    "saved_embedding": "../data/saved_dnn_embedding",
    "max_steps": 200000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.2,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "dnn",

    "saved_pb": "../data1/saved_model",

    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}


def fm_fn(inputs, is_test):
    # 取特征和y值，feature为：user_id 和 movie_id
    input_embedding = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, config['feature_len']*config['embedding_dim']])  # [batch, f_nums, weight_dim]
    print(input_embedding)
    weight_ = tf.reshape(
        input_embedding,
        [-1, config['feature_len'], config['embedding_dim']])
    print(weight_)
    # split linear weight and cross weight
    weight_ = tf.split(weight_, num_or_size_splits=[config['embedding_dim'] - 1, 1], axis=2)

    # linear part
    bias_part = tf.get_variable(
        "bias", [1, ],
        initializer=tf.zeros_initializer())  # 1*1

    linear_part = tf.nn.bias_add(
        tf.reduce_sum(weight_[1], axis=1),
        bias_part)  # batch*1

    # cross part
    # cross sub part : sum_square part
    summed_square = tf.square(tf.reduce_sum(weight_[0], axis=1))  # batch*embed
    # cross sub part : square_sum part
    square_summed = tf.reduce_sum(tf.square(weight_[0]), axis=1)  # batch*embed
    cross_part = 0.5 * tf.reduce_sum(
        tf.subtract(summed_square, square_summed),
        axis=1, keepdims=True)  # batch*1
    out_ = linear_part + cross_part
    out_tmp = tf.sigmoid(out_)  # batch
    if is_test:
        tf.add_to_collections("input_tensor", input_embedding)
        tf.add_to_collections("output_tensor", out_tmp)

    # 损失函数loss label = inputs["label"]  # [batch, 1]
    loss_ = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=out_, labels=inputs["label"]))

    out_dic = {
        "loss": loss_,
        "ground_truth": inputs["label"][:, 0],
        "prediction": out_[:, 0]
    }

    return out_dic


# define graph
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope("net_graph", reuse=is_test):
        # init graph
        net_out_dic = fm_fn(inputs, is_test)

        loss = net_out_dic["loss"]

        result["out"] = net_out_dic
        if is_test:
            return result

        # ps - sgd
        emb_grad = tf.gradients(
            loss, [inputs["feature_embedding"]], name="feature_embedding")[0]

        result["feature_new_embedding"] = \
            inputs["feature_embedding"] - config['learning_rate'] * emb_grad

        result["feature_embedding"] = inputs["feature_embedding"]
        result["feature"] = inputs["feature"]

        # net - sgd
        tvars1 = tf.trainable_variables()
        grads1 = tf.gradients(loss, tvars1)
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=config['learning_rate'],
            use_locking=True)
        train_op = opt.apply_gradients(zip(grads1, tvars1))
        result["train_op"] = train_op

        return result
