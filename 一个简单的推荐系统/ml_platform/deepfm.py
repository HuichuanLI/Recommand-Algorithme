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
    "feature_len": 8,
    "embedding_dim": 7,
    "label_len": 1,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 16,
    "learning_rate": 0.01,

    "dnn_hidden_units": [64, 32],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.1,

    "train_file": "../data/train",
    "test_file": "../data/val",
    "saved_embedding": "../data/saved_dfm_embedding",
    "max_steps": 80000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "deepfm",

    "saved_pb": "../data/saved_dfm_model",
    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}


def nn_tower(
        name, nn_input, hidden_units,
        activation=tf.nn.relu, use_bias=False,
        l2=0.0):
    out = nn_input
    for i, num in enumerate(hidden_units):
        out = tf.layers.dense(
            out,
            units=num,
            kernel_initializer=tf.truncated_normal_initializer(),
            use_bias=use_bias,
            activation=activation,
            name=name + "/layer_" + str(i),
        )
    return out


def deepfm_fn(inputs, is_test):
    # 取特征和y值，feature为：[batch, f_nums, weight_dim]
    input_embedding = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, config['feature_len'] * config['embedding_dim']])  # [batch, f_nums, weight_dim]

    weight_ = tf.reshape(
        input_embedding,
        [-1, config['feature_len'], config['embedding_dim']])
    weight_ = tf.split(
        weight_,
        num_or_size_splits=[config['embedding_dim'] - 1, 1],
        axis=2)

    # ================================================================
    #
    # linear part
    bias_part = tf.get_variable(
        "bias", [1, ],
        initializer=tf.zeros_initializer())  # 1*1

    linear_part = tf.nn.bias_add(
        tf.reduce_sum(weight_[1], axis=1),
        bias_part)  # batch*1

    # cross part
    # cross sub part : sum_square part
    summed_square = tf.square(tf.reduce_sum(weight_[0], axis=1))  #

    # batch*embed
    # cross sub part : square_sum part
    square_summed = tf.reduce_sum(tf.square(weight_[0]), axis=1)  # batch*embed
    cross_part = tf.subtract(summed_square, square_summed)

    feature_with_embedding_concat = tf.reshape(
        weight_[0],
        [-1, config['feature_len'] * (config['embedding_dim'] - 1)])

    dnn_out_ = nn_tower(
        'dnn_hidden',
        feature_with_embedding_concat, config['dnn_hidden_units'],
        use_bias=True, activation=config['activation_function'],
        l2=config['dnn_l2']
    )
    print(linear_part)
    print(cross_part)
    print(dnn_out_)
    out_ = tf.concat([linear_part, cross_part, dnn_out_], axis=1)
    print(out_)
    out_ = nn_tower('out', out_, [1], activation=None)

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
        net_out_dic = deepfm_fn(inputs, is_test)

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
