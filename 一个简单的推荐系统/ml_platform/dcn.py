# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/9/25 | 15:32
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = {
    "feature_len": 10,
    "embedding_dim": 6,
    "label_len": 1,

    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 16,

    "learning_rate": 0.01,
    "cross_layer_num": 2,
    "dnn_hidden_units": [64, 32],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.1,

    "train_file": "../data1/train",
    "test_file": "../data1/val",
    "saved_embedding": "../data1/saved_dnn_embedding",
    "max_steps": 80000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "dcn",

    "saved_pb": "../data1/saved_model",
    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}


def nn_layer(
        name, nn_input, hidden_units,
        activation=tf.nn.relu, use_bias=False):
    out = nn_input   # [batch, embedding*F]  -> [batch, 64] -> [batch, 32]
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


def cross_layer(name_scope, inputs, cross_layer_num):
    """
    Input shape : [batch, field, embed]
    Output shape : [batch, field*embed]
    Params num : 3*embed*att_dims * head_num
    layer_num: cross 层数，交叉的阶数
    """

    with tf.variable_scope(name_scope):
        inputs_shape = inputs.get_shape().as_list()  # [batch, embedding*F]

        kernels = []
        bias = []
        for i in range(cross_layer_num):
            kernels.append(
                tf.get_variable(
                    "w_" + str(i), [inputs_shape[-1], 1], # embedding*F
                    initializer=tf.glorot_normal_initializer(seed=2020))
            )
            bias.append(
                tf.get_variable(
                    "b_" + str(i), [inputs_shape[-1], 1],
                    initializer=tf.zeros_initializer())
            )

        #
        x_0 = tf.expand_dims(inputs, axis=2)  # batch*[field*embed]*1 # X_0 -> batch, embedding*F, 1  batch, 1, embedding*F
        x_l = x_0
        for i in range(cross_layer_num):
            # dot_ = tf.matmul(x_l.T, kernels[i])
            xl_w = tf.tensordot(x_l, kernels[i], axes=(1, 0))  # [batch,[embedding*F], 1] | [embedding*F, 1]   batch,1,1
            dot_ = tf.matmul(x_0, xl_w)  # batch*[field*embed]*1
            print(dot_.get_shape().as_list())
            print(bias[i].get_shape().as_list())
            print(x_l.get_shape().as_list())

            x_l = dot_ + bias[i] + x_l  # batch*[field*embed]*1  拟合残差

        x_l = tf.squeeze(x_l, axis=2)

    return x_l  # [batch, embedding*F]


def dcn_fn(inputs, is_test):
    # 取特征和y值，feature为：[batch, f_nums, weight_dim]
    input_embedding = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, config['feature_len'], config['embedding_dim']])  # [batch, f_nums, weight_dim]

    # ================================================================
    feature_with_embedding_concat = tf.reshape(
        input_embedding,
        [-1, config['feature_len'] * config['embedding_dim']]) # [batch, 60]
    print(feature_with_embedding_concat)
    # deep network
    dnn_out_ = nn_layer(
        'dnn_hidden',
        feature_with_embedding_concat, config['dnn_hidden_units'],
        use_bias=True, activation=config['activation_function'],
    )
    print(dnn_out_)
    # deep cross
    cross_out_ = cross_layer(
        "cross",
        inputs=feature_with_embedding_concat,
        cross_layer_num=config['cross_layer_num'])

    out_ = tf.concat([dnn_out_, cross_out_], axis=1)
    print(out_)
    out_ = nn_layer('out', out_, [1], activation=None)

    out_tmp = tf.sigmoid(out_)  # batch
    if is_test:
        tf.add_to_collections("input_tensor", feature_with_embedding_concat)
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
        net_out_dic = dcn_fn(inputs, is_test)

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
