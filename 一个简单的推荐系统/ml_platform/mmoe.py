# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/8/10 | 11:22
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = {
    "feature_len": 11,
    "embedding_dim": 5,
    "label_len": 2,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 32,
    "learning_rate": 0.01,
    "target_num": 2,
    "expert_num": 2,
    "dnn_hidden_units": [32, 16],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.0,

    "train_file": "../data6/train",
    "test_file": "../data6/val",
    "saved_embedding": "../data6/saved_dnn_embedding",
    "max_steps": 1000,
    "train_log_iter": 100,
    "test_show_step": 100,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "doubletower",

    "saved_pb": "../data6/saved_model",

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
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2),
            use_bias=use_bias,
            activation=activation,
            name=name + "/layer_" + str(i),
        )
    return out


def mmoe_fn(inputs, is_test):
    # 取特征和y值，feature为：[batch, f_nums, weight_dim]
    input_embedding = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, config['feature_len'] * config['embedding_dim']])

    # ================================================================
    #
    # ----- build Mixture-of-Experts -----
    # expert 的输出大小必须保持一致，后面需要对每个expert 进行加权求和
    with tf.variable_scope("Mixture-Of-Experts"):
        experts = []
        for i in range(config['expert_num']):
            expert_out = nn_tower(
                'dnn_hidden' + str(i),
                input_embedding, config['dnn_hidden_units'],
                use_bias=True, activation=config['activation_function'],
                l2=config['dnn_l2'])
            experts.append(expert_out)
        # experts -> list([batch, expert_out_size]*expert_num)
        # 获得每个expert的输出结果
        experts = tf.transpose(tf.stack(experts, axis=0), perm=[1, 0, 2])  # [batch, expert_num, expert_out_size]

    # ----- build Gates -----
    with tf.variable_scope("Gate-Weight"):
        # 基于输入embed层进行一个dense网络，获得每个expert的权重
        gates = []
        for i in range(config['target_num']):  # 有几个目标就有几个权重门
            gate_out = nn_tower(
                'dnn_hidden' + str(i),
                input_embedding, [config['expert_num']],
                use_bias=True, activation=config['activation_function'],
                l2=config['dnn_l2'])  # [batch, expert_num]
            gates.append(tf.expand_dims(tf.nn.softmax(gate_out), axis=1))  # list([batch, 1, expert_num]*target_num)

    # ----- build target -----
    with tf.variable_scope("Target-Weight"):
        # 基于每个target进行加权
        target_input = []
        for i in range(config['target_num']):  # gates[i] -> [batch,1,expert_num]; experts -> [batch,expert_num,export_out_size]
            target_input.append(tf.reshape(tf.matmul(gates[i], experts), shape=[-1, config['dnn_hidden_units'][-1]]))
        # print(target_input)  -> target_input: list([batch, export_out_size]*target_num)

    # ---- build customize -----
    target_out = []  # 进入各自的tower里面
    for i in range(config['target_num']):
        with tf.variable_scope("Target-{}".format(i)):
            x = nn_tower(
                'dnn_hidden',
                target_input[i], config['dnn_hidden_units'],
                use_bias=True, activation=config['activation_function'],
                l2=config['dnn_l2'])  # [batch, expert_num]
            y = nn_tower(
                '_out_dnn_hidden',
                x, [1],
                use_bias=True, activation=None,
                l2=config['dnn_l2'])  # [batch, expert_num]
            target_out.append(y)

    with tf.variable_scope("mmoe_out"):  # 假设只有两个目标，对应ctr，cvr时
        pctr = target_out[0]
        pctcvr = target_out[0] * target_out[1]

    pctr_ = tf.reshape(pctr, [-1])  # batch
    pctcvr_ = tf.reshape(pctcvr, [-1])  # batch
    loss1 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pctr_, labels=inputs["label"][:, 0]))

    loss2 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pctcvr_, labels=inputs["label"][:, 1]))

    loss = 0.5 * loss1 + 0.5 * loss2
    pred = tf.sigmoid(tf.concat([pctr, pctcvr], axis=1))
    if is_test:
        tf.add_to_collections("input_tensor", input_embedding)
        tf.add_to_collections("output_tensor", pred)

    out_dic = {
        "loss": loss,
        "ground_truth": inputs["label"][:, 1],
        "prediction": pctcvr_
    }

    return out_dic


# define graph
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope("net_graph", reuse=is_test):
        # init graph
        net_out_dic = mmoe_fn(inputs, is_test)

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
