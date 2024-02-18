# -*- coding: utf-8 -*-
# @Time   : 2021/11/7 11:01
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

config = {
    "feature_len": 5,
    "user_feature_len": 1,
    "item_feature_len": 2,
    "embedding_dim": 10,
    "label_len": 1,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 32,
    "learning_rate": 0.01,

    "dnn_hidden_units": [32, 16],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.0,

    "train_file": "../data5/train",
    "test_file": "../data5/val",
    "saved_embedding": "../data5/saved_dnn_embedding",
    "max_steps": 1000,
    "train_log_iter": 100,
    "test_show_step": 100,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "dat",

    "saved_pb": "../data5/saved_model",

    "input_tensor": ["input_tensor"],
    "output_tensor": ["user_output_tensor", "item_output_tensor"]
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


def dat_fn(inputs, is_test):
    # 取特征和y值，feature为：[batch, f_nums, weight_dim]
    f_nums = config['user_feature_len'] + config['item_feature_len'] + 3
    input_embedding = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, f_nums, config['embedding_dim']])
    # ================================================================
    # 用户id, 用户id_au, 物品id, 物品cate_id, 物品id_av, batch内物品major_cate
    # 切分用户和物品  【batch, 6, embedding】
    user_input_embedding, item_input_embedding = tf.split(
        input_embedding,
        num_or_size_splits=[config['user_feature_len'] + 1, config['item_feature_len'] + 2],
        axis=1)
    #
    # 用户tower   【batch, 2, embedding】
    user_input_embedding = tf.reshape(
        user_input_embedding,
        [-1, (config['user_feature_len']+1) * config['embedding_dim']])

    # au   【batch, embedding】
    au = user_input_embedding[:, config['embedding_dim']:config['embedding_dim']*2]

    user_out = nn_tower(
        'user_dnn_hidden',
        user_input_embedding, config['dnn_hidden_units'][:-1],
        use_bias=True, activation=config['activation_function'],
        l2=config['dnn_l2'])
    user_out = nn_tower(
        'user_dnn_out',
        user_out, config['dnn_hidden_units'][-1:],
        use_bias=True, activation=None,
        l2=config['dnn_l2'])

    # 物品tower
    item_input_embedding, major_cate = tf.split(
        item_input_embedding,
        num_or_size_splits=[config['item_feature_len'] + 1, 1],
        axis=1)
    item_input_embedding = tf.reshape(
        item_input_embedding,
        [-1, (config['item_feature_len']+1) * config['embedding_dim']])

    av = item_input_embedding[:, config['embedding_dim']*2:config['embedding_dim']*3]
    self_cate = item_input_embedding[:, config['embedding_dim']*1:config['embedding_dim']*2]

    item_out = nn_tower(
        'item_dnn_hidden',
        item_input_embedding, config['dnn_hidden_units'][:-1],
        use_bias=True, activation=config['activation_function'],
        l2=config['dnn_l2'])

    item_out = nn_tower(
        'item_dnn_out',
        item_out, config['dnn_hidden_units'][-1:],
        use_bias=True, activation=None,
        l2=config['dnn_l2'])


    out_ = tf.reduce_sum(user_out*item_out, axis=1)
    label_ = tf.reshape(inputs["label"], [-1])
    out_tmp_ = tf.sigmoid(out_)  # batch
    if is_test:
        tf.add_to_collections("input_tensor", input_embedding)
        tf.add_to_collections("user_output_tensor", user_out)
        tf.add_to_collections("item_output_tensor", item_out)

    lossu = tf.losses.mean_squared_error(label_ * au + (1 - label_) * item_out, item_out)
    lossv = tf.losses.mean_squared_error(label_ * av + (1 - label_) * user_out, user_out)
    lossC = tf.losses.mean_squared_error(major_cate, self_cate)
    # 损失函数loss label = inputs["label"]  # [batch, 1]
    loss_ = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=out_, labels=label_))
    loss_ = 0.2*loss_ + 0.1*lossu + 0.1*lossv + 0.1*lossC

    out_dic = {
        "loss": loss_,
        "ground_truth": label_,
        "prediction": out_tmp_
    }

    return out_dic


# define graph
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope("net_graph", reuse=is_test):
        # init graph
        net_out_dic = dat_fn(inputs, is_test)

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
