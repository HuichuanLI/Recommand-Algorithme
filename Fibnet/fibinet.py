# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/8/21|15:28
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import sys
import tensorflow as tf
import itertools

#
# PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, PACKAGE_DIR)


config = {
    "feature_len": 10,
    "embedding_dim": 5,
    "label_len": 1,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 16,
    "learning_rate": 0.01,
    "reduction_ratio": 0.5,
    "bilinear_type": "all",
    "dnn_hidden_units": [32, 16],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.1,

    "train_file": "../data1/train",
    "test_file": "../data1/val",
    "saved_embedding": "../data1/saved_dnn_embedding",
    "max_steps": 10000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "fibinet",

    "saved_pb": "../data1/saved_model",

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


# SENet
# 想通过控制scale的大小，
# 把重要的特征增强，不重要的特征减弱，从而让提取的特征指向性更强
def squeeze_excitation_layer(name_scope, inputs, ratio=16):
    """
    Input shape : [batch, field, embed]
    Output shape : [batch, embed]
    Params num : 0
    ratio: 压缩率
    """
    with tf.variable_scope(name_scope):
        inputs_shape = inputs.get_shape().as_list()
        sqeeze = tf.reduce_sum(inputs, axis=2)  # batch*field

        # 按比例压缩
        excitation = nn_tower(
            name="ex01", nn_input=sqeeze, hidden_units=[inputs_shape[1] // ratio], use_bias=True)

        excitation = nn_tower(
            name="ex02", nn_input=excitation, hidden_units=[inputs_shape[1]],
            use_bias=True, activation=tf.nn.sigmoid)  # batch*field

        excitation = tf.tile(tf.reshape(
            excitation, [-1, inputs_shape[1], 1]), [1, 1, inputs_shape[2]])  # batch*field*embed
        scale = inputs * excitation  # batch*field*embed
    return scale


# Bi-linear Interaction Layer used in FiBiNET
def bi_linear_interaction_layer(name_scope, inputs, bilinear_type):
    """
    Input shape : [batch, field, embed]
    Output shape : [batch, field*embed]
    Params num :

    fields: [0,0,0,1,1,1,2]
    """

    with tf.variable_scope(name_scope):
        inputs_shape = inputs.get_shape().as_list()
        embedding_size = inputs_shape[2]

        inputs_list = tf.split(
            inputs, inputs_shape[1], axis=1)
        list_num = len(inputs_list)

        if bilinear_type == "all":
            w = tf.get_variable(
                "bilinear_weight",
                [embedding_size, embedding_size],
                initializer=tf.glorot_normal_initializer(seed=2020))

            p = [tf.multiply(tf.tensordot(v_i, w, axes=(-1, 0)), v_j)
                 for v_i, v_j in itertools.combinations(inputs_list, 2)]

        elif bilinear_type == "each":
            w_list = []
            for i in range(list_num - 1):
                w_list.append(tf.get_variable(
                    "bilinear_weight_" + str(i),
                    [embedding_size, embedding_size],
                    initializer=tf.glorot_normal_initializer(seed=2020)))

            p = [tf.multiply(tf.tensordot(inputs_list[i], w_list[i], axes=(-1, 0)), inputs_list[j])
                 for i, j in itertools.combinations(range(list_num), 2)]

        elif bilinear_type == "interaction":
            w_list = []
            for i, j in itertools.combinations(range(list_num), 2):
                w_list.append(tf.get_variable(
                    "bilinear_weight_" + str(i) + '_' + str(j),
                    [embedding_size, embedding_size],
                    initializer=tf.glorot_normal_initializer(seed=2020)))

                p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                     for v, w in zip(itertools.combinations(inputs_list, 2), w_list)]

        else:
            raise ValueError("bilinear_type not in (all, each, interaction)")

        p = tf.concat(p, axis=1)
        return p


def fibinet_model(inputs, is_test=False):
    # 取特征和y值，feature为：[batch, f_nums, weight_dim]
    input_embedding = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, config['feature_len'], config['embedding_dim']])

    # se_net weight
    senet_embedding = squeeze_excitation_layer(
        name_scope="senet", inputs=input_embedding, ratio=config['reduction_ratio'])

    #
    senet_bilinear_out_ = bi_linear_interaction_layer(
        name_scope="senet_bilinear",
        inputs=senet_embedding,
        bilinear_type=config['bilinear_type'])

    input_bilinear_out_ = bi_linear_interaction_layer(
        name_scope="input_bilinear",
        inputs=input_embedding,
        bilinear_type=config['bilinear_type'])

    out_ = tf.concat([senet_bilinear_out_, input_bilinear_out_], axis=1)

    out = nn_tower(
        'dnn_hidden',
        out_, config['dnn_hidden_units'],
        use_bias=True, activation=config['activation_function'],
        l2=config['dnn_l2']
    )

    out_ = nn_tower('out', out_, [1], activation=None)
    # =================================================
    # 统一模型output部分
    # =================================================
    out_ = tf.reshape(out_, [-1])  # batch
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=out_, labels=inputs["label"][:, 0]))
    if is_test:
        tf.add_to_collections("input_tensor", input_embedding)
        tf.add_to_collections("output_tensor", out_)
    net_dic = {
        "loss": loss,
        "ground_truth": inputs["label"][:, 0],
        "prediction": out_}
    # ==================================================
    return net_dic


# if __name__ == '__main__':
#     Invoke("conf/job.conf.dlbox.ini", fm_model).run()

# define graph
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope("net_graph", reuse=is_test):
        # init graph
        net_out_dic = fibinet_model(inputs, is_test)

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
