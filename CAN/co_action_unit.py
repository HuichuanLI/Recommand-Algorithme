# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/5/11|下午 04:30
# @Moto   : Knowledge comes from decomposition

import tensorflow as tf
from tensorflow.layers.activation import activation_layer


def co_action_unit(name_scope, inputs, weight, layer_num, order, activation):
    """
    假设： inputs -> [batch, 4]
          weight -> [batch, (4*4+4)*8]
          layer_num -> 8
          order -> 2
          对应的是8层神经网络，每层的参数量是：4*4+4 （权重+偏置）
          inputs -> [batch, 4] 实际是 4 + 4的order
    """
    with tf.variable_scope(name_scope):
        # todo : 对输入部分进行order处理
        input_shape = inputs.get_shape().as_list()[-1]
        _inputs = inputs
        for i in range(order - 1):
            _inputs += tf.pow(inputs, i + 2)

        # todo : 对权重部分进行拆分处理

        _weight_and_bias = tf.split(
            weight, num_or_size_splits=layer_num, axis=-1)
        # 基于mlp的计算逻辑进行每层计算
        fc = tf.expand_dims(_inputs, 1)
        for _w_b in _weight_and_bias:
            _w_b = tf.split(tf.reshape(
                _w_b, [-1, input_shape, input_shape+1]),
                [input_shape, 1], axis=-1)
            _w = _w_b[0]
            _b = tf.transpose(_w_b[1], (0,2,1))

            # 进行运算
            fc = tf.matmul(fc, _w) + _b
            #
            fc = activation_layer(activation)(fc)

    return tf.layers.flatten(fc)
