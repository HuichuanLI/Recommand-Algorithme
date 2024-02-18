# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/7/20|17:23
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import itertools
import numpy as np


def batch_norm(
        name=None, inputs=None, training=True, axis=-1,
        momentum=0.99, epsilon=1e-3, scale=True, center=True):
    """ Batch Normalization
    注意：训练时,需要更新moving_mean和moving_variance.
    默认情况下,更新操作被放入tf.GraphKeys.UPDATE_OPS,
    因此需要将它们作为依赖项添加到train_op.
    此外,在获取update_ops集合之前,请务必添加任何batch_normalization操作.
    否则,update_ops将为空,并且训练/推断将无法正常工作, 例如：

    x_norm = tf.layers.batch_normalization(x, training=training)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    axis的值取决于按照input的哪一个维度进行BN
    """
    return tf.compat.v1.layers.batch_normalization(
        name=name,
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        scale=scale,
        center=center,
        training=training
    )


def layer_norm(
        name=None,
        inputs=None,
        center=True,
        scale=True,
        training=True):
    """ Layer Normalization
    与Batch Normalization 不同点，在于不依赖batch，是对于节点维度进行Normalization

    """

    return tf.contrib.layers.layer_norm(
        scope=name,
        inputs=inputs,
        center=center,
        scale=scale,
        trainable=training
    )


# 根据不同的方式进行激活函数的选择
def activation_layer(act):
    if act == "rule":
        return tf.nn.relu
    elif act == "sigmoid":
        return tf.nn.sigmoid
    elif act == "softmax":
        return tf.nn.softmax
    elif act == "tanh":
        return tf.nn.tanh
    else:
        return tf.nn.sigmoid



# MLP : Multi-Layer Perceptron
def mlp_layer(
        name_scope=None, inputs=None, units=None,
        use_bn=True, bn_decay=0.999, bn_epsilon=1e-3,
        use_drop=False, drop_rate=0.5,
        activation=None, is_training=False):
    """ 多层MLP
    Input shape : [batch, concat_embed]
    Output shape : [batch, units[-1]]
    Params num : concat_embed*units[0] + units[1]*units[2] +...+units[n-1]*units[n]

    units = [128, 64]
    activation = ["relu", "relu"]
    use_bn bn层
    use_drop dropout层

    """
    if not isinstance(units, list) or len(units) < 1:
        raise ValueError('A dnn_layer' 'units on a list of at least 1')
    if not isinstance(activation, list) or len(activation) < 1:
        raise ValueError('A dnn_layer' 'activation on a list of at least 1')

    inputs_shape = inputs.get_shape().as_list()
    if len(inputs_shape) == 3:
        inputs = tf.reshape(inputs, [-1, inputs_shape[1] * inputs_shape[2]])

    with tf.variable_scope(name_scope):
        logits = inputs
        for i, unit in enumerate(units):
            logits = tf.layers.dense(
                name="dense_" + str(i),
                activation=None,
                units=unit,
                inputs=logits
            )

            # 注意： batch normalize、relu、dropout 等的相对顺序
            if use_bn:
                logits = batch_norm(
                    name="bn_" + str(i), inputs=logits, training=is_training,
                    momentum=bn_decay, epsilon=bn_epsilon)
            if activation[i] is not None:
                logits = activation_layer(activation[i])(logits)

            if use_drop:
                logits = tf.layers.dropout(
                    logits, rate=drop_rate, training=is_training)

    return logits


# out: format out
def out_layer(name_scope, inputs=None, units=1, activation="sigmoid"):
    """
    Input shape : [batch, concat_embed]
    Output shape : [batch, units]
    Params num : concat_embed*units

    """
    with tf.variable_scope(name_scope):
        logits = tf.layers.dense(
            name='wx',
            activation=None,
            units=units,
            inputs=inputs
        )
        logits = activation_layer(activation)(logits)

    return logits


