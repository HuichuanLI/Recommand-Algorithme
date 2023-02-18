# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
import tensorflow as tf


class FMLayer(tf.keras.layers.Layer):

    def __init__(self, user_feature_name, item_feature_name, feature_size, embedding_size, **kwargs):
        super(FMLayer, self).__init__(**kwargs)
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.user_feature_name = user_feature_name
        self.item_feature_name = item_feature_name

    def build(self, input_shape):
        # bias
        self.bias = self.add_weight(name='bias',
                                    shape=(1, ),
                                    trainable=True)
        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_size,
                                               self.embedding_size,
                                               embeddings_regularizer="l2")
        self.w = tf.keras.layers.Embedding(self.feature_size,
                                           1,
                                           embeddings_regularizer="l2")
        self.built = True

    def call(self, inputs, is_train=False):
        # 结构
        emb_dic = {}
        emb_list = []
        for k, v in inputs.items():
            tmp = self.embed(v)
            emb_dic[k] = tmp
            emb_list.append(tmp)
        
        w_list = []
        for k, v in inputs.items():
            w_list.append(self.w(v))

        # concat
        emb_list = tf.concat(emb_list, axis=1)
        w_list = tf.concat(w_list, axis=1)

        # sum(weight) -- 一次项
        # 一次项
        first_order = tf.reduce_sum(w_list, axis=1)  # [batch, 1]
        first_order = tf.add(first_order, self.bias)
        # 二次项FM部分 -- 1/2 * (sum_square - square_sum)
        sum_square = tf.square(tf.reduce_sum(emb_list, axis=1))  # [batch, emb]
        square_sum = tf.reduce_sum(tf.square(emb_list), axis=1)  # [batch, emb]
        second_order = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum),
                                           axis=1,
                                           keepdims=True)  # [batch, 1]
        logit = tf.nn.sigmoid(second_order + first_order)
        # 召回使用
        user_emb = []
        item_emb = []
        for k, v in emb_dic.items():
            if k in self.user_feature_name:
                user_emb.append(v)
            if k in self.item_feature_name:
                item_emb.append(v)
        
        user_emb = tf.reduce_sum(tf.concat(user_emb, axis=1), axis=1)
        item_emb = tf.reduce_sum(tf.concat(item_emb, axis=1), axis=1)
        result = {
            "pred": logit,
            "user_emb": user_emb,
            "item_emb": item_emb
        }
        return result
