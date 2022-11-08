# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
import tensorflow as tf
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Dropout


class FMLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, emb_list, w_list, is_train=False):
        # sum(weight) -- 一次项
        # 一次项
        first_order = tf.reduce_sum(w_list, axis=1)  # [batch, 1]
        # 二次项FM部分 -- 1/2 * (sum_square - square_sum)
        sum_square = tf.square(tf.reduce_sum(emb_list, axis=1))  # [batch, emb]
        square_sum = tf.reduce_sum(tf.square(emb_list), axis=1)  # [batch, emb]
        second_order = tf.subtract(sum_square, square_sum)  # [batch, emb]
        out_ = tf.concat([first_order, second_order], axis=1)
        return out_


class MLPLayer(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 is_batch_norm=False,
                 is_dropout=0,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        self.units = [units] if not isinstance(units, list) else units
        if len(self.units) <= 0:
            raise ValueError(
                f'Received an invalid value for `units`, expected '
                f'a positive integer, got {units}.')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        dims = [last_dim] + self.units

        self.kernels = []
        self.biases = []
        self.bns = []
        for i in range(len(dims) - 1):
            self.kernels.append(
                self.add_weight(f'kernel_{i}',
                                shape=[dims[i], dims[i + 1]],
                                initializer=self.kernel_initializer,
                                trainable=True))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(f'bias_{i}',
                                    shape=[
                                        dims[i + 1],
                                    ],
                                    initializer=self.bias_initializer,
                                    trainable=True))
            self.bns.append(tf.keras.layers.BatchNormalization())
        self.built = True

    def call(self, inputs, is_train=False):
        _input = inputs
        for i in range(len(self.units)):
            _input = MatMul(a=_input, b=self.kernels[i])
            if self.use_bias:
                _input = BiasAdd(value=_input, bias=self.biases[i])
            # BN
            if self.is_batch_norm:
                _input = self.bns[i](_input)
            # ACT
            if self.activation is not None:
                _input = self.activation(_input)
            # DROP
            if is_train and self.is_dropout > 0:
                _input = Dropout(self.is_dropout)(_input)

        return _input


class DSSMLayer(tf.keras.layers.Layer):

    def __init__(self, user_feature_name, item_feature_name, feature_size,
                 embedding_size, **kwargs):
        super(DSSMLayer, self).__init__(**kwargs)
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.user_feature_name = user_feature_name
        self.item_feature_name = item_feature_name

    def build(self, input_shape):
        self.embed = tf.keras.layers.Embedding(self.feature_size,
                                               self.embedding_size,
                                               embeddings_regularizer="l2")
        # user tower
        self.user_embed_tower = MLPLayer(units=[64], activation='relu')
        self.item_embed_tower = MLPLayer(units=[64], activation='relu')
        self.user_embed_out = MLPLayer(units=[8], activation=None)
        self.item_embed_out = MLPLayer(units=[8], activation=None)

        self.built = True

    def call(self, inputs, is_train=False):
        # 结构
        emb_dic = {}
        for k, v in inputs.items():
            tmp = self.embed(v)
            emb_dic[k] = tmp

        user_emb = []
        item_emb = []
        for k, v in emb_dic.items():
            if k in self.user_feature_name:
                user_emb.append(v)
            if k in self.item_feature_name:
                item_emb.append(v)

        # concat
        user_emb = tf.keras.layers.Flatten()(tf.concat(user_emb, axis=1))
        item_emb = tf.keras.layers.Flatten()(tf.concat(item_emb, axis=1))

        user_emb = self.user_embed_tower(user_emb)
        item_emb = self.item_embed_tower(item_emb)

        user_emb = self.user_embed_out(user_emb)
        item_emb = self.item_embed_out(item_emb)

        out_ = tf.nn.sigmoid(tf.reduce_sum(user_emb * item_emb, axis=1))
        result = {"pred": out_, "user_emb": user_emb, "item_emb": item_emb}
        return result
