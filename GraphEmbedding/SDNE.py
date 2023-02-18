# -*- coding: utf-8 -*-
# @Time    : 2021/11/5 7:32 PM
# @Author  : zhangchaoyang
# @File    : SDNE.py

import math
import tensorflow as tf
from keras.layers import Dense, Embedding, Input, Reshape
from keras.models import Model, Sequential
from keras.regularizers import l2
import numpy as np


def build_adjacency_matrix(edges, node_num):
    # 构建邻接矩阵（高度稀疏的二维矩阵）
    adjacency_matrix = np.zeros(shape=(node_num, node_num), dtype=np.float32)  # SDNE非常耗内存，用float32不要用float64
    for row in edges[:]:
        vi, vj = row[0], row[1]
        adjacency_matrix[vi][vj] = 1.0  # 边权重都是1
        adjacency_matrix[vj][vi] = 1.0
    return adjacency_matrix


def reconstruction_loss(beta):
    def loss(y_true, y_pred):
        B = tf.ones_like(y_true, dtype=tf.float32)
        A = tf.ones_like(y_true, dtype=tf.float32) * beta
        B = tf.where(y_true > 0, A, B)  # 权重默认都是1，y_true中的非0项权重为beta(beta>1)
        sub = tf.subtract(y_true, y_pred) ** 2 * By
        return tf.reduce_mean(tf.reduce_sum(sub, axis=1), axis=0)  # 行内（向量的各个元素）求和，再行间（一个批次内的多个样本）求平均

    return loss


def embedding_loss():
    def loss(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.subtract(y_pred, y_true), axis=1),
                              axis=0)  # y_true全是0，所以tf.subtract(y_pred, y_true)就是y_pred，y_pred已经把误差算好了

    return loss


def sample_generator(edges, adjacency_matrix, embedding_dim, batch_size=32):
    N = edges.shape[0]
    for step in range(math.ceil(1.0 * N / batch_size)):
        begin = step * batch_size
        end = (step + 1) * batch_size
        if end > N:
            end = N
        batch = edges[begin:end]
        v_i = batch[:, 0]
        v_j = batch[:, 1]
        x_i = adjacency_matrix[v_i]
        x_j = adjacency_matrix[v_j]
        dummy_output = tf.zeros(shape=[end - begin, embedding_dim], dtype=tf.float32)
        yield ([v_i, v_j], [x_i, x_j, dummy_output])  # 输入：相邻的2个顶点的编号。输出：这2个顶点的重构向量以及一个全0的tensor，这里就是y_true


def create_model(alpha, beta, node_num, hidden_dims, v, adjacency_matrix):
    input_i = Input(shape=(1,), dtype=tf.int32)  # 在定义Layer的时候是不考虑batch_size的
    input_j = Input(shape=(1,), dtype=tf.int32)

    # 从顶点编号，得到它的邻接向量
    embedding_layer = Embedding(node_num, node_num,
                                weights=[adjacency_matrix], trainable=False)

    # 把embedding_layer也算到encoder里面去
    fcs = [embedding_layer, Reshape(
        target_shape=(
        node_num,))]  # 在定义Layer的时候是不考虑batch_size的。input经过embedding_layer会多出一维，通过Reshape把多出的这一维squeeze掉(在本模型中，Embedding层的输入序列长度为1)
    for dim in hidden_dims:
        fcs.append(Dense(units=dim, use_bias=True, activation='sigmoid',
                         kernel_regularizer=l2(v), bias_regularizer=l2(v)))
    encoder = Sequential(fcs)

    # decoder和encoder对称
    fcs = []
    for dim in reversed(hidden_dims[:-1]):
        fcs.append(Dense(units=dim, use_bias=True, activation='sigmoid',
                         kernel_regularizer=l2(v), bias_regularizer=l2(v)))
    fcs.append(Dense(units=node_num, use_bias=True, activation='sigmoid',
                     kernel_regularizer=l2(v), bias_regularizer=l2(v)))
    decoder = Sequential(fcs)

    y_i = encoder(input_i)  # encode结果就是结点的向量表示(Embedding层的输入是input_i或input_j，它们的shape都是1)
    x_i_hat = decoder(y_i)
    y_j = encoder(input_j)
    x_j_hat = decoder(y_j)

    y_diff = tf.subtract(y_i, y_j) ** 2  # 一阶相似度的误差
    model = Model(inputs=[input_i, input_j],
                  outputs=[x_i_hat, x_j_hat, y_diff])  # outputs就是y_pred
    model.compile(
        # 3个output，3个loss，一一对应
        loss=[reconstruction_loss(beta), reconstruction_loss(beta), embedding_loss()],  # y_true和y_pred都三元组，对应3个loss
        loss_weights=[1, 1, alpha])  # 二阶相似度权重为1，一阶相似度权重为alpha
    model.encoder = encoder
    return model


def train_model(alpha, beta, node_num, hidden_dims, v, edges, epochs):
    adjacency_matrix = build_adjacency_matrix(edges, node_num)
    model = create_model(alpha, beta, node_num,
                         hidden_dims, v, adjacency_matrix)
    model.fit(sample_generator(
        edges, adjacency_matrix, hidden_dims[-1], batch_size=128), epochs=epochs)
    return model


def get_embedding(model, x):
    # 从顶点编号得到顶点的向量表示
    return model.encoder(x)


if __name__ == "__main__":
    alpha = 2.0
    beta = 2.0
    v = 0.01
    epochs = 1
    hidden_dims = [128, 64]

    index = 0
    node2index = dict()
    edges = []
    with open("data/trust_data_small.txt") as fin:  ## SDNE非常耗内存，用data/trust_data.txt可能内存不够用
        for line in fin:
            arr = line.strip().split()
            if len(arr) >= 2:
                v_i = int(arr[0])
                v_j = int(arr[1])
                if node2index.get(v_i) is None:
                    node2index[v_i] = index
                    index += 1
                if node2index.get(v_j) is None:
                    node2index[v_j] = index
                    index += 1
                edges.append([node2index[v_i], node2index[v_j]])
    node_num = len(node2index)

    model = train_model(alpha, beta, node_num, hidden_dims,
                        v, np.array(edges), epochs)

    node = 1
    node_index = node2index.get(node)
    if node_index is not None:
        embedding = get_embedding(
            model, tf.convert_to_tensor([[node_index]], dtype=tf.int32))
        print(embedding)
