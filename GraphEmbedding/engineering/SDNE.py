# encoding: utf-8
# @author: ZhangChaoyang
# @file: SDNE.py
# @time: 2022-04-28

import os
import math

import tensorflow as tf
from keras.layers import Dense, Embedding, Input, Reshape
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.optimizers
import numpy as np
import mlflow
from mlflow import log_metric, log_param, log_artifact
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0, '优化器的初始学习率')
flags.DEFINE_float('embed', 100, '顶点向量的维度')
flags.DEFINE_string('data_path', '', 'data目录路径')
flags.DEFINE_string('model_path', '', 'model目录路径')


def build_adjacency_matrix(edges, node_num):
    # 构建邻接矩阵（高度稀疏的二维矩阵）
    adjacency_matrix = np.zeros(shape=(node_num, node_num), dtype=np.float32)  # SDNE非常耗内存，用float32不要用float64
    for row in edges[:]:
        vi, vj = row[0], row[1]
        adjacency_matrix[vi][vj] = 1.0  # 边权重都是1
        adjacency_matrix[vj][vi] = 1.0
    return adjacency_matrix


def reconstruction_loss(beta, y_true, y_pred):
    B = tf.ones_like(y_true, dtype=tf.float32)
    A = tf.ones_like(y_true, dtype=tf.float32) * beta
    B = tf.where(y_true > 0, A, B)  # 权重默认都是1，y_true中的非0项权重为beta(beta>1)
    sub = tf.subtract(y_true, y_pred) ** 2 * B
    return tf.reduce_mean(tf.reduce_sum(sub, axis=1), axis=0)  # 行内（向量的各个元素）求和，再行间（一个批次内的多个样本）求平均


def embedding_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.subtract(y_pred, y_true), axis=1),
                          axis=0)  # y_true全是0，所以tf.subtract(y_pred, y_true)就是y_pred，y_pred已经把误差算好了


def sample_generator(edges, adjacency_matrix, embedding_dim, batch_size=32):
    N = edges.shape[0]
    for step in range(math.ceil(1.0 * N / batch_size)):
        begin = step * batch_size
        end = (step + 1) * batch_size
        if end > N:
            end = N
        batch = edges[begin:end]
        v_i = tf.convert_to_tensor(batch[:, 0])
        v_j = tf.convert_to_tensor(batch[:, 1])
        x_i = tf.convert_to_tensor(adjacency_matrix[batch[:, 0]])
        x_j = tf.convert_to_tensor(adjacency_matrix[batch[:, 1]])
        dummy_output = tf.zeros(shape=[end - begin, embedding_dim], dtype=tf.float32)
        yield ([v_i, v_j], [x_i, x_j, dummy_output])  # 输入：相邻的2个顶点的编号。输出：这2个顶点的重构向量以及一个全0的tensor，这里就是y_true


class SDNE(Model):
    def __init__(self, alpha, beta, node_num, hidden_dims, v, adjacency_matrix):
        super(SDNE, self).__init__()
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
        self.encoder = Sequential(fcs)

        # decoder和encoder对称
        fcs = []
        for dim in reversed(hidden_dims[:-1]):
            fcs.append(Dense(units=dim, use_bias=True, activation='sigmoid',
                             kernel_regularizer=l2(v), bias_regularizer=l2(v)))
        fcs.append(Dense(units=node_num, use_bias=True, activation='sigmoid',
                         kernel_regularizer=l2(v), bias_regularizer=l2(v)))
        self.decoder = Sequential(fcs)

    @tf.function(
        input_signature=[tf.TensorSpec([None, ], tf.int32, name="vi"),
                         tf.TensorSpec([None, ], tf.int32, name="vj")])
    def call(self, v_i, v_j):
        # v_i, v_j = inputs
        y_i = self.encoder(v_i)  # encode结果就是结点的向量表示(Embedding层的输入是input_i或input_j，它们的shape都是1)
        x_i_hat = self.decoder(y_i)
        y_j = self.encoder(v_j)
        x_j_hat = self.decoder(y_j)
        y_diff = tf.subtract(y_i, y_j) ** 2  # 一阶相似度的误差
        return y_i, y_j, x_i_hat, x_j_hat, y_diff


def trainSDNE(alpha, beta, node_num, hidden_dims, v, edges, epochs):
    adjacency_matrix = build_adjacency_matrix(edges, node_num)
    model = SDNE(alpha, beta, node_num, hidden_dims, v, adjacency_matrix)
    init_learning_rate = FLAGS.lr
    # log_param("lr", init_learning_rate)# 如果通过mlflow run运行代码，则不要执行该行代码，因为lr这个param已经log过了，param不能重复log
    optimizer = keras.optimizers.adam_v2.Adam(learning_rate=init_learning_rate)
    step = 0
    for epoch in range(epochs):
        for [v_i, v_j], [x_i, x_j, dummy_output] in sample_generator(edges, adjacency_matrix, hidden_dims[-1],
                                                                     batch_size=128):
            with tf.GradientTape() as tape:
                _, _, x_i_hat, x_j_hat, y_diff = model.call(v_i, v_j)
                recon_loss_i = reconstruction_loss(beta, x_i, x_i_hat)
                recon_loss_j = reconstruction_loss(beta, x_j, x_j_hat)
                embed_loss = embedding_loss(y_diff, dummy_output)
                loss = recon_loss_i + recon_loss_j + alpha * embed_loss
                log_metric('recon_loss_i', tf.reduce_mean(recon_loss_i).numpy().tolist(), step)
                log_metric('recon_loss_j', tf.reduce_mean(recon_loss_j).numpy().tolist(), step)
                log_metric('embed_loss', tf.reduce_mean(embed_loss).numpy().tolist(), step)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            step += 1

    return model


def main(argv):
    alpha = 2.0
    beta = 2.0
    v = 0.01
    epochs = 1
    hidden_dims = [128, 64]

    index = 0
    node2index = dict()
    edges = []
    if FLAGS.data_path[0] == "'":
        train_file = os.path.join(FLAGS.data_path[1:-1], "trust_data_small.txt")
        # log_param("data_path", FLAGS.data_path[1:-1])# 如果通过mlflow run运行代码，则不要执行该行代码，因为data_path这个param已经log过了，param不能重复log
    else:
        train_file = os.path.join(FLAGS.data_path, "trust_data_small.txt")
        # log_param("data_path", FLAGS.data_path)# 如果通过mlflow run运行代码，则不要执行该行代码，因为data_path这个param已经log过了，param不能重复log
    with open(train_file) as fin:  # SDNE非常耗内存，用data/trust_data.txt可能内存不够用
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

    model = trainSDNE(alpha, beta, node_num, hidden_dims,
                      v, np.array(edges), epochs)

    if FLAGS.model_path[0] == "'":
        tf_saved_model_dir = os.path.join(FLAGS.model_path[1:-1], "sdne")
        # log_param('model_path',
        #           FLAGS.model_path[1:-1])  # 如果通过mlflow run运行代码，则不要执行该行代码，因为model_path这个param已经log过了，param不能重复log
    else:
        tf_saved_model_dir = os.path.join(FLAGS.model_path, "sdne")
        # log_param('model_path', FLAGS.model_path)  # 如果通过mlflow run运行代码，则不要执行该行代码，因为model_path这个param已经log过了，param不能重复log
    print(tf_saved_model_dir)
    tf.saved_model.save(model, tf_saved_model_dir)  # saved_model_cli show --dir model/sdne --all查看tags和signature_def
    # log_artifact(tf_saved_model_dir, "sdne")
    mlflow.tensorflow.log_model(tf_saved_model_dir=tf_saved_model_dir,
                                tf_meta_graph_tags=['serve'],
                                tf_signature_def_key='serving_default',
                                artifact_path="sdne",
                                # 模型文件会存到mlruns/<experiment_id>/<run_id>/artifacts/<artifact_path>目录下
                                )# log_model 会调用log_artifact
    print(f'run_id {mlflow.active_run().info.run_id}')


if __name__ == "__main__":
    app.run(main)

# python .\engineering\SDNE.py -lr 0.001 -data_path ./data -model_path ./model
# mlflow models serve -m runs:/d363f87e7bc842b8a15da62087b06c73/sdne

# -P 会调用log_param
# mlflow run ./engineering -P lr=0.001 -P data_path=./data -P model_path=./model
# mlflow models serve -m runs:/f9a88cda04504b9d93ea9b64f6f7da59/sdne
