# -*- coding: utf-8 -*-
# @Time    : 2021/10/24 5:57 PM
# @Author  : lihuichuan
# @File    : LINE.py

import math
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Embedding, Input
from alias_sample import AliasSample


def create_model(numNodes, embedding_dim, order='all'):
    # 模型只需要输入边上两个顶点的编号，通过边采样技术，所有的权重已经都化为1
    v_i = Input(shape=(1,), dtype=tf.int32)
    v_j = Input(shape=(1,), dtype=tf.int32)

    # 每个顶点有3个向量：计算一阶相似度时的向量，计算二阶相似度时作为中心节点的向量，计算二阶相似度时作为背景节点的向量
    first_embed = Embedding(numNodes, embedding_dim, name="first_embed")
    second_embed = Embedding(numNodes, embedding_dim, name="second_embed")
    second_context_embed = Embedding(numNodes, embedding_dim, name="second_context_embed")
    # 1阶相似度只能用于无向图当中
    v_i_embed = first_embed(v_i)
    v_j_embed = first_embed(v_j)
    # 在有向图当中，v_i是中心节点，v_j是背景节点，即边由v_i指向v_j
    v_i_embed_second = second_embed(v_i)
    v_j_embed_context = second_context_embed(v_j)

    # 两个顶点同时出现的联系概率
    join_prob = tf.reduce_sum(v_i_embed * v_j_embed, axis=-1, keepdims=False)
    # 由中心节点得到背景节点的条件概率
    cond_prob = tf.reduce_sum(v_i_embed_second * v_j_embed_context, axis=-1, keepdims=False)
    # outputs的维度跟_batch_generator()的输出维度保持一致
    if order == 'first':
        outputs = [join_prob]
    elif order == 'second':
        outputs = [cond_prob]
    else:
        outputs = [join_prob, cond_prob]

    model = Model(inputs=[v_i, v_j], outputs=outputs)
    return model, {'first_embed': first_embed, 'second_embed': second_embed}


def kl_dist(y_true, y_pred):
    '''
    在一阶相似度中，正样本y_true=1，负样本y_true=0；在二阶相似度中，正样本y_true=1，负样本y_true=-1
    '''
    # y_true=0时把loss置为0，因为一阶相似度中不需要负样本
    dot = tf.where(tf.equal(y_true, 0), 1E9, y_true * y_pred)
    return -tf.math.log(tf.sigmoid(dot))


class LINE(object):
    def __init__(self, graph, nodesNum, batch_size, embedding_dim=100, negative_k=5, order='all'):
        '''
        :param graph: 二维数组，每行有3个元素[中心顶点的编号，背景顶点的编号，边的权重]
        :param nodesNum: 顶点的总数
        :param batch_size: mini-batch的大小
        :param embedding_dim: 顶点向量的维度
        :param negative_k: 负采样次数
        :param order: 训练几阶相似度
        '''
        if order not in ('first', 'second', 'all'):
            raise ValueError('mode must be first, second or all')
        self.graph = graph
        self.nodesNum = nodesNum
        self.batch_size = batch_size
        self.edgeNum = len(graph)
        self.embedding_dim = embedding_dim
        self.order = order
        self.negative_k = negative_k
        self._gen_sample_table()
        self.model, self.embedding = create_model(nodesNum, embedding_dim, order)
        optimizer = tf.keras.optimizers.RMSprop()
        self.model.compile(optimizer, loss=kl_dist)  # 自定义损失函数。不论output有一个还是两个，都使用这一种loss

    def _gen_sample_table(self):
        power = 0.75
        # 创建顶点采样表（负采样）
        node_degree = [0.0] * self.nodesNum
        for triple in self.graph:
            v_i = triple[0]
            weight = triple[2]
            node_degree[v_i] += weight  # 计算顶点的出度之和
        degree_sum = math.fsum([math.pow(d, power) for d in node_degree])  # 0.75次方是为了降低高频顶点被选中的概率
        norm_prob = [math.pow(d, power) / degree_sum for d in node_degree]
        self.NodeSampler = AliasSample(norm_prob)
        # 创建边采样表
        weight_sum = math.fsum([triple[2] for triple in self.graph])  # 边采样时不需要0.75次方，因为边采样不是负采样
        norm_prob = [float(triple[2]) / weight_sum for triple in self.graph]
        self.EdgeSampler = AliasSample(norm_prob)

    def _batch_generator(self):
        '''
        样本生成器，根据self.graph生成一个mini-batch的样本
        :return:
        '''
        start_index = 0
        while True:  # 这是个无限循环，可以一直生成样本
            end_index = min(start_index + self.batch_size, self.edgeNum)
            source = []  # 源顶点
            sink = []  # 汇顶点
            # 正采样（一阶和二阶相似度都需要）
            for i in range(start_index, end_index):
                edge_index = self.EdgeSampler.sample_i(i)  # 通过采样，此时的等价权重是1
                edge = self.graph[edge_index]
                vi = edge[0]
                vj = edge[1]
                source.append(vi)
                sink.append(vj)
            y = np.ones(len(source))
            if self.order == 'all':
                yield ([np.array(source), np.array(sink)], [y, y])  # 一阶loss和二阶loss是分开优化的，没有做融合
            else:
                yield ([np.array(source), np.array(sink)], [y])

            if self.order == 'all' or self.order == 'second':
                # 负采样（仅二阶相似度都需要）
                for i in range(self.negative_k):
                    sink = []  # 汇顶点
                    for i in range(len(source)):
                        sink.append(self.NodeSampler.sample())  # 随机选一个顶点作为背景顶点（负采样）
                    y = np.ones(len(source)) * -1
                    if self.order == 'all':
                        yield ([np.array(source), np.array(sink)], [np.zeros(len(source)), y])
                    else:
                        yield ([np.array(source), np.array(sink)], [y])
            start_index = end_index
            if start_index >= self.edgeNum:
                start_index = 0

    def train(self, epochs):
        sample_per_epoch = self.edgeNum * (1 + self.negative_k)  # 正采样每进行一次，负采样就要进行K次
        steps_per_epoch = (sample_per_epoch - 1) // self.batch_size + 1  # 每个epoch进行几次mini-batch训练
        self.model.fit(self._batch_generator(),
                       steps_per_epoch=steps_per_epoch,
                       epochs=epochs,
                       validation_freq=100)  # 可以把样本做成迭代器传给fit函数

    def get_embeddings(self):
        if self.order == 'first':
            return self.embedding['first_embed'].get_weights()[0]
        elif self.order == 'second':
            return self.embedding['second_embed'].get_weights()[0]
        else:
            return np.hstack(
                (self.embedding['first_embed'].get_weights()[0], self.embedding['second_embed'].get_weights()[0]))


def train_epinions(corpus_file, epochs, order):
    origin_graph = []
    node_set = set()
    with open(corpus_file) as fin:
        for line in fin:
            arr = line.strip().split()
            if len(arr) == 3:
                source = int(arr[0])
                sink = int(arr[1])
                weight = float(arr[2])
                node_set.add(source)
                node_set.add(sink)
                origin_graph.append((source, sink, weight))
    # 对每个node进行编号
    node2index = {node: index for index, node in enumerate(list(node_set))}
    graph = []
    for triple in origin_graph:
        source = node2index[triple[0]]
        sink = node2index[triple[1]]
        weight = triple[2]
        graph.append((source, sink, weight))

    line = LINE(graph, len(node_set), 64, order=order)
    line.train(epochs)
    return line, node2index


def test(line, node2index, u1, u2):
    embedding = line.get_embeddings()
    source1 = node2index[u1]
    source2 = node2index[u2]
    source_embed1 = embedding[source1]
    source_embed2 = embedding[source2]
    if source_embed1.shape[0] == line.embedding_dim:
        print(
            "simlarity {}".format(tf.reduce_sum(source_embed1 * source_embed2)))
    elif source_embed1.shape[0] == 2 * line.embedding_dim:
        first_sim = tf.reduce_sum(source_embed1[:line.embedding_dim] * source_embed2[:line.embedding_dim])
        second_sim = tf.reduce_sum(source_embed1[line.embedding_dim:] * source_embed2[line.embedding_dim:])
        print("first order simlarity {},second order simlarity {}".format(first_sim, second_sim))


if __name__ == "__main__":
    line, node2index = train_epinions("data/trust_data_small.txt", 2, 'all')
    test(line, node2index, 22605, 25278)
    test(line, node2index, 22605, 3195)
    test(line, node2index, 22605, 244)
    test(line, node2index, 22605, 5052)
