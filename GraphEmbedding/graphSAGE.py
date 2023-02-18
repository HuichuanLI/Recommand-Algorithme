# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 7:33 PM
# @Author  : zhangchaoyang
# @File    : graphSAGE.py

import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Dense
from keras.models import Model
from collections import defaultdict
from tensorflow_addons.optimizers import AdamW


class GraphSAGE(Model):
    def __init__(self, adjacency_dict, neighbor_cnt_list, hidden_dim_list, embedding_dim, node_cnt):
        super(GraphSAGE, self).__init__()
        assert len(neighbor_cnt_list) == len(hidden_dim_list)
        self.adjacency_dict = adjacency_dict
        self.layer_num = len(neighbor_cnt_list)
        self.neighbor_cnt_list = neighbor_cnt_list
        self.hidden_dim_list = hidden_dim_list
        self.embed = Embedding(input_dim=node_cnt + 1, output_dim=embedding_dim)  # 初始，顶点的向量表示通过Embedding层得到
        fc_layers = []
        for i, hidden_dim in enumerate(hidden_dim_list[:-1]):
            fc_layers.append(Dense(units=hidden_dim, use_bias=True, activation='sigmoid',
                                   name="fc" + str(i)))  # 每过一层Dense，顶点的向量表示维度就要变
        fc_layers.append(Dense(units=hidden_dim_list[-1], use_bias=False, activation=None,
                               name="fc" + str(len(hidden_dim_list) - 1)))  # 最后一层先不用激活函数
        self.fc_layers = fc_layers

    def update_vector(self, vertex, k, embedding_dict):
        '''
        计算顶点vertex在第k层的向量表示
        :param vertex:
        :param k:
        :param embedding_dict: 存储每一个顶点在每一层的向量（同一个顶点在每一层的向量维度都不一样）
        :return:
        '''
        sampled_neighbors = np.random.choice(self.adjacency_dict[int(vertex)], size=(self.neighbor_cnt_list[k],))
        if k == len(self.neighbor_cnt_list) - 1:  # 如果已经到最外层
            neighbor_vectors = self.embed(sampled_neighbors)  # 通过embedding_lookup得到邻居的向量
        else:  # 如果还没到最外层，则neighbor的向量也需要通过递归调用aggregate函数来生成(更新)
            neighbor_vectors = []
            for neighbor in sampled_neighbors:
                neighbor_vectors.append(self.update_vector(tf.convert_to_tensor(neighbor), k + 1, embedding_dict))
        neighbor_aggregate = tf.reduce_mean(neighbor_vectors, axis=0, keepdims=True)  # 采样取平均的方法对邻居进行聚合
        h_v_k_plus_1 = embedding_dict[k + 1].get(int(vertex))
        if h_v_k_plus_1 is None:
            embedding = tf.expand_dims(self.embed(vertex), axis=0)
            h_v_k_plus_1 = embedding
        else:
            h_v_k_plus_1 = tf.convert_to_tensor(h_v_k_plus_1)
        con = tf.concat([h_v_k_plus_1, neighbor_aggregate], axis=1)  # 把vertex的向量和邻居聚合后的向量拼接起来
        h_v_k = tf.squeeze(self.fc_layers[k](con), axis=0)  # 过一层Dense，得到里面一层vertex的向量表示
        embedding_dict[k][int(vertex)] = h_v_k.numpy().tolist()
        return h_v_k

    def call(self, inputs):
        embeddings = []
        print(inputs)
        for vertex in inputs:  # 对一个batch里的每一个顶点，求其向量表示
            print(vertex)
            embeddings.append(self.update_vector(vertex, 0, embedding_dict=defaultdict(dict)))
        return tf.convert_to_tensor(embeddings)

    def train(self, x, y, batch_size, epochs):
        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)  # 优化算法
        # self.compile(optimizer=optimizer, loss=tf.nn.softmax_cross_entropy_with_logits)  # 这个地方会过softmax，所以最后一层Dense没用激活函数
        # self.fit(x, y, batch_size=batch_size, epochs=epochs)  #如果用fit去训练模型，则在aggregate函数里adjacency_dict[int(vertex)]会报错：Tensor is unhashable. Instead, use tensor.ref() as the key

        total = x.shape[0]
        for epoch in range(epochs):
            for begin in range(0, total, batch_size):
                end = begin + batch_size
                if end > total:
                    end = total
                batch_x = x[begin:end]
                batch_y = y[begin:end]
                with tf.GradientTape() as tape:
                    y_hat = self(batch_x)
                    loss = tf.nn.softmax_cross_entropy_with_logits(batch_y, y_hat)  # 多分类任务，过softmax后求交叉熵损失
                    grads = tape.gradient(loss, self.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.trainable_weights))
                    print("epoch {} step {} cross entropy loss {}".format(epoch, begin // batch_size,
                                                                          tf.reduce_mean(loss).numpy().tolist()))

if __name__ == "__main__":
    adjacency_dict = {1: [3, 5], 2: [3, 7, 4], 3: [1, 2, 6], 4: [2, 5, 6], 5: [1, 4], 6: [3, 4], 7: [2]}  # 邻接矩阵
    node_cnt = 7  # 总共有7个顶点
    embedding_dim = 4  # 初始每个顶点的向量维度是4
    neighbor_cnt_list = [3, 2]  # index=0表示最内层，即最内层每个顶点选3个邻居，再往外一层每个顶点选2个邻居
    cls_num = 10  # 一共10个类别，即最终每个顶点被表示为10维向量，进行softmax多分类
    hidden_dim_list = [cls_num, 8]  # index=0表示最内层。最内层顶点向量是10维，往外一层是8维，最初是4维

    x = tf.convert_to_tensor([1, 2, 3, 4, 5, 6, 7], dtype=tf.int32)
    y = tf.one_hot([0, 4, 4, 2, 1, 1, 9], cls_num)

    model = GraphSAGE(adjacency_dict, neighbor_cnt_list, hidden_dim_list, embedding_dim, node_cnt)
    # print(model(x))
    model.train(x, y, batch_size=2, epochs=4)
