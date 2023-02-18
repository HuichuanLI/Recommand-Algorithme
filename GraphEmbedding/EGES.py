# -*- coding: utf-8 -*-
# @Time    : 2021/10/31 9:54 PM
# @Author  : zhangchaoyang
# @File    : EGES.py

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding
from tensorflow_addons.optimizers import AdamW


class EGES(Model):
    '''Enhanced Graph Embedding with Side Information'''

    def __init__(self, itemNum, sideInfoNum, embedding_dim):
        super(EGES, self).__init__()
        self.itemNum = itemNum
        self.sideInfoCount = len(sideInfoNum)
        self.item_embed = Embedding(itemNum, embedding_dim)#item_id对应一个Embedding层
        self.side_embeds = []
        for num in sideInfoNum:#每一个side info也对应一个Embedding层
            self.side_embeds.append(Embedding(num, embedding_dim))
        self.W = Embedding(itemNum, 1 + self.sideInfoCount)  # 权重矩阵用Embedding层来实现

    def weighted_pooling(self, info_i):
        item_id = info_i[:, 0]  # info_i第0维上是batch_size
        a = self.W(item_id)  # 取得该商品上各个属性的重要度
        weight = tf.expand_dims(tf.nn.softmax(a, axis=0), axis=-1)  # 取softmax，确保权重大于0

        item_vector = tf.expand_dims(self.item_embed(item_id), axis=1)#取得item_id的embedding
        vectors = [item_vector]
        for i in range(1, self.sideInfoCount + 1):#取得各side info的embedding
            side_vector = tf.expand_dims(self.side_embeds[i - 1](info_i[:, i]),
                                         axis=1)
            vectors.append(side_vector)
        vectors = tf.concat(vectors, axis=1)  # 各个属性的向量

        sum = tf.reduce_sum(tf.multiply(weight, vectors), axis=1)  # 对各属性的向量求加权和
        return sum

    def call(self, inputs, training=None, mask=None):
        info_i = inputs[:, :1 + self.sideInfoCount]  # x的左半部分是中心词，itemId+各SideInfo的ID
        info_j = inputs[:, 1 + self.sideInfoCount:]  # x的右半部分是背景词，itemId+各SideInfo的ID
        vector_i = self.weighted_pooling(info_i)
        vector_j = self.weighted_pooling(info_j)
        cond_prob = tf.reduce_sum(vector_i * vector_j, axis=-1, keepdims=True)  # 中心向量和背景向量求内积。简单地就用向量内积来表示两个顶点共现的概率
        return cond_prob


def train_GESI(x, y, itemNum, sideInfoNum, embedding_dim):
    model = EGES(itemNum, sideInfoNum, embedding_dim)
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)  # 优化算法
    model.compile(loss=sample_loss, optimizer=optimizer)
    model.fit(x, y, batch_size=16, epochs=3)
    return model


def sample_loss(y_true, y_pred):
    # y_true=1是正样本，y_true=0是负样本
    dot = tf.where(tf.equal(y_true, 0), 1E9, y_true * y_pred)
    return -tf.math.log(tf.sigmoid(dot))


if __name__ == "__main__":
    itemNum = 10000  # itemID的个数
    sideInfoNum = [20000, 30000]  # 有两个side info，对应它们取值的个数
    total_sample = 1000  # 样本总数
    embedding_dim = 100

    # x的左半部分是中心词
    x = np.random.randint(low=0, high=itemNum - 1, size=(total_sample, 1))
    for cnt in sideInfoNum:
        x = np.hstack([x, np.random.randint(low=0, high=cnt - 1, size=(total_sample, 1))])

    # x的右半部分是背景词
    x = np.hstack([x, np.random.randint(low=0, high=itemNum - 1, size=(total_sample, 1))])
    for cnt in sideInfoNum:
        x = np.hstack([x, np.random.randint(low=0, high=cnt - 1, size=(total_sample, 1))])

    # 生成一些正样本和负样本
    y = np.random.randn(total_sample, 1)
    y = np.where(y >= 0, 1, 0)

    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    model = train_GESI(x, y, itemNum, sideInfoNum, embedding_dim)

    unseen = tf.convert_to_tensor([[8, 3, 5]], dtype=tf.float64)
    vector = model.weighted_pooling(unseen)
    print(vector)
