# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 9:05 AM
# @Author  : zhangchaoyang
# @File    : attention.py

import os, sys

sys.path.insert(0, os.getcwd())
import tensorflow as tf
import fasttext
from fast_text import JOB_WV_FILE, EMBEDDING_DIM

MAX_SEQUENCE_LENGTH = 7  # 句子中最多只能有7个词


def attention(V, K, Q):
    prod = tf.matmul(Q,
                     tf.transpose(K, perm=(0, 2, 1)))  # multiply是逐元素相剩，matmul是矩阵相乘。第2维和第1维互换，第0维是batch-size的大小，即样本的个数
    weights = tf.nn.softmax(prod / tf.sqrt(float(V.shape[2])), axis=2)  # 行内归一化
    result = tf.matmul(weights, V)
    return weights, result


def test_attention():
    V = tf.random.normal(shape=[1, 4, EMBEDDING_DIM])  # 随机初始化tensor。1个样本，序列长度为4
    K = tf.random.normal(shape=V.shape)  # K和V的shape需要保持一致
    Q = tf.random.normal(shape=[1, 6, EMBEDDING_DIM])
    W, output = attention(K, V, Q)
    print(W.shape)
    print(output.shape)
    print(W)


def attention_encoding(sentences):
    wv_model = fasttext.load_model(JOB_WV_FILE)
    value = []
    for sent in sentences:
        if len(sent) > MAX_SEQUENCE_LENGTH:
            sent = sent[:MAX_SEQUENCE_LENGTH]  # 长度超过MAX_SEQ_LEN则截断
        vectors = []
        for word in sent:
            vectors.append(wv_model[word])
        for i in range(len(sent), MAX_SEQUENCE_LENGTH):
            vectors.append([0.0] * EMBEDDING_DIM)  # 用0向量padding
        value.append(vectors)
    V = tf.convert_to_tensor(value)  # convert_to_tensor从numpy或list构建tensor
    print(V[0][0])  # 第一个句子第一个词对应的向量
    W, output = attention(V, V, V)
    print(W)  # W的后几列是padding位，我们希望它是0，怎么实现？DIN课里再讲
    print(output[0][0])  # attention重新编码后，第一个句子第一个词对应的向量


if __name__ == "__main__":
    test_attention()
    attention_encoding(sentences=[["服务端", "测试", "开发", "负责人"], ["java", "后端"]])
