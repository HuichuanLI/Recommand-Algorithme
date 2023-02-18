# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 6:06 PM
# @Author  : zhangchaoyang
# @File    : din2.py

import os, sys

sys.path.insert(0, os.getcwd())
import tensorflow as tf
from keras.activations import sigmoid, relu
from fast_text import EMBEDDING_DIM
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Embedding, Input, Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from rec_common import load_corpus, BATCH, MAX_LEN_OF_USER_LIKEWORDS, MAX_LEN_OF_USER_CLICKS, MAX_LEN_OF_ITEM_KEYWORDS, \
    load_word_index, WORD_INDEX_FILE, load_itemid_index, ID_INDEX_FILE, cal_auc
from term_weight.common import draw_train_history


class SumAttentionLayer(Model):
    def __init__(self):
        super(SumAttentionLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return tf.reduce_sum(inputs, axis=1)  # inputs最后几行是0行，用mean会把0向量也平均进去


class DotAttentionLayer(Model):
    def __init__(self):
        super(DotAttentionLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        if "query" not in kwargs:
            raise Exception("must have a input argument named query")
        query = kwargs["query"]
        prod = tf.matmul(
            query,
            tf.transpose(inputs, perm=(0, 2, 1)))  # multiply是逐元素相剩，matmul是矩阵相乘。第2维和第1维互换，第0维是batch-size的大小，即样本的个数
        weights = tf.nn.softmax(prod / tf.sqrt(float(inputs.shape[2])), axis=2)  # 行内归一化
        return tf.reduce_sum(tf.matmul(weights, inputs), axis=1)  # inputs最后几行是0行，用mean会把0向量也平均进去


class MlpAttentionLayer(Model):
    def __init__(self, embedding_dim):
        super(MlpAttentionLayer, self).__init__()
        self.fc = Sequential([
            BatchNormalization(),
            Dense(units=embedding_dim, use_bias=False, activation=relu),
            BatchNormalization(),
            Dense(units=1, use_bias=False, activation=sigmoid)
        ])

    def call(self, inputs, *args, **kwargs):
        if "query" not in kwargs:
            raise Exception("must have a input argument named query")
        query = kwargs["query"]
        weights = []
        for i in range(inputs.shape[1]):
            input = inputs[:, i, :]  # 第i个历史点击
            sub = tf.subtract(input, query)
            cat = tf.concat([input, sub, query], axis=1)
            weight = self.fc(cat)
            weights.append(weight)
        weights = tf.expand_dims(tf.concat(weights, axis=1), axis=2)  # tensorflow中的expand_dims相当于torch中的unsqueeze
        weighted_sum = tf.reduce_sum(tf.multiply(inputs, weights), axis=1)
        return weighted_sum  # inputs最后几行是0行，用mean会把0向量也平均进去


class DIN(object):
    def __init__(self, word2index, id2index, attention_type="sum"):
        word_count = len(word2index)
        id_count = len(id2index)
        # print("word count {}, id count {}".format(word_count, id_count))
        word_embedding_matrix = np.zeros(
            shape=(word_count + 1, EMBEDDING_DIM))  # embedding_matrix的第0行为0向量，padding时就是用的第0行
        # embedding_matrix中存储所有的word vector
        for word, tuple in word2index.items():
            i = tuple[0]
            vector = tuple[1]  # 预先训练好的word vector
            word_embedding_matrix[i] = vector
        self.word_embed_layer = Embedding(input_dim=word_count + 1,  # weights的第一维大小
                                          output_dim=EMBEDDING_DIM,  # weights的第二维大小
                                          weights=[word_embedding_matrix],  # 使用预先训练好的word vector
                                          trainable=True,  # 允许再训练
                                          )
        self.id_embed_layer = Embedding(input_dim=id_count + 1,  # weights的第一维大小
                                        output_dim=EMBEDDING_DIM,  # weights的第二维大小
                                        trainable=True,  # 允许再训练
                                        )
        if attention_type == "mlp":
            self.word_attention_layer = MlpAttentionLayer(EMBEDDING_DIM)
            self.id_attention_layer = MlpAttentionLayer(EMBEDDING_DIM)
        elif attention_type == "dot":
            self.word_attention_layer = DotAttentionLayer()
            self.id_attention_layer = DotAttentionLayer()
        elif attention_type == "sum":
            self.word_attention_layer = SumAttentionLayer()
            self.id_attention_layer = SumAttentionLayer()
        else:
            raise Exception("unsupported attention type {}".format(attention_type))
        self.fc = Sequential([
            BatchNormalization(), Dense(units=EMBEDDING_DIM, activation=relu),
            BatchNormalization(), Dense(units=EMBEDDING_DIM // 2, activation=relu),
            BatchNormalization(), Dense(units=EMBEDDING_DIM // 4, activation=relu),
            BatchNormalization(), Dense(units=1, activation=sigmoid),
        ])

    def train(self, train_files, valid_files):
        label_file, user_likewords_file, user_clicks_file, item_id_file, item_keywords_file = train_files
        label, user_likewords, user_clicks, item_id, item_keywords = load_corpus(label_file,
                                                                                 user_likewords_file,
                                                                                 user_clicks_file,
                                                                                 item_id_file,
                                                                                 item_keywords_file)
        print("total {} positive {}".format(label.shape[0], tf.reduce_sum(label, axis=0)))
        # total 854287 positive 85965
        user_clicks_input = Input(shape=(MAX_LEN_OF_USER_CLICKS,), dtype=tf.int32)
        user_likewords_input = Input(shape=(MAX_LEN_OF_USER_LIKEWORDS,), dtype=tf.int32)
        item_id_input = Input(shape=(1,), dtype=tf.int32)
        item_keywords_input = Input(shape=(MAX_LEN_OF_ITEM_KEYWORDS,), dtype=tf.int32)

        target_id = tf.squeeze(self.id_embed_layer(item_id_input), 1)
        click_ids = self.id_embed_layer(user_clicks_input)
        click_id_sum = self.id_attention_layer(click_ids, query=target_id)

        target_words = self.word_embed_layer(item_keywords_input)
        target_word = tf.reduce_sum(target_words, axis=1)
        user_words = self.word_embed_layer(user_likewords_input)
        user_word_sum = self.word_attention_layer(user_words, query=target_word)

        out = tf.concat([click_id_sum, user_word_sum, target_id, target_word], axis=1)
        out = tf.squeeze(self.fc(out), 1)

        self.model = Model(inputs=[user_clicks_input, user_likewords_input, item_id_input, item_keywords_input],
                           outputs=out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse'])
        early_stop = EarlyStopping(monitor='loss', min_delta=1, patience=2)  # loss变化很小时，提前停止迭代

        label_file, user_likewords_file, user_clicks_file, item_id_file, item_keywords_file = valid_files
        val_label, val_user_likewords, val_user_clicks, val_item_id, val_item_keywords = load_corpus(label_file,
                                                                                                     user_likewords_file,
                                                                                                     user_clicks_file,
                                                                                                     item_id_file,
                                                                                                     item_keywords_file)
        history = self.model.fit(x=[user_clicks, user_likewords, item_id, item_keywords], y=label, batch_size=BATCH,
                                 epochs=2,
                                 validation_data=(
                                     [val_user_clicks, val_user_likewords, val_item_id, val_item_keywords], val_label),
                                 callbacks=[early_stop])
        return history


if __name__ == "__main__":
    word2index = load_word_index(WORD_INDEX_FILE)
    id2index = load_itemid_index(ID_INDEX_FILE)
    din = DIN(word2index, id2index, attention_type="mlp")
    train_files = ("data/rec_train/label.npy",
                   "data/rec_train/user_likewords.npy",
                   "data/rec_train/user_clicks.npy",
                   "data/rec_train/item_id.npy",
                   "data/rec_train/item_keywords.npy")
    valid_files = ("data/rec_valid/label.npy",
                   "data/rec_valid/user_likewords.npy",
                   "data/rec_valid/user_clicks.npy",
                   "data/rec_valid/item_id.npy",
                   "data/rec_valid/item_keywords.npy")
    history = din.train(train_files, valid_files)
    draw_train_history(history, "data/din_history.png")
    cal_auc(din.model, valid_files)