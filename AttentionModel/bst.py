# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 6:06 PM
# @Author  : lihuichuan
# @File    : bst.py

import os, sys

sys.path.insert(0, os.getcwd())
import tensorflow as tf
from keras.activations import sigmoid, elu
from fast_text import EMBEDDING_DIM
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Embedding, Dense, ReLU, LayerNormalization, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping
from rec_common import load_corpus, BATCH, MAX_LEN_OF_USER_LIKEWORDS, MAX_LEN_OF_USER_CLICKS, MAX_LEN_OF_ITEM_KEYWORDS, \
    load_word_index, WORD_INDEX_FILE, load_itemid_index, ID_INDEX_FILE, cal_auc
from term_weight.common import draw_train_history
from tensorflow_addons.layers import multihead_attention


class MaskLayer(Model):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        if "mask" in kwargs:
            mask = kwargs["mask"]
            if mask is not None:
                return tf.expand_dims(tf.cast(mask, tf.float32), axis=-1) * inputs
        return inputs


class PositionalEncodingLayer(Model):
    """通过位置编码，学习序列信息"""

    def __init__(self, d_model, max_len):
        super(PositionalEncodingLayer, self).__init__()
        self.encoding = tf.Variable(tf.zeros(shape=[max_len, d_model], dtype=tf.float32))
        pos = tf.expand_dims(tf.convert_to_tensor(np.arange(start=0, stop=max_len, step=1), dtype=tf.float32),
                             1)
        _2i = tf.expand_dims(tf.convert_to_tensor(np.arange(start=0, stop=d_model, step=2), dtype=tf.float32),
                             0)
        self.encoding[:, 0::2].assign(tf.sin(pos / (10000 ** (
                _2i / d_model))))  # 需要通过assign赋值，否则会报错TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
        self.encoding[:, 1::2].assign(tf.cos(pos / (10000 ** (
                _2i / d_model))))  # pos是max_len行1列，_2i是1行d_model/2列，pos/_2i是max_len行d_model/2列（会自动进行broadcast）
        self.mask_layer = MaskLayer()

    def call(self, inputs, *args, **kwargs):
        return self.mask_layer(self.encoding, mask=kwargs.get("mask"))  # 确保padding位对应的位置编码是0向量


def masked_fill(t, mask, value):
    '''mask为0/False的位置替换为value，其它位置保持t中本来的值'''
    return value * (1 - tf.cast(mask, tf.float32)) + t * tf.cast(mask, tf.float32)


class ScaleDotAttentionLayer(Model):
    def __init__(self):
        super(ScaleDotAttentionLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        batch_size, head, seq_len, embedding_dim = inputs.shape
        if "query" not in kwargs:
            raise Exception("must have a input argument named query")
        query = kwargs["query"]
        if "key" not in kwargs:
            raise Exception("must have a input argument named key")
        key = kwargs["key"]
        prod = tf.matmul(query, tf.transpose(key, perm=(0, 1, 3, 2))) / tf.sqrt(float(embedding_dim))
        if "mask" in kwargs:
            mask = kwargs["mask"]
            prod = masked_fill(prod, mask,
                               -1e9)  # padding的位置替换为-1e9，这样它经过`之后是0，即在padding位置上不应该分配注意力，确保padding位经过attention后是0向量
        weights = tf.nn.softmax(prod, axis=3)  # 行内归一化
        return tf.matmul(weights, inputs)


class MultiHeadAttention(Model):
    '''多头注意力机制'''

    def __init__(self, embedding_dim, head):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head = head
        self.head_dim = embedding_dim // head  # 多头最后要拼接起来，所以每个头的dim是embedding_dim的1/head
        self.linear_k = Dense(units=self.head * self.head_dim, use_bias=False)  # 一次生成head个W_k
        self.linear_q = Dense(units=self.head * self.head_dim, use_bias=False)
        self.linear_v = Dense(units=self.head * self.head_dim, use_bias=False)
        self.attention = ScaleDotAttentionLayer()
        if self.head * self.head_dim != self.embedding_dim:  # 如果embedding_dim不能被head整除，则head_dim*head会小于embedding_dim
            self.restore_dim_layer = Dense(units=self.embedding_dim,
                                           use_bias=False)  # 还原输入维度。bias=False确保输入是0向量的时候输出也是0向量
        else:
            self.restore_dim_layer = None

    def call(self, inputs, *args, **kwargs):
        Q = self.linear_q(inputs)
        K = self.linear_q(inputs)
        V = self.linear_q(inputs)
        _, seq_len, embedding_dim = inputs.shape
        Q = tf.transpose(tf.reshape(Q, shape=[-1, seq_len, self.head, self.head_dim]),
                         perm=[0, 2, 1,
                               3])  # 三维变四维，把head这一维拆出来，然后放到batch_size这一维后面。因为ScaleDotAttentionLayer要求数据的最后两维是(seq_len, embedding_dim)
        K = tf.transpose(tf.reshape(K, shape=[-1, seq_len, self.head, self.head_dim]), perm=[0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, shape=[-1, seq_len, self.head, self.head_dim]), perm=[0, 2, 1, 3])
        mask = None
        if "mask" in kwargs:
            mask = kwargs["mask"]
            # mask的shape本来是(batch_size,seq_len),把它扩到(batch_size,head,seq_len,seq_len)--attention里权值矩阵的shape
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.expand_dims(mask, axis=1)
        context = self.attention(inputs=V, query=Q, key=K, mask=mask)
        context = tf.transpose(context, perm=[0, 2, 1, 3])  # 把head和seq_len这2个维度交换一下，即再把seq_len放到batch_size后面
        output = tf.reshape(context, shape=[-1, seq_len, self.head * self.head_dim])
        if self.restore_dim_layer:
            outputs = []
            for i in range(seq_len):
                vec = tf.expand_dims(output[:, i, :], axis=1)  # 等价于output[:, i:i+1, :]
                y = self.restore_dim_layer(vec)
                outputs.append(y)
            output = tf.concat(outputs, axis=1)
        return output


class PointwiseFeedForward(Model):
    def __init__(self, embedding_dim):
        super(PointwiseFeedForward, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = Sequential([
            # 输入维度是embedding_dim，即每一个小向量独立地过全连接网络，所以叫Pointwise
            Dense(units=embedding_dim * 4, use_bias=False),  # 原始论文中就是先把维度扩到4倍，过一个ReLU，再还原维度
            ReLU(),
            # use_bias=False 确保0向量经过fc后还是0向量
            Dense(units=embedding_dim, use_bias=False),
            Dropout(0.1)
        ])
        self.norm = LayerNormalization(axis=[1, 2])  # LayerNorm是行内归一化；而BatchNorm是行间归一化，即样本间归一化

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        output = self.fc(inputs)
        return self.norm(residual + output)


class Transformer(Model):
    def __init__(self, embedding_layer, embedding_dim, head, max_len):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.mask_layer = MaskLayer()
        self.pe = PositionalEncodingLayer(embedding_dim, max_len)
        self.embed = embedding_layer
        self.attention = MultiHeadAttention(embedding_dim, head)
        self.norm = LayerNormalization(axis=[1, 2])
        self.dropout = Dropout(0.1)
        self.ffn = PointwiseFeedForward(embedding_dim)

    def call(self, inputs, *args, **kwargs):
        key = self.embed(inputs)  # padding位是0，0经过embedding层后是0向量
        key = self.dropout(key)
        mask = inputs > 0
        key *= tf.sqrt(1.0 * self.embedding_dim)  # 原始论文的做法，embedding向量太小，为了让它和PositionEmbedding处于同一个scale，这里做了放大
        key = self.mask_layer(key, mask=mask)
        key = tf.add(key, self.pe(inputs, mask=mask))  # padding位是0，0经过PositionalEncodingLayer后是0向量
        output = self.attention(key, mask=mask)  # 要保证padding位经过attention后是0向量，就需要给padding位分配的注意力权重是0
        output = self.dropout(output)
        output = self.norm(output + key)
        output = self.ffn(output)
        return output


class DirectTransmit(Model):
    def __init__(self, embedding_layer, embedding_dim):
        super(DirectTransmit, self).__init__()
        self.embed = embedding_layer
        self.ffn = PointwiseFeedForward(embedding_dim)

    def call(self, inputs, *args, **kwargs):
        output = self.embed(inputs)
        output = self.ffn(output)
        return output


class BST(object):
    def __init__(self, embedding_dim, word2index, id2index, head, trans_type="transformer"):
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
        word_embed_layer = Embedding(input_dim=word_count + 1,  # weights的第一维大小
                                     output_dim=EMBEDDING_DIM,  # weights的第二维大小
                                     weights=[word_embedding_matrix],  # 使用预先训练好的word vector
                                     trainable=True,  # 允许再训练
                                     )
        id_embed_layer = Embedding(input_dim=id_count + 1,  # weights的第一维大小
                                   output_dim=EMBEDDING_DIM,  # weights的第二维大小
                                   trainable=True,  # 允许再训练
                                   )
        if trans_type == "transformer":
            self.word_transformer = Transformer(embedding_layer=word_embed_layer, embedding_dim=embedding_dim,
                                                head=head,
                                                max_len=MAX_LEN_OF_USER_LIKEWORDS + MAX_LEN_OF_ITEM_KEYWORDS)
            self.id_transformer = Transformer(embedding_layer=id_embed_layer, embedding_dim=embedding_dim, head=head,
                                              max_len=MAX_LEN_OF_USER_CLICKS + 1)
        else:
            self.word_transformer = DirectTransmit(embedding_layer=word_embed_layer, embedding_dim=embedding_dim)
            self.id_transformer = DirectTransmit(embedding_layer=id_embed_layer, embedding_dim=embedding_dim)
        self.fc = Sequential([
            BatchNormalization(), Dropout(0.5), Dense(units=1024, activation=elu),
            BatchNormalization(), Dropout(0.5), Dense(units=512, activation=elu),
            BatchNormalization(), Dropout(0.5), Dense(units=256, activation=elu),
            BatchNormalization(), Dropout(0.5), Dense(units=1, activation=sigmoid)
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

        ids = tf.concat([user_clicks_input, item_id_input], axis=1)
        ids = self.id_transformer(ids)
        words = tf.concat([user_likewords_input, item_keywords_input], axis=1)
        words = self.word_transformer(words)
        fc_input = tf.concat([ids, words], axis=1)
        fc_input = tf.reduce_sum(fc_input, axis=1)  # transform的输出向量按位相加
        out = tf.squeeze(self.fc(fc_input), axis=1)

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


def testTransformer():
    embedding_dim = 10
    max_len = 5
    inputs = tf.convert_to_tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
                                  dtype=tf.int32)
    embedding_layer = Embedding(input_dim=2, output_dim=embedding_dim)
    head = 3
    transformer = Transformer(embedding_layer, embedding_dim, head, max_len)
    output = transformer(inputs)
    print(output)


def testMultiHeadAttention():
    inputs = tf.convert_to_tensor([[1, 2], [1, 0]], dtype=tf.int32)
    lengths = [2, 1]
    seq_len = np.max(lengths)
    embedding_dim = 10
    head = 3

    embed_layer = Embedding(input_dim=3, output_dim=embedding_dim)
    mask_layer = MaskLayer()
    pl = PositionalEncodingLayer(embedding_dim, seq_len)

    key = embed_layer(inputs)  # padding位是0，0经过embedding层后是0向量
    mask = inputs > 0
    key = mask_layer(key, mask=mask)
    pe_result = pl(inputs, mask=mask)
    key = tf.add(key, pe_result)

    multihead_attention = MultiHeadAttention(embedding_dim, head)
    output = multihead_attention(key, mask=mask)
    print(output.shape)


def testPointwiseFeedForward():
    batch_size = 5
    seq_len = 5
    embedding_dim = 10
    input = tf.random.normal(shape=[batch_size, seq_len, embedding_dim])
    pnn = PointwiseFeedForward(embedding_dim)
    output = pnn(input)
    print(output.shape)


if __name__ == "__main__":
    # testMultiHeadAttention()
    # testPointwiseFeedForward()
    # testTransformer()

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
    word2index = load_word_index(WORD_INDEX_FILE)
    id2index = load_itemid_index(ID_INDEX_FILE)
    head = 5
    bst = BST(EMBEDDING_DIM, word2index, id2index, head,
              # trans_type="transformer"
              )
    history = bst.train(train_files, valid_files)
    draw_train_history(history, "data/bst_history.png")
    cal_auc(bst.model, valid_files)
