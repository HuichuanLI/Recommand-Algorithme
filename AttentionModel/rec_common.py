# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 5:34 PM
# @Author  : lihuichuan
# @File    : rec_common.py

import os, sys

sys.path.insert(0, os.getcwd())
from fast_text import REC_WV_FILE, EMBEDDING_DIM
import fasttext
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score

BATCH = 256
TRAIN_FILE = "data/train_rec.txt"
TEST_FILE = "data/test_rec.txt"
WORD_INDEX_FILE = "data/rec_word_index.txt"
ID_INDEX_FILE = "data/rec_id_index.txt"
MAX_LEN_OF_USER_LIKEWORDS = 20
MAX_LEN_OF_USER_CLICKS = 10
MAX_LEN_OF_ITEM_KEYWORDS = 4


def gen_corpus(corpus_file, word_index_file, id_index_file, label_file, user_likewords_file, user_clicks_file,
               item_id_file, item_keywords_file):
    '''
    :param corpus_file: label,user_likewords,user_clicks,item_id,item_keywords
    :param word_index_file:
    :param id_index_file:
    :param label_file:
    :param user_likewords_file:
    :param user_clicks_file:
    :param item_id_file:
    :param item_keywords_file:
    :return:
    '''
    word_index_vector = load_word_index(word_index_file)
    itemid_index = load_itemid_index(id_index_file)

    Y = []
    user_likewords = []
    user_clicks = []
    item_id = []
    item_keywords = []
    with open(corpus_file, encoding="utf-8") as fin:
        next(fin)  # 跳过第一行
        for line in fin:
            arr = line.strip().split(",")
            if len(arr) == 5:
                Y.append(int(arr[0]))
                user_likeword = []
                for ele in arr[1].split("|"):
                    user_likeword.append(word_index_vector.get(ele, (0, 0))[0])
                user_likewords.append(user_likeword)
                user_click = []
                for ele in arr[2].split("|"):
                    user_click.append(itemid_index.get(ele, 0))
                user_clicks.append(user_click)
                item_id.append(itemid_index.get(arr[3], 0))
                item_keyword = []
                for ele in arr[4].split("|"):
                    item_keyword.append(word_index_vector.get(ele, (0, 0))[0])
                item_keywords.append(item_keyword)

    if len(Y) == len(user_clicks) and len(Y) == len(user_clicks) and len(Y) == len(item_id) and len(Y) == len(
            item_keywords):
        np.save(label_file, np.array(Y))
        np.save(user_likewords_file, np.array(user_likewords))
        np.save(user_clicks_file, np.array(user_clicks))
        np.save(item_id_file, np.array(item_id))
        np.save(item_keywords_file, np.array(item_keywords))

    else:
        raise Exception("label count not equal to feature count")


def load_corpus(label_file, user_likewords_file, user_clicks_file, item_id_file, item_keywords_file):
    Y = np.load(label_file, allow_pickle=True)
    user_likewords = np.load(user_likewords_file, allow_pickle=True)
    user_clicks = np.load(user_clicks_file, allow_pickle=True)
    item_id = np.load(item_id_file, allow_pickle=True)
    item_keywords = np.load(item_keywords_file, allow_pickle=True)
    user_likewords = pad_sequences(user_likewords, maxlen=MAX_LEN_OF_USER_LIKEWORDS, padding="post", truncating="post",
                                   dtype="int", value=0)
    user_clicks = pad_sequences(user_clicks, maxlen=MAX_LEN_OF_USER_CLICKS, padding="post", truncating="post",
                                dtype="int", value=0)
    item_keywords = pad_sequences(item_keywords, maxlen=MAX_LEN_OF_ITEM_KEYWORDS, padding="post", truncating="post",
                                  dtype="int", value=0)
    return (Y, user_likewords, user_clicks, item_id, item_keywords)


def index_words(corpus_file, index_file):
    '''
    :param corpus_file: label,user_likewords,user_clicks,item_id,item_keywords
    :param index_file:
    :return:
    '''

    word_set = set()
    with open(corpus_file, encoding="utf-8") as fin:
        next(fin)
        for line in fin:
            arr = line.strip().split(",")
            if len(arr) == 5:
                for ele in arr[1].split("|"):
                    if ele:
                        word_set.add(ele)
                for ele in arr[4].split("|"):
                    if ele:
                        word_set.add(ele)

    wv_model = fasttext.load_model(REC_WV_FILE)
    with open(index_file, "w", encoding="utf-8") as fout:
        for idx, word in enumerate(list(word_set)):
            vector = wv_model[word]
            fout.write("{}\t{}\t{}\n".format(word, idx + 1, "\t".join(map(str, vector))))  # index从1开始，0留给未出现过的word


def load_word_index(index_file):
    word_index_vector = {}
    with open(index_file, encoding="utf-8") as f_in:
        for line in f_in:
            arr = line.strip().split()
            if len(arr) == 2 + EMBEDDING_DIM:
                vector = [0.] * EMBEDDING_DIM
                for i in range(2, 2 + EMBEDDING_DIM):
                    vector[i - 2] = float(arr[i])
                word_index_vector[arr[0]] = (int(arr[1]), vector)
    return word_index_vector


def index_itemid(corpus_file, index_file):
    id_set = set()
    with open(corpus_file, encoding="utf-8") as fin:
        next(fin)
        for line in fin:
            arr = line.strip().split(",")
            if len(arr) == 5:
                for ele in arr[2].split("|"):
                    if ele:
                        id_set.add(int(ele))
                if arr[3]:
                    id_set.add(int(arr[3]))
    with open(index_file, "w", encoding="utf-8") as fout:
        for idx, id in enumerate(list(id_set)):
            fout.write("{}\t{}\n".format(id, idx + 1))  # index从1开始，0留给未出现过的itemid


def load_itemid_index(index_file):
    rect = {}
    with open(index_file, encoding="utf-8") as fin:
        for line in fin:
            arr = line.strip().split("\t")
            if len(arr) == 2:
                rect[arr[0]] = int(arr[1])
    return rect


def pad_zero(input, max_len):
    if len(input) >= max_len:
        return input[:max_len]
    else:
        input.extend([0] * (max_len - len(input)))
        return input


def gen_rec_word_corpus(corpus_file):
    with open(corpus_file, "w", encoding="utf-8") as fout:
        with open(TRAIN_FILE, encoding="utf-8") as fin:
            fin.readline()  # 跳过第一行
            for line in fin:
                arr = line.strip().split(",")
                words = arr[1].split("|")
                fout.write("\t".join(words) + "\n")


def cal_auc(model, valid_files):
    label_file, user_likewords_file, user_clicks_file, item_id_file, item_keywords_file = valid_files
    val_label, val_user_likewords, val_user_clicks, val_item_id, val_item_keywords = load_corpus(label_file,
                                                                                                 user_likewords_file,
                                                                                                 user_clicks_file,
                                                                                                 item_id_file,
                                                                                                 item_keywords_file)
    y_pred = model.predict(x=[val_user_clicks, val_user_likewords, val_item_id, val_item_keywords])
    auc = roc_auc_score(val_label, y_pred)
    print("auc={:.4f}".format(auc))


if __name__ == "__main__":
    gen_rec_word_corpus("data/like_words.txt")

    index_words(TRAIN_FILE, WORD_INDEX_FILE)
    index_itemid(TRAIN_FILE, ID_INDEX_FILE)

    label_file = "data/rec_train/label.npy"
    user_likewords_file = "data/rec_train/user_likewords.npy"
    user_clicks_file = "data/rec_train/user_clicks.npy"
    item_id_file = "data/rec_train/item_id.npy"
    item_keywords_file = "data/rec_train/item_keywords.npy"
    gen_corpus(TRAIN_FILE, WORD_INDEX_FILE, ID_INDEX_FILE, label_file, user_likewords_file, user_clicks_file,
               item_id_file, item_keywords_file)

    label_file = "data/rec_valid/label.npy"
    user_likewords_file = "data/rec_valid/user_likewords.npy"
    user_clicks_file = "data/rec_valid/user_clicks.npy"
    item_id_file = "data/rec_valid/item_id.npy"
    item_keywords_file = "data/rec_valid/item_keywords.npy"
    gen_corpus(TEST_FILE, WORD_INDEX_FILE, ID_INDEX_FILE, label_file, user_likewords_file, user_clicks_file,
               item_id_file, item_keywords_file)
