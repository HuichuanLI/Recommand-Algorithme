# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import random
# step01
user_feature_path = "user_feature.dat"
item_feature_path = "item_feature.dat"
shop = "shop.dat"
train_path = "mini_ec.train.tfrecord"
test_path = "mini_ec.test.tfrecord"
index_path = "index"

user_feature_dic = {}
lines = open(user_feature_path, encoding="utf8")
for line in lines:
    line = line.strip().split(",")
    user_feature_dic[line[0]] = [line[1], line[2]]

item_feature_dic = {}
lines = open(item_feature_path, encoding="utf8")
for line in lines:
    line = line.strip().split(",")
    item_feature_dic[line[0]] = [line[1], line[2], line[3]]

train_writer = tf.io.TFRecordWriter(train_path)
test_writer = tf.io.TFRecordWriter(test_path)
index_writer = open(index_path, "w")

feature_name_list = [
    'user', 'item', 'ctr', 'utag1', 'utag2', 'itag1', 'itag2', 'itag3'
]
slot = {}


def write_index(slot: dict, writer):
    for k, v in slot.items():
        writer.write(f"{k}\t{v}\n")


def to_tfrecord(line, writer):
    sample = {}
    user = line[1]
    item = line[2]
    ctr = float(line[3])
    sample["ctr"] = tf.train.Feature(float_list=tf.train.FloatList(
        value=[ctr]))
    # user
    if user not in slot.keys():
        slot[user] = len(slot)
    value = [slot[user]]
    sample["user"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=value))
    # userfeature
    tmp = ["utag1", "utag2"]
    uf = user_feature_dic.get(user, ["utag1", "utag2"])
    for i, f in enumerate(uf):
        if f not in slot.keys():
            slot[f] = len(slot)
        value = [slot[f]]
        sample[tmp[i]] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=value))

    # item
    if item not in slot.keys():
        slot[item] = len(slot)
    value = [slot[item]]
    sample["item"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=value))
    # itemfeature
    tmp = ['itag1', 'itag2', 'itag3']
    uf = item_feature_dic.get(user, ['itag1', 'itag2', 'itag3'])
    for i, f in enumerate(uf):
        if f not in slot.keys():
            slot[f] = len(slot)
        value = [slot[f]]
        sample[tmp[i]] = tf.train.Feature(int64_list=tf.train.Int64List(
            value=value))

    sample = tf.train.Example(features=tf.train.Features(feature=sample))
    writer.write(sample.SerializeToString())


# slot -> dict
data_num = 100000
lines = open(shop, encoding="utf8")
for i, line in enumerate(lines):
    if i > data_num:
        break
    line = line.strip().split(",")
    if random.randint(1, 10) > 8:
        to_tfrecord(line, test_writer)
    else:
        to_tfrecord(line, train_writer)

write_index(slot, index_writer)

index_writer.close()
train_writer.close()
test_writer.close()
