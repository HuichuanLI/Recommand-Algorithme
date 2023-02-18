# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import mmh3


# step01: 读取数据
# 对数据进行处理，hash化
# 数据值hash化，有两个作用，1.特征唯一化； 2.特征值较多的时候有一定降维效果
# hash方法有很多，可以采用MurmurHash
def mmh32hash64(str01):
    seed = 2021
    tmp = mmh3.hash(str01, seed=seed, signed=False)
    return tmp


names = [
    'id', 'date', 'user_id', 'product', 'campaign_id', 'webpage_id', 'product_category_id',
    'user_group_id', 'gender', 'age_level', 'user_depth', 'var_1'
]


def to_hash(line, is_test=False):
    line_json = line.strip().split(",")
    result = []
    for i, f in enumerate(line_json):
        if i == 0 or i == 1:
            continue
        if not is_test and i == len(line_json) - 1:
            result.append(f)
            continue
        w = mmh32hash64(names[i] + "=" + str(f))
        result.append(str(w))
    return ",".join(result)


def ext_train_test_to_hash():
    raw_data_train = "./raw/train.csv"

    raw_data_val = "./raw/test.csv"

    raw_data_lines = open(raw_data_train, encoding='utf8')
    raw_data_write_to_hash = open(raw_data_train + ".hash", 'w')
    val_data_write_to_hash = open("raw/val.csv.hash", 'w')
    for i, line in enumerate(raw_data_lines):
        if i == 0:
            continue
        tmp = to_hash(line, False)
        if tmp is not None:
            if random.randint(1, 10) > 8:
                val_data_write_to_hash.write(tmp + '\n')
            else:
                raw_data_write_to_hash.write(tmp + '\n')
    raw_data_write_to_hash.close()
    val_data_write_to_hash.close()
    raw_data_lines = open(raw_data_val, encoding='utf8')
    raw_data_write_to_hash = open(raw_data_val + ".hash", 'w')
    for i, line in enumerate(raw_data_lines):
        if i == 0:
            continue
        tmp = to_hash(line, True)
        if tmp is not None:
            raw_data_write_to_hash.write(tmp + '\n')
    raw_data_write_to_hash.close()


def get_tfrecords_example(featrue, label):
    tfrecords_features = {
        'feature': tf.train.Feature(
            int64_list=tf.train.Int64List(value=featrue)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=tfrecords_features))


def totfrecords():
    raw_data_train = "./raw/train.csv.hash"
    raw_data_val = "./raw/val.csv.hash"
    writer = tf.io.TFRecordWriter(raw_data_train + ".tfrecords")
    lines = open(raw_data_train)
    for i, line in enumerate(lines):
        line_arr = line.strip().split(",")
        # print(line_arr)
        feature = []
        label = []
        for j, feat in enumerate(line_arr):
            if j == len(line_arr) - 1:
                # print(feat)
                label.append(float(feat))
                continue
            feature.append(int(feat))

        example = get_tfrecords_example(feature, label)
        writer.write(example.SerializeToString())

    print("Process To tfrecord File: %s End" % raw_data_train)
    writer.close()

    writer = tf.io.TFRecordWriter(raw_data_val + ".tfrecords")
    lines = open(raw_data_val)
    for i, line in enumerate(lines):
        line_arr = line.strip().split(",")
        feature = []
        label = []
        for j, feat in enumerate(line_arr):
            if j == len(line_arr) - 1:
                label.append(float(feat))
                continue
            feature.append(int(feat))
        example = get_tfrecords_example(feature, label)
        writer.write(example.SerializeToString())
    print("Process To tfrecord File: %s End" % raw_data_val)
    writer.close()


def main():
    # step01: 处理训练数据, hash
    # ext_train_test_to_hash()
    # step02: tfrecord
    totfrecords()


if __name__ == '__main__':
    main()
