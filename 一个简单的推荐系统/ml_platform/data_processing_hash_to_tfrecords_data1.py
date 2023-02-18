# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import json
import random
import redis


# step01: 读取数据
# 对数据进行处理，hash化
# 数据值hash化，有两个作用，1.特征唯一化； 2.特征值较多的时候有一定降维效果
# hash方法有很多，可以采用MurmurHash, 这里我们采用一种简单的方法实现hash
def bkdr2hash64(str01):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str01:
        hash = hash * seed + ord(s)
    return hash & mask60


def get_fm_recall_data(data_path, user_pool):
    ds = pd.read_csv(data_path)
    ds = ds[['user_id', 'article_id']]

    items = list(ds['article_id'].unique())

    ds = zip(ds['user_id'], ds['article_id'])
    data = []

    for u, i in ds:
        user_info = json.loads(user_pool.get(u))
        data.append(
            (u, user_info['environment'], user_info['region'], i, 1))
        for t in random.sample(items, 10):
            data.append(
                (u, user_info['environment'], user_info['region'], t, 0))

    random.shuffle(data)
    num = int(len(data) * 0.8)
    train = data[: num]
    test = data[num:]
    return train, test


def tohash(data, save_path):
    print("start write in {}...".format(save_path))
    wf = open(save_path, "w")
    for line in data:
        user_id = bkdr2hash64("user_id=" + str(line[0]))
        environment = bkdr2hash64("environment=" + str(line[1]))
        region = bkdr2hash64("region=" + str(line[2]))
        item_id = bkdr2hash64("article_id=" + str(line[3]))
        wf.write(str(user_id) + "," + str(environment) + ","
                 + str(region) + "," + str(item_id)
                 + "," + str(line[4]) + "\n")

    wf.close()


def get_tfrecords_example(featrue, label):
    tfrecords_features = {
        'feature': tf.train.Feature(
            int64_list=tf.train.Int64List(value=featrue)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=tfrecords_features))


def totfrecords(file, save_dir):
    print("Process To tfrecord File: %s ..." % file)
    num = 0
    writer = tf.io.TFRecordWriter(
        save_dir + "/" + "part-0000" + str(num) + ".tfrecords")
    lines = open(file)
    for i, line in enumerate(lines):
        tmp = line.strip().split(",")
        feature = [int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])]
        label = [float(tmp[4])]
        example = get_tfrecords_example(feature, label)
        writer.write(example.SerializeToString())
        if (i + 1) % 100000 == 0:
            writer.close()
            num += 1
            writer = tf.io.TFRecordWriter(
                save_dir + "/" + "part-0000" + str(num) + ".tfrecords")
    print("Process To tfrecord File: %s End" % file)
    writer.close()


def main():
    # step01: 处理训练数据
    click_path = "../data/click_log.csv"
    redis_url = 'redis://127.0.0.1:6379/1'
    pool = redis.from_url(redis_url)
    train, test = get_fm_recall_data(click_path, pool)

    train_tohash = "../data/train_tohash"
    test_tohash = "../data/test_tohash"
    tohash(train, train_tohash)
    tohash(test, test_tohash)
    print("data to hash processing end...")

    #
    train_tfrecod_path = "../data/train"
    val_tfrecod_path = "../data/val"
    totfrecords(train_tohash, train_tfrecod_path)
    totfrecords(test_tohash, val_tfrecod_path)
    print("data hash to tfrecords processing end ...")


if __name__ == '__main__':
    main()
