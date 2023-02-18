# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import os


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


def ext_train_test(raw_data_path):
    names = ['goods_id', 'user_id', 'user_session', 'user_type', 'user_cate1',
             'user_cate2', 'user_cate3', 'user_env', 'label']

    raw_data_path = "../data/rawdata"
    files_path = []
    for file in os.listdir(raw_data_path):
        files_path.append(os.path.join(raw_data_path, file))

    print("all file: ", files_path)

    files_path.sort()
    train_data_file = files_path[:-1]
    test_data_file = files_path[-1:]

    train_ds_list = []
    for f in train_data_file:
        train_ds_list.append(pd.read_csv(f, header=None, names=names))

    train_ds = pd.concat(train_ds_list, axis=0, ignore_index=True)
    train_ds = train_ds.astype(str)

    print(train_ds.head(5))

    test_ds_list = []
    for f in test_data_file:
        test_ds_list.append(pd.read_csv(f, header=None, names=names))

    test_ds = pd.concat(test_ds_list, axis=0, ignore_index=True)
    test_ds = test_ds.astype(str)
    print(test_ds.head(5))

    return train_ds, test_ds


def tohash(data, save_path):
    # names = ['goods_id', 'user_id', 'user_session', 'user_type',
    # 'goods_cate1', 'goods_state', 'user_pref', 'user_env', 'label']
    print("start write in {}...".format(save_path))
    wf = open(save_path, "w")

    for line in data.values:
        goods_id = bkdr2hash64("goods_id=" + str(line[0]))
        user_id = bkdr2hash64("user_id=" + str(line[1]))
        user_session = bkdr2hash64("user_session=" + str(line[2]))
        user_type = bkdr2hash64("user_type=" + str(line[3]))
        user_cate1 = bkdr2hash64("user_cate1=" + str(line[4]))
        user_cate2 = bkdr2hash64("user_cate2=" + str(line[5]))
        user_cate3 = bkdr2hash64("user_cate3=" + str(line[6]))
        user_env = bkdr2hash64("user_env=" + str(line[7]))

        wf.write(str(goods_id) + "," + str(user_id) + ","
                 + str(user_session) + "," + str(user_type) + ","
                 + str(user_cate1) + "," + str(user_cate2) + ","
                 + str(user_cate3) + "," + str(user_env)
                 + "," + str(line[8]) + "\n")

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
        feature = [int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]),
                   int(tmp[4]), int(tmp[5]), int(tmp[6]), int(tmp[7])]
        label = [float(tmp[8])]
        example = get_tfrecords_example(feature, label)
        writer.write(example.SerializeToString())
        if (i + 1) % 20000 == 0:
            writer.close()
            num += 1
            writer = tf.io.TFRecordWriter(
                save_dir + "/" + "part-0000" + str(num) + ".tfrecords")
    print("Process To tfrecord File: %s End" % file)
    writer.close()


def main():
    # step01: 处理训练数据
    click_path = "rawdata"
    train, test = ext_train_test(click_path)

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
