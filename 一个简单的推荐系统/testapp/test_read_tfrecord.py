# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/5/6|23:14
# @Moto   : Knowledge comes from decomposition

import tensorflow as tf
import argparse
import struct
parser = argparse.ArgumentParser()
parser.add_argument('--tfrecord_path',
                    default="17.hash.tfrecords")
args = parser.parse_args()

tfrecord_path = args.tfrecord_path
def read_tf():
    for serialized_example in tf.compat.v1.python_io.tf_record_iterator(tfrecord_path):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        # x = example.features
        # print(x)
        feature = example.features.feature["feature"].int64_list.value
        print("feature: ")
        print(feature)
        label = example.features.feature["label"].float_list.value
        print("label: ")
        print(label)
        break
read_tf()





