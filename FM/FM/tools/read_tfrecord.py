# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string('tfrecord_path',
                    default='../data/mini_ec.test.tfrecord',
                    help='tfrecord_path')


def read_tf(_):
    ind = 0
    for serialized_example in tf.compat.v1.python_io.tf_record_iterator(
            FLAGS.tfrecord_path):
        ind += 1
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        sample = example.features.feature
        print(sample)
        print("================")
        if ind > 1:
            break


if __name__ == "__main__":
    app.run(read_tf)
