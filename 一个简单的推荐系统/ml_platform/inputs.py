# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/11/11|10:17
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os


class InputFn:
    def __init__(self, local_ps, config):
        self.feature_len = config['feature_len']
        self.label_len = config['label_len']
        self.n_parse_threads = config['n_parse_threads']
        self.shuffle_buffer_size = config['shuffle_buffer_size']
        self.prefetch_buffer_size = config['prefetch_buffer_size']
        self.batch = config['batch']
        self.local_ps = local_ps

    def input_fn(self, data_dir, is_test=False):
        def _parse_example(example):
            features = {
                "feature": tf.io.FixedLenFeature([4], tf.int64),
                "label": tf.io.FixedLenFeature([1], tf.float32),
            }
            return tf.io.parse_single_example(example, features)

        def _get_embedding(parsed):
            keys = parsed["feature"]
            keys_array = tf.py_func(
                self.local_ps.pull, [keys],
                tf.float32)
            result = {
                "feature": parsed["feature"],
                "label": parsed["label"],
                "feature_embedding": keys_array,
            }
            return result

        file_list = os.listdir(data_dir)
        files = []
        for i in range(len(file_list)):
            files.append(os.path.join(data_dir, file_list[i]))

        dataset = tf.data.Dataset.list_files(files)
        # 数据复制多少份
        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()
        # 读取tfrecord数据
        dataset = dataset.interleave(
            lambda _: tf.data.TFRecordDataset(_),
            cycle_length=1
        )
        # 对tfrecord的数据进行解析
        dataset = dataset.map(
            _parse_example,
            num_parallel_calls=self.n_parse_threads)

        # batch data
        dataset = dataset.batch(
            self.batch, drop_remainder=True)

        dataset = dataset.map(
            _get_embedding,
            num_parallel_calls=self.n_parse_threads)

        # 对数据进行打乱
        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)

        # 数据预加载
        dataset = dataset.prefetch(
            buffer_size=self.prefetch_buffer_size)

        # 迭代器
        iterator = tf.data.make_initializable_iterator(dataset)
        return iterator, iterator.get_next()


if __name__ == '__main__':
    from ps import PS

    local_ps = PS(8)
    config = {
        "feature_len": 4,
        "embedding_dim": 17,
        "label_len": 1,
        "n_parse_threads": 4,
        "shuffle_buffer_size": 1024,
        "prefetch_buffer_size": 1,
        "batch": 16,
        "learning_rate": 0.01,

        "train_file": "/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个简单的推荐系统/data/train",
        "test_file": "/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个简单的推荐系统/data/val",
        "saved_embedding": "../data/saved_dnn_embedding",
        "max_steps": 200000,
        "train_log_iter": 1000,
        "test_show_step": 1000,
        "last_test_auc": 0.2,

        "saved_checkpoint": "checkpoint",
        "checkpoint_name": "dnn",

        "saved_pb": "../data1/saved_model",

        "input_tensor": ["input_tensor"],
        "output_tensor": ["output_tensor"]
    }

    inputs = InputFn(local_ps, config)
    data_dir = "/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个简单的推荐系统/data/train"
    train_itor, train_inputs = inputs.input_fn(data_dir, is_test=False)
    with tf.Session() as sess:
        sess.run(train_itor.initializer)
        for i in range(1):
            print(sess.run(train_inputs))
