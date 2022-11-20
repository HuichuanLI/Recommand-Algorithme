from pathlib import Path

import tensorflow as tf

class InputFn:

    def __init__(self, parameter_server, feature_num=4, label_len=1,
                 n_parse_threads=4, shuffle_buffer_size=1024, batch_size=64):
        self.feature_len = feature_num
        self.label_len = label_len
        self.n_parse_threads = n_parse_threads
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.parameter_server = parameter_server
        self.prefetch_buffer_size = self.batch_size
        self.features_format = {
            "feature": tf.io.FixedLenFeature(self.feature_len, tf.int64),
            "label": tf.io.FixedLenFeature(self.label_len, tf.float32),
        }

    def _parse_example(self, example):
        return tf.io.parse_single_example(example, self.features_format)

    def _get_embedding(self, parsed):
        keys = parsed["feature"]
        embedding = tf.compat.v1.py_func(self.parameter_server.pull, [keys],
                                         tf.float32)
        result = {
            "feature": keys,
            "label": parsed["label"],
            "feature_embedding": embedding
        }
        return result

    def input_fn(self, data_dir, is_test=False):

        data_dir = Path(data_dir)
        files = [str(path) for path in list(data_dir.glob("*"))]

        dataset = tf.compat.v1.data.Dataset.list_files(files)
        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            lambda x: tf.compat.v1.data.TFRecordDataset(x), cycle_length=1)
        dataset = dataset.map(self._parse_example,
                              num_parallel_calls=self.n_parse_threads)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self._get_embedding,
                              num_parallel_calls=self.n_parse_threads)
        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)

        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        return iterator, iterator.get_next()