# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
from base_layer import CrossLayer, MLPLayer


class DCN(object):

    def __init__(self):
        # 特征
        self.feature_name_list = [
            'uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3', 'itag4'
        ]
        self.label_name = 'ctr'
        self.feature_size = 20000
        self.embedding_size = 8
        self.lr = 0.01
        self.checkpoint_dir = "../model/dcn"
        self.checkpoint_interval = True

        self.file_dir = "../data"
        self.n_threads = 2
        self.shuffle = 50
        self.batch = 4

        self.epochs = 1
        self.current_model_path = "../model/save_model"
        self.restore = False

    # 定义model
    def init_model(self):
        # 输入
        input_dic = {}
        for f in self.feature_name_list:
            input_dic[f] = tf.keras.Input(shape=(1,), name=f, dtype=tf.int64)

        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_size,
                                               self.embedding_size,
                                               embeddings_regularizer="l2")

        # 结构
        emb_list = []
        for k, v in input_dic.items():
            tmp = self.embed(v)
            emb_list.append(tmp)

        # concat
        emb_list = tf.concat(emb_list, axis=1)
        _input_out = tf.keras.layers.Flatten()(emb_list)
        _cross_out = CrossLayer(3)(_input_out)
        _deep_out = MLPLayer(units=[64, 8], activation='relu')(_input_out)
        _out = tf.concat([_cross_out, _deep_out], axis=1)

        _out = MLPLayer(units=[1], activation='sigmoid')(_out)
        self.model = tf.keras.Model(input_dic, _out)
        self.model.summary()

    # 定义loss
    def init_loss(self):
        self.loss = tf.keras.losses.BinaryCrossentropy()

    # 定义optam
    def init_opt(self):
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

    # 定义metric
    def init_metric(self):
        self.metric_auc = tf.keras.metrics.AUC(name="auc")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    # 保存模型
    def init_save_checkpoint(self):
        checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=2,
            checkpoint_interval=self.checkpoint_interval,
            step_counter=self.opt.iterations)

    # 准备输入数据
    def init_dataset(self, file_dir, is_train=True):

        def _parse_example(example):
            feats = {}
            feats["ctr"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
            for f in self.feature_name_list:
                feats[f] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
            feats = tf.io.parse_single_example(example, feats)
            return feats

        file_name_list = os.listdir(file_dir)
        files = []
        for i in range(len(file_name_list)):
            files.append(os.path.join(file_dir, file_name_list[i]))

        dataset = tf.data.Dataset.list_files(files)
        #
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=self.n_threads)
        #
        if is_train:
            dataset.shuffle(self.shuffle)
        #
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=self.n_threads)

        dataset = dataset.batch(self.batch)

        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    # train
    def train_step(self, ds):
        # start
        def train_loop_begin():
            self.metric_auc.reset_states()
            self.metric_loss.reset_states()

        # end
        def train_loop_end():
            result = {
                self.metric_auc.name: self.metric_auc.result().numpy(),
                self.metric_loss.name: self.metric_loss.result().numpy()
            }

            return result

        # loop
        def train_loop(inputs):
            with tf.GradientTape() as tape:
                target = inputs.pop(self.label_name)
                logits = self.model(inputs, training=True)
                scaled_loss = tf.reduce_sum(self.loss(target, logits))
                gradients = tape.gradient(scaled_loss,
                                          self.model.trainable_variables)

                self.opt.apply_gradients(
                    list(zip(gradients, self.model.trainable_variables)))

                self.metric_loss.update_state(scaled_loss)
                self.metric_auc.update_state(target, logits)

        # training
        step = 0
        try:
            inputs = iter(ds)
            train_loop_begin()
            while True:
                train_loop(next(inputs))
                step += 1
                if step % 500 == 0:
                    result = train_loop_end()
                    print(f"train over-{step} : {result}")
        except (StopIteration, tf.errors.OutOfRangeError):
            print(
                f"The train dataset iterator is exhausted after {step} steps.")

        result = train_loop_end()
        print(f"train over-{step} : {result}")
        return result

    # eval
    def eval_step(self, ds):
        # start
        def eval_loop_begin():
            self.metric_auc.reset_states()

        # end
        def eval_loop_end():
            result = {self.metric_auc.name: self.metric_auc.result().numpy()}
            return result

        # loop
        def eval_loop(inputs):
            target = inputs.pop(self.label_name)
            logits = self.model(inputs, training=True)
            self.metric_auc.update_state(target, logits)

        # eval
        step = 0
        try:
            inputs = iter(ds)
            eval_loop_begin()
            while True:
                eval_loop(next(inputs))
                step += 1
                if step % 500 == 0:
                    result = eval_loop_end()
                    print(f"eval over-{step} : {result}")
        except (StopIteration, tf.errors.OutOfRangeError):
            print(
                f"The eval dataset iterator is exhausted after {step} steps.")

        result = eval_loop_end()
        print(f"eval over-{step} : {result}")
        return result

    # run
    def run(self, train_ds, test_ds, mode="train_and_eval"):
        if self.restore:
            self.manager.restore_or_initialize()
        if mode == "train_and_eval" and train_ds is not None and test_ds is not None:
            for epoch in range(self.epochs):
                print(f"=====epoch: {epoch}=====")
                train_result = self.train_step(train_ds)
                eval_result = self.eval_step(test_ds)
                if train_result[self.metric_auc.name] > 0.5 or eval_result[
                    self.metric_auc.name] > 0.5:
                    self.manager.save(checkpoint_number=epoch,
                                      check_interval=True)

        if mode == "train":
            for epoch in range(self.epochs):
                print(f"=====epoch: {epoch}=====")
                train_result = self.train_step(train_ds)
                if train_result[self.metric.name] > 0.5:
                    self.manager.save(checkpoint_number=epoch,
                                      check_interval=True)

        if mode == "eval":
            eval_result = self.eval_step(test_ds)

    # export
    def export_model(self):
        tf.saved_model.save(self.model, self.current_model_path)

    def load_model(self):
        self.imported = tf.saved_model.load(self.current_model_path)
        # self.f = imported.signatures["serving_default"]

    # infer
    def infer(self, x):
        result = self.imported(x)
        return result


if __name__ == "__main__":
    fm = DCN()
    train_ds = fm.init_dataset("../data/train")
    test_ds = fm.init_dataset("../data/test", is_train=False)
    # init model info
    fm.init_model()
    fm.init_loss()
    fm.init_opt()
    fm.init_metric()
    fm.init_save_checkpoint()

    # train
    fm.run(train_ds, test_ds, mode="train_and_eval")

    # save
    fm.export_model()
