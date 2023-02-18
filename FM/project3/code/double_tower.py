# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
from base_layer import MLPLayer


class DoubleTower(object):

    def __init__(self):
        # 特征
        self.feature_name_list = [
            'user', 'item', 'utag1', 'utag2', 'itag1', 'itag2', 'itag3'
        ]
        self.user_feature_name = ['user', 'utag1', 'utag2']
        self.item_feature_name = ['item', 'itag1', 'itag2', 'itag3']
        self.label_name = 'ctr'
        self.feature_size = 10000
        self.embedding_size = 16
        self.lr = 0.01
        self.checkpoint_dir = "../model/dssm"
        self.checkpoint_interval = True

        self.file_dir = "../data"
        self.n_threads = 2
        self.shuffle = 50
        self.batch = 16

        self.epochs = 1
        self.current_model_path = "../model/save_model_dssm"
        self.restore = False

    # 定义model
    def init_model(self):
        # 输入
        input_dic = {}
        for f in self.feature_name_list:
            input_dic[f] = tf.keras.Input(shape=(1,), name=f, dtype=tf.int64)
        embed = tf.keras.layers.Embedding(self.feature_size,
                                          self.embedding_size,
                                          embeddings_regularizer="l2")
        user_input_dic = {}
        item_input_dic = {}
        for k, v in input_dic.items():
            if k in self.user_feature_name:
                user_input_dic[k] = v
            if k in self.item_feature_name:
                item_input_dic[k] = v

        user_emb = []
        item_emb = []
        for k, v in user_input_dic.items():
            user_emb.append(embed(v))
        for k, v in item_input_dic.items():
            item_emb.append(embed(v))

        user_emb = tf.keras.layers.Flatten()(tf.concat(user_emb, axis=1))
        item_emb = tf.keras.layers.Flatten()(tf.concat(item_emb, axis=1))

        user_emb = MLPLayer(units=[64], activation='relu')(user_emb)
        user_emb = MLPLayer(units=[8], activation=None)(user_emb)

        item_emb = MLPLayer(units=[64], activation='relu')(item_emb)
        item_emb = MLPLayer(units=[8], activation=None)(item_emb)

        out_ = tf.nn.sigmoid(tf.reduce_sum(user_emb * item_emb, axis=1))

        self.user_model = tf.keras.Model(user_input_dic, user_emb)
        self.item_model = tf.keras.Model(item_input_dic, item_emb)
        self.model = tf.keras.Model(input_dic, out_)
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
        # tf.saved_model.save(self.model, self.current_model_path)
        tf.saved_model.save(self.user_model, self.current_model_path + "_user")
        tf.saved_model.save(self.item_model, self.current_model_path + "_item")

    def load_model_user(self):
        self.imported_user = tf.saved_model.load(self.current_model_path + "_user")
        # self.f = imported.signatures["serving_default"]

    def load_model_item(self):
        self.imported_item = tf.saved_model.load(self.current_model_path + "_item")

    # infer
    def infer_user(self, x):
        result = self.imported_user(x)
        return result

    def infer_item(self, x):
        result = self.imported_item(x)
        return result


if __name__ == "__main__":
    m = DoubleTower()
    train_ds = m.init_dataset("../data/train")
    test_ds = m.init_dataset("../data/test", is_train=False)
    # init model info
    m.init_model()
    m.init_loss()
    m.init_opt()
    m.init_metric()
    m.init_save_checkpoint()

    # train
    m.run(train_ds, test_ds, mode="train_and_eval")

    # save
    m.export_model()
