# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/11/10|17:24
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ps import PS
from inputs import InputFn
from auc import AUCUtil
from mf import *

local_ps = PS(config["embedding_dim"])
# 数据输入
inputs = InputFn(local_ps, config)

# 训练
train_metric = AUCUtil()
test_metric = AUCUtil()

# train_file = "../data0/rawdata/train"
# test_file = "../data0/rawdata/val"
#
# saved_embedding = "../data0/saved_embedding"

train_itor, train_inputs = inputs.input_fn(config["train_file"], is_test=False)
train_dic = setup_graph(train_inputs, is_test=False)

test_itor, test_inputs = inputs.input_fn(config["test_file"], is_test=True)
test_dic = setup_graph(test_inputs, is_test=True)

train_log_iter = 1000
last_test_auc = 0.5

def train():
    _step = 0
    print("#" * 80)
    # 建立sess，进行训练
    with tf.compat.v1.Session() as sess:
        # init global & local variables
        sess.run([tf.compat.v1.global_variables_initializer(),
                  tf.compat.v1.local_variables_initializer()])
        # 开始训练
        sess.run(train_itor.initializer)
        while _step < config["max_steps"]:
            feature_old_embedding, feature_new_embedding, keys, out = sess.run(
                [train_dic["feature_embedding"],
                 train_dic["feature_new_embedding"],
                 train_dic["feature"],
                 train_dic["out"]]
            )

            train_metric.add(
                out["loss"],
                out["ground_truth"],
                out["prediction"])

            local_ps.push(keys, feature_new_embedding)
            _step += 1

            # 每训练多少个batch的训练数据，就打印一次训练的这些batch的auc等信息
            if _step % train_log_iter == 0:

                print("Train at step %d: %s", _step, train_metric.calc_str())
                train_metric.reset()
            if _step % config["test_show_step"] == 0:
                valid_step(sess, test_itor, test_dic)


def valid_step(sess, test_itor, test_dic):
    test_metric.reset()
    sess.run(test_itor.initializer)
    global last_test_auc
    while True:
        try:

            out = sess.run(test_dic["out"])

            test_metric.add(
                out["loss"],
                out["ground_truth"],
                out["prediction"])

        except tf.errors.OutOfRangeError:
            print("Test at step: %s", test_metric.calc_str())
            if test_metric.calc()["auc"] > last_test_auc:
                last_test_auc = test_metric.calc()["auc"]
                local_ps.save(config["saved_embedding"])

            break

if __name__ == '__main__':

    train()
