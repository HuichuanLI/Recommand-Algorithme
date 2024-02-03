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

# from dnn import *
# from can import *
# from mind import *
from ple import *
# from dat import *
# from fibinet import *



# 初始化对应
local_ps = PS(config['embedding_dim'])  # 参数服务
train_metric = AUCUtil()  # 评估
test_metric = AUCUtil()  # 评估
inputs = InputFn(local_ps, config)  # 输入

# 定义模型网络【输入部分和网络部分】
train_itor, train_inputs = inputs.input_fn(config['train_file'], is_test=False)
train_dic = setup_graph(train_inputs, is_test=False)
test_itor, test_inputs = inputs.input_fn(config['test_file'], is_test=True)
test_dic = setup_graph(test_inputs, is_test=True)

# 训练参数
max_steps = config['max_steps']
train_log_iter = config['train_log_iter']
test_show_step = config['test_show_step']
last_test_auc = config['last_test_auc']


#
def train():
    _step = 0
    print("#" * 80)
    saver = tf.train.Saver(max_to_keep=2)  # 保存
    # 建立sess，进行训练
    with tf.Session() as sess:
        # init global & local variables
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        # 开始训练
        sess.run(train_itor.initializer)
        while _step < max_steps:
            old_embedding, new_embedding, keys, out, _ = sess.run(
                [train_dic["feature_embedding"],
                 train_dic["feature_new_embedding"],
                 train_dic["feature"],
                 train_dic["out"],
                 train_dic["train_op"]]
            )

            train_metric.add(
                out["loss"],
                out["ground_truth"],
                out["prediction"])

            local_ps.push(keys, new_embedding)
            _step += 1

            # 每训练多少个batch的训练数据，就打印一次训练的这些batch的auc等信息
            if _step % train_log_iter == 0:
                print("Train at step %d: %s", _step, train_metric.calc_str())
                train_metric.reset()
            if _step % test_show_step == 0:
                valid_step(sess, test_itor, test_dic, saver, _step)


def valid_step(sess, test_itor, test_dic, saver, _step):
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
            break


if __name__ == '__main__':
    train()
