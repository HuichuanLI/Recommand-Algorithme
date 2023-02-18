# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/4|下午 03:03
# @Moto   : Knowledge comes from decomposition

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.saved_model import tag_constants


# todo: 基于tensorboard 展示图的结构和分析
#  tensorboard --logdir log
#  http://localhost:6006
def tensorboard_show_graph(model_dir):
    # model_dir = "../../../data/saved_model/dnn/1609124014"
    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(
        sess, [tag_constants.SERVING], model_dir)
    print(meta_graph_def)
    graph = tf.get_default_graph()
    summary_write = tf.summary.FileWriter("./log", graph)


def main():
    model_dir = "../data2/saved_model"
    tensorboard_show_graph(model_dir)


if __name__ == '__main__':
    main()