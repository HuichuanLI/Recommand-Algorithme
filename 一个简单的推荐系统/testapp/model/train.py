from pathlib import Path

import tensorflow as tf

from model.fm import setup_graph
from model.input_function import InputFn
from model.parameter_server import ParameterSever
from model.score_utils import ScoreUtil


def valid(sess, paramter_server, save_path, test_iter, test_result, test_metric, step, last_test_auc):
    test_metric.reset()
    sess.run(test_iter.initializer)

    while True:
        try:
            output = sess.run(test_result["result"])
            test_metric.add(output["loss"], output["label"],
                            output["prediction"])
        except tf.errors.OutOfRangeError:
            result = test_metric.get_score()
            print("Test step {}: {}".format(step, test_metric.to_string(result)))
            auc = result["auc"]
            if last_test_auc < auc:
                print("AUC improved, saving embedding\n")
                paramter_server.save(save_path)
                return auc
            else:
                print()
                return last_test_auc

def train(save_path, train_iter, result, test_iter, test_result, train_metric, test_metric,
          parameter_server, max_step, last_test_auc, log_step=1000, valid_step=1000):
    step = 0
    with tf.compat.v1.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(),
                  tf.compat.v1.local_variables_initializer()])

        sess.run(train_iter.initializer)
        while step < max_step:
            feature_embedding, new_feature_embedding, keys, output = sess.run(
                [result["feature_embedding"],
                 result["new_feature_embedding"],
                 result["feature"],
                 result["result"]]
            )

            # feature_embedding, new_feature_embedding, keys, output, gradient = sess.run(
            #     [result["feature_embedding"],
            #      result["new_feature_embedding"],
            #      result["feature"],
            #      result["result"], result["gradient"]]
            # )

            # movie_id_embedding = output["movie"]
            # user_id_embedding = output["user"]
            # out = output["output"]
            # pred = tf.sigmoid(tf.reduce_mean(user_id_embedding * movie_id_embedding, axis=1))
            # loss_ = tf.reduce_mean(tf.square(output["label"] - pred))
            train_metric.add(output["loss"], output["label"],
                             output["prediction"])
            parameter_server.push(keys, new_feature_embedding)
            step += 1

            if step % log_step == 0:
                print("Train step {}: {}".format(step, train_metric))
                train_metric.reset()
            if step % valid_step == 0:
                last_test_auc = valid(sess, parameter_server, save_path, test_iter, test_result, test_metric, step, last_test_auc)


learning_rate = 10
batch_size = 64
weight_dim = 17
feature_num = 4
parameter_server = ParameterSever(weight_dim)
n_parse_threads = 4
shuffle_buffer_size = 1024
prefetch_buffer_size = 128
max_steps = 100000
train_log_iter = 1000
test_show_step = 1000
last_train_auc = 0.0
last_test_auc = 0.5

inputs = InputFn(parameter_server, feature_num=feature_num,
                 label_len=1,
                 n_parse_threads=n_parse_threads,
                 shuffle_buffer_size=shuffle_buffer_size,
                 batch_size=batch_size)
data_dir = Path("../data_test/tf_records")
train_dir = data_dir / "train"
test_dir = data_dir / "test"
train_metric = ScoreUtil()
test_metric = ScoreUtil()

train_iter, train_inputs = inputs.input_fn(train_dir, is_test=False)
test_iter, test_inputs = inputs.input_fn(test_dir, is_test=True)

train_result = setup_graph(train_inputs, is_test=False)
test_result = setup_graph(test_inputs, is_test=True)

save_path = Path("../data_test") / "trained_embedding" / "embedding.dat"
train(save_path, train_iter, train_result, test_iter, test_result, train_metric,
      test_metric,
      parameter_server, max_steps, last_test_auc)

