import logging
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn import neighbors

from common_utils import get_redis_connection, get_from_redis, \
    save_to_redis_dict, init_logging, USER_DB, USER_CF_DB, MATRIX_CF_DB, \
    ARTICLE_DB, hash_function
from feature_server.feature_util import read_click_log, ARTICLE_HISTORY_COLUMN, \
    USER_HISTORY_COLUMN

def create_click_list(group):
    return group.values.tolist()


def generate_fm_sample(save_path, article_redis, user_redis, threshold=0.8):
    df = pd.read_csv(save_path)[["user_id", "article_id"]]
    user_list = [int(user_id) for user_id in df["user_id"].unique()]
    article_list = [int(article_id) for article_id in df["article_id"].unique()]
    user_info_list = get_from_redis(user_redis, user_list)
    # df_article_list = df.groupby("user_id").agg(create_click_list)

    samples = []
    for index, user_id, article_id in df.itertuples():
        user_info = user_info_list[user_id]
        environment = user_info["environment"]
        region = user_info["region"]
        samples.append((user_id, environment, region, article_id, 1))
        for article_id in random.sample(article_list, 10):
            samples.append((user_id, environment, region, article_id, 0))

    random.shuffle(samples)
    train_size = int(len(samples) * 0.8)
    train_samples = samples[:train_size]
    test_samples = samples[train_size:]

    return train_samples, test_samples


def save_hash_file(samples, hash_file_path, filename="train.dat"):
    hash_file_path = Path(hash_file_path)
    hash_file_path.mkdir(exist_ok=True, parents=True)
    hash_file_path /= filename
    logger.debug("Saving samples to {}".format(hash_file_path))
    with open(hash_file_path, "w") as save_f:
        for data in samples:
            user_id = hash_function("user_id=" + str(data[0]))
            environment = hash_function("environment=" + str(data[1]))
            region = hash_function("region=" + str(data[2]))
            article_id = hash_function("article_id=" + str(data[3]))
            label = str(data[4])
            final_str = ",".join(
                (user_id, environment, region, article_id, label)) + "\n"
            save_f.write(final_str)

def get_tf_examples(feature, label):
    tf_feature = {
        "feature": tf.train.Feature(int64_list=tf.train.Int64List(value=feature)),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    tf_features = tf.train.Features(feature=tf_feature)
    return tf.train.Example(features=tf_features)

def get_tfrecords(file, save_dir):
    file_path = Path(file)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    num = 0
    record_filename = "part-0000" + str(num) + ".record"
    save_path = str(save_dir / record_filename)
    writer = tf.io.TFRecordWriter(save_path)
    logger.debug("writting to %s" % save_path)
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            user_id, environment, region, article_id, label = line.strip().split(",")
            feature_list = [int(user_id), int(environment), int(region), int(article_id)]
            label_list = [float(label)]
            tf_example = get_tf_examples(feature_list, label_list)
            writer.write(tf_example.SerializeToString())
            if (i+1) % 100000 == 0:
                num += 1
                writer.close()
                record_filename = "part-0000" + str(num) + ".record"
                save_path = str(save_dir /  record_filename)
                writer = tf.io.TFRecordWriter(save_path)
                logger.debug("writting to %s" % save_path)

    writer.close()


if __name__ == '__main__':
    article_redis = get_redis_connection(ARTICLE_DB)
    user_redis = get_redis_connection(USER_DB)
    init_logging("../logging.conf")
    logger = logging.getLogger(__name__)
    train_samples, test_samples = generate_fm_sample(
        "../data_test/click_log.csv", article_redis, user_redis)
    save_hash_file(train_samples, "../data_test/hash_data")
    save_hash_file(test_samples, "../data_test/hash_data", "test.dat")
    
    get_tfrecords("../data_test/hash_data/train.dat", "../data_test/tf_records/train")
    get_tfrecords("../data_test/hash_data/test.dat", "../data_test/tf_records/test")
