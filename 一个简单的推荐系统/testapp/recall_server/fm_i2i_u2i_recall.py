import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common_utils import get_redis_connection, get_from_redis, \
    read_embedding_files, save_to_redis_dict, init_logging, USER_DB, USER_CF_DB, \
    MATRIX_CF_DB, hash_function, FM_ARTICLE_EMBEDDING_DB, FM_USER_EMBEDDING_DB, \
    dict_to_2_lists, FM_I2I_DB
from feature_util import read_click_log, ARTICLE_HISTORY_COLUMN, \
    USER_HISTORY_COLUMN
from matrix_cf_recall import balltree_similarity_score_for_item


def read_hash_id_to_id(file_path, feature_key_to_hash_keys):
    # logger.info('Start processing hash id to id')
    feature_to_hash_id_dict = {}

    df = pd.read_csv(file_path)
    for feature_key, hash_key in feature_key_to_hash_keys.items():
        id_to_hash_id = {}
        unique_id_list = list(df[feature_key].unique())
        for id in unique_id_list:
            hash_id = hash_function("{}={}".format(hash_key, id))
            id_to_hash_id[hash_id] = id
        feature_to_hash_id_dict[feature_key] = id_to_hash_id

    return feature_to_hash_id_dict


def get_user_article_embedding(id_list, embedding_list,
                               feature_to_hash_id_dict,
                               feature_key_to_hash_keys):
    logger.debug("Start to mapping embeddings to the true id")
    id_to_item_embedding = {}
    id_to_user_embedding = {}

    user_keys = {"user_id", "environment", "region"}
    item_keys = {"article_id"}
    for hash_id, embedding in zip(id_list, embedding_list):
        found = False
        for item_key in item_keys:
            hash_id_to_id = feature_to_hash_id_dict[item_key]
            if hash_id in hash_id_to_id:
                id = hash_id_to_id[hash_id]
                id_to_item_embedding[int(id)] = embedding
                found = True
                continue

        if not found:
            for user_key in user_keys:
                hash_id_to_id = feature_to_hash_id_dict[user_key]
                if hash_id in hash_id_to_id:
                    id = hash_id_to_id[hash_id]
                    hash_key = feature_key_to_hash_keys[user_key]
                    key = "{}={}".format(hash_key, id)
                    id_to_user_embedding[key] = embedding
                    continue
    logger.debug("Item embedding size: {}".format(len(id_to_item_embedding)))
    logger.debug("User embedding size: {}".format(len(id_to_user_embedding)))
    return id_to_item_embedding, id_to_user_embedding


def u2i_recall(id_to_item_embedding, id_to_user_embedding):
    save_to_redis_dict(id_to_item_embedding, FM_ARTICLE_EMBEDDING_DB)
    save_to_redis_dict(id_to_user_embedding, FM_USER_EMBEDDING_DB)


def i2i_recall(id_to_item_embedding, threshold=20):
    id_list, embedding_list = dict_to_2_lists(id_to_item_embedding)
    item_to_item_to_score = balltree_similarity_score_for_item(id_list, embedding_list, threshold)
    save_to_redis_dict(item_to_item_to_score, FM_I2I_DB)


if __name__ == '__main__':
    init_logging("../logging.conf")
    logger = logging.getLogger(__name__)
    hash_id_list, embedding_list = read_embedding_files(
        "../data_test/trained_embedding/embedding.dat",
     return_np_array=False)
    features = {"user_id": "user_id", "article_id": "article_id",
                "environment": "environment", "region": "region"}
    feature_to_hash_id_dict = read_hash_id_to_id("../data_test/click_log.csv",
                                                 features)
    id_to_item_embedding, id_to_user_embedding = get_user_article_embedding(
        hash_id_list, embedding_list,
        feature_to_hash_id_dict,
        features)
    u2i_recall(id_to_item_embedding, id_to_user_embedding)
    i2i_recall(id_to_item_embedding)
