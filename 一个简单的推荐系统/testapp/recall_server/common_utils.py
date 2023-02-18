import traceback
import json
import logging
import logging.config
from pathlib import Path

import yaml
import redis
import numpy as np

REDIS_URL = "redis://@127.0.0.1:6379/"
USER_DB = 1
ARTICLE_DB = 2
MATRIX_CF_DB = 3
ITEM_CF_DB = 4
USER_CF_DB = 5
FM_I2I_DB = 6
FM_ARTICLE_EMBEDDING_DB = 7
FM_USER_EMBEDDING_DB = 8
DB_LIST = [USER_DB,
           ARTICLE_DB,
           MATRIX_CF_DB,
           ITEM_CF_DB,
           USER_CF_DB,
           FM_I2I_DB,
           FM_ARTICLE_EMBEDDING_DB,
           FM_USER_EMBEDDING_DB]
logger = logging.getLogger(__name__)


def init_logging(path="logging.conf"):
    with open(path) as f:
        logging.config.dictConfig(yaml.load(f))
    return


def get_redis_connection(db):
    redis_db_url = REDIS_URL + str(db)
    client = redis.from_url(redis_db_url)
    return client


def get_from_redis(client, key):
    try:
        if isinstance(key, (list, tuple)):
            value_json_list = client.mget(key)
            return {k: json.loads(value_json) for value_json, k in
                    zip(value_json_list, key)}
        else:
            value_json = client.get(key)
            return json.loads(value_json)
    except:
        if isinstance(key, (list, tuple)):
            print(key, value_json_list)
        else:
            print(key, value_json)
        traceback.print_exc()


def save_to_redis_dataframe(data, db):
    logger.debug("Writting to redis\n")
    client = get_redis_connection(db)
    try:
        for key, value in data.iterrows():
            value_json = json.dumps(value.to_dict())
            client.set(key, value_json)
    except:
        traceback.print_exc()


def save_to_redis_dict(data, db):
    logger.debug("Writting to redis\n")
    client = get_redis_connection(db)
    try:
        for key, value in data.items():
            value_json = json.dumps(value)
            client.set(key, value_json)
    except:
        traceback.print_exc()


def save_to_redis(keys, values, db):
    logger.debug("Writting to redis\n")
    client = get_redis_connection(db)
    try:
        client.set(keys, values)
    except:
        traceback.print_exc()


def dict_to_2_lists(original_dict):
    return list(original_dict.keys()), list(original_dict.values())


def read_embedding_files(save_path, return_np_array=True):
    path = Path(save_path)
    id_list = []
    embedding_list = []

    with open(path, "r") as f:
        for line in f.readlines():
            article_id, embedding_str = line.split("\t")
            id_list.append(article_id)
            embedding = [float(embed) for embed in embedding_str.split(",")]
            if return_np_array:
                embedding = [np.array(embed) for embed in embedding]
            embedding_list.append(embedding)

    return id_list, embedding_list


def hash_function(data_str):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash_val = 0
    for character in data_str:
        hash_val = hash_val * seed + ord(character)
    return str(hash_val & mask60)
