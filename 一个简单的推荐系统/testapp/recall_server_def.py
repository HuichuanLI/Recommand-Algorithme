import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import redis
from tqdm import tqdm

from common_utils import get_redis_connection, get_from_redis, \
    read_embedding_files, save_to_redis_dict, init_logging, USER_DB, USER_CF_DB, \
    MATRIX_CF_DB, hash_function, FM_ARTICLE_EMBEDDING_DB, FM_USER_EMBEDDING_DB, \
    dict_to_2_lists, FM_I2I_DB, REDIS_URL, ARTICLE_DB, ITEM_CF_DB
from feature_server.feature_util import read_click_log, ARTICLE_HISTORY_COLUMN, \
    USER_HISTORY_COLUMN
from recall_server.matrix_cf_recall import balltree_similarity_score_for_item
from vector_server.vector_server import VectorServer

CLICK_TIME_FACTOR = 0.7
TYPE_WEIGHT = 1.0
DISCOUNTED_TYPE_WEIGHT = 0.7


class RecallSever:

    def __init__(self):
        self.user_redis = redis.from_url("{}{}".format(REDIS_URL, USER_DB))
        self.item_redis = redis.from_url("{}{}".format(REDIS_URL, ARTICLE_DB))
        self.matrix_cf_redis = redis.from_url(
            "{}{}".format(REDIS_URL, MATRIX_CF_DB))
        self.item_cf_redis = redis.from_url(
            "{}{}".format(REDIS_URL, ITEM_CF_DB))
        self.user_cf_redis = redis.from_url(
            "{}{}".format(REDIS_URL, USER_CF_DB))
        self.fm_i2i_redis = redis.from_url("{}{}".format(REDIS_URL, FM_I2I_DB))
        self.fm_item_redis = redis.from_url(
            "{}{}".format(REDIS_URL, FM_ARTICLE_EMBEDDING_DB))
        self.fm_user_redis = redis.from_url(
            "{}{}".format(REDIS_URL, FM_USER_EMBEDDING_DB))
        self.redis_list = [self.user_redis,
                           self.item_redis,
                           self.matrix_cf_redis,
                           self.item_cf_redis,
                           self.user_cf_redis,
                           self.fm_i2i_redis,
                           self.fm_item_redis,
                           self.fm_user_redis]
        self.user_info = {}
        self.item_info = {}

        self.current_user_feature = {}
        self.vector_server = VectorServer(self.fm_item_redis)

    def set_current_user_info(self, user_info):
        user_id = user_info["user_id"]

        if user_id not in self.user_info:
            self.user_info[user_id] = self._get_from_redis(USER_DB, user_id)
        self.current_user_feature = self.user_info[user_id]

    def _get_from_redis(self, db, key):
        return json.loads(self.redis_list[db - 1].get(key))

    def _mget_from_redis(self, db, keys):
        return [json.loads(value_json) for value_json in
                self.redis_list[db - 1].mget(keys)]

    def _get_info_and_cache(self, id, cache, db):
        if id not in cache:
            cache[id] = self._get_from_redis(db, id)
        return cache[id]

    def get_item_info(self, article_id):
        return self._get_info_and_cache(article_id, self.item_info, ARTICLE_DB)

    def get_user_info(self, user_id):
        return self._get_info_and_cache(user_id, self.user_info, USER_DB)

    def _get_time_weight(self, i_timestamp, j_timestamp):
        return np.exp(CLICK_TIME_FACTOR ** np.abs(i_timestamp - j_timestamp))

    def _get_category_weight(self, type_i, type_j):
        return TYPE_WEIGHT if type_i == type_j else DISCOUNTED_TYPE_WEIGHT

    def _get_click_location_weight(self, index, click_list):
        return 0.9 ** (len(click_list) - index)

    def _i2i_recommend(self, db, recall_num):
        history = self.current_user_feature["article_timestamp"]
        clicked_articles = [x[0] for x in history]
        itemcf_i2i_scores = self._mget_from_redis(db, clicked_articles)
        id_to_score = {}
        # sim_score = 所有用户计算得到的文章相似分数之和
        # （时间权重*类别权重/log（点击历史长度）+1）
        for i, (article_id_i, sim_scores) in enumerate(
                zip(clicked_articles, itemcf_i2i_scores)):
            item_info_i = self.get_item_info(article_id_i)
            for article_id_j, sim_score in sim_scores:
                # 不推荐已有文章
                if article_id_j in clicked_articles:
                    continue
                item_info_j = self.get_item_info(article_id_j)
                # 对全局计算的相似度根据用户点击历史进行再调整
                if article_id_j not in id_to_score:
                    id_to_score[article_id_j] = 0
                id_to_score[
                    article_id_j] += sim_score * self._get_time_weight(
                    item_info_i[
                        "created_at_ts"],
                    item_info_j[
                        "created_at_ts"]) * self._get_click_location_weight(
                    i, clicked_articles) * self._get_category_weight(
                    item_info_i["category_id"], item_info_j["category_id"])

        return sorted(id_to_score.items(), key=lambda x: x[1], reverse=True)[
               :recall_num]

    def get_item_cf_recommendation(self, recall_num=30):
        return self._i2i_recommend(ITEM_CF_DB, recall_num)

    def get_matrix_cf_recommendation(self, recall_num=30):
        return self._i2i_recommend(MATRIX_CF_DB, recall_num)

    def get_fm_i2i_recommendation(self, recall_num=30):
        return self._i2i_recommend(FM_I2I_DB, recall_num)

    def get_user_cf_recommendation(self, recall_num=30):
        cur_user_id = self.current_user_feature["user_id"]
        history = self.current_user_feature["article_timestamp"]
        clicked_articles = [x[0] for x in history]
        user_cf_u2u_scores = self._get_from_redis(USER_CF_DB, cur_user_id)
        id_to_score = {}
        # sim_score = 所有用户计算得到的文章相似分数之和
        # （时间权重*类别权重/log（点击历史长度）+1）
        for i, (article_id_i, timestamp_i) in enumerate(history):
            item_info_i = self.get_item_info(article_id_i)
            for user_id_i, sim_score in user_cf_u2u_scores:
                user_info_i = self.get_user_info(user_id_i)
                for article_id_j, timestamp_j in user_info_i[
                    "article_timestamp"]:
                    # 不推荐已有文章
                    if article_id_j in clicked_articles:
                        continue
                    item_info_j = self.get_item_info(article_id_j)
                    # 对全局计算的相似度根据用户点击历史进行再调整
                    if article_id_j not in id_to_score:
                        id_to_score[article_id_j] = 0
                    id_to_score[
                        article_id_j] += sim_score * self._get_time_weight(
                        item_info_i[
                            "created_at_ts"],
                        item_info_j[
                            "created_at_ts"]) * self._get_click_location_weight(
                        i, clicked_articles) * self._get_category_weight(
                        item_info_i["category_id"], item_info_j["category_id"])

        return sorted(id_to_score.items(), key=lambda x: x[1], reverse=True)[
               :recall_num]

    def get_fm_u2i_recommendation(self, recall_num=30):
        user_id_str = "user_id={}".format(self.current_user_feature["user_id"])
        env_str = "environment={}".format(
            self.current_user_feature["environment"])
        region_str = "region={}".format(self.current_user_feature["region"])
        embedding_list = self._mget_from_redis(FM_USER_EMBEDDING_DB,
                                               [user_id_str, env_str,
                                                region_str])
        embedding = np.sum(np.array(embedding_list), axis=0, keepdims=True)

        return self.vector_server.get_similar_items(embedding, recall_num)[0]

    def merge_results(self, recall_result_and_weight):
        item_rank = {}

        for rank_list, weight in recall_result_and_weight:
            scores = [x[1] for x in rank_list]
            max_score = max(scores)
            min_score = min(scores)
            diff = max_score - min_score
            for item, score in rank_list:
                if item not in item_rank:
                    item_rank[item] = 0
                item_rank[item] += weight * (score - min_score) / diff

        return item_rank

recall_server = RecallSever()
user_id = {"user_id": 190000}
recall_server.set_current_user_info(user_id)
item_cf_result = recall_server.get_item_cf_recommendation()
user_cf_result = recall_server.get_user_cf_recommendation()
matrix_cf_result = recall_server.get_matrix_cf_recommendation()
fm_i2i_result = recall_server.get_fm_i2i_recommendation()
fm_u2i_result = recall_server.get_fm_u2i_recommendation()
merge_result = recall_server.merge_results([(item_cf_result, 1.0),
                                            (user_cf_result, 1.0),
                                            (matrix_cf_result, 1.0),
                                            (fm_i2i_result, 1.0),
                                            (fm_u2i_result, 1.0),
                                            ])
print(item_cf_result)
print(user_cf_result)
print(matrix_cf_result)
print(fm_i2i_result)
print(fm_u2i_result)
