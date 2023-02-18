import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common_utils import get_redis_connection, get_from_redis, save_to_redis_dict, init_logging, ARTICLE_DB, ITEM_CF_DB
from feature_server.feature_util import read_click_log, ARTICLE_HISTORY_COLUMN

CLICK_TIME_FACTOR = 0.7
TYPE_WEIGHT = 1
DISCOUNTED_TYPE_WEIGHT = 0.7

def get_item_cf_score(user_click_log_df, redis_connection, threshold=20):
    article_to_info = {}
    article_to_i2i_score = defaultdict(dict)
    article_to_count = defaultdict(int)
    logger.debug("start calculate weight")
    for user_id, article_timestamp in tqdm(
            user_click_log_df[ARTICLE_HISTORY_COLUMN].iteritems()):

        for i, (article_id_i, i_timestamp) in enumerate(article_timestamp):
            article_to_count[article_id_i] += 1
            if article_id_i not in article_to_info:
                article_to_info[article_id_i] = get_from_redis(redis_connection,
                                                               article_id_i)
            for j, (article_id_j, j_timestamp) in enumerate(article_timestamp):
                if i == j:
                    continue
                if article_id_j not in article_to_info:
                    article_to_info[article_id_j] = get_from_redis(
                        redis_connection, article_id_j)


                article_i = article_to_info[article_id_i]
                article_j = article_to_info[article_id_j]
                # 时间权重，点击时间间隔越长越低
                click_time_weight = np.exp(
                    CLICK_TIME_FACTOR ** np.abs(i_timestamp - j_timestamp))
                # 类别权重，类别相同为1否则0.7
                type_weight = TYPE_WEIGHT if article_i["category_id"] == \
                                             article_j[
                                                 "category_id"] else DISCOUNTED_TYPE_WEIGHT

                if j not in article_to_i2i_score[article_id_i]:
                    article_to_i2i_score[article_id_i][article_id_j] = 0

                article_to_i2i_score[article_id_i][article_id_j] += click_time_weight * type_weight / np.log(len(article_timestamp) + 1)

    logger.debug("start calculating scores")
    article_to_i2i_score_final = article_to_i2i_score.copy()
    for i, i2i_scores in article_to_i2i_score_final.items():
        tmp = {}
        for j, i2i_score in i2i_scores.items():
            tmp[j] = i2i_score / np.sqrt(article_to_count[i] * article_to_count[j])
        article_to_i2i_score_final[i] = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:threshold]

    save_to_redis_dict(article_to_i2i_score_final, ITEM_CF_DB)


if __name__ == '__main__':
    init_logging(Path("..") / "logging.conf")
    logger = logging.getLogger(__name__)
    data_path = Path("..") / "data_test"
    train_click_log_path = data_path / "click_log.csv"
    user_click_log = read_click_log(train_click_log_path, save_redis=False)

    client = get_redis_connection(ARTICLE_DB)
    get_item_cf_score(user_click_log, client, 200)
