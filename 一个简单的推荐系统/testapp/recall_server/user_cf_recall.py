import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common_utils import get_redis_connection, get_from_redis, \
    save_to_redis_dict, init_logging, USER_DB, USER_CF_DB
from feature_server.feature_util import read_click_log, ARTICLE_HISTORY_COLUMN, \
    USER_HISTORY_COLUMN


def get_user_cf_score(user_click_log_df, redis_connection, threshold=20):
    user_to_info = {}
    user_to_i2i_score = defaultdict(dict)
    user_to_count = defaultdict(int)
    logger.debug("start calculate weight")
    for item_id, user_timestamp in tqdm(
            user_click_log_df[USER_HISTORY_COLUMN].iteritems()):

        for i, (user_id_i, i_timestamp) in enumerate(user_timestamp):
            user_to_count[user_id_i] += 1
            if user_id_i not in user_to_info:
                user_to_info[user_id_i] = get_from_redis(redis_connection,
                                                         user_id_i)
            for j, (user_id_j, j_timestamp) in enumerate(user_timestamp):
                if user_id_j not in user_to_i2i_score[user_id_i]:
                    user_to_i2i_score[user_id_i][user_id_j] = 0
                if i == j:
                    continue
                if user_id_j not in user_to_info:
                    user_to_info[user_id_j] = get_from_redis(
                        redis_connection, user_id_j)
                user_i = user_to_info[user_id_i]
                user_j = user_to_info[user_id_j]



                user_to_i2i_score[user_id_i][
                    user_id_j] += 0.1 * 0.5 * (
                            len(user_i[ARTICLE_HISTORY_COLUMN]) + len(
                        user_j[ARTICLE_HISTORY_COLUMN])) / np.log(
                    len(user_timestamp) + 1)

    logger.debug("start calculating scores")
    article_to_i2i_score_final = user_to_i2i_score.copy()
    for i, i2i_scores in article_to_i2i_score_final.items():
        tmp = {}
        for j, i2i_score in i2i_scores.items():
            tmp[j] = i2i_score / np.sqrt(user_to_count[i] * user_to_count[j])
        article_to_i2i_score_final[i] = sorted(tmp.items(), key=lambda x: x[1],
                                               reverse=True)[:threshold]

    save_to_redis_dict(article_to_i2i_score_final, USER_CF_DB)


if __name__ == '__main__':
    init_logging(Path("..") / "logging.conf")
    logger = logging.getLogger(__name__)
    data_path = Path("..") / "data_test"
    train_click_log_path = data_path / "click_log.csv"
    item_click_log = read_click_log(train_click_log_path,
                                    major_key="article_id", save_redis=False)

    client = get_redis_connection(USER_DB)
    get_user_cf_score(item_click_log, client, 20)
