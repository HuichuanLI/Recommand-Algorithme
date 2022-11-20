import json
import traceback
from pathlib import Path
import logging

import redis
import pandas as pd

from common_utils import save_to_redis_dataframe, ARTICLE_DB, USER_DB

logger = logging.getLogger(__name__)
ARTICLE_HISTORY_SOURCE_COLUMNS = ["article_id", "timestamp"]
ARTICLE_HISTORY_COLUMN = "article_timestamp"
USER_HISTORY_SOURCE_COLUMNS = ["user_id", "timestamp"]
USER_HISTORY_COLUMN = "user_timestamp"
USER_COLUMN = "user_id"

def read_article_attribute(article_file_path, save_redis=True):
    logging.debug("Start reading article embeddings")
    df = pd.read_csv(article_file_path)
    df.index.name = None
    df = df.set_index("article_id", drop=False)
    if save_redis:
        save_to_redis_dataframe(df, ARTICLE_DB)
    return df

def create_click_list(group):
    return group.values.tolist()


def read_click_log(click_log_path, major_key="user_id", save_redis=True, db=USER_DB):
    logging.debug("Start reading click logs")
    df = pd.read_csv(click_log_path)
    df = df.set_index(major_key, drop=False)
    df.index.name = None
    df = df.sort_values("timestamp")
    if major_key == "user_id":
        df[ARTICLE_HISTORY_COLUMN] = df[ARTICLE_HISTORY_SOURCE_COLUMNS].values.tolist()
        df.drop(columns=ARTICLE_HISTORY_SOURCE_COLUMNS, inplace=True)
        df_group = df.groupby(major_key, sort=False, as_index=False)
        df_new = df_group.agg({"user_id": "first",
                               ARTICLE_HISTORY_COLUMN: create_click_list,
                               "environment": "first",
                               "region": "first"
                               })
    elif major_key == "article_id":
        df[USER_HISTORY_COLUMN] = df[USER_HISTORY_SOURCE_COLUMNS].values.tolist()
        df.drop(columns=USER_HISTORY_SOURCE_COLUMNS, inplace=True)
        df_group = df.groupby(major_key, sort=False, as_index=False)
        df_new = df_group.agg({"article_id": "first",
                               USER_HISTORY_COLUMN: create_click_list,
                               "environment": "first",
                               "region": "first"
                               })
    else:
        raise NotImplementedError
    df_new = df_new.set_index(major_key, drop=False)
    df.index.name = None
    if save_redis:
        save_to_redis_dataframe(df_new, db)
    return df_new


def main():
    train_path = "../../data1/"
    test_path = "../../data1/"
    train_article_path = train_path + "articles.csv"
    test_article_path = test_path + "articles.csv"
    train_click_log_path = train_path + "click_log.csv"
    test_click_log_path = test_path + "click_log.csv"
    #
    # read_article_attribute(train_article_path)
    # read_click_log(train_click_log_path)

    read_article_attribute(test_article_path)
    read_click_log(test_click_log_path)


if __name__ == '__main__':
    main()
