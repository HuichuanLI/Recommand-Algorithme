import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn import neighbors

from common_utils import get_redis_connection, get_from_redis, \
    read_embedding_files, save_to_redis_dict, init_logging, USER_DB, USER_CF_DB, \
    MATRIX_CF_DB
from feature_server.feature_util import read_click_log, ARTICLE_HISTORY_COLUMN, \
    USER_HISTORY_COLUMN

logger = logging.getLogger(__name__)
def balltree_similarity_score_for_item(article_id_list, article_embedding_list, threshold=20):
    logger.debug("Start to build BallTree")
    article_embeddings = np.array(article_embedding_list)
    normalized_article_embeddings = article_embeddings / np.linalg.norm(
        article_embeddings, axis=1, keepdims=True)

    article_tree = neighbors.BallTree(normalized_article_embeddings,
                                      leaf_size=40)

    logger.debug("Start to build dict of results")
    similarity, indices = article_tree.query(normalized_article_embeddings,
                                             threshold)
    item_to_item_to_score = {}
    for article_id, sim_score_list, index_list in zip(article_id_list,
                                                      similarity, indices):
        tmp = {}
        for sim_score, index in zip(sim_score_list[1:], index_list[1:]):
            tmp[article_id_list[index]] = sim_score
        item_to_item_to_score[article_id] = sorted(tmp.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True)[:threshold]
    return item_to_item_to_score


if __name__ == '__main__':
    init_logging(Path("..") / "logging.conf")
    logger = logging.getLogger(__name__)
    article_id_list, article_embedding_list = read_embedding_files(
        Path("../data_test/matrixcf_articles_emb.csv"))
    item_to_item_to_score = balltree_similarity_score_for_item(article_id_list, article_embedding_list)
    save_to_redis_dict(item_to_item_to_score, MATRIX_CF_DB)