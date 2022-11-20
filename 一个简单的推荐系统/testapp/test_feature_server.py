import logging

from feature_server.feature_util import read_click_log, read_article_attribute
from common_utils import init_logging
from pathlib import Path

if __name__ == '__main__':
    init_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Start testing feature server")
    train_path = Path("data")
    test_path = Path("data_test")
    train_article_path = train_path / "articles.csv"
    test_article_path = test_path / "articles.csv"
    train_click_log_path = train_path / "click_log.csv"
    test_click_log_path = test_path / "click_log.csv"
    #
    # read_article_embedding(train_article_path)
    # read_click_log(train_click_log_path)

    read_article_attribute(test_article_path)
    read_click_log(test_click_log_path)