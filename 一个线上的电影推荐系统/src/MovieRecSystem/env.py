import os

# 路径信息初始化
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("MOVIE_REC_OUTPUT_DIR") or os.path.join(ROOT_DIR, 'output')
LOG_FILE = os.path.join(OUTPUT_DIR, 'log', 'log.log')
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE))
