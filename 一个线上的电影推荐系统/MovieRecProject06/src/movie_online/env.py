"""
定义一些系统级别的环境变量
"""

import os

# 路径信息初始化
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get('MOVIE_REC_ONLINE_OUTPUT_DIR', os.path.join(ROOT_DIR, "..", 'output'))
LOG_FILE = os.path.join(OUTPUT_DIR, 'log', 'rec.log')
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE))
