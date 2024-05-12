"""
一个简单的基于Flask的电影推荐系统，提供简单的对外接口
模拟的就是一个推荐的服务
"""

from .config import init_config
from .env import *
from .utils.logger_util import logger

# 相关初始化操作
init_config()
logger.info(f'PID:{os.getpid()}, Parent PID:{os.getppid()}')
