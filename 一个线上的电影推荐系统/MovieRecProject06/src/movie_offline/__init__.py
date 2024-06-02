# -*- coding: utf-8 -*-

from .env import *
from .utils.logger_util import logger
from .config import init_config

# 初始化参数
init_config()
logger.info(f"PID:{os.getpid()}, Parent PID:{os.getppid()}")
