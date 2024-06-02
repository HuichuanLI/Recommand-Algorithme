import logging
from logging.handlers import TimedRotatingFileHandler

from .. import LOG_FILE


def __log():
    # 设置基础格式
    _fmt = '%(asctime)s %(filename)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=_fmt)
    # create a logger, 如果参数为空则返回root logger
    _logger = logging.getLogger("REC_SYSTEM")
    _logger.setLevel(logging.INFO)  # 设置logger日志等级

    # 这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not _logger.handlers:
        # 创建handler
        fh = TimedRotatingFileHandler(LOG_FILE, encoding='UTF-8', when="D", interval=1, backupCount=7)
        ch = logging.StreamHandler()

        # 设置formatter
        fh.setFormatter(fmt=logging.Formatter(fmt=_fmt))
        ch.setFormatter(fmt=logging.Formatter(fmt=_fmt))

        # 为logger添加的日志处理器
        _logger.addHandler(fh)
        # _logger.addHandler(ch)

    return _logger  # 直接返回logger


logger = __log()
