import threading
from typing import List, Set

import redis

from ..config import redis_config


def _encode(v, encoding='UTF-8') -> str:
    return str(v, encoding=encoding)


class RedisUtil(object):
    lock = threading.Lock()  # 锁
    instance = None

    def __init__(self):
        self._pool = redis.ConnectionPool(**redis_config.cfg)

    def create_redis(self) -> redis.Redis:
        """
        创建一个操作redis的对象
        :return: redis对象
        """
        return redis.Redis(connection_pool=self._pool)

    @staticmethod
    def _get_redis() -> redis.Redis:
        if RedisUtil.instance is None:
            # 获取锁
            RedisUtil.lock.acquire()
            try:
                if RedisUtil.instance is None:
                    RedisUtil.instance = RedisUtil()
            finally:
                # 释放锁
                RedisUtil.lock.release()
        return RedisUtil.instance.create_redis()

    # ---------------------------------------------------------------------
    # 主要对外提供的方法

    @staticmethod
    def get_slist(key: str, sep: str = ",") -> List[str]:
        """
        提取key对应的value，并将value按照给定分隔符划分为list字符串列表
        :param key: key字符串
        :param sep: 分隔符
        :return: 结果对象
        """
        with RedisUtil._get_redis() as client:
            value = client.get(key)
            if value is None:
                return []
            value = str(value, encoding='UTF-8')
            return value.split(sep=sep)

    @staticmethod
    def get_slist_int(key: str, sep: str = ",") -> List[int]:
        """
        在get_slist的基础上，再次将字符串转换为int类型
        :param key: key字符串
        :param sep: 分隔符
        :return: 结果对象
        """
        _list = RedisUtil.get_slist(key, sep)
        if _list is None:
            return []
        return [int(_v) for _v in _list]

    @staticmethod
    def get_list(key: str, start: int = 0, end: int = -1) -> List[str]:
        """
        直接提取redis中list类型对象的结果
        :param key: key对象
        :param start: 截取的起始位置
        :param end: 截取的结束位置
        :return: list结果
        """
        with RedisUtil._get_redis() as client:
            if not client.exists(key):
                return []
            values = client.lrange(key, start, end)
            if values is None:
                return []
            return [str(v, encoding='UTF-8') for v in values]

    @staticmethod
    def get_list_int(key: str, start: int = 0, end: int = -1) -> List[int]:
        """
        基于get_list提取int列表
        :param key: key对象
        :param start: 截取的起始位置
        :param end: 截取的结束位置
        :return: list结果
        """
        _list = RedisUtil.get_list(key, start, end)
        if _list is None:
            return []
        return [int(_v) for _v in _list]

    @staticmethod
    def get_set(key: str) -> List[str]:
        with RedisUtil._get_redis() as client:
            if not client.exists(key):
                return []
            values = client.smembers(key)
            if values is None:
                return []
            return [_encode(v) for v in values]

    @staticmethod
    def get_set_int(key: str) -> List[int]:
        _list = RedisUtil.get_set(key)
        if _list is None:
            return []
        return [int(_v) for _v in _list]

