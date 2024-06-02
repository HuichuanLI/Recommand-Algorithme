import threading
from typing import List, Dict, Optional

import redis

from ..config import redis_config


def _encode(v):
    return str(v, encoding='UTF-8')


class RedisUtil(object):
    lock = threading.Lock()
    instance = None

    def __init__(self):
        # redis连接池对象的创建
        self.pool = redis.ConnectionPool(**redis_config.cfg)

    def create_redis(self) -> redis.Redis:
        return redis.Redis(connection_pool=self.pool)

    @staticmethod
    def get_redis() -> redis.Redis:
        """
        对外/内提供的获取redis操作对象的方法
        :return:
        """
        if RedisUtil.instance is None:
            RedisUtil.lock.acquire()  # 获得锁对象 --> 每次只会有一个线程获得锁
            try:
                if RedisUtil.instance is None:
                    RedisUtil.instance = RedisUtil()  # 创建操作 --> 只希望创建一次
            finally:
                # 释放锁
                RedisUtil.lock.release()
        return RedisUtil.instance.create_redis()

    # %% 接下来的代码就是具体的对外提供操作

    @staticmethod
    def get_list(key: str, start: int = 0, end: int = -1) -> List[str]:
        """
        从redis数据库中获取key对应的list数据
        :param key: key字符串
        :param start: 起始下标
        :param end: 结束下标
        :return:
        """
        with RedisUtil.get_redis() as client:
            if not client.exists(key):
                return []
            values = client.lrange(name=key, start=start, end=end)
            if values is None:
                return []
            return [_encode(v) for v in values]

    @staticmethod
    def get_list_int(key: str, start: int = 0, end: int = -1) -> List[int]:
        values = RedisUtil.get_list(key, start, end)
        return [int(v) for v in values]

    @staticmethod
    def get_set(key: str) -> List[str]:
        """
        获取set类型的数据值
        :param key: redis key
        :return: value 列表值
        """
        with RedisUtil.get_redis() as client:
            if not client.exists(key):
                return []
            values = client.smembers(name=key)
            if values is None:
                return []
            return [_encode(v) for v in values]

    @staticmethod
    def get_set_int(key: str) -> List[int]:
        values = RedisUtil.get_set(key)
        return [int(v) for v in values]

    @staticmethod
    def get_zset(key: str, desc=False) -> List[str]:
        with RedisUtil.get_redis() as client:
            if not client.exists(key):
                return []
            values = client.zrange(name=key, start=0, end=-1, desc=desc)
            if values is None:
                return []
            # noinspection PyTypeChecker
            return [_encode(v) for v in values]

    @staticmethod
    def get_zset_int(key: str, desc=False) -> List[int]:
        values = RedisUtil.get_zset(key, desc=desc)
        return [int(v) for v in values]

    @staticmethod
    def get_hash(key: str, fields: Optional[List[str]] = None) -> Dict[str, str]:
        with RedisUtil.get_redis() as client:
            if not client.exists(key):
                return {}
            if fields is None or len(fields) == 0:
                values = client.hgetall(name=key)
                values = values.items()
            else:
                values = client.hmget(name=key, keys=fields)
                values = zip(fields, values)
            return {k: _encode(v) for k, v in values}
