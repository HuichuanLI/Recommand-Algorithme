from typing import List

from cachetools import cached, TTLCache

from ..utils.redis_util import RedisUtil


class UserFeatureService:
    USER_VIEW_ID_LIST_REDIS_KEY = "user:views:"

    def __init__(self):
        pass

    @staticmethod
    @cached(cache=TTLCache(maxsize=10000, ttl=5))
    def get_user_view_ids(user_id: int, start: int = 0, end: int = -1) -> List[int]:
        """
        提取用户最近浏览商品id列表
        :param user_id: 用户id
        :param start: 起始下标， 0表示第一个开始
        :param end: 结束下表，-1表示所有
        :return: 列表，可能为[]
        """
        # 1. 从redis提取数据
        _key: str = f"{UserFeatureService.USER_VIEW_ID_LIST_REDIS_KEY}{user_id}"
        _values: List[str] = RedisUtil.get_list(_key, start, end)
        # 2. 进行数据拆分，每个数据类似: id:timestamp的形式
        return [int(v.split(":")[0]) for v in _values]

    @staticmethod
    @cached(cache=TTLCache(maxsize=1000, ttl=5))
    def get_user_blacklist_ids(user_id: int) -> List[int]:
        _key: str = f"user:blacklist:{user_id}"
        return RedisUtil.get_set_int(_key)
