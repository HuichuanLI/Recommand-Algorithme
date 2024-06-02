# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple

from cachetools import TTLCache, cached

from ..utils.mysql_util import DB
from ..utils.redis_util import RedisUtil


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
class UserFeatureService(object):
    @staticmethod
    @cached(cache=TTLCache(maxsize=10000, ttl=10))
    def get_view_spu_ids(user_id: int) -> List[int]:
        """
        获取user_id用户对应的最近浏览商品列表
        :param user_id:
        :return:
        """
        key = f"rec:user:views:{user_id}"
        print(f"调用redis:{key}")
        # 按照播放的时间戳/也就是zset的score降序排列
        return RedisUtil.get_zset_int(key, desc=True)

    @staticmethod
    @cached(cache=TTLCache(maxsize=10000, ttl=60))
    def get_blacklist_spu_ids(user_id: int) -> List[int]:
        key = f"rec:user:blacklist:{user_id}"
        print(f"调用redis:{key}")
        return RedisUtil.get_set_int(key)

    @staticmethod
    @cached(cache=TTLCache(maxsize=10000, ttl=60))
    def get_no_like_category_names(user_id: int) -> List[str]:
        key = f"rec:user:nolikecategory:{user_id}"
        print(f"调用redis:{key}")
        return RedisUtil.get_set(key)

    @staticmethod
    def get_u2i_recall_spus(user_id: int, field_names: List[str]) -> Dict[str, List[Tuple[int, float]]]:
        def _parse_value(_value: str):
            # 851:5.000,1449:4.943,868:4.789,408:4.728,483:4.673
            vs = []
            for t in _value.strip().split(","):
                spu_id, spu_score = t.split(":")
                vs.append((int(spu_id), float(spu_score)))
            return vs

        key = f"rec:recall:u2i:{user_id}"
        print(f"调用redis:{key}")
        values: Dict[str, str] = RedisUtil.get_hash(key, fields=field_names)
        return {_k: _parse_value(_v) for _k, _v in values.items()}

    @staticmethod
    @cached(cache=TTLCache(maxsize=1000, ttl=60))
    def get_stat_mean_rating(user_id: int) -> dict:
        _list = DB.query_sql(
            sql="select mean_rating as user_mean_rating from user_mean_rating_stat where user_id = %(id)s limit 1",
            id=user_id
        )
        if len(_list) > 0:
            return _list[0]
        return {'user_mean_rating': 0.0}

    @staticmethod
    @cached(cache=TTLCache(maxsize=1000, ttl=60))
    def get_stat_movie_genre_mean_rating(user_id: int) -> dict:
        _list = DB.query_sql(
            sql="select * from user_movie_genre_mean_rating_stat where user_id=%(id)s limit 1",
            id=user_id
        )
        if len(_list) > 0:
            return _list[0]
        return {}

    @staticmethod
    @cached(cache=TTLCache(maxsize=1000, ttl=60))
    def get_base_info(user_id: int) -> dict:
        _list = DB.query_sql(
            sql="select id as user_id, age, gender, occupation, zip_code from users where id=%(id)s limit 1",
            id=user_id
        )
        if len(_list) > 0:
            return _list[0]
        else:
            return {}
