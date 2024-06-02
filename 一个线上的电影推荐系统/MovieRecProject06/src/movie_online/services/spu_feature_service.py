# -*- coding: utf-8 -*-
"""
负责商品相关特征的提取服务
"""
from typing import List, Dict, Tuple

from cachetools import cached, TTLCache

from ..entity.spu_feature import SpuFeatureEntity
from ..utils.mysql_util import DB
from ..utils.redis_util import RedisUtil


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
class SpuFeatureService(object):
    spu_cache = TTLCache(maxsize=10000, ttl=120)
    spu_null_cache = TTLCache(maxsize=10000, ttl=60)

    @staticmethod
    @cached(cache=TTLCache(maxsize=1, ttl=600))
    def get_new_ids() -> List[int]:
        """
        获取新品列表
        cached这个装饰器的作用就是，当重复调用的时候，不会执行实际的代码逻辑，而且直接从内存中获取上一次的结果数据，可以减少对数据库的压力
        :return:
        """
        key = "rec:new_spu"
        print(f"调用redis:{key}")
        return RedisUtil.get_list_int(key)

    @staticmethod
    @cached(cache=TTLCache(maxsize=1, ttl=600))
    def get_hot_ids() -> List[int]:
        """
        获取热映品列表
        cached这个装饰器的作用就是，当重复调用的时候，不会执行实际的代码逻辑，而且直接从内存中获取上一次的结果数据，可以减少对数据库的压力
        :return:
        """
        key = "rec:hot_spu"
        print(f"调用redis:{key}")
        return RedisUtil.get_list_int(key)

    @staticmethod
    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def get_location_new_ids(location_id: int) -> List[int]:
        """
        获取地域新品列表
        cached这个装饰器的作用就是，
            当重复调用(入参一样)的时候，不会执行实际的代码逻辑，而且直接从内存中获取对应入参上一次的结果数据，可以减少对数据库的压力
        :param location_id 地域唯一标识符id
        :return:
        """
        key = f"rec:new_spu:loc_{location_id}"
        print(f"调用redis:{key}")
        return RedisUtil.get_list_int(key)

    @staticmethod
    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def get_location_hot_ids(location_id: int) -> List[int]:
        key = f"rec:hot_spu:loc_{location_id}"
        print(f"调用redis:{key}")
        return RedisUtil.get_list_int(key)

    @staticmethod
    def get_effect_spu_features(spu_ids: List[int]) -> Dict[int, SpuFeatureEntity]:
        """
        基于给定的商品id列表，从数据库&第三方的服务获取有效商品的特征属性
        :param spu_ids:
        :return:
        """

        def _get_spu_features(_miss_spu_ids):
            _list = DB.query_sql(
                sql="SELECT * FROM movies WHERE id in %(ids)s",
                ids=_miss_spu_ids
            )
            _result = {}
            for _record in _list:
                _result[_record['id']] = SpuFeatureEntity(_record)
            return _result

        if spu_ids is None or len(spu_ids) == 0:
            return {}

        result: Dict[int, SpuFeatureEntity] = {}
        # 1. 从缓存中获取数据
        miss_spu_ids = []
        miss_spu_cache_keys = []
        for spu_id in spu_ids:
            cache_key = f"spu_{spu_id}"
            try:
                spu: SpuFeatureEntity = SpuFeatureService.spu_cache[cache_key]
                if spu.viewable:
                    result[spu_id] = spu  # 添加到结果列表中
            except KeyError:
                if cache_key in SpuFeatureService.spu_null_cache:
                    # 表示当前spu id在源数据库MySQL中也不存在
                    continue
                # 当前spu id需要重新从数据源获取
                miss_spu_ids.append(spu_id)
                miss_spu_cache_keys.append(cache_key)

        # 2. 针对缓存中不存在的数据，从数据库获取数据，并完成添加缓存、数据合并
        if len(miss_spu_ids) > 0:
            miss_spus = _get_spu_features(miss_spu_ids)  # 从原始数据源获取
            for spu_id, cache_key in zip(miss_spu_ids, miss_spu_cache_keys):
                if spu_id in miss_spus:
                    spu = miss_spus[spu_id]
                    SpuFeatureService.spu_cache[cache_key] = spu  # 添加缓存
                    if spu.viewable:
                        result[spu_id] = spu  # 添加到结果列表中
                else:
                    SpuFeatureService.spu_null_cache[cache_key] = 1  # 表示当前spu不存在

        # 3. 结果返回
        return result

    @staticmethod
    def get_i2i_recall_spus(spu_id: int, field_names: List[str]) -> Dict[str, List[Tuple[int, float]]]:
        def _parse_value(_value: str):
            # 851:5.000,1449:4.943,868:4.789,408:4.728,483:4.673
            vs = []
            for t in _value.strip().split(","):
                spu_id, spu_score = t.split(":")
                vs.append((int(spu_id), float(spu_score)))
            return vs

        key = f"rec:recall:i2i:{spu_id}"
        print(f"调用redis:{key}")
        values: Dict[str, str] = RedisUtil.get_hash(key, fields=field_names)
        return {_k: _parse_value(_v) for _k, _v in values.items()}
