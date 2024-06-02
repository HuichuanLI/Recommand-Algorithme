# -*- coding: utf-8 -*-
from typing import List, Optional

from .base import BaseRecall
from ...entity import InnerRecItem
from ...services.spu_feature_service import SpuFeatureService


class HotsRecall(BaseRecall):
    def __init__(self):
        super(HotsRecall, self).__init__('hots')

    def get_candidates(self, k: Optional[int] = None, **kwargs) -> List[InnerRecItem]:
        """
        获取候选集
        :param k: 获取前多少个热映商品
        :param kwargs:
        :return: 热映商品列表
        """
        # 1. 从redis中获取热映商品列表
        spu_ids = SpuFeatureService.get_hot_ids()

        # 2. 从列表中随机选择k个商品返回
        return self.fetch_random_candidates(spu_ids, k)


class LocHotsRecall(BaseRecall):
    def __init__(self):
        super(LocHotsRecall, self).__init__('loc_hots')

    def get_candidates(self,
                       k: Optional[int] = None, location_id: Optional[int] = None, **kwargs
                       ) -> List[InnerRecItem]:
        """
        获取候选集
        :param k: 获取前多少个热映商品
        :param location_id: 地域唯一标识符id
        :param kwargs:
        :return: 热映商品列表
        """
        if location_id is None:
            return []

        # 1. 从redis中获取地域热映商品列表
        spu_ids = SpuFeatureService.get_location_hot_ids(location_id)

        # 2. 从列表中随机选择k个商品返回
        return self.fetch_random_candidates(spu_ids, k)
