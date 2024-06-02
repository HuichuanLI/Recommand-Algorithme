# -*- coding: utf-8 -*-
from typing import List, Optional

from .base import BaseRecall
from ...entity import InnerRecItem
from ...services.spu_feature_service import SpuFeatureService


class NewsRecall(BaseRecall):
    def __init__(self):
        super(NewsRecall, self).__init__('news')

    def get_candidates(self, k: Optional[int] = None, **kwargs) -> List[InnerRecItem]:
        """
        获取候选集
        :param k: 获取前多少个新商品
        :param kwargs:
        :return: 新商品列表
        """
        # 1. 从redis中获取新品列表
        spu_ids = SpuFeatureService.get_new_ids()

        # 2. 从列表中随机选择k个商品返回
        return self.fetch_random_candidates(spu_ids, k)


class LocNewsRecall(BaseRecall):
    def __init__(self):
        super(LocNewsRecall, self).__init__('loc_news')

    def get_candidates(self,
                       k: Optional[int] = None, location_id: Optional[int] = None, **kwargs
                       ) -> List[InnerRecItem]:
        """
        获取候选集
        :param k: 获取前多少个新商品
        :param location_id: 位置id
        :param kwargs:
        :return: 新商品列表
        """
        if location_id is None:
            return []

        # 1. 从redis中获取地域新品列表
        spu_ids = SpuFeatureService.get_location_new_ids(location_id)

        # 2. 从列表中随机选择k个商品返回
        return self.fetch_random_candidates(spu_ids, k)
