# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import List, Dict, Tuple

from .filter.user_blacklist_filter import UserBlacklistFilter
from .filter.user_nolike_category_filter import UserNoLikeCategoryFilter
from .filter.user_views_filter import UserViewsFilter
from .. import logger
from ..entity import InnerRecItem
from ..entity.config_param import ConfigParams
from ..entity.scene_meta import SceneMeta
from ..entity.spu_feature import SpuFeatureEntity


class FilterRunner(object):
    def __init__(self):
        self.user_views_filter = UserViewsFilter()
        self.user_blacklist_filter = UserBlacklistFilter()
        self.user_nolike_category_filter = UserNoLikeCategoryFilter()

    def filter_candidates_items(self,
                                scene: SceneMeta, config: ConfigParams,
                                items: List[InnerRecItem], spus: Dict[int, SpuFeatureEntity]
                                ) -> Tuple[List[InnerRecItem], List[InnerRecItem]]:
        if items is None or len(items) == 0:
            logger.warn("召回商品列表为空，无法进行过滤操作!")
            return [], []

            # 1. 获取所有无效的商品id列表 --> 待过滤的商品id列表
        filtered_spu_ids = defaultdict(list)  # 商品id为key，所有触发过滤该商品的策略字符串列表为value
        for _filter in scene.filters:
            _ids = self._get_filtered_spu_ids(_filter, config, items, spus)
            if _ids is None or len(_ids) == 0:
                continue
            for _id in _ids:
                filtered_spu_ids[_id].append(_filter)

        # 2. 遍历所有商品，提取得到有效商品和无效商品列表
        effect_items, no_effect_items = [], []
        for item in items:
            if not item.is_effect:
                # 当前商品本来就是无效商品
                no_effect_items.append(item)
                continue

            if item.spu_id in filtered_spu_ids:
                item.add_filter(';'.join(filtered_spu_ids[item.spu_id]))  # 商品过滤
                no_effect_items.append(item)
            else:
                effect_items.append(item)

        return effect_items, no_effect_items

    def _get_filtered_spu_ids(self,
                              filter_strategy: str, config: ConfigParams,
                              items: List[InnerRecItem], spus: Dict[int, SpuFeatureEntity]) -> List[int]:
        spu_ids = []
        try:
            if filter_strategy.startswith('user_views_filter:'):
                k = int(filter_strategy[len('user_views_filter:'):])
                spu_ids = self.user_views_filter.get_spu_ids(user=config.user, k=k)
            elif filter_strategy == 'user_blacklist_filter':
                spu_ids = self.user_blacklist_filter.get_spu_ids(user=config.user)
            elif filter_strategy == 'user_nolike_category_filter':
                spu_ids = self.user_nolike_category_filter.get_spu_ids(spus, config.user)
            else:
                logger.error(f"当前系统还不支持过滤策略:{filter_strategy}")
        except Exception as e:
            logger.error(f"过滤策略:{filter_strategy}执行异常.", exc_info=e)
        return spu_ids
