# -*- coding: utf-8 -*-
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np

from .base import BaseRank
from ... import logger
from ...entity import InnerRecItem
from ...entity.spu_feature import SpuFeatureEntity
from ...services.spu_feature_service import SpuFeatureService


class NewItemWeightingRank(BaseRank):
    def __init__(self):
        super(NewItemWeightingRank, self).__init__("new_item")
        self.k = 2

    def rank_items(self,
                   rank_num: int,
                   items: List[InnerRecItem],
                   spus: Optional[Dict[int, SpuFeatureEntity]] = None,
                   **kwargs
                   ) -> List[InnerRecItem]:
        """
        针对待排序的商品列表进行新商品的加权
        新商品权重计算的规则：
        方式一：分段计算权重，上映时间在10天以前的情况，直接设定权重0.8，在10天到3天之间的权重为1.0，在3天以内的权重直接为1.2
        方式二：可以自行构建一个函数，这个函数能够体现和上映时间之间的转换关系即可
             w = np.exp(alpha) * (1.0 + k / (current_date - release_date + k))
             w = alpha ^ (k / (current_date - release_date + k))
             w = np.exp(alpha) * (1.0 + k / (beta * (current_date - release_date) + k))
             current_date: 当前时间，就是一个固定值
             release_date: 上映时间，该值越小表示商品越旧
        :param rank_num:
        :param items:
        :param spus: 商品信息dict
        :param kwargs:
        :return:
        """
        if spus is None or len(spus) != len(items):
            spus = SpuFeatureService.get_effect_spu_features(
                spu_ids=[item.spu_id for item in items]
            )
        update_func = self.fetch_update_score_func(rank_num)

        def _calc_weight(_item, _spu, alpha=1.0, **kwargs):
            try:
                if _spu is None:
                    return 1.0
                current_date = datetime.now()
                release_date = datetime.strptime(_spu.release_date, '%d-%b-%Y')
                offset_seconds = (current_date - release_date).total_seconds()
                return np.exp(alpha) * (1.0 + self.k / (offset_seconds + self.k))
            except Exception as e:
                logger.error(f"计算权重信息失败:{item} - {_spu}", exc_info=e)
                return 1.0

        for item in items:
            # 1. 计算权重信息
            score_w = _calc_weight(item, spus.get(item.spu_id), **kwargs)
            # 2. 基于权重信息对商品的评分进行合并
            score = update_func(v1=item.score, v2=score_w)
            # 3. 赋值
            item.add_rank(f"{self.name}_{score:.3f}_{score_w:.3f}", self.name)
            item.score = score
        return items
