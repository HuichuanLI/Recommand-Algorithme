# -*- coding: utf-8 -*-
from typing import List, Optional, Dict

from .base import BaseRank
from ... import logger
from ...entity import InnerRecItem
from ...entity.config_param import ConfigParams
from ...entity.spu_feature import SpuFeatureEntity
from ...services.spu_feature_service import SpuFeatureService


# noinspection DuplicatedCode
class CategoryWeightingRank(BaseRank):
    def __init__(self):
        super(CategoryWeightingRank, self).__init__('category')

    def rank_items(self,
                   rank_num: int,
                   items: List[InnerRecItem],
                   config: Optional[ConfigParams] = None,
                   spus: Optional[Dict[int, SpuFeatureEntity]] = None,
                   category_weight: float = 1.1,
                   **kwargs) -> List[InnerRecItem]:
        if config is None or config.spu is None or config.spu.categorys is None or len(config.spu.categorys) == 0:
            return items
        if spus is None or len(spus) != len(items):
            spus = SpuFeatureService.get_effect_spu_features(
                spu_ids=[item.spu_id for item in items]
            )
        current_spu_categorys = config.spu.categorys
        update_func = self.fetch_update_score_func(rank_num)

        def _calc_weight(_item, _spu, **kwargs):
            try:
                if _spu is None:
                    return 1.0
                _spu_categorys = _spu.categorys  # 推荐候选商品所属类别的名称列表
                if _spu_categorys is None or len(_spu_categorys) == 0:
                    return 1.0
                # 只要推荐候选商品类别名称列表和当前实际商品类别名称列表有重叠，那么进行加权
                for cate in current_spu_categorys:
                    if cate in _spu_categorys:
                        return category_weight
                return 1.0
            except Exception as e:
                logger.error(f"计算权重信息失败:{self.name} - {item} - {_spu}", exc_info=e)
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
