# -*- coding: utf-8 -*-
from typing import List, Dict

from .rank.dummy_rank import DummyRank
from .rank.recall_weighting_rank import RecallWeightingRank
from .. import logger
from ..entity import InnerRecItem
from ..entity.config_param import ConfigParams
from ..entity.scene_meta import SceneMeta
from ..entity.spu_feature import SpuFeatureEntity


class RankRunner(object):
    def __init__(self):
        self.dummy = DummyRank()
        self.recall_weighting = RecallWeightingRank()

    def rank_candidates_items(self,
                              scene: SceneMeta, config: ConfigParams,
                              items: List[InnerRecItem], spus: Dict[int, SpuFeatureEntity]
                              ) -> List[InnerRecItem]:
        # 如果候选商品列表为空，直接返回
        if items is None or len(items) == 0:
            logger.warn("候选商品列表为空，不需要进行排序操作!")
            return []
        # 遍历所有的排序策略，获取最终排序结果
        for _rank in scene.ranks:
            items = self._rank_items(_rank, config, items, spus)
        return items

    def _rank_items(self,
                    rank, config: ConfigParams,
                    items: List[InnerRecItem], spus: Dict[int, SpuFeatureEntity]
                    ) -> List[InnerRecItem]:
        """
        基于具体的排序策略对候选商品列表进行排序操作
        :param rank: 排序策略名称字符串
        :param config: 请求配置对象
        :param items: 候选商品列表
        :param spus: 商品属性mapping对象
        :return:
        """
        try:
            if rank == 'dummy':
                items = self.dummy.rank_items(items)
            elif rank == 'recall_weighting':
                items = self.recall_weighting.rank_items(items)
            else:
                logger.error(f"当前系统不支持该排序策略:{rank}")
        except Exception as e:
            logger.error(f"当前排序策略:{rank}支持出现异常:{config} - {list(spus.keys())}", exc_info=e)
        return items
