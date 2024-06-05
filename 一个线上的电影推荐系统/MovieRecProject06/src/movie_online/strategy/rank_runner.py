# -*- coding: utf-8 -*-
from typing import List, Dict

from .rank.bpr_rank import BPRRank
from .rank.category_weighting_rank import CategoryWeightingRank
from .rank.deepfm_rank import DeepFMRank
from .rank.dummy_rank import DummyRank
from .rank.fm_rank import FMRank
from .rank.gbdt_lr_rank import GBDTLrRank
from .rank.lr_rank import LrRank
from .rank.new_item_weighting_rank import NewItemWeightingRank
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
        self.fm = FMRank()
        self.lr = LrRank()
        self.gbdt_lr = GBDTLrRank()
        self.bpr = BPRRank()
        self.new_item = NewItemWeightingRank()
        self.deepfm = DeepFMRank()
        self.cate = CategoryWeightingRank()

    def rank_candidates_items(self,
                              scene: SceneMeta, config: ConfigParams,
                              items: List[InnerRecItem], spus: Dict[int, SpuFeatureEntity]
                              ) -> List[InnerRecItem]:
        # 如果候选商品列表为空，直接返回
        if items is None or len(items) == 0:
            logger.warn("候选商品列表为空，不需要进行排序操作!")
            return []
        # 遍历所有的排序策略，获取最终排序结果
        for _rank_num, _rank in enumerate(scene.ranks):
            items = self._rank_items(_rank_num, _rank, config, items, spus)
        # 最终排序一下
        items.sort(key=lambda t: t.score, reverse=True)
        return items

    def _rank_items(self,
                    rank_num: int,
                    rank: str,
                    config: ConfigParams,
                    items: List[InnerRecItem],
                    spus: Dict[int, SpuFeatureEntity]
                    ) -> List[InnerRecItem]:
        """
        基于具体的排序策略对候选商品列表进行排序操作
        :param rank_num: 当前是第几个排序策略
        :param rank: 排序策略名称字符串
        :param config: 请求配置对象
        :param items: 候选商品列表
        :param spus: 商品属性mapping对象
        :return:
        """
        try:
            if rank == 'dummy':
                items = self.dummy.rank_items(rank_num, items)
            elif rank == 'recall_weighting':
                items = self.recall_weighting.rank_items(rank_num, items)
            elif rank == 'fm':
                items = self.fm.rank_items(rank_num, items, user=config.user, spus=spus)
            elif rank == 'deepfm':
                items = self.deepfm.rank_items(rank_num, items, user=config.user, spus=spus)
            elif rank == 'lr':
                items = self.lr.rank_items(rank_num, items, user=config.user, spus=spus)
            elif rank == 'gbdt_lr':
                items = self.gbdt_lr.rank_items(rank_num, items, user=config.user, spus=spus)
            elif rank == 'bpr':
                items = self.bpr.rank_items(rank_num, items, user=config.user, spus=spus)
            elif rank == 'new_item':
                items = self.new_item.rank_items(rank_num, items, spus=spus, alpha=1.0)
            elif rank.startswith("new_item:"):
                alpha = float(rank[len("new_item:"):])
                items = self.new_item.rank_items(rank_num, items, spus=spus, alpha=alpha)
            elif rank == 'category':
                items = self.cate.rank_items(rank_num, items, config=config, spus=spus)
            elif rank.startswith("category:"):
                category_weight = float(rank[len("category:"):])
                items = self.cate.rank_items(rank_num, items, config=config, spus=spus, category_weight=category_weight)
            else:
                logger.error(f"当前系统不支持该排序策略:{rank}")
        except Exception as e:
            logger.error(f"当前排序策略:{rank}支持出现异常:{config} - {list(spus.keys())}", exc_info=e)
        return items
