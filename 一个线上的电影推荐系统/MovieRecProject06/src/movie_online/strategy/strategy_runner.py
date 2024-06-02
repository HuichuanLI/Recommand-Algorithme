# -*- coding: utf-8 -*-
import json
from typing import List, Dict

from .filter_runner import FilterRunner
from .rank_runner import RankRunner
from .recall_runner import RecallRunner
from .. import logger
from ..entity import RecItem, InnerRecItem
from ..entity.config_param import ConfigParams
from ..entity.scene_meta import SceneMeta
from ..entity.spu_feature import SpuFeatureEntity
from ..services.spu_feature_service import SpuFeatureService

class StrategyRunner(object):
    def __init__(self):
        self.recall_runner = RecallRunner()
        self.filter_runner = FilterRunner()
        self.rank_runner = RankRunner()

    @staticmethod
    def _fetch_effect_spu_features_and_update_candidates(items: List[InnerRecItem]) -> Dict[int, SpuFeatureEntity]:
        # 1. 实际情况下，应该是调用其它项目组提供的商品接口，获取有效商品集合(特征属性集合)
        # NOTE: 有效的定义 --> 通用场景下，可以推送给用户的商品
        spus: Dict[int, SpuFeatureEntity] = SpuFeatureService.get_effect_spu_features(
            spu_ids=[item.spu_id for item in items]
        )
        # 2. 针对未返回的商品id，直接标记为filter状态
        for item in items:
            if item.spu_id not in spus:
                item.add_filter('effective_filter')  # 商品过滤
        return spus

    def get_rec_items_by_scene(self, scene: SceneMeta, config: ConfigParams) -> List[RecItem]:
        """
        基于入参获取推荐商品列表
        :param scene: 场景对象
        :param config: 请求对象
        :return: 返回的商品列表
        """
        # 1. 获取召回商品列表
        items: List[InnerRecItem] = self.recall_runner.fetch_candidates_items(scene, config)

        # 2. 过滤商品
        # 2.1 调用其它模块，获取有效商品对象属性
        spus: Dict[int, SpuFeatureEntity] = self._fetch_effect_spu_features_and_update_candidates(items)
        # 2.2 具体的过滤策略进行过滤
        effect_items, no_effect_items = self.filter_runner.filter_candidates_items(scene, config, items, spus)

        # 3. 商品排序
        sorted_items: List[InnerRecItem] = self.rank_runner.rank_candidates_items(scene, config, effect_items, spus)

        # 4. 日志记录&结果数据的转换返回
        logger.info(
            f"当前参数:[{scene.name} - {config.user_id} - {config.spu_id} - {config.number_push}]，"
            f"推荐返回结果为:{json.dumps([item.to_dict() for item in sorted_items])}，"
            f"过滤掉的商品列表为:{json.dumps([item.to_dict() for item in no_effect_items])}"
        )

        # 将内部商品对象转换为外部实际商品对象
        result_items = [RecItem.build_with_inner_rec_item(item) for item in sorted_items if item.is_effect]
        result_items = result_items[:config.number_push]
        return result_items
