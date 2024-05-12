import json
from typing import List, Optional

from .entity.config_param import ConfigParams
from .entity.scene_meta import SceneMeta
from .filter_runner import FilterRunner
from .rank_runner import RankRunner
from .recall_runner import RecallRunner
from ..entity.rec_item import RecItem
from ..entity.spu_feature import SpuFeatureEntity
from ..entity.user_feature import UserFeatureEntity
from ..utils.logger_util import logger


class StrategyRunner(object):
    def __init__(self):
        self.recall_runner = RecallRunner()
        self.filter_runner = FilterRunner()
        self.rank_runner = RankRunner()

    def get_rec_spu_ids_by_scene(self,
                                 config: ConfigParams,
                                 scene: SceneMeta,
                                 user: Optional[UserFeatureEntity],
                                 spu: Optional[SpuFeatureEntity]
                                 ) -> List[RecItem]:
        """
        基于场景配置信息提取对应的推荐结果
        :param config: 配置参数, 也就是搜索相关的参数
        :param scene: 场景对象，定义了召回、过滤、精排、重排等策略的元数据
        :param user: 用户对象，当前场景对应的用户对象，有可能为None
        :param spu: 商品对象，当前场景对应的物品对象，有可能为None
        :return:
        """
        if scene is None:
            logger.error("进行推荐操作的时候，场景入参不允许为None.")
            return []
        # 1. 召回
        items = self.recall_runner.fetch_candidates_items(config, scene, user, spu)
        # 1.1 TODO: 针对items进行商品特征的补全，同时过滤不可见商品
        # TODO: 获取不可见商品id列表，也需要进行删除 --> 作业：类似写redis的形式，弄一个mysql的数据库连接池，然后在这个位置检查具体有那些视频是不可见的
        # 2. 过滤
        items = self.filter_runner.filter_candidates_items(items, config, scene, user, spu)
        # 3. 排序
        items = self.rank_runner.rank_candidates_items(items, config, scene, user, spu)
        # 4. 重排

        logger.info(
            "当前参数[%s - %s - %s - %s]情况下推荐结果为:%s",
            config.number_push, scene.name, None if user is None else user.id,
            None if spu is None else spu.id, json.dumps([dict(item) for item in items])
        )
        # 5. 获取推荐结果(前n个推荐结果)
        return items[:config.number_push]
