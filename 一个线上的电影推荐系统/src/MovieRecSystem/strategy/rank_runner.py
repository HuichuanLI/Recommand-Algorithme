from typing import List, Optional

from .entity.config_param import ConfigParams
from .entity.scene_meta import SceneMeta
from .rank.random_rank import RandomRank
from ..entity.rec_item import RecItem
from ..entity.spu_feature import SpuFeatureEntity
from ..entity.user_feature import UserFeatureEntity
from ..utils.logger_util import logger


class RankRunner(object):
    def __init__(self):
        self.random_rank = RandomRank()

    def rank_candidates_items(self,
                              candidates_items: List[RecItem],
                              config: ConfigParams,
                              scene: SceneMeta,
                              user: Optional[UserFeatureEntity],
                              spu: Optional[SpuFeatureEntity]
                              ) -> List[RecItem]:
        """
        对候选集采用给定
        :param candidates_items: 候选集
        :param config: 配置对象
        :param scene: 场景对象
        :param user: 用户对象
        :param spu: 商品对象
        :return:
        """
        if candidates_items is None or len(candidates_items) == 0:
            return []
        # 根据每个排序策略进行排序
        items = candidates_items
        for _rank in scene.ranks:
            items = self._rank_items(_rank, config, user, spu, items)
        # 结果返回
        return items

    def _rank_items(self,
                    rank_strategy: str,
                    config: ConfigParams,
                    user: Optional[UserFeatureEntity],
                    spu: Optional[SpuFeatureEntity],
                    candidates_items: List[RecItem]
                    ):
        """
        使用给定的策略对列表进行排序
        :param rank_strategy: 策略字符串
        :param config: 配置对象
        :param user: 用户对象
        :param spu: 商品对象
        :param candidates_items: 候选集列表
        :return: 排序后结果列表
        """
        try:
            if "random" == rank_strategy:
                return self.random_rank.rank_items(items=candidates_items)
            else:
                logger.warn(f"不支持该排序策略:{rank_strategy}")
        except Exception as e:
            logger.error(f"基于给定的排序策略进行商品排序异常:{rank_strategy}.", exc_info=e)
        return candidates_items
