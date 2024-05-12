from typing import Optional, List

from .filter.blacklist_filter import BlacklistFilter
from .filter.user_views import UserViewsFilter
from ..entity.rec_item import RecItem
from ..entity.spu_feature import SpuFeatureEntity
from .entity.config_param import ConfigParams
from .entity.scene_meta import SceneMeta
from ..entity.user_feature import UserFeatureEntity
from ..utils.logger_util import logger


class FilterRunner(object):
    def __init__(self):
        self.user_views = UserViewsFilter(start=0, end=9)
        self.blacklist_filter = BlacklistFilter()
        pass

    def filter_candidates_items(self,
                                candidates_items: List[RecItem],
                                config: ConfigParams,
                                scene: SceneMeta,
                                user: Optional[UserFeatureEntity],
                                spu: Optional[SpuFeatureEntity]
                                ) -> List[RecItem]:
        """
        针对候选列表商品数据进行过滤操作
        NOTE: 实现效果上来讲，最好不要改变顺序
        :param candidates_items: 待过滤的候选推荐列表数据
        :param config: 配置对象
        :param scene: 场景对象
        :param user: 用户
        :param spu: 商品
        :return: 过滤后的推荐列表
        """
        if candidates_items is None or len(candidates_items) == 0:
            return []
        filtered_spu_ids = set()
        # 遍历每个过滤策略进行数据过滤，得到需要删除的id列表
        candidates_spu_ids = [v.spu_id for v in candidates_items]
        for _filter in scene.filters:
            # 返回的是该过滤策略决定那些id是需要删除的
            _ids = self._filter_item_ids(_filter, config, user, spu, candidates_spu_ids)
            for _id in _ids:
                filtered_spu_ids.add(_id)
        # 过滤提取商品列表
        return [v for v in candidates_items if v.spu_id not in filtered_spu_ids]

    def _filter_item_ids(self,
                         filter_strategy: str,
                         config: ConfigParams,
                         user: Optional[UserFeatureEntity],
                         spu: Optional[SpuFeatureEntity],
                         candidates_item_ids: List[int]
                         ):
        """
        提取需要删除的商品id列表
        :param filter_strategy: 策略字符串
        :param config: 配置信息
        :param user: 用户对象
        :param spu: 商品对象
        :param candidates_item_ids: 候选集商品id列表
        :return: 待删除的商品id列表
        """
        try:
            if "views" == filter_strategy:
                return self.user_views.get_user_view_spu_ids(user)
            elif "blacklist" == filter_strategy:
                return self.blacklist_filter.get_user_blacklist_spu_ids(user)
            else:
                logger.warn(f"不支持该过滤策略:{filter_strategy}")
        except Exception as e:
            logger.error(f"基于给定的过滤策略进行商品过滤异常:{filter_strategy}.", exc_info=e)
        return []
