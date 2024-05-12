from typing import List, Dict, Optional

from .entity.config_param import ConfigParams
from .entity.scene_meta import SceneMeta
from .recall import *
from ..entity.rec_item import RecItem
from ..entity.spu_feature import SpuFeatureEntity
from ..entity.user_feature import UserFeatureEntity
from ..utils.logger_util import logger


class RecallRunner(object):
    def __init__(self):
        self.hot = HotBasedRecall()
        self.new = NewBasedRecall()
        self.loc_hot = LocationHotBasedRecall()
        self.loc_new = LocationNewBasedRecall()
        self.cate_hot = CategoryHotBasedRecall()
        self.cate_new = CategoryNewBasedRecall()

    def fetch_candidates_items(self,
                               config: ConfigParams,
                               scene: SceneMeta,
                               user: Optional[UserFeatureEntity],
                               spu: Optional[SpuFeatureEntity]
                               ) -> List[RecItem]:
        """
        基于多路召回策略提取候选商品列表

        :param config:
        :param scene:
        :param user:
        :param spu:
        :return: 候选商品列表，可能为空列表
        """
        recalls = scene.recalls
        if recalls is None or len(recalls) == 0:
            raise ValueError("参数recalls为空，必须给定非空值!")
        # 遍历提取每个策略对应的召回结果
        recall_dict: Dict[int, RecItem] = {}
        for _recall in recalls:
            items = self._get_recommend_candidates(
                recall=_recall, user=user, spu=spu, config=config
            )
            if items is None:
                continue
            for item in items:
                if item.spu_id == 0:
                    logger.warn("召回异常:%s %s", _recall, item)
                    continue
                if item.spu_id in recall_dict:
                    # 考虑合并(多个策略的合并)
                    recall_dict[item.spu_id].merge_explain(item.explains)
                else:
                    recall_dict[item.spu_id] = item
        # 按照score评分降顺排列
        result: List[RecItem] = list(recall_dict.values())
        result.sort(key=lambda t: t.score, reverse=True)
        return result

    def _get_recommend_candidates(self,
                                  recall: str,
                                  user: Optional[UserFeatureEntity],
                                  spu: Optional[SpuFeatureEntity],
                                  config: ConfigParams
                                  ) -> List[RecItem]:
        """
        进行具体召回策略的商品召回
        :param recall: 召回策略字符串
        :param user: 用户对象，可能为None
        :param spu: 商品对象，可能为None
        :param config: 配置信息，不能为None
        :return:
        """
        try:
            if "hots" == recall:
                return self.hot.get_candidates(number_push=config.number_push)
            elif "new" == recall:
                return self.new.get_candidates(number_push=config.number_push)
            elif "loc_hots" == recall:
                return self.loc_hot.get_candidates(None if user is None else user.location_id, config.number_push)
            elif "loc_news" == recall:
                return self.loc_new.get_candidates(None if user is None else user.location_id, config.number_push)
            elif "cate_news" == recall:
                if spu is None:
                    return []
                return self.cate_new.get_candidates(spu.category_id, config.number_push)
            elif "cate_hots" == recall:
                if spu is None:
                    return []
                return self.cate_hot.get_candidates(spu.category_id, config.number_push)
            elif "sim_cate_news" == recall:
                if spu is None or len(spu.sim_category_ids) == 0:
                    return []
                _rs = []
                for _category_id in spu.sim_category_ids:
                    _rs.extend(self.cate_new.get_candidates(_category_id, config.number_push, "sim_cate_news"))
                return _rs
            elif "sim_cate_hots" == recall:
                if spu is None or len(spu.sim_category_ids) == 0:
                    return []
                _rs = []
                for _category_id in spu.sim_category_ids:
                    _rs.extend(self.cate_hot.get_candidates(_category_id, config.number_push, "sim_cate_hots"))
                return _rs
            else:
                logger.warn(f"不支持该召回策略:{recall}")
        except Exception as e:
            logger.error(f"基于给定的召回策略提取商品列表信息失败:{recall}.", exc_info=e)
        return []
