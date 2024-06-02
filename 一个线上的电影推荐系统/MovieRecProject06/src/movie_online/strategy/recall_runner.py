# -*- coding: utf-8 -*-
from typing import List, Dict

from .recall.fm_recall import FMRecall
from .recall.hots_recall import HotsRecall, LocHotsRecall
from .recall.i2i_recall import I2IRecall, UserViewI2IRecall
from .recall.news_recall import NewsRecall, LocNewsRecall
from .recall.u2i_recall import U2IRecall
from .. import logger
from ..entity import InnerRecItem
from ..entity.config_param import ConfigParams
from ..entity.scene_meta import SceneMeta


class RecallRunner(object):
    def __init__(self):
        self.news = NewsRecall()
        self.hots = HotsRecall()
        self.loc_news = LocNewsRecall()
        self.loc_hots = LocHotsRecall()
        self.u2i = U2IRecall()
        self.i2i = I2IRecall()
        self.user_views_i2i = UserViewI2IRecall()
        self.fm = FMRecall()


    def fetch_candidates_items(self, scene: SceneMeta, config: ConfigParams) -> List[InnerRecItem]:
        """
        召回策略执行器
        :param scene: 场景对象
        :param config: 请求对象
        :return: 召回列表
        """
        recalls = scene.recalls
        if recalls is None or len(recalls) == 0:
            logger.warn(f"当前召回策略为空，直接使用news和hots作为默认召回:{scene.name} - {recalls}")
            recalls = ['news', 'hots']
        recall_dict = self._inner_fetch_candidates_items(recalls, config)
        if len(recall_dict) == 0:
            logger.error(f"召回策略的执行没有召回出任何商品, 请检查:{scene.name} - {recalls} - {config}")
            recall_dict = self._inner_fetch_candidates_items(['hots', 'news'], config)
        return list(recall_dict.values())

    def _inner_fetch_candidates_items(self, recalls: List[str], config: ConfigParams) -> Dict[int, InnerRecItem]:
        """
        内部的获取召回商品列表的方法
        :param recalls: 召回策略字符串列表
        :param config: 配置信息
        :return:
        """
        # recall_dict中保存的是所有召回路径的召回商品列表
        recall_dict: Dict[int, InnerRecItem] = {}
        # 遍历所有策略，合并每个策略的召回商品列表
        for _recall in recalls:
            # 获取当前召回策略对应的召回商品列表
            _items = self._get_recommend_candidates(_recall, config)
            if _items is None or len(_items) == 0:
                logger.info(f"召回策略{_recall}返回结果为空，相关请求参数:{config}.")
                continue
            # 合并召回的商品序列
            for _item in _items:
                if _item.spu_id in recall_dict:
                    # 同一个商品被多个策略召回，那么进行策略合并
                    recall_dict[_item.spu_id].merge_explain(_item.explains, _item.score)
                else:
                    recall_dict[_item.spu_id] = _item
        return recall_dict

    def _get_recommend_candidates(self, recall: str, config: ConfigParams) -> List[InnerRecItem]:
        """
        基于给定的具体召回策略名称字符串，执行对应的召回逻辑，返回对应的召回结果
        :param recall: 召回策略名称字符串
        :param config: 请求参数配置信息
        :return:
        """
        items = []
        try:
            k = config.number_push * 2
            location_id = config.location_id  # 地域id
            if recall == 'news':
                items = self.news.get_candidates(k=k)
            elif recall == 'hots':
                items = self.hots.get_candidates(k=k)
            elif recall == 'loc_news':
                items = self.loc_news.get_candidates(k=k, location_id=location_id)
            elif recall == 'loc_hots':
                items = self.loc_hots.get_candidates(k=k, location_id=location_id)
            elif recall.startswith('u2i:'):
                u2i_names = recall[4:].split(":")
                print(u2i_names)
                items = self.u2i.get_candidates(k=k, user=config.user, u2i_names=u2i_names)
            elif recall.startswith("i2i:"):
                i2i_names = recall[4:].split(":")
                items = self.i2i.get_candidates(k=k, spu_id=config.spu_id, i2i_names=i2i_names)
            elif recall.startswith("user_views_i2i:"):
                i2i_names = recall[len('user_views_i2i:'):].split(":")
                items = self.user_views_i2i.get_candidates(k=k, user=config.user, i2i_names=i2i_names)
            elif recall.startswith("fm:"):
                model_version = recall[len("fm:"):]
                items = self.fm.get_candidates(user=config.user, k=k, model_version=model_version)
            elif recall == 'fm':
                items = self.fm.get_candidates(user=config.user, k=k)
            else:
                logger.error(f"当前系统不支持该召回策略:{recall}.")
        except Exception as e:
            logger.error(f"当前召回策略执行出现异常:{recall} - {config}.", exc_info=e)
        return items
