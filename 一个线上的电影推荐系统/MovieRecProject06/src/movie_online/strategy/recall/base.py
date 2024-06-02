# -*- coding: utf-8 -*-
import copy
import random
from typing import List, Optional

from ...entity import InnerRecItem


class BaseRecall(object):
    def __init__(self, name: str):
        """
        构造方法
        :param name: 当前策略名称，全局唯一
        """
        self.name = name

    def get_candidates(self, **kwargs) -> List[InnerRecItem]:
        raise NotImplementedError("当前类没有实现获取候选集的方法!")

    def fetch_random_candidates(self, spu_ids: List[int], k: Optional[int] = None) -> List[InnerRecItem]:
        """
        随机从spu_ids中获取k个商品，并组成返回结果
        :param spu_ids: 商品id列表
        :param k: 待获取的商品数量
        :return:
        """
        if spu_ids is None or len(spu_ids) == 0:
            return []

        # 随机获取k个
        if k is not None and k > 0:
            spu_ids = copy.deepcopy(spu_ids)
            random.shuffle(spu_ids)
            spu_ids = spu_ids[:k]

        # 构造返回结果
        items = []
        for spu_id in spu_ids:
            items.append(InnerRecItem.get_recall_rec_item(spu_id, self.name))
        return items
