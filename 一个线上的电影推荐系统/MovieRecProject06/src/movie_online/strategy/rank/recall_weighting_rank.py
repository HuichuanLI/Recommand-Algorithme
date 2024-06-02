# -*- coding: utf-8 -*-
from typing import List

from .base import BaseRank
from ...entity import InnerRecItem


class RecallWeightingRank(BaseRank):
    def __init__(self):
        super(RecallWeightingRank, self).__init__("recall_weighting")

    def rank_items(self, items: List[InnerRecItem], **kwargs) -> List[InnerRecItem]:
        for item in items:
            item.add_rank(self.name, self.name)
            item.score = item.score * item.recall_explains()  # 直接召回数量加权
        items.sort(key=lambda t: t.score, reverse=True)
        return items
