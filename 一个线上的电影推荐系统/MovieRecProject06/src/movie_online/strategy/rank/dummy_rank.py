# -*- coding: utf-8 -*-

from typing import List

from .base import BaseRank
from ...entity import InnerRecItem


class DummyRank(BaseRank):
    def __init__(self):
        super(DummyRank, self).__init__(name="dummy")

    def rank_items(self, items: List[InnerRecItem], **kwargs) -> List[InnerRecItem]:
        for item in items:
            item.add_rank(self.name, self.name)
            item.score = 1.0  # 权重置信度全部重置为0, 防止有一些召回路径中给定的score并不是1.0
        items.sort(key=lambda t: t.score, reverse=True)
        return items
