# -*- coding: utf-8 -*-
from typing import List

from ...entity import InnerRecItem


class BaseRank(object):
    def __init__(self, name: str):
        """
        构造方法
        :param name: 当前策略名称，全局唯一
        """
        self.name = name

    def rank_items(self, items: List[InnerRecItem], **kwargs) -> List[InnerRecItem]:
        raise NotImplementedError("当前子类未实现具体的排序方法!")
