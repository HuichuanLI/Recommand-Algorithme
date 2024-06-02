# -*- coding: utf-8 -*-
from typing import List


class BaseFilter(object):
    def __init__(self, name: str):
        """
        构造方法
        :param name: 当前策略名称，全局唯一
        """
        self.name = name

    def get_spu_ids(self, **kwargs) -> List[int]:
        """
        获取待过滤的商品id列表 ---> 这些id对应的商品不应该推荐给用户
        :param kwargs:
        :return:
        """
        raise NotImplementedError(f"当前子类没有实现具体过滤策略方法:{self.name}")
