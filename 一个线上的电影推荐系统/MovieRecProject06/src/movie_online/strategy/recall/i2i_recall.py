# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Tuple, Union

from .base import BaseRecall
from ...entity import InnerRecItem
from ...entity.user_feature import UserFeatureEntity
from ...services.spu_feature_service import SpuFeatureService
from ...services.user_feature_service import UserFeatureService


class I2IRecall(BaseRecall):
    def __init__(self):
        super(I2IRecall, self).__init__("i2i")

    def get_candidates(self,
                       k: Optional[int] = -1,
                       spu_id: Optional[int] = None,
                       i2i_names: Optional[List[str]] = None,
                       **kwargs) -> List[InnerRecItem]:
        if spu_id is None:
            return []
        if i2i_names is None or len(i2i_names) == 0:
            return []
        k = None if k <= 0 else k
        return self.inner_get_candidates(self.name, spu_id, i2i_names, k)

    @staticmethod
    def inner_get_candidates(
            name: str, spu_id: int, i2i_names: List[str], k: Union[int],
            result: List[InnerRecItem] = None
    ) -> List[InnerRecItem]:
        # 1. 从数据源获取当前用户对应的各个策略的召回列表
        values: Dict[str, List[Tuple[int, float]]] = SpuFeatureService.get_i2i_recall_spus(
            spu_id=spu_id,
            field_names=i2i_names
        )

        # 2. 拼接结果并返回
        if result is None:
            result = []
        for field_name, rec_spus in values.items():
            explain = f"{name}_{field_name}_{spu_id}"
            for spu_id, rec_score in rec_spus[:k]:
                result.append(InnerRecItem.get_recall_rec_item(
                    spu_id, explain, score=rec_score
                ))
        return result


class UserViewI2IRecall(BaseRecall):
    def __init__(self):
        super(UserViewI2IRecall, self).__init__('user_views_i2i')

    def get_candidates(self,
                       k: Optional[int] = -1,
                       user: Optional[UserFeatureEntity] = None,
                       i2i_names: Optional[List[str]] = None,
                       **kwargs) -> List[InnerRecItem]:
        if user is None or user.id is None:
            return []
        if i2i_names is None or len(i2i_names) == 0:
            return []

        # 0. 从数据源获取当前用户最近浏览的商品id列表
        view_spu_ids = UserFeatureService.get_view_spu_ids(user_id=user.id)
        if len(view_spu_ids) == 0:
            return []

        # 1. 遍历结果数据
        result = []
        for spu_id in view_spu_ids:
            result = I2IRecall.inner_get_candidates(
                self.name, spu_id, i2i_names, k, result
            )
        return result
