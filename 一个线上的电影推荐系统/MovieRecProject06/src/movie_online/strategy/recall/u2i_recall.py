# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Tuple

from .base import BaseRecall
from ...entity import InnerRecItem
from ...entity.user_feature import UserFeatureEntity
from ...services.user_feature_service import UserFeatureService


class U2IRecall(BaseRecall):
    def __init__(self):
        super(U2IRecall, self).__init__("u2i")

    def get_candidates(self,
                       k: Optional[int] = -1,
                       user: Optional[UserFeatureEntity] = None,
                       u2i_names: Optional[List[str]] = None,
                       **kwargs) -> List[InnerRecItem]:
        if user is None or user.id is None:
            return []
        if u2i_names is None or len(u2i_names) == 0:
            return []
        # 1. 从数据源获取当前用户对应的各个策略的召回列表
        values: Dict[str, List[Tuple[int, float]]] = UserFeatureService.get_u2i_recall_spus(
            user_id=user.id,
            field_names=u2i_names
        )

        # 2. 拼接结果并返回
        k = None if k <= 0 else k
        result = []
        for field_name, rec_spus in values.items():
            explain = f"{self.name}_{field_name}_{user.id}"
            for spu_id, rec_score in rec_spus[:k]:
                result.append(InnerRecItem.get_recall_rec_item(
                    spu_id, explain, score=rec_score
                ))
        return result
