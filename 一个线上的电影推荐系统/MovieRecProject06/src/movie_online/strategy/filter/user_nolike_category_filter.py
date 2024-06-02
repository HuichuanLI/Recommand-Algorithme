# -*- coding: utf-8 -*-
from typing import List, Optional, Dict

from .base import BaseFilter
from ...entity.spu_feature import SpuFeatureEntity
from ...entity.user_feature import UserFeatureEntity
from ...services.user_feature_service import UserFeatureService


class UserNoLikeCategoryFilter(BaseFilter):
    def __init__(self):
        super(UserNoLikeCategoryFilter, self).__init__('user_nolike_category_filter')

    def get_spu_ids(self,
                    spus: Dict[int, SpuFeatureEntity],
                    user: Optional[UserFeatureEntity] = None, **kwargs
                    ) -> List[int]:
        if user is None or user.id is None:
            return []
        # 1. 获取当前用户不喜欢的商品类别名称列表
        nolike_category_names = UserFeatureService.get_no_like_category_names(user_id=user.id)
        if len(nolike_category_names) == 0:
            return []
        # 2. 遍历所有候选商品，判断商品的类别是否在用户不喜欢的类别名称列表中，如果在，那么当前商品id需要返回
        spu_ids = []
        for spu_id, spu in spus.items():
            for spu_category in spu.categorys:
                if spu_category in nolike_category_names:
                    spu_ids.append(spu_id)
                    break
        return spu_ids
