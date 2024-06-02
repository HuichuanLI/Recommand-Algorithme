# -*- coding: utf-8 -*-
from typing import List, Optional

from .base import BaseFilter
from ...entity.user_feature import UserFeatureEntity
from ...services.user_feature_service import UserFeatureService


class UserViewsFilter(BaseFilter):
    def __init__(self):
        super(UserViewsFilter, self).__init__("user_views_filter")

    def get_spu_ids(self,
                    k: Optional[int] = None,
                    user: Optional[UserFeatureEntity] = None,
                    **kwargs
                    ) -> List[int]:
        if user is None or user.id is None:
            return []
        if k == 0:
            return []
        # 1. 从redis中获取当前用户最近浏览的商品id列表
        view_spu_ids = UserFeatureService.get_view_spu_ids(user.id)
        # 2. 截断
        if k > 0:
            view_spu_ids = view_spu_ids[:k]
        return view_spu_ids
