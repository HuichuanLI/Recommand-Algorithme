# -*- coding: utf-8 -*-
from typing import List, Optional

from .base import BaseFilter
from ...entity.user_feature import UserFeatureEntity
from ...services.user_feature_service import UserFeatureService


class UserBlacklistFilter(BaseFilter):
    def __init__(self):
        super(UserBlacklistFilter, self).__init__('user_blacklist_filter')

    def get_spu_ids(self, user: Optional[UserFeatureEntity] = None, **kwargs) -> List[int]:
        if user is None or user.id is None:
            return []
        # 1. 获取当前用户对应的商品黑名单列表
        return UserFeatureService.get_blacklist_spu_ids(user.id)
