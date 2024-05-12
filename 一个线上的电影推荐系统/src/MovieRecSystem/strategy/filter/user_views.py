from typing import Optional, List

from ...entity.user_feature import UserFeatureEntity
from ...services.user_feature import UserFeatureService


class UserViewsFilter(object):
    def __init__(self, start: int = 0, end: int = -1):
        self.start = start
        self.end = end

    def get_user_view_spu_ids(self, user: Optional[UserFeatureEntity]) -> List[int]:
        """
        提取用户最近浏览过的商品id列表
        :param user: 用户对象
        :return: 用户商品id列表
        """
        if user is None or user.id is None:
            return []
        # 提取10个最近浏览商品id列表
        return UserFeatureService.get_user_view_ids(user_id=user.id, start=self.start, end=self.end)
