from typing import Optional, List

from ...entity.user_feature import UserFeatureEntity
from ...services.user_feature import UserFeatureService


class BlacklistFilter(object):
    def __init__(self):
        pass

    # noinspection PyMethodMayBeStatic
    def get_user_blacklist_spu_ids(self, user: Optional[UserFeatureEntity]) -> List[int]:
        """
        获取当前用户设定的黑名单商品列表
        :param user: 用户对象
        :return: 商品id列表
        """
        if user is None or user.id is None:
            return []
        # 提取10个最近浏览商品id列表
        return UserFeatureService.get_user_blacklist_ids(user_id=user.id)
