import json
from typing import Optional, List

from .spu_feature import SpuFeatureEntity
from .user_feature import UserFeatureEntity


class ConfigParams(object):
    def __init__(self,
                 number_push=5,
                 user: Optional[UserFeatureEntity] = None, spu: Optional[SpuFeatureEntity] = None,
                 **kwargs
                 ):
        # 推荐需要的商品数量
        self.number_push = number_push

        # ============当前用户特征===================================================
        self.user: Optional[UserFeatureEntity] = user
        self.location_id: Optional[int] = kwargs.get('location_id')
        self.user_id = None if self.user is None else self.user.id

        # ============当前物品特征===================================================
        self.spu: Optional[SpuFeatureEntity] = spu
        self.spu_id = None if self.spu is None else self.spu.id
        self.category_ids: List[str] = []
        self.sim_category_ids: List[str] = []

        # ============相关参数初始化=================================================
        if user is not None:
            self.location_id = int(user.location_id)
        if spu is not None:
            self.category_ids = spu.categorys
            self.sim_category_ids = spu.sim_categorys

    def to_dict(self):
        return {
            'location_id': self.location_id
        }

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)
