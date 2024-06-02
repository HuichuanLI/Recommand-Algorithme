# -*- coding: utf-8 -*-
from typing import List, Optional, Dict

from ...entity import InnerRecItem
from ...entity.spu_feature import SpuFeatureEntity
from ...entity.user_feature import UserFeatureEntity
from ...models.model_service import ModelService
from ...services.spu_feature_service import SpuFeatureService


class BaseRank(object):
    def __init__(self, name: str):
        """
        构造方法
        :param name: 当前策略名称，全局唯一
        """
        self.name = name

    def rank_items(self, items: List[InnerRecItem], **kwargs) -> List[InnerRecItem]:
        raise NotImplementedError("当前子类未实现具体的排序方法!")


class ModelBaseRank(BaseRank):
    def __init__(self, name: str, model_register_name: str):
        super(ModelBaseRank, self).__init__(name)
        self.model_register_name = model_register_name

    def rank_items(self,
                   items: List[InnerRecItem],
                   user: Optional[UserFeatureEntity] = None,
                   spus: Optional[Dict[int, SpuFeatureEntity]] = None,
                   **kwargs) -> List[InnerRecItem]:
        # 1. 入参判断
        if items is None or len(items) == 0:
            return []
        if user is None:
            return []
        if spus is None or len(spus) != len(items):
            spus = SpuFeatureService.get_effect_spu_features(
                spu_ids=[item.spu_id for item in items]
            )

        # 2. 获取每个商品对应的预测评分
        result = ModelService.fetch_predict_result(
            model_register_name=self.model_register_name,
            user=user, spus=spus
        )
        if result is None:
            return items
        version = result[0]['version']
        spu_id2score = result[1]

        # 3. 进行数据的处理
        item_dict: Dict[int, InnerRecItem] = {v.spu_id: v for v in items}
        for spu_id, item in item_dict.items():
            spu_score = spu_id2score.get(spu_id, -1.0)
            item.add_rank(f'{self.name}_{version}_{spu_score:.3f}', self.name)
            item.score = spu_score

        # 4. 数据排序返回
        items = list(item_dict.values())
        items.sort(key=lambda t: t.score, reverse=True)
        return items
