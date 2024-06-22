# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseRecall
from ... import logger
from ...entity import InnerRecItem
from ...entity.user_feature import UserFeatureEntity
from ...models.model_service import ModelService
from ...vectors.vector_service import VectorService


class DSSMRecall(BaseRecall):
    def __init__(self):
        super(DSSMRecall, self).__init__('dssm')
        self.model_register_name = "dssm_user"

    def _fetch_user_vector(self,
                           user: UserFeatureEntity, model_version: Optional[str] = None
                           ) -> Tuple[List[float], str]:
        """
        基于给定的用户id，获取对应的特征向量信息
        :param user: 当前用户
        :param model_version: 模型版本字符串
        :return: 特征向量,模型版本
        """
        result = ModelService.fetch_predict_result(
            model_register_name=self.model_register_name,
            model_version=model_version,
            user=user
        )
        if result is None:
            return [], ""
        vectors = result[1]
        version = result[0]['version']
        return vectors, version

    @staticmethod
    def _fetch_similar_spu_ids(model_version: str, user_vector: List[float], k: int) -> List[int]:
        """
        基于给定的用户特征向量，获取对应最相似的商品id列表
        :param model_version: 模型版本字符串
        :param user_vector: 用户特征向量
        :param k: 获取最相似的k个值
        :return: 相似商品id列表
        """
        return VectorService.search(model_version, vector=user_vector, k=k)

    def get_candidates(self,
                       user: Optional[UserFeatureEntity], k: int,
                       model_version: Optional[str] = None, **kwargs
                       ) -> List[InnerRecItem]:
        """
        DSSM向量召回
        :param user: 当前用户
        :param k: 获取前k个匹配商品
        :param model_version: 模型版本字符串
        :param kwargs:
        :return:
        """
        if user is None or user.id is None:
            return []
        if k <= 0:
            return []
        # 1. 基于用户id获取用户特征向量
        user_vector, model_version = self._fetch_user_vector(user=user, model_version=model_version)
        if len(user_vector) == 0:
            logger.warn(f"当前用户的特征向量未获取得到:{self.model_register_name} - {user.id}")
            return []

        # 2. 调用faiss向量检索服务获取最相似的K个向量对应的商品id列表
        spu_ids = self._fetch_similar_spu_ids(model_version=model_version, user_vector=user_vector, k=k)

        # 3. 结果拼接处理
        result = []
        _source = f"{self.name}_{model_version}"
        _total_spu_nums = 1.0 * len(spu_ids)
        for _idx, _spu_id in enumerate(spu_ids):
            _score = np.exp((_total_spu_nums - _idx) / _total_spu_nums)  # 基于顺序位置计算一个权重，随着位置越往后，权重越小
            result.append(InnerRecItem.get_recall_rec_item(_spu_id, f"{_source}_{_score:.3f}", _score, _source))
        return result
