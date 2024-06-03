# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple, Dict, Any

import torch

from . import BasePredictor
from .. import logger
from ..entity.spu_feature import SpuFeatureEntity
from ..entity.user_feature import UserFeatureEntity
from ..utils import SimpleMapping


class BPRLocalPredictor(BasePredictor):
    def __init__(self, model_dir: str, model_version: str):
        super(BPRLocalPredictor, self).__init__(model_dir, model_version)
        logger.info(f"开始加载恢复BPR rank模型:{model_dir}")

        self.user_id_mapping = SimpleMapping(os.path.join(model_dir, "bpr_dict", "user_id.dict"))
        self.spu_id_mapping = SimpleMapping(os.path.join(model_dir, "bpr_dict", "spu_id.dict"))

        extra_files = {
            'model_version': ''
        }
        self.net = torch.jit.load(
            os.path.join(model_dir, "model.pt"),
            map_location='cpu', _extra_files=extra_files
        )
        self.net.eval().cpu()
        if self.model_version is None:
            self.model_version = extra_files['model_version']
            if not isinstance(self.model_version, str):
                self.model_version = str(self.model_version, encoding='utf-8')

    def _fetch_base_features(self, user: UserFeatureEntity, spus: Dict[int, SpuFeatureEntity]):
        inner_user_id_list = []
        inner_spu_id_list = []
        spu_id_list = []
        miss_spu_id_list = []

        if self.user_id_mapping.in_mapping(user.id):
            for spu_id, _ in spus.items():
                if self.spu_id_mapping.in_mapping(spu_id):
                    spu_id_list.append(spu_id)
                    inner_spu_id_list.append(self.spu_id_mapping.get(spu_id))  # 传入模型的内部商品id
                    inner_user_id_list.append(self.user_id_mapping.get(user.id))  # 传入模型的内部用户id
                else:
                    miss_spu_id_list.append(spu_id)
        else:
            # 当前用户不在模型用户列表中 --> 所有商品均无法进行预测
            for spu_id, _ in spus.items():
                miss_spu_id_list.append(spu_id)

        return inner_user_id_list, inner_spu_id_list, spu_id_list, miss_spu_id_list

    @torch.no_grad()
    def internal_predict(self,
                         user: UserFeatureEntity, spus: Dict[int, SpuFeatureEntity]
                         ) -> Tuple[Dict[int, float], str]:
        # 1. 获取当前商品对应的特征属性
        inner_user_id_list, inner_spu_id_list, spu_id_list, miss_spu_id_list = self._fetch_base_features(user, spus)

        # 2. 调用模型获取预测结果
        result = {}
        if len(inner_user_id_list) > 0:
            scores = self.net(
                torch.tensor(inner_user_id_list, dtype=torch.long),
                torch.tensor(inner_spu_id_list, dtype=torch.long)
            )  # [?]
            scores = torch.sigmoid(scores)  # 希望所有的评分都是正数
            scores = list(map(float, scores.numpy()))
            result = dict(zip(spu_id_list, scores))

        # 针对不在模型支持的商品id列表直接填充0
        if len(miss_spu_id_list) > 0:
            for miss_spu_id in miss_spu_id_list:
                result[miss_spu_id] = 0.0

        # 3. 结果处理并返回
        return result, self.model_version

    def predict(self, **kwargs) -> Optional[Tuple[Dict[str, Any], Any]]:
        # 获取入参
        user = kwargs.get('user')
        spus = kwargs.get('spus')
        if (user is None) or (not isinstance(user, UserFeatureEntity)):
            logger.warn(f"调用BPR排序模型必须传入UserFeatureEntity类型参数:user，当前为:{user}")
            return None
        if (spus is None) or (not isinstance(spus, dict)) or (len(spus) == 0):
            logger.warn(f"调用BPR排序模型必须传入SpuFeatureEntity类型参数:spus，当前为:{spus}")
            return None
        # 调用模型
        scores, version = self.internal_predict(user, spus)
        infos = {
            'version': version
        }
        return infos, scores
