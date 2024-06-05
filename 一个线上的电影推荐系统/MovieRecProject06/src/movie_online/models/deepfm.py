# -*- coding: utf-8 -*-
import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

from . import BasePredictor
from .pai_caller import PaiCaller
from .. import logger
from ..entity.spu_feature import SpuFeatureEntity
from ..entity.user_feature import UserFeatureEntity
from ..services.spu_feature_service import SpuFeatureService
from ..services.user_feature_service import UserFeatureService
from ..utils import SimpleMapping

USER_SPARSE_COLUMNS = [
    'user_id', 'age', 'gender', 'occupation', 'location', 'max_rating_genre', 'max_rete_items_genre'
]
USER_DENSE_COLUMNS = [
    'action_mean_rating', 'adventure_mean_rating', 'animation_mean_rating', 'children_mean_rating',
    'comedy_mean_rating', 'crime_mean_rating', 'documentary_mean_rating', 'drama_mean_rating',
    'fantasy_mean_rating', 'film_noir_mean_rating', 'horror_mean_rating', 'musical_mean_rating',
    'mystery_mean_rating', 'romance_mean_rating', 'sci_fi_mean_rating', 'thriller_mean_rating',
    'unknown_mean_rating', 'war_mean_rating', 'western_mean_rating', 'user_mean_rating'
]
USER_COLUMNS = USER_SPARSE_COLUMNS + USER_DENSE_COLUMNS

SPU_SPARSE_COLUMNS = [
    'movie_id',
    'unknown', 'action', 'adventure', 'animation',
    'children', 'comedy', 'crime', 'documentary',
    'drama', 'fantasy', 'film_noir', 'horror',
    'musical', 'mystery', 'romance', 'sci_fi',
    'thriller', 'war', 'western'
]
SPU_DENSE_COLUMNS = [
    'movie_mean_rating', 'm_mean_rating', 'f_mean_rating'
]
SPU_COLUMNS = SPU_SPARSE_COLUMNS + SPU_DENSE_COLUMNS

COLUMNS = USER_COLUMNS + SPU_COLUMNS


# noinspection DuplicatedCode
class DeepFMLocalPredictor(BasePredictor):
    def __init__(self, model_dir, model_version):
        super(DeepFMLocalPredictor, self).__init__(model_dir, model_version)
        logger.info(f"开始加载恢复DeepFM rank模型:{model_dir}")

        self.user_id_mapping = SimpleMapping(os.path.join(model_dir, "dict", "user_id.dict"))
        self.age_mapping = SimpleMapping(os.path.join(model_dir, "dict", "age.dict"))
        self.gender_mapping = SimpleMapping(os.path.join(model_dir, "dict", "gender.dict"))
        self.occupation_mapping = SimpleMapping(os.path.join(model_dir, "dict", "occupation.dict"))
        self.location_mapping = SimpleMapping(os.path.join(model_dir, "dict", "location.dict"))
        self.movie_genre_mapping = SimpleMapping(os.path.join(model_dir, "dict", "movie_genre.dict"))
        self.movie_id_mapping = SimpleMapping(os.path.join(model_dir, "dict", "movie_id.dict"))

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

    @staticmethod
    def _fetch_base_features(user: UserFeatureEntity, spus: Dict[int, SpuFeatureEntity]):
        # 1. 获取用户额外特征
        user_mean_rating = UserFeatureService.get_stat_mean_rating(user_id=user.id)
        user_movie_genre_rating = UserFeatureService.get_stat_movie_genre_mean_rating(user_id=user.id)

        # 2. 获取商品额外特征
        spu_ids = list(spus.keys())
        movie_mean_rating = SpuFeatureService.get_stat_mean_rating(spu_ids)
        movie_user_gender_mean_rating = SpuFeatureService.get_stat_user_gender_mean_rating_v2(spu_ids)

        # 2. 合并构造数据
        data_list = []
        spu_ids = []
        for spu_id, spu in spus.items():
            data = {}
            # 合并用户特征
            data.update(user.record)
            data['location'] = data['zip_code'][:2]
            data.update(user_mean_rating)
            data.update(user_movie_genre_rating)
            # 合并当前商品特征
            data.update(spu.record)
            data['movie_id'] = data['id']
            data.update(movie_mean_rating.get(spu_id, {}))
            data.update(movie_user_gender_mean_rating.get(spu_id, {}))

            # 填充默认值
            for c in COLUMNS:
                if c not in data:
                    data[c] = 0
            data_list.append(data)
            spu_ids.append(spu_id)

        # 3. 构造DataFrame对象
        df = pd.DataFrame(data_list)
        df = df[COLUMNS].copy()

        # 3. 针对某些类型的，需要这种为str类型

        return df, spu_ids

    def _parse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['user_id'] = df.user_id.apply(lambda t: self.user_id_mapping.get(t))
        df['age'] = df.age.apply(lambda t: self.age_mapping.get(t))
        df['gender'] = df.gender.apply(lambda t: self.gender_mapping.get(t))
        df['occupation'] = df.occupation.apply(lambda t: self.occupation_mapping.get(t))
        df['location'] = df.location.apply(lambda t: self.location_mapping.get(t))
        df['max_rating_genre'] = df.max_rating_genre.apply(lambda t: self.movie_genre_mapping.get(t))
        df['max_rete_items_genre'] = df.max_rete_items_genre.apply(lambda t: self.movie_genre_mapping.get(t))
        df['movie_id'] = df.movie_id.apply(lambda t: self.movie_id_mapping.get(t))
        return df

    @torch.no_grad()
    def internal_predict(self,
                         user: UserFeatureEntity, spus: Dict[int, SpuFeatureEntity]
                         ) -> Tuple[Dict[int, float], str]:
        # 1. 获取当前商品对应的特征属性 --> 当前模型需要什么原始特征属性就获取什么原始特征属性
        df, spu_ids = self._fetch_base_features(user, spus)

        # 2. 特征处理转换
        df: pd.DataFrame = self._parse_features(df)

        # 3. 调用模型获取预测结果
        spu_sparse = np.asarray(df[SPU_SPARSE_COLUMNS], dtype=np.int64)
        spu_dense = np.asarray(df[SPU_DENSE_COLUMNS], dtype=np.float32)
        user_sparse = np.asarray(df[USER_SPARSE_COLUMNS], dtype=np.int64)
        user_dense = np.asarray(df[USER_DENSE_COLUMNS], dtype=np.float32)
        scores = self.net(
            torch.from_numpy(user_sparse), torch.from_numpy(user_dense),
            torch.from_numpy(spu_sparse), torch.from_numpy(spu_dense)
        )  # [?]
        scores = torch.sigmoid(scores)  # 针对输出的置信度做了一个sigmoid概率转换
        scores = list(map(float, scores.numpy()))

        # 3. 结果处理并返回
        return dict(zip(spu_ids, scores)), self.model_version

    def predict(self, **kwargs) -> Optional[Tuple[Dict[str, Any], Any]]:
        # 获取入参
        user = kwargs.get('user')
        spus = kwargs.get('spus')
        if (user is None) or (not isinstance(user, UserFeatureEntity)):
            logger.warn(f"调用FM排序模型必须传入UserFeatureEntity类型参数:user，当前为:{user}")
            return None
        if (spus is None) or (not isinstance(spus, dict)) or (len(spus) == 0):
            logger.warn(f"调用FM排序模型必须传入SpuFeatureEntity类型参数:spus，当前为:{spus}")
            return None
        # 调用模型
        scores, version = self.internal_predict(user, spus)
        infos = {
            'version': version
        }
        return infos, scores
