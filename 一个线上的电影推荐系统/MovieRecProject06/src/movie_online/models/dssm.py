# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch

from . import BasePredictor
from .. import logger
from ..entity.spu_feature import SpuFeatureEntity
from ..entity.user_feature import UserFeatureEntity
from ..services.spu_feature_service import SpuFeatureService
from ..services.user_feature_service import UserFeatureService
from ..utils import SimpleMapping

# noinspection DuplicatedCode
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
def fetch_dssm_user_base_features(user: UserFeatureEntity) -> pd.DataFrame:
    # 1. 从数据库提取数据
    user_mean_rating = UserFeatureService.get_stat_mean_rating(user_id=user.id)
    # 1.2 从数据库获取每个用户每个商品类别的平均评分
    user_movie_genre_rating = UserFeatureService.get_stat_movie_genre_mean_rating(user_id=user.id)

    # 2. 合并构造数据
    data = {}
    data.update(user.record)
    data['location'] = data['zip_code'][:2]
    data.update(user_mean_rating)
    data.update(user_movie_genre_rating)
    for c in USER_COLUMNS:
        if c not in data:
            data[c] = 0
    df = pd.DataFrame([data])
    df = df[USER_COLUMNS].copy()
    return df


# noinspection DuplicatedCode
def fetch_dssm_spu_base_features(spu: SpuFeatureEntity) -> pd.DataFrame:
    # 1.1 提取所有物品对应的平均评分
    movie_mean_rating = SpuFeatureService.get_stat_mean_rating([spu.id])
    # 1.4 提取所有物品对应各个用户性别的平均评分
    movie_user_gender_mean_rating = SpuFeatureService.get_stat_user_gender_mean_rating_v2([spu.id])

    # 2. 合并构造数据
    data = {}
    data.update(spu.record)
    data['movie_id'] = data['id']
    data.update(movie_mean_rating.get(spu.id, {}))
    data.update(movie_user_gender_mean_rating.get(spu.id, {}))
    for c in SPU_COLUMNS:
        if c not in data:
            data[c] = 0
    df = pd.DataFrame([data])
    df = df[SPU_COLUMNS].copy()
    return df


# noinspection DuplicatedCode
class DSSMUserSideLocalPredictor(BasePredictor):
    def __init__(self, model_dir, model_version):
        super(DSSMUserSideLocalPredictor, self).__init__(model_dir, model_version)
        logger.info(f"开始创建DSSM本地user侧向量子模型:{model_dir}")

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
            os.path.join(model_dir, "user_model.pt"),
            map_location='cpu', _extra_files=extra_files
        )
        self.net.eval().cpu()
        if self.model_version is None:
            self.model_version = extra_files['model_version']
            if not isinstance(self.model_version, str):
                self.model_version = str(self.model_version, encoding='utf-8')

    def _parse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['user_id'] = df.user_id.apply(lambda t: self.user_id_mapping.get(t))
        df['age'] = df.age.apply(lambda t: self.age_mapping.get(t))
        df['gender'] = df.gender.apply(lambda t: self.gender_mapping.get(t))
        df['occupation'] = df.occupation.apply(lambda t: self.occupation_mapping.get(t))
        df['location'] = df.location.apply(lambda t: self.location_mapping.get(t))
        df['max_rating_genre'] = df.max_rating_genre.apply(lambda t: self.movie_genre_mapping.get(t))
        df['max_rete_items_genre'] = df.max_rete_items_genre.apply(lambda t: self.movie_genre_mapping.get(t))
        return df

    @torch.no_grad()
    def internal_predict(self, user: UserFeatureEntity) -> Tuple[List[float], str]:
        # 1. 获取当前用户对应的特征属性
        df: pd.DataFrame = fetch_dssm_user_base_features(user)

        # 2. 调用模型获取预测结果
        df: pd.DataFrame = self._parse_features(df)
        user_sparse = np.asarray(df[USER_SPARSE_COLUMNS], dtype=np.int64)
        user_dense = np.asarray(df[USER_DENSE_COLUMNS], dtype=np.float32)
        vectors = self.net(torch.from_numpy(user_sparse), torch.from_numpy(user_dense))  # [1,?]

        # 3. 结果处理并返回
        return list(map(float, vectors.numpy()[0])), self.model_version

    def predict(self, **kwargs) -> Optional[Tuple[Dict[str, Any], Any]]:
        user = kwargs.get('user')
        if user is None:
            logger.warn("调用DSSM用户侧向量子模型必须传入UserFeatureEntity类型参数:user，当前为空。")
            return None
        if not isinstance(user, UserFeatureEntity):
            logger.warn("调用DSSM用户侧向量子模型必须传入UserFeatureEntity类型参数:user，当前类型不对。")
            return None
        vectors, version = self.internal_predict(user)
        infos = {
            'version': version
        }
        return infos, vectors


# noinspection DuplicatedCode
class DSSMSpuSideLocalPredictor(BasePredictor):
    def __init__(self, model_dir, model_version):
        super(DSSMSpuSideLocalPredictor, self).__init__(model_dir, model_version)
        logger.info(f"开始创建DSSM本地spu侧向量子模型:{model_dir}")

        self.movie_id_mapping = SimpleMapping(os.path.join(model_dir, "dict", "movie_id.dict"))

        extra_files = {
            'model_version': ''
        }
        self.net = torch.jit.load(
            os.path.join(model_dir, "spu_model.pt"),
            map_location='cpu', _extra_files=extra_files
        )
        self.net.eval().cpu()
        if self.model_version is None:
            self.model_version = extra_files['model_version']
            if not isinstance(self.model_version, str):
                self.model_version = str(self.model_version, encoding='utf-8')

    def _parse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['movie_id'] = df.movie_id.apply(lambda t: self.movie_id_mapping.get(t))
        return df

    @torch.no_grad()
    def internal_predict(self, spu: SpuFeatureEntity) -> Tuple[List[float], str]:
        # 1. 获取当前商品对应的特征属性
        df: pd.DataFrame = fetch_dssm_spu_base_features(spu)

        # 2. 调用模型获取预测结果
        df: pd.DataFrame = self._parse_features(df)
        spu_sparse = np.asarray(df[SPU_SPARSE_COLUMNS], dtype=np.int64)
        spu_dense = np.asarray(df[SPU_DENSE_COLUMNS], dtype=np.float32)
        vectors = self.net(torch.from_numpy(spu_sparse), torch.from_numpy(spu_dense))  # [1,?]

        # 3. 结果处理并返回
        return list(map(float, vectors.numpy()[0])), self.model_version

    def predict(self, **kwargs) -> Optional[Tuple[Dict[str, Any], Any]]:
        spu = kwargs.get('spu')
        if spu is None:
            logger.warn("调用DSSM商品侧向量子模型必须传入SpuFeatureEntity类型参数:spu，当前为空。")
            return None
        if not isinstance(spu, SpuFeatureEntity):
            logger.warn("调用FM商品侧向量子模型必须传入SpuFeatureEntity类型参数:spu，当前类型不对。")
            return None
        vectors, version = self.internal_predict(spu)
        infos = {
            'version': version
        }
        return infos, vectors
