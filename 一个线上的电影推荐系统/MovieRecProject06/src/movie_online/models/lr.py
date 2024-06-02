# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from . import BasePredictor
from .. import logger
from ..entity.spu_feature import SpuFeatureEntity
from ..entity.user_feature import UserFeatureEntity
from ..services.spu_feature_service import SpuFeatureService
from ..services.user_feature_service import UserFeatureService

COLUMNS = [
    'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
    'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
    'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western',
    'year', 'age', 'gender', 'occupation', 'location', 'movie_mean_rating',
    'movie_gender_mean_rating', 'user_mean_rating', 'action_mean_rating',
    'action_items', 'adventure_mean_rating', 'adventure_items',
    'animation_mean_rating', 'animation_items', 'children_mean_rating',
    'children_items', 'comedy_mean_rating', 'comedy_items',
    'crime_mean_rating', 'crime_items', 'documentary_mean_rating',
    'documentary_items', 'drama_mean_rating', 'drama_items',
    'fantasy_mean_rating', 'fantasy_items', 'film_noir_mean_rating',
    'film_noir_items', 'horror_mean_rating', 'horror_items',
    'musical_mean_rating', 'musical_items', 'mystery_mean_rating',
    'mystery_items', 'romance_mean_rating', 'romance_items',
    'sci_fi_mean_rating', 'sci_fi_items', 'thriller_mean_rating',
    'thriller_items', 'unknown_mean_rating', 'unknown_items',
    'war_mean_rating', 'war_items', 'western_mean_rating', 'western_items',
    'max_rating_genre', 'max_rete_items_genre'
]


class LrLocalPredictor(BasePredictor):
    def __init__(self, model_dir: str, model_version: str):
        super(LrLocalPredictor, self).__init__(model_dir, model_version)
        logger.info(f"开始加载恢复LR rank模型:{model_dir}")
        models = joblib.load(os.path.join(model_dir, "model.pkl"))
        self.onehot = models['onehot']
        self.lr = models['lr']
        if self.model_version is None:
            self.model_version = models['version']
        self.current_year = models['current_year']

    @staticmethod
    def _fetch_base_features(user: UserFeatureEntity, spus: Dict[int, SpuFeatureEntity]):
        # 1. 获取用户额外特征
        user_mean_rating = UserFeatureService.get_stat_mean_rating(user_id=user.id)
        user_movie_genre_rating = UserFeatureService.get_stat_movie_genre_mean_rating(user_id=user.id)

        # 2. 获取商品额外特征
        spu_ids = list(spus.keys())
        movie_mean_rating = SpuFeatureService.get_stat_mean_rating(spu_ids)
        movie_user_gender_mean_rating = SpuFeatureService.get_stat_user_gender_mean_rating(spu_ids, user.gender)

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
            data['year'] = str(data.get('release_date', ''))[-4:]
            if len(data['year']) == 0:
                data['year'] = '1995'
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

    def parse_dataset(self, xdf, onehot):
        """
        进行特征处理
        :param xdf: DataFrame对象
        :param onehot: onehot对象
        :return:
        """
        # 提取当前年份和电影上映年份间隔
        xdf.fillna({'year': self.current_year}, inplace=True)
        xdf['year'] = xdf.year.astype(np.int32)
        xdf['year'] = 1.0 * (self.current_year - xdf.year) / 100
        # 年龄做一个截断
        xdf['age'] = xdf.age.apply(lambda t: max(min(int(t), 80), 1))
        # 对部分列进行onehot编码
        _columns = ['age', 'gender', 'occupation', 'location', 'max_rating_genre', 'max_rete_items_genre']
        _onehot_columns = []
        for c in _columns:
            if c in xdf:
                _onehot_columns.append(c)
                xdf[c] = xdf[c].astype('str')
        xdf1 = onehot.transform(xdf[_onehot_columns])
        # 删除做过onehot的列
        for c in _columns:
            if c in xdf:
                del xdf[c]
        # 合并
        for c in xdf.columns:
            xdf[c] = xdf[c].astype(np.float32)
        xdf = np.hstack([xdf, xdf1])
        return xdf

    def internal_predict(self,
                         user: UserFeatureEntity, spus: Dict[int, SpuFeatureEntity]
                         ) -> Tuple[Dict[int, float], str]:
        # 1. 获取当前商品对应的特征属性
        df, spu_ids = self._fetch_base_features(user, spus)

        # 2. 调用模型获取预测结果
        xdf = self.parse_dataset(df, self.onehot)
        scores = self.lr.predict(xdf)
        scores = list(map(float, scores))

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
