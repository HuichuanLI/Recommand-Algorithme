# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from surprise import accuracy
from surprise.prediction_algorithms.predictions import Prediction
from tqdm import tqdm

# region 模型定义
GBDT_LR_COLUMNS = [
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


class GBDTLrModel(object):
    def __init__(self, model_dir):
        super(GBDTLrModel, self).__init__()
        self.current_year = 2024
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    @staticmethod
    def get_dataset(data_path):
        df = pd.read_csv(data_path, low_memory=False)
        # 我把不用的列删除，减少内存占用
        _del_columns = [
            'timestamp', 'title', 'release_date', 'actors', 'zip_code', 'viewable'
        ]
        for c in _del_columns:
            if c in df:
                del df[c]
        return df

    def parse_dataset(self, xdf):
        # 提取当前年份和电影上映年份间隔
        xdf.fillna({'year': self.current_year}, inplace=True)
        xdf['year'] = xdf.year.astype(np.int32)
        xdf['year'] = 1.0 * (self.current_year - xdf.year) / 100
        # 年龄做一个截断
        xdf['age'] = xdf.age.apply(lambda t: max(min(int(t), 80), 1))
        # 对部分列进行标签化处理 --> 转换成从0开始的数字
        label_encoders = {}
        _columns = ['gender', 'occupation', 'location', 'max_rating_genre', 'max_rete_items_genre']
        for c in xdf.columns:
            if c in _columns:
                label_encoder = preprocessing.LabelEncoder()
                label_encoders[c] = label_encoder
                xdf[c] = label_encoder.fit_transform(xdf[c])
            else:
                xdf[c] = xdf[c].astype(np.float32)

        return xdf, label_encoders

    def training(self, data_path):
        # 1. 加载数据集
        df = self.get_dataset(data_path)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        y_df = df['rating']
        x_df = df[GBDT_LR_COLUMNS].copy()

        # 2. 特征工程
        onehot = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
        x_df, label_encoders = self.parse_dataset(x_df)

        # 3. 模型训练
        y = y_df.astype(np.int32)
        x = x_df

        gbdt = GradientBoostingClassifier(n_estimators=10, max_depth=3)
        gbdt.fit(x, y)
        x = gbdt.apply(x)  # 得到叶子节点位置信息
        x = np.reshape(x, (x.shape[0], -1))

        onehot = preprocessing.OneHotEncoder(handle_unknown='ignore')
        x = onehot.fit_transform(x)

        lr = Ridge()  # 做一个回归模型
        # lr = LogisticRegression() # 正常做CTR或者CVR应该是一个分类模型
        lr.fit(x, y)
        y_ = lr.predict(x)

        # 4. 评估
        y2_ = []
        predictions = []
        for idx, (tr, pr) in tqdm(enumerate(zip(y, y_)), total=len(y)):
            idf = df.loc[idx]
            predictions.append(Prediction(uid=idf.user_id, iid=idf.movie_id, r_ui=tr, est=pr, details=""))
            if pr < 1.5:
                pr = 1
            elif pr < 2.5:
                pr = 2
            elif pr < 3.5:
                pr = 3
            elif pr < 4.5:
                pr = 4
            else:
                pr = 5
            y2_.append(pr)
        print(f"MSE:{sklearn.metrics.mean_squared_error(y, y_)}")
        print(f"准确率:{sklearn.metrics.accuracy_score(y, y2_)}")
        print(sklearn.metrics.confusion_matrix(y2_, y))
        print(f"fcp:{accuracy.fcp(predictions)}")
        print(f"mae:{accuracy.mae(predictions)}")
        print(f"mse:{accuracy.mse(predictions)}")
        print(f"rmse:{accuracy.rmse(predictions)}")

        # 模型保存
        now = datetime.now()
        model_version = f'gbdt_lr_{now.strftime("%Y%m%d_%H%M%S")}'
        joblib.dump(
            {
                'onehot': onehot,
                'lr': lr,
                'gbdt': gbdt,
                'labels': label_encoders,
                'version': model_version,
                'current_year': self.current_year
            },
            os.path.join(self.model_dir, "model.pkl")
        )
        extra_files = {
            "model_version": model_version,
            'current_year': self.current_year
        }
        json.dump(dict(extra_files), open(os.path.join(self.model_dir, 'info.json'), 'w', encoding='utf-8'))


# endregion

# region 模型训练

def training(root_dir, output_dir):
    model = GBDTLrModel(model_dir=output_dir)

    model.training(data_path=os.path.join(root_dir, "feature_lr.csv"))


# endregion

# region 模型上传

def upload(model_dir):
    """
    将本地文件夹中的内容上传到服务器上
    :param model_dir: 本地待上传的文件夹路径
    :return:
    """
    base_url = "http://127.0.0.1:5051"
    # base_url = "http://121.40.96.93:9999"
    name = 'gbdt_lr'  # 当前必须为gbdt_lr
    sess = requests.session()

    # 1. version信息恢复
    extra_files = json.load(open(os.path.join(model_dir, 'info.json'), 'r', encoding='utf-8'))
    model_version = extra_files['model_version']

    # 删除文件夹
    sess.get(f"{base_url}/deleter", params={"version": model_version, "name": name})

    # 2. 上传文件
    def upload_file(_f, pname=None, fname=None, sub_dir_names=None):
        data = {
            "version": model_version,
            "name": pname or name
        }
        if fname is not None:
            data['filename'] = fname
        if sub_dir_names is not None:
            data['sub_dir_names'] = sub_dir_names
        res1 = sess.post(
            url=f"{base_url}/uploader",
            data=data,
            files={
                "file": open(_f, 'rb')
            }
        )
        if res1.status_code == 200:
            _data = res1.json()
            if _data['code'] != 200:
                raise ValueError(f"上传文件失败，异常信息为:{_data['msg']}")
            else:
                print(f"上传成功，version:'{_data['version']}'，filename:'{_data['filename']}'")
        else:
            raise ValueError("网络异常!")

    def upload(_f, pname=None, fname=None, sub_dir_names=None):
        if os.path.isfile(_f):
            upload_file(_f, pname, fname, sub_dir_names)
        else:
            cur_dir_name = os.path.basename(_f)
            fname = fname or cur_dir_name  # 选择外部给定的名称，或者当前自身的名称
            if sub_dir_names is None:
                sub_dir_names = f"{fname}"
            else:
                sub_dir_names = f"{sub_dir_names},{fname}"
            # 子文件的处理
            for _name in os.listdir(_f):
                upload(
                    _f=os.path.join(_f, _name),
                    pname=pname,
                    fname=None,  # 子文件无法重命名
                    sub_dir_names=sub_dir_names
                )

    upload(_f=os.path.join(model_dir, "model.pkl"))
    upload(_f=os.path.join(model_dir, "info.json"))

# endregion
