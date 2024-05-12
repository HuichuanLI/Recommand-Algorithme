import heapq

import numpy as np
import surprise
from surprise import Trainset
from surprise.model_selection import GridSearchCV
from tqdm import tqdm

from MovieRecOffline.utils.redis_util import RedisUtil


def get_dataset(path):
    # 1. 加载数据
    reader = surprise.Reader(
        name=None,
        line_format='user item rating timestamp',
        sep='\t',
        rating_scale=(1, 5),
        skip_lines=0
    )
    dateset = surprise.Dataset.load_from_file(path, reader)
    return dateset


def search_best_params(path):
    """
    网格交叉验证的方式来选择模型参数
    :param path: 数据路径
    :return:
    """
    # 1. 加载数据
    dataset = get_dataset(path)

    grid = {
        'n_factors': [4, 8, 10, 20],
        'n_epochs': [30, 40, 50],
        'biased': [True],
        'lr_all': [0.005],
        'reg_all': [0.02]
    }
    model = GridSearchCV(
        algo_class=surprise.SVD,
        param_grid=grid,
        measures=['rmse', 'mae', 'fcp'],
        cv=5,
        n_jobs=5,
        joblib_verbose=True
    )
    print(f"总实验参数对:{len(model.param_combinations)}")
    model.fit(dataset)

    print(model.best_score)
    print(model.best_params)


def cosine_func(x, y):
    a = 0.0
    b = 0.0
    c = 0.0
    for v1, v2 in zip(x, y):
        a += v1 * v2
        b += v1 ** 2
        c += v2 ** 2
    return a / np.sqrt(b * c)


def training(path, k=100, sim_func=cosine_func, min_sim=0.0):
    # 1. 数据加载
    dateset = get_dataset(path)
    trainset: Trainset = dateset.build_full_trainset()

    # 2. 训练ItemCF
    model = surprise.SVD(
        n_factors=10,
        n_epochs=30,
        biased=True,
        lr_all=0.005,
        reg_all=0.02
    )
    model.fit(trainset)

    # 3. 提取物品-隐因子矩阵
    p = model.qi  # [1682, 10]

    # 4. 两两计算相似度并保存
    # noinspection PyProtectedMember
    with RedisUtil._get_redis() as client:
        values = []

        def save_to_redis():
            if len(values) == 0:
                return
            _pipe = client.pipeline()
            for t in values:
                _pipe.hset(t[0], 'mf', t[1])
                _pipe.expire(t[0], 7 * 24 * 60 * 60)
            _pipe.execute()

        # 遍历所有数据
        n_items = trainset.n_items
        for iiid in tqdm(range(n_items)):
            iid_sim = []
            # 实际物品id
            iid = trainset.to_raw_iid(iiid)
            # 遍历其它物品计算相似度
            for iiid2 in range(n_items):
                if iiid2 == iiid:
                    continue
                iid2 = trainset.to_raw_iid(iiid2)
                sim = sim_func(p[iiid], p[iiid2])
                if sim <= min_sim:
                    continue
                iid_sim.append((iid2, sim))
            # 提取前k个最相似度的物品及相似度
            top_k_iid_sim = heapq.nlargest(k, iid_sim, key=lambda t: t[1])
            if len(top_k_iid_sim) == 0:
                continue
            # 转换为保存到redis的key\value值
            key = f"rec:recall:i2i:{iid}"
            value = ",".join([f"{t[0]}:{t[1]:.2f}" for t in top_k_iid_sim])
            values.append([key, value])
            # 检查是否需要保存redis
            if len(values) > 100:
                save_to_redis()
                values = []
        # 最终保存
        save_to_redis()


if __name__ == '__main__':
    training(
        r'C:\Users\gerry_17578261252713\PycharmProjects\MovieRecSystem\data\u.data',
        min_sim=0.5
    )
