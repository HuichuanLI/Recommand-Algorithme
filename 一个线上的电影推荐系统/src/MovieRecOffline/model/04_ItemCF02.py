import heapq

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
        'k': [40],
        'min_k': [1, 3],
        'bsl_options': {
            'method': ['sgd'],
            'n_epochs': [10, 20]
        },
        'sim_options': {
            'name': ['msd', 'cosine'],
            'user_based': [False]
        }
    }
    model = GridSearchCV(
        algo_class=surprise.KNNBaseline,
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


def training(path, k=100):
    # 1. 数据加载
    dateset = get_dataset(path)
    trainset: Trainset = dateset.build_full_trainset()

    # 2. 训练ItemCF
    model = surprise.KNNBaseline(
        k=40,
        min_k=3,
        sim_options={
            'name': 'cosine',
            'user_based': False,
            'min_support': 10  # 要求两个物品同时有十个用户评分才计算
        },
        bsl_options={
            'method': 'sgd',
            'reg': 0.02,
            'learning_rate': 0.005,
            'n_epochs': 20
        }
    )
    model.fit(trainset)

    # 3. 针对每个用户产生推荐列表(前k个，要求这k个商品，之前用户没有访问过)
    # 3. 提取物品相似度矩阵，计算每个物品的最相似的k个物品作为推荐结果
    # noinspection PyProtectedMember
    with RedisUtil._get_redis() as client:
        values = []

        def save_to_redis():
            if len(values) == 0:
                return
            _pipe = client.pipeline()
            for t in values:
                _pipe.hset(t[0], 'item_cf', t[1])
                _pipe.expire(t[0], 7 * 24 * 60 * 60)
            _pipe.execute()

        sim = model.sim  # 相似度矩阵
        n_items = trainset.n_items
        # 内部物品id和外部物品id的映射mapping
        iiid_2_iid_dict = {}
        for iiid in range(n_items):
            iiid_2_iid_dict[iiid] = trainset.to_raw_iid(iiid)
        # 遍历每个物品提取其相似度信息
        for iiid in tqdm(range(n_items)):
            iid = iiid_2_iid_dict[iiid]
            # 当前物品和其它物品的相似度向量（其它物品实际id，相似度）
            iid_sim = []
            for _iiid, _sim in enumerate(sim[iiid]):
                if _iiid == iiid:
                    continue
                if _sim <= 0.0:
                    continue
                iid_sim.append((iiid_2_iid_dict[_iiid], _sim))
            # 提取前k个最相似度的物品及相似度
            top_k_iid_sim = heapq.nlargest(k, iid_sim, key=lambda t: t[1])
            # 转换为保存到redis的key\value值
            key = f"rec:recall:i2i:{iid}"
            value = ",".join([f"{t[0]}:{t[1]:.2f}" for t in top_k_iid_sim])
            values.append([key, value])
            # 检查是否需要保存redis
            if len(values) > 100:
                save_to_redis()
                values = []
        # 针对剩余的进行保存
        save_to_redis()


if __name__ == '__main__':
    training(
        r'C:\Users\gerry_17578261252713\PycharmProjects\MovieRecSystem\data\u.data'
    )
