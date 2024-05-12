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
            'user_based': False
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
    # noinspection PyProtectedMember
    with RedisUtil._get_redis() as client:
        n_users = trainset.n_users
        n_items = trainset.n_items
        for iuid in tqdm(range(n_users)):
            uid = trainset.to_raw_uid(iuid)
            item_ratings = []
            # 提取当前用户已经有操作行为的商品列表
            user_view_iiids = [t[0] for t in trainset.ur[iuid]]
            # 遍历所有商品，提取对应的预测评分
            for iiid in range(n_items):
                if iiid in user_view_iiids:
                    continue
                iid = trainset.to_raw_iid(iiid)
                rating = model.predict(uid, iid).est
                # 保存预测评分及实际物品id
                item_ratings.append([rating, iid])
            # 排序获取评分最高的前K个值
            top_k_item_ratings = heapq.nlargest(k, item_ratings, key=lambda t: t[0])
            # 合并保存到redis中
            key = f"rec:recall:u2i:{uid}"
            field = "item_cf"
            value = ",".join([f"{t[1]}:{t[0]:.2f}" for t in top_k_item_ratings])
            client.hset(key, field, value)
            # 过期时间设置为7天
            client.expire(key, time=7 * 24 * 60 * 60)


if __name__ == '__main__':
    training(
        r'C:\Users\gerry_17578261252713\PycharmProjects\MovieRecSystem\data\u.data'
    )
