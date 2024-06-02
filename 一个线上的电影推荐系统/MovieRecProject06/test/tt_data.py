# -*- coding: utf-8 -*-
import os

import sys
sys.path.append(os.path.abspath("../src"))

def init_redis_data():
    from movie_online.utils.redis_util import RedisUtil

    user_id = 196

    with RedisUtil.get_redis() as redis:
        if not redis.exists(f'rec:user:nolikecategory:{user_id}'):
            redis.sadd(f'rec:user:nolikecategory:{user_id}', 'action', 'adventure', 'animation', 'children', 'comedy')

        if not redis.exists(f'rec:user:blacklist:{user_id}'):
            redis.sadd(f'rec:user:blacklist:{user_id}', 102, 202, 302, 402, 502, 602, 702)

        if not redis.exists('rec:new_spu'):
            redis.lpush('rec:new_spu', 101, 102, 103, 104, 105, 106, 107)

        if not redis.exists('rec:hot_spu'):
            redis.lpush('rec:hot_spu', 106, 107, 201, 202, 203, 204, 205)

        if not redis.exists('rec:new_spu:loc_40'):
            redis.lpush('rec:new_spu:loc_40', 106, 301, 302, 303, 304, 305, 201)

        if not redis.exists('rec:hot_spu:loc_40'):
            redis.lpush('rec:hot_spu:loc_40', 202, 105, 301, 401, 402, 403, 404, 405, 406)

        if not redis.exists(f'rec:user:views:{user_id}'):
            redis.zadd(
                f'rec:user:views:{user_id}',
                {
                    '105': 1705191015,
                    '205': 1705191215,
                    '305': 1705191315,
                    '405': 1705191415,
                    '505': 1705192515,
                    '605': 1705191615,
                    '705': 1705191715,
                    '706': 1705193715,
                    '707': 1705194715,
                    '708': 1705195715,
                    '709': 1705196715,
                    '710': 1705197715
                }
            )


def download_movie_author():
    from movie_offline.data import movie_author_precess

    movie_author_precess.append_authors(
        root_dir="../data",
        name="u.item",
        new_name="u.new.item"
    )


def init_model_features():
    from movie_offline.data import movie_feature_process

    root_dir = "../data/features"
    os.makedirs(root_dir, exist_ok=True)

    # 构建FM模型训练的特征属性矩阵
    movie_feature_process.merge_feature_v3(root_dir)


if __name__ == '__main__':
    # init_redis_data()
    # download_movie_author()
    # init_model_features()
