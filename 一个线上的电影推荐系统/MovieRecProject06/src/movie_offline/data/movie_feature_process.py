# -*- coding: utf-8 -*-
"""
特征数据构造：从原始数据源下载数据，并进行必要的数据转换处理，得到模型可以进行训练的数据格式
"""

import os

import numpy as np
import pandas as pd
import surprise


def write_dict_mapping(path, values):
    _dir = os.path.dirname(path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    if isinstance(values, dict):
        with open(path, 'w', encoding='utf-8') as writer:
            for _k, _v in values.items():
                writer.writelines(f"{_k}\t{_v}\n")
    else:
        with open(path, 'w', encoding='utf-8') as writer:
            values = list(values)
            values.sort()
            writer.writelines("unk\t0\n")
            for idx, value in enumerate(values):
                writer.writelines(f"{value}\t{idx + 1}\n")


def merge_feature_v1(root_dir):
    """
    最简单合并成特征属性矩阵的方式
    :param root_dir: 数据根路径
    :return:
    """
    # 1. 加载电影基础特征
    movie_base_info = pd.read_csv(os.path.join(root_dir, "movies.csv"), sep="\t")
    movie_base_info['year'] = movie_base_info.release_date.apply(lambda t: str(t)[-4:])
    # 2. 加载用户基础特征
    user_base_info = pd.read_csv(os.path.join(root_dir, "users.csv"), sep="\t")
    user_base_info['location'] = user_base_info.zip_code.apply(lambda t: str(t)[:2])
    # 3. 加载评分基础数据
    user_movie_rating = pd.read_csv(os.path.join(root_dir, "user_movie_rating.csv"), sep="\t")
    # 4. 合并
    df = pd.merge(user_movie_rating, movie_base_info, left_on='movie_id', right_on='id')
    del df['id']
    df = pd.merge(df, user_base_info, left_on='user_id', right_on='id')
    del df['id']
    # 5. 部分特征扩展处理
    df.to_csv(os.path.join(root_dir, "feature.csv"), index=False)


def merge_feature_v2(root_dir):
    # 1. 加载电影基础特征
    movie_base_info = pd.read_csv(os.path.join(root_dir, "movies.csv"), sep="\t")
    movie_base_info['year'] = movie_base_info.release_date.apply(lambda t: str(t)[-4:])
    # 2. 加载用户基础特征
    user_base_info = pd.read_csv(os.path.join(root_dir, "users.csv"), sep="\t")
    user_base_info['location'] = user_base_info.zip_code.apply(lambda t: str(t)[:2])
    # 3. 加载评分基础数据
    user_movie_rating = pd.read_csv(os.path.join(root_dir, "user_movie_rating.csv"), sep="\t")
    # 4. 合并
    df = pd.merge(user_movie_rating, movie_base_info, left_on='movie_id', right_on='id')
    del df['id']
    df = pd.merge(df, user_base_info, left_on='user_id', right_on='id')
    del df['id']
    # 5. 均值特征信息加载
    mean_df = pd.read_csv(os.path.join(root_dir, 'movie_mean_rating_stat.csv'), sep="\t")
    mean_df.columns = ['movie_id', 'movie_mean_rating', 'total_rate_users']
    mean_df = mean_df[['movie_id', 'movie_mean_rating']]
    df = pd.merge(df, mean_df, on='movie_id')

    mean_df = pd.read_csv(os.path.join(root_dir, 'movie_user_gender_mean_rating_stat.csv'), sep="\t")
    mean_df.columns = ['movie_id', 'gender', 'movie_gender_mean_rating', 'total_rate_users']
    mean_df = mean_df[['movie_id', 'gender', 'movie_gender_mean_rating']]
    df = pd.merge(df, mean_df, on=['movie_id', 'gender'])

    mean_df = pd.read_csv(os.path.join(root_dir, 'user_mean_rating_stat.csv'), sep="\t")
    mean_df.columns = ['user_id', 'user_mean_rating', 'total_rate_items']
    mean_df = mean_df[['user_id', 'user_mean_rating']]
    df = pd.merge(df, mean_df, on='user_id')

    mean_df = pd.read_csv(os.path.join(root_dir, 'user_movie_genre_mean_rating_stat.csv'), sep="\t")
    df = pd.merge(df, mean_df, on='user_id')

    # 5. 部分特征扩展处理
    df.to_csv(os.path.join(root_dir, "feature.csv"), index=False)


# noinspection DuplicatedCode
def merge_feature_v3(root_dir):
    # 1. 加载电影基础特征
    movie_base_info = pd.read_csv(os.path.join(root_dir, "movies.csv"), sep="\t")
    # 2. 加载用户基础特征
    user_base_info = pd.read_csv(os.path.join(root_dir, "users.csv"), sep="\t")
    user_base_info['location'] = user_base_info.zip_code.apply(lambda t: str(t)[:2])
    # 3. 加载评分基础数据
    user_movie_rating = pd.read_csv(os.path.join(root_dir, "user_movie_rating.csv"), sep="\t")
    # 4. 合并
    df = pd.merge(user_movie_rating, movie_base_info, left_on='movie_id', right_on='id')
    del df['id']
    df = pd.merge(df, user_base_info, left_on='user_id', right_on='id')
    del df['id']
    # 5. 均值特征信息加载
    mean_df = pd.read_csv(os.path.join(root_dir, 'movie_mean_rating_stat.csv'), sep="\t")
    mean_df.columns = ['movie_id', 'movie_mean_rating', 'total_rate_users']
    mean_df = mean_df[['movie_id', 'movie_mean_rating']]
    df = pd.merge(df, mean_df, on='movie_id')

    mean_df = pd.read_csv(os.path.join(root_dir, 'movie_user_gender_mean_rating_stat2.csv'), sep="\t")
    df = pd.merge(df, mean_df, on=['movie_id'])

    mean_df = pd.read_csv(os.path.join(root_dir, 'user_mean_rating_stat.csv'), sep="\t")
    mean_df.columns = ['user_id', 'user_mean_rating', 'total_rate_items']
    mean_df = mean_df[['user_id', 'user_mean_rating']]
    df = pd.merge(df, mean_df, on='user_id')

    mean_df = pd.read_csv(os.path.join(root_dir, 'user_movie_genre_mean_rating_stat.csv'), sep="\t")
    df = pd.merge(df, mean_df, on='user_id')

    # 提取部分列
    df = df[[
        'user_id', 'age', 'gender', 'occupation', 'location', 'max_rating_genre', 'max_rete_items_genre',
        'action_mean_rating', 'adventure_mean_rating', 'animation_mean_rating', 'children_mean_rating',
        'comedy_mean_rating', 'crime_mean_rating', 'documentary_mean_rating', 'drama_mean_rating',
        'fantasy_mean_rating', 'film_noir_mean_rating', 'horror_mean_rating', 'musical_mean_rating',
        'mystery_mean_rating', 'romance_mean_rating', 'sci_fi_mean_rating', 'thriller_mean_rating',
        'unknown_mean_rating', 'war_mean_rating', 'western_mean_rating', 'user_mean_rating',

        'movie_id',
        'unknown', 'action', 'adventure', 'animation',
        'children', 'comedy', 'crime', 'documentary',
        'drama', 'fantasy', 'film_noir', 'horror',
        'musical', 'mystery', 'romance', 'sci_fi',
        'thriller', 'war', 'western',
        'movie_mean_rating', 'm_mean_rating', 'f_mean_rating',

        'rating'
    ]]

    # 部分映射表输出
    write_dict_mapping(os.path.join(root_dir, "dict", f"age.dict"), range(100))
    for c in [
        'user_id', 'gender', 'occupation', 'location', 'movie_id'
    ]:
        write_dict_mapping(os.path.join(root_dir, "dict", f"{c}.dict"), np.unique(df[c]))

    # 5. 部分特征扩展处理
    df.to_csv(os.path.join(root_dir, "feature_fm.csv"), index=False)


def merge_feature_v4(root_dir):
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

    dataset = get_dataset(os.path.join(root_dir, "u.data"))
    trainset = dataset.build_full_trainset()

    user_id_2_inner_id = {}  # 实际用户id到模型内部用户id的映射表
    spu_id_2_inner_id = {}  # 实际物品id到模型内部物品id的映射表
    for uiid in trainset.all_users():
        user_id_2_inner_id[trainset.to_raw_uid(uiid)] = uiid
    for iiid in trainset.all_items():
        spu_id_2_inner_id[trainset.to_raw_iid(iiid)] = iiid
    write_dict_mapping(os.path.join(root_dir, "bpr_dict", f"user_id.dict"), user_id_2_inner_id)
    write_dict_mapping(os.path.join(root_dir, "bpr_dict", f"spu_id.dict"), spu_id_2_inner_id)
    with open(os.path.join(root_dir, "feature_bpr.csv"), "w", encoding="utf-8") as writer:
        with open(os.path.join(root_dir, "feature_bpr2.csv"), "w", encoding="utf-8") as writer2:
            for uiid in trainset.all_users():
                user_ratings = trainset.ur[uiid]  # 得到当前用户评论的所有商品以及对应评分列表
                user_ratings.sort(key=lambda t: t[1], reverse=True)  # 按照评分降序排列
                n = len(user_ratings)
                for i in range(n):
                    for j in range(n - 1, i, -1):
                        if i == j:
                            continue
                        ui = user_ratings[i]  # 当前用户第i个评论物品以及对应的评分
                        uj = user_ratings[j]  # 当前用户第j个评论物品以及对应的评分
                        if ui[1] > uj[1]:  # 第i个物品的评分比第j个的物品的评分高
                            writer.writelines(f"{uiid}\t{ui[0]}\t{uj[0]}\n")
                            writer2.writelines(f"{uiid}\t{ui[0]}\t{uj[0]}\t{ui[1]}\t{uj[1]}\n")
                        else:
                            break


def merge_feature_v5(root_dir):
    movie_base_info = pd.read_csv(os.path.join(root_dir, "movies.csv"), sep="\t")
    movie_base_info = movie_base_info[['id', 'title', 'actors']]
    movie_base_info.dropna(inplace=True)
    with open(os.path.join(root_dir, "feature_movies_neo4j.csv"), "w", encoding="utf-8") as writer:
        for value in movie_base_info.values:
            _id = value[0]
            _title = str(value[1]).replace(",", " ")
            actors = str(value[2]).split(",")
            for actor in actors:
                writer.writelines(f"{_id},{_title},{actor.strip()}\n")
