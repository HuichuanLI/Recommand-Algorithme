# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/1|下午 10:54
# @Moto   : Knowledge comes from decomposition


import redis
import traceback
import warnings
import json

warnings.filterwarnings('ignore')


def save_redis(items, db=1):
    redis_url = 'redis://:123456@127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items.items():
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()


# 读取文件
def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")]
            dic[tmp[0]] = json.dumps(embedding)
    return dic


def main():
    data_path = "../data/"
    embedding_file = data_path + "fm_articles_emb"
    save_redis(read_embedding_file(embedding_file), db=7)
    embedding_file = data_path + "fm_user_emb"
    save_redis(read_embedding_file(embedding_file), db=8)


if __name__ == '__main__':
    main()
