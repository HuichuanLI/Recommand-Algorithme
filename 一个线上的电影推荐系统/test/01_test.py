import json
import time
from urllib import request

import flask
import requests
from cachetools import cached, TTLCache

from MovieRecOffline.data import movie_author_process
from MovieRecSystem.entity.rec_item import RecItem
from MovieRecSystem.utils.redis_util import RedisUtil


def t1():
    item = RecItem.get_recall_rec_item(12, 1.0, "hot", "hot")
    print(json.dumps(dict(item)))
    # print(item)
    # app = flask.Flask(__name__)
    # print(flask.jsonify(item))


def t2():
    @cached(cache=TTLCache(maxsize=1, ttl=6))
    def add(a, b):
        print(f"执行add:{a} -- {b}")
        return a + b

    print(add(1, 2))
    time.sleep(5)
    print(add(1, 2))
    time.sleep(2)
    print(add(1, 2))
    print(add(1, 4))
    print(add(1, 4))
    print(add(1, 2))


def t3():
    with RedisUtil._get_redis() as client:
        client.set("rec:hot_spu", "1,3,5,7,9,11,13,15,17,19,21,23,25,27,29")
        client.set("rec:new_spu", "2,4,6,8,10,1,3,5,7,9")
        client.lpush('user:views:10001', 1, 3, 6, 10, 14, 18)


def t4():
    movie_author_process.append_authors(
        r"D:\授课\01_推荐系统\2022\20220626\ml-100k",
        "u.item.txt",
        "u_new.item2"
    )


if __name__ == '__main__':
    t4()
