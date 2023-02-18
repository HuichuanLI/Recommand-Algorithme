# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import mmh3
import random


def mhash(valu):
    return mmh3.hash64(valu, signed=False)[1]


lines = open("articles.csv")

wfile = open("item_feature.dat", "w")

items = []
for i, line in enumerate(lines):
    if i == 0:
        continue
    tmp = line.strip().split(",")
    tag1 = tmp[0]
    tag2 = tmp[1]
    tag3 = tmp[2]
    tag4 = tmp[3]
    item_id = mhash("item_id=" + str(tag1))
    items.append(item_id)
    category_id = mhash("category_id=" + str(tag2))
    created_at_ts = mhash("created_at_ts=" + str(tag3))
    words_count = mhash("words_count=" + str(tag4))
    wfile.write(
        str(item_id) + "," + str(category_id) + "," + str(created_at_ts) +
        "," + str(words_count) + "\n")

lines = open("click_log.csv")


samplefile = open("shop.dat", "w")
user_info = {}
sample_info = []
for i, line in enumerate(lines):
    if i == 0:
        continue
    tmp = line.strip().split(",")
    uid = tmp[0]
    iid = tmp[1]
    ts = tmp[2]
    tag2 = tmp[3]
    tag3 = tmp[4]
    user_id = mhash("user_id=" + str(uid))
    item_id = mhash("item_id=" + str(iid))
    samplefile.write(
        str(ts) + "," + str(user_id) + "," + str(item_id) + ",1\n")
    for item in random.sample(items, 10):
        samplefile.write(
            str(ts) + "," + str(user_id) + "," + str(item) + ",0\n")

    tag2 = mhash("tag2=" + str(tag2))
    tag3 = mhash("tag3=" + str(tag3))
    user_info[user_id] = [tag2, tag3]

wfile = open("user_feature.dat", "w")
for k, v in user_info.items():
    wfile.write(str(k) + "," + str(v[0]) + "," + str(v[1]) + "\n")