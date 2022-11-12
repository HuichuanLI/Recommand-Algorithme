# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import json

user_feat_dic = {}
ad_feat_dic = {}
train_sample = []
test_sample = []

lines = open("train.csv")
for line in lines:
    line = line.strip().split(",")
    id = line[0]
    ts = line[1]
    uid = line[2]
    iid = line[3]
    itag1 = int(line[4])
    itag2 = int(line[5])
    itag3 = int(line[6])
    itag4 = float(line[7]) if line[7] != "" else 0
    utag1 = line[8]
    utag2 = int(float(line[9])) if line[9] != "" else 0
    utag3 = int(float(line[10])) if line[10] != "" else 0
    utag4 = int(line[11])
    label = float(line[12])
    user_feat_dic[uid] = {
        "uid": uid,
        "utag1": utag1,
        "utag2": utag2,
        "utag3": utag3,
        "utag4": utag4
    }
    ad_feat_dic[iid] = {
        "iid": iid,
        "itag1": itag1,
        "itag2": itag2,
        "itag3": itag3,
        "itag4": itag4
    }
    train_sample.append({
        "uid": uid,
        "utag1": utag1,
        "utag2": utag2,
        "utag3": utag3,
        "utag4": utag4,
        "iid": iid,
        "itag1": itag1,
        "itag2": itag2,
        "itag3": itag3,
        "itag4": itag4,
        "label": label
    })

lines = open("test.csv")
for line in lines:
    line = line.strip().split(",")
    id = line[0]
    ts = line[1]
    uid = line[2]
    iid = line[3]
    itag1 = int(line[4])
    itag2 = int(line[5])
    itag3 = int(line[6])
    itag4 = float(line[7]) if line[7] != "" else 0
    utag1 = line[8]
    utag2 = int(float(line[9])) if line[9] != "" else 0
    utag3 = int(float(line[10])) if line[10] != "" else 0
    utag4 = int(line[11])
    label = float(line[12])
    user_feat_dic[uid] = {
        "uid": uid,
        "utag1": utag1,
        "utag2": utag2,
        "utag3": utag3,
        "utag4": utag4
    }
    ad_feat_dic[iid] = {
        "iid": iid,
        "itag1": itag1,
        "itag2": itag2,
        "itag3": itag3,
        "itag4": itag4
    }
    test_sample.append({
        "uid": uid,
        "utag1": utag1,
        "utag2": utag2,
        "utag3": utag3,
        "utag4": utag4,
        "iid": iid,
        "itag1": itag1,
        "itag2": itag2,
        "itag3": itag3,
        "itag4": itag4,
        "label": label
    })

file1 = open("user_feature.dat", 'w')
for u, f in user_feat_dic.items():
    file1.write(json.dumps(f) + "\n")
file1.close()

file2 = open("ad_feature.dat", 'w')
for u, f in ad_feat_dic.items():
    file2.write(json.dumps(f) + "\n")
file2.close()

file3 = open("train_sample.dat", 'w')
for sam in train_sample:
    file3.write(json.dumps(sam) + "\n")
file3.close()

file4 = open("test_sample.dat", 'w')
for sam in test_sample:
    file4.write(json.dumps(sam) + "\n")
file4.close()
