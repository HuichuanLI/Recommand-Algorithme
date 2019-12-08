import os
import csv


def get_user_click(rating_file):
    """
    get user click list
    :param rating_file:
    :return:
    dict,key:userid ,value:[itemid1,itemid2]
    """
    if not os.path.exists(rating_file):
        return {}
    csvFile = open(rating_file, "r")
    reader = csv.reader(csvFile)
    user_click = {}
    for item in reader:
        if reader.line_num == 1:
            continue
        if len(item) < 4:
            continue
        if float(item[2]) < 3.0:
            continue
        if item[0] not in user_click:
            user_click[item[0]] = []
        user_click[item[0]].append(item[1])
    return user_click

def get_item_info(item_file):
    """
    get item info [title,genres]
    :param item_file:
    :return:
    a dict, key itemid value:[titre,genre]
    """
    if not os.path.exists(item_file):
        return {}
    csvFile = open(item_file, "r")
    reader = csv.reader(csvFile)
    item_info = {}
    for item in reader:
        if reader.line_num == 1:
            continue
        if len(item) < 3:
            continue
        if item[0] not in item_info:
            item_info[item[0]] = [item[1],item[2]]
    return item_info


