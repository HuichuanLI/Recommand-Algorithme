# -*-coding:utf8-*-
"""
author:Huichuan
date:2019****
get up and online recommendation
"""

import os


def get_up(item_cate, input_file):
    """
    Args:
        item_cate:key itemid, value: dict , key category value ratio
        input_file:user rating file
    Return:
    User 对每个Category的喜欢
        a dict: key userid, value [(category, ratio), (category1, ratio1)]
    """
    if not os.path.exists(input_file):
        return {}
