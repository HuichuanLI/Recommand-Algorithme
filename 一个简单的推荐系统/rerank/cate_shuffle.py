# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/2/24|上午 09:47
# @Moto   : Knowledge comes from decomposition


"""
items = [
    {'item_id': 'N2031', 'cate': 01, 'score': 0.92},
    {'item_id': 'N2032', 'cate': 01, 'score': 0.71},
    {'item_id': 'N2033', 'cate': 01, 'score': 0.70},
    {'item_id': 'N2034', 'cate': 02, 'score': 0.65},
    {'item_id': 'N2035', 'cate': 02, 'score': 0.64},
    {'item_id': 'N2036', 'cate': 03, 'score': 0.63},
    {'item_id': 'N2037', 'cate': 03, 'score': 0.61},
]


"""


def cate_shuffle(items):
    cate_items = {}
    cate_sort = []

    for item in items:
        cate = item['cate']
        cate_items.setdefault(cate, [])
        cate_items[cate].append(item)
        if cate not in cate_sort:
            cate_sort.append(cate)
    #
    result = []
    for i in range(len(items)):
        for c in cate_sort:
            res = cate_items[c]
            if i > len(res) - 1:
                continue
            result.append(res[i])

    return result


if __name__ == '__main__':
    items = [
        {'item_id': 'N2031', 'cate': '01', 'score': 0.92},
        {'item_id': 'N2032', 'cate': '01', 'score': 0.71},
        {'item_id': 'N2033', 'cate': '01', 'score': 0.70},
        {'item_id': 'N2034', 'cate': '02', 'score': 0.65},
        {'item_id': 'N2035', 'cate': '02', 'score': 0.64},
        {'item_id': 'N2036', 'cate': '03', 'score': 0.63},
        {'item_id': 'N2037', 'cate': '03', 'score': 0.61},
    ]

    result = cate_shuffle(items)

    for re in result:
        print(re)
