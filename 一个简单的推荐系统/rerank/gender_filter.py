# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/2/24|上午 09:48
# @Moto   : Knowledge comes from decomposition


# target_gender = ['sports', 'lifestyle'] / ['health']
# items = [{'item_id': N2073, 'cate': 'sports'}]
def gender_filter(target_gender, items):
    items_tmp = []
    for it in items:
        if it['cate'] in target_gender:
            items_tmp.append(it)

    return items_tmp


if __name__ == '__main__':
    target_gender = ['01', '03']
    items = [
        {'item_id': 'N2031', 'cate': '01', 'score': 0.92},
        {'item_id': 'N2032', 'cate': '01', 'score': 0.71},
        {'item_id': 'N2033', 'cate': '01', 'score': 0.70},
        {'item_id': 'N2034', 'cate': '02', 'score': 0.65},
        {'item_id': 'N2035', 'cate': '02', 'score': 0.64},
        {'item_id': 'N2036', 'cate': '03', 'score': 0.63},
        {'item_id': 'N2037', 'cate': '03', 'score': 0.61},
    ]

    items = gender_filter(target_gender, items)
    for item in items:
        print(item)


