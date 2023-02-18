# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/2/24|上午 09:49
# @Moto   : Knowledge comes from decomposition


def forced_insertion(new_doc, items, nums):
    items_tmp = []
    max_score = items[0]['score']
    if nums == 1:
        for i, n in enumerate(new_doc):
            items_tmp.append(
                {'item_id': n, 'score': max_score + (len(new_doc) - i) * 0.01})
        for it in items:
            items_tmp.append(it)
        return items_tmp
    else:
        max_score = items[nums - 2]['score']
        min_score = items[nums - 1]['score']
        score = (max_score - min_score - 0.01) / len(new_doc)

        for i, it in enumerate(items):
            if i == nums - 1:
                for j, n in enumerate(new_doc):
                    items_tmp.append(
                        {'item_id': n, 'score': max_score - (j + 1) * score})
            items_tmp.append(it)

    return items_tmp


if __name__ == '__main__':
    new_doc = ['N2073', 'N2075']
    items = [
        {'item_id': 'N2031', 'cate': '01', 'score': 0.92},
        {'item_id': 'N2032', 'cate': '01', 'score': 0.71},
        {'item_id': 'N2033', 'cate': '01', 'score': 0.70},
        {'item_id': 'N2034', 'cate': '02', 'score': 0.65},
        {'item_id': 'N2035', 'cate': '02', 'score': 0.64},
        {'item_id': 'N2036', 'cate': '03', 'score': 0.63},
        {'item_id': 'N2037', 'cate': '03', 'score': 0.61},
    ]

    items = forced_insertion(new_doc, items, 2)
    for item in items:
        print(item)
