"""
item cf algo
"""

import math

import read
import operator


def base_contribure_score():
    return 1


def cal_item_sim(user_click):
    """

    :param
    user_clcik: dict userid value [itemid1,itemid2]
    :return:
    dict key:itemid_i , value dict,value_key item_j ,value_value
    """

    co_appear = {}
    item_user_click_time = {}
    for user, itemlist in user_click.items():
        for index_i in range(0, len(itemlist)):
            itemid_i = itemlist[index_i]
            item_user_click_time.setdefault(itemid_i, 0)
            item_user_click_time[itemid_i] += 1
            for index_j in range(index_i + 1, len(itemlist)):
                itemid_j = itemlist[index_j]
                co_appear.setdefault(itemid_i, {})
                co_appear[itemid_i].setdefault(itemid_j, 0)
                co_appear[itemid_i][itemid_j] += base_contribure_score()

                co_appear.setdefault(itemid_j, {})
                co_appear[itemid_j].setdefault(itemid_i, 0)
                co_appear[itemid_j][itemid_i] += base_contribure_score()

    item_sim_score = {}
    item_sim_score_sort = {}
    for itemid_i, relate_item in co_appear.items():
        for itemid_j, co_time in relate_item.items():
            sim_score = co_time / math.sqrt(item_user_click_time[itemid_i] * item_user_click_time[itemid_j])
            item_sim_score.setdefault(itemid_i, {})
            item_sim_score[itemid_i].setdefault(itemid_j, {})
            item_sim_score[itemid_i][itemid_j] = sim_score

    # 为了排好序 每个 i最好的 几个相似的
    for itemid in item_sim_score:
        item_sim_score_sort[itemid] = sorted(item_sim_score[itemid].items(), key=operator.itemgetter(1),
                                             reverse=True)

    return item_sim_score_sort


def debugsim(item_info,sim_info,index):
    """
    shiw itemsim info
    :param item_info: dict key itemid value:[title,ggenres]
    :param sim_info: dict key item value list[(item1,simscore1)]
    :return:
    """
    fixed_itemid = index;
    if fixed_itemid not in item_info:
        print("invalid index")
        return
    [title_fix,genres_fix] = item_info[fixed_itemid]
    for zuhe in sim_info[fixed_itemid][:5]:
        itemid_sim = zuhe[0]
        sim_score = zuhe[1]
        [title,genres] = item_info[itemid_sim]
        print("{}\t+{}\t+{}\t+{}\t+{}".format(title_fix,genres_fix,title,genres,str(sim_score)))


def debug_recomresult(recomsult,item_info,user_id):
    """
    将输出ID 转换为名字
    :param recomsult: userid : item_id item_score
    :param item_info:
    :return:
    """

    if user_id not in recomsult:
        print("invalid index")
        return
    for zuhe in sorted(recomsult[user_id].items(),key=operator.itemgetter(1),reverse=True):
        itemid,score = zuhe
        if itemid not in item_info:
            continue
        print("{}+{}\n".format(item_info[itemid],score))





def cal_recom_result(sim_info, user_clcik):
    """

    :param sim_info: item sim dict
    :param user_clcik:
    :return:
    Return:
        dict,key item_id ,value_key :item_id value_value recom_score
    """
    recent_click_number = 3
    topk = 5
    recom_info = {}
    for user, itemlist in user_clcik.items():
        recom_info.setdefault(user, {})
        for itemid in itemlist[:recent_click_number]:
            if itemid not in sim_info:
                continue
            for itemidsimset in sim_info[itemid][:topk]:
                if itemidsimset[0] in user_clcik[user]:
                    continue
                itemsimid = itemidsimset[0]
                itemsimidscore = itemidsimset[1]
                recom_info[user][itemsimid] = itemsimidscore

    return recom_info


def main_flow():
    """
    main flow
    :return:
    """
    user_click = read.get_user_click("ratings.csv")
    item_info = read.get_item_info("movies.csv")
    sim_info = cal_item_sim(user_click)

    recom_result = cal_recom_result(sim_info, user_click)

    # debugsim(item_info, sim_info, "1")
    debug_recomresult(recom_result,item_info,"1")
    # print(recom_result["1"])


main_flow()
