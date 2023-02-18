# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/2|下午 12:49
# @Moto   : Knowledge comes from decomposition

import redis
import json
import numpy as np
from vector_server import vector_gen


class RecallServer:
    def __init__(self):
        # redis
        self.user_feature_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/1')

        self.item_feature_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/2')

        self.matrixcf_i2i_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/3')

        self.itemcf_i2i_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/4')

        self.usercf_u2u_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/5')

        self.fm_i2i_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/6')

        self.fm_item_embedding_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/7')

        self.fm_user_feature_embedding_pool = \
            redis.from_url('redis://:123456@127.0.0.1:6379/8')

        # 定义一个缓存item信息的缓存区
        self.user_info = {}
        self.item_info = {}

        # 当前用户
        self.current_user_feature = {}

        # 构建u2i需要向量服务
        print("init vector server ...")
        self.vectorserver = vector_gen.VectorServer(self.fm_item_embedding_pool)

    # {'user_id': 12345}
    def set_user_info(self, user_info):
        u = user_info['user_id']
        self.current_user_feature = self.user_info.get(str(u), None)
        if self.current_user_feature is None:
            self.current_user_feature = \
                json.loads(self.user_feature_pool.get(str(u)))
            self.user_info[str(u)] = self.current_user_feature

        print("current user feature include: ...")
        for k, v in self.current_user_feature.items():
            print(k, ": ", v)
        print("=" * 80)

    # u2i2i
    def get_item_cf_recommend_result(self, recall_num=30):
        if len(self.current_user_feature) <= 0:
            print("item_cf is error ...")
            return

        item_rank = {}
        hists = self.current_user_feature['hists']
        for loc, (i, _) in enumerate(hists):
            item_sim_ = json.loads(self.itemcf_i2i_pool.get(str(i)))
            print("=" * 80)
            print("from redis get item_cf_i2i-> item: "
                  "{} \n sim_item: {} ".format(i, item_sim_))
            print("=" * 80)

            for j, wij in item_sim_:
                if j in hists:
                    continue

                # 两篇文章的类别的权重，其中类别相同权重大
                item_i_info = self.item_info.get(i, None)
                if item_i_info is None:
                    item_i_info = json.loads(self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info
                item_j_info = self.item_info.get(j, None)
                if item_j_info is None:
                    item_j_info = json.loads(self.item_feature_pool.get(str(j)))
                    self.item_info[j] = item_j_info

                # 两篇文章的类别的权重，其中类别相同权重大
                type_weight = 1.0 if item_i_info['category_id'] == item_j_info[
                    'category_id'] else 0.7

                # 时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
                created_time_weight = np.exp(
                    0.7 ** np.abs(item_i_info['created_at_ts']
                                  - item_j_info['created_at_ts']))

                # 相似文章和历史点击文章序列中历史文章所在的位置权重
                loc_weight = (0.9 ** (len(hists) - loc))

                item_rank.setdefault(j, 0)
                item_rank[j] += loc_weight * type_weight * \
                                created_time_weight * type_weight * wij

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[
                    :recall_num]

        print("=" * 80)
        print("【recall】 item_cf_i2i: {}".format(item_rank))
        print("=" * 80)

        return item_rank

    # u2u2i
    def get_user_cf_recommend_result(self, recall_num=30):
        if len(self.current_user_feature) <= 0:
            print("item_cf is error ...")
            return
        user_id = self.current_user_feature['user_id']
        user_hist_item = [i for i, _ in self.current_user_feature['hists']]

        u2u_sim = json.loads(self.usercf_u2u_pool.get(str(user_id)))

        print("=" * 80)
        print("from redis get user_cf_u2u-> user: "
              "{} \n sim_user: {} ".format(user_id, u2u_sim))
        print("=" * 80)

        item_rank = {}
        for sim_u, wuv in u2u_sim:
            sim_user_feature = self.user_info.get(str(sim_u), None)
            if sim_user_feature is None:
                sim_user_feature = \
                    json.loads(self.user_feature_pool.get(str(sim_u)))
                self.user_info[str(sim_u)] = sim_user_feature

            for i, _ in sim_user_feature['hists']:

                if i in user_hist_item:
                    continue
                item_rank.setdefault(i, 0)
                # 两篇文章的类别的权重，其中类别相同权重大
                item_i_info = self.item_info.get(i, None)
                if item_i_info is None:
                    item_i_info = json.loads(
                        self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info

                for loc, (j, click_time) in enumerate(
                        self.current_user_feature['hists']):

                    item_j_info = self.item_info.get(j, None)
                    if item_j_info is None:
                        item_j_info = json.loads(
                            self.item_feature_pool.get(str(j)))
                        self.item_info[j] = item_j_info

                    # 两篇文章的类别的权重，其中类别相同权重大
                    type_weight = 1.0 if item_i_info['category_id'] == \
                                         item_j_info[
                                             'category_id'] else 0.7

                    # 时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
                    created_time_weight = np.exp(
                        0.7 ** np.abs(item_i_info['created_at_ts']
                                      - item_j_info['created_at_ts']))

                    # 相似文章和历史点击文章序列中历史文章所在的位置权重
                    loc_weight = (0.9 ** (len(user_hist_item) - loc))
                    item_rank[i] += loc_weight * type_weight * \
                                    created_time_weight * type_weight * wuv

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[
                    :recall_num]

        print("=" * 80)
        print("【recall】 user_cf_u2u: {}".format(item_rank))
        print("=" * 80)
        return item_rank

    # u2i2i
    def get_matrix_cf_recommend_result(self, recall_num=30):
        if len(self.current_user_feature) <= 0:
            print("item_cf is error ...")
            return

        item_rank = {}
        hists = self.current_user_feature['hists']
        for loc, (i, _) in enumerate(hists):
            item_sim_ = json.loads(self.matrixcf_i2i_pool.get(str(i)))
            print("=" * 80)
            print("from redis get matrix_cf_i2i-> item: "
                  "{} \n sim_item: {} ".format(i, item_sim_))
            print("=" * 80)

            for j, wij in item_sim_:
                if j in hists:
                    continue
                # 两篇文章的类别的权重，其中类别相同权重大
                item_i_info = self.item_info.get(i, None)
                if item_i_info is None:
                    item_i_info = json.loads(
                        self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info
                item_j_info = self.item_info.get(j, None)
                if item_j_info is None:
                    item_j_info = json.loads(
                        self.item_feature_pool.get(str(j)))
                    self.item_info[j] = item_j_info

                # 两篇文章的类别的权重，其中类别相同权重大
                type_weight = 1.0 if item_i_info['category_id'] == \
                                     item_j_info[
                                         'category_id'] else 0.7

                # 时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
                created_time_weight = np.exp(
                    0.7 ** np.abs(item_i_info['created_at_ts']
                                  - item_j_info['created_at_ts']))

                # 相似文章和历史点击文章序列中历史文章所在的位置权重
                loc_weight = (0.9 ** (len(hists) - loc))

                item_rank.setdefault(j, 0)
                item_rank[j] += loc_weight * type_weight * \
                                created_time_weight * type_weight * wij

        item_rank = sorted(item_rank.items(), key=lambda x: x[1],
                           reverse=True)[
                    :recall_num]

        print("=" * 80)
        print("【recall】 matrix_cf_i2i: {}".format(item_rank))
        print("=" * 80)

        return item_rank

    # u2i2i
    def get_fm_i2i_recommend_result(self, recall_num=30):
        if len(self.current_user_feature) <= 0:
            print("item_cf is error ...")
            return

        item_rank = {}
        hists = self.current_user_feature['hists']
        for loc, (i, _) in enumerate(hists):
            item_sim_ = json.loads(self.fm_i2i_pool.get(str(i)))
            print("=" * 80)
            print("from redis get fm_i2i-> item: "
                  "{} \n sim_item: {} ".format(i, item_sim_))
            print("=" * 80)

            for j, wij in item_sim_:
                if j in hists:
                    continue
                # 两篇文章的类别的权重，其中类别相同权重大
                item_i_info = self.item_info.get(i, None)
                if item_i_info is None:
                    item_i_info = json.loads(
                        self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info
                item_j_info = self.item_info.get(j, None)
                if item_j_info is None:
                    item_j_info = json.loads(
                        self.item_feature_pool.get(str(j)))
                    self.item_info[j] = item_j_info

                # 两篇文章的类别的权重，其中类别相同权重大
                type_weight = 1.0 if item_i_info['category_id'] == \
                                     item_j_info[
                                         'category_id'] else 0.7

                # 时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
                created_time_weight = np.exp(
                    0.7 ** np.abs(item_i_info['created_at_ts']
                                  - item_j_info['created_at_ts']))

                # 相似文章和历史点击文章序列中历史文章所在的位置权重
                loc_weight = (0.9 ** (len(hists) - loc))

                item_rank.setdefault(j, 0)
                item_rank[j] += loc_weight * type_weight * \
                                created_time_weight * type_weight * wij

        item_rank = sorted(item_rank.items(), key=lambda x: x[1],
                           reverse=True)[
                    :recall_num]

        print("=" * 80)
        print("【recall】 fm_i2i: {}".format(item_rank))
        print("=" * 80)

        return item_rank

    # u2i
    def fm_u2i_recommend_result(self, recall_num=30):
        user_id = "user_id=" + str(self.current_user_feature['user_id'])
        env = "env=" + str(self.current_user_feature['environment'])
        region = "region=" + str(self.current_user_feature['region'])

        emb = self.fm_user_feature_embedding_pool.mget([user_id, env, region])
        emb = [json.loads(_) for _ in emb]
        emb = np.sum(np.asarray(emb), axis=0, keepdims=True)
        items_rec_ = self.vectorserver.get_sim_item(emb, recall_num)
        item_rank = items_rec_[0]
        print("=" * 80)
        print("【recall】 fm_u2i: {}".format(item_rank))
        print("=" * 80)
        return item_rank

    def merge_recall_result(self, item_ranks):
        item_rec = {}
        for item_rank, weight in item_ranks:
            tmp = [_[1] for _ in item_rank]
            max_value = max(tmp)
            min_value = min(tmp)
            for i, w in item_rank:
                item_rec.setdefault(i, 0)
                item_rec[i] += weight * (w - min_value) / (
                            max_value - min_value)

        return item_rec


def main():
    print("start recall server ... ")
    rs = RecallServer()

    rec_user = {'user_id': 190000}

    rs.set_user_info(rec_user)
    itemcf_item_rank = rs.get_item_cf_recommend_result()
    usercf_item_rank = rs.get_user_cf_recommend_result()
    matrixcf_item_rank = rs.get_matrix_cf_recommend_result()
    fm_i2i_item_rank = rs.get_fm_i2i_recommend_result()
    fm_u2i_item_rank = rs.fm_u2i_recommend_result()

    item_rec = rs.merge_recall_result(
        [(itemcf_item_rank, 1.0), (usercf_item_rank, 1.0),
         (matrixcf_item_rank, 1.0),
         (fm_i2i_item_rank, 1.0), (fm_u2i_item_rank, 1.0)]
    )
    print("+" * 80)
    print("current_user: {}, recommend item: {}".format(
        rec_user['user_id'], item_rec))


if __name__ == '__main__':
    main()
