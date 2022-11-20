# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/1/1|上午 10:41
# @Moto   : Knowledge comes from decomposition
import json
import numpy as np
from sklearn import neighbors


class VectorServer:
    def __init__(self, pool):
        # redis_url = 'redis://:123456@127.0.0.1:6379/' + str(redis_db)

        self.pool = pool
        self.keys_index = []
        self.vector_matrix = []

        keys = self.pool.keys()
        pipe = self.pool.pipeline()
        key_list = []
        s = 0

        for key in keys:
            key_list.append(key)
            pipe.get(key)
            if s < 10000:
                s += 1
            else:
                for k, v in zip(key_list, pipe.execute()):
                    vec = json.loads(v)
                    self.keys_index.append(int(k))
                    self.vector_matrix.append(vec)
                s = 0
                key_list = []

        for k, v in zip(key_list, pipe.execute()):
            vec = json.loads(v)
            self.keys_index.append(int(k))
            self.vector_matrix.append(vec)

        item_emb_np = np.asarray(self.vector_matrix)
        item_emb_np = item_emb_np / np.linalg.norm(
            item_emb_np, axis=1, keepdims=True)

        # 建立faiss/BallTree索引
        print("start build tree ... ")
        self.item_tree = neighbors.BallTree(item_emb_np, leaf_size=40)
        print("build tree end")

    # todo: items: [vector, vector, vector] -> n*embedding的矩阵
    def get_sim_item(self, items, cut_off):
        sim, idx = self.item_tree.query(items, cut_off)

        items_result = []
        for i in range(len(sim)):
            items = [self.keys_index[_] for _ in idx[i]]
            item_sim_score = dict(zip(items, sim[i]))
            item_sim_score = sorted(
                item_sim_score.items(), key=lambda _: _[1],
                reverse=True)[: cut_off]
            items_result.append(item_sim_score)

        return items_result


if __name__ == '__main__':
    print()
