import json

import numpy as np
from sklearn import neighbors

class VectorServer:

    def __init__(self, redis_connection):
        self.redis_connection = redis_connection
        self.id_list = [int(key) for key in self.redis_connection.keys()]
        self.embedding_array = np.array([json.loads(value_json) for value_json in
                               self.redis_connection.mget(self.id_list)])

        normalized_embedding_array = self.embedding_array / np.linalg.norm(
            self.embedding_array, axis=1, keepdims=True)

        self.item_tree = neighbors.BallTree(normalized_embedding_array,
                                            leaf_size=40)

    def get_similar_items(self, embeddings, threshold=20):
        similarity, indices = self.item_tree.query(embeddings,
                                                   threshold)
        list_item_to_score = []
        for sim_score_list, index_list in zip(similarity, indices):
            tmp = {}
            for sim_score, index in zip(sim_score_list[1:], index_list[1:]):
                tmp[self.id_list[index]] = sim_score
            id_to_score = sorted(tmp.items(), key=lambda x: x[1],
                                                       reverse=True)[:threshold]
            list_item_to_score.append(id_to_score)
        return list_item_to_score