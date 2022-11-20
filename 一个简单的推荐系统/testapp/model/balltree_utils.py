import numpy as np
from sklearn import neighbors

def build_balltree(embedding_list):
    article_embeddings = np.array(embedding_list)
    normalized_article_embeddings = article_embeddings / np.linalg.norm(
        article_embeddings, axis=1, keepdims=True)

    article_tree = neighbors.BallTree(normalized_article_embeddings,
                                      leaf_size=40)
    return article_tree, normalized_article_embeddings

def query_tree(balltree, normalized_embeddings, id_list, threshold=20):
    similarity, indices = balltree.get_similar_items(normalized_embeddings,
                                                     threshold)
    item_to_item_to_score = {}
    for article_id, sim_score_list, index_list in zip(id_list,
                                                      similarity, indices):
        tmp = {}
        for sim_score, index in zip(sim_score_list[1:], index_list[1:]):
            tmp[id_list[index]] = sim_score
        item_to_item_to_score[article_id] = sorted(tmp.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True)[:threshold]
    return item_to_item_to_score
