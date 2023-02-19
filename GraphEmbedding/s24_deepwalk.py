import networkx as nx
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec


def walkOneTime(g, start_node, walk_length):
    walk = [str(start_node)]  # 初始化游走序列
    for _ in range(walk_length):  # 最大长度范围内进行采样
        current_node = int(walk[-1])
        successors = list(g.successors(current_node))  # graph.successor: 获取当前节点的后继邻居
        if len(successors) > 0:
            next_node = np.random.choice(successors, 1)
            walk.extend([str(n) for n in next_node])
        else:
            break
    return walk


def getDeepwalkSeqs(g, walk_length, num_walks):
    seqs = []
    for _ in tqdm(range(num_walks)):
        start_node = np.random.choice(g.nodes)
        w = walkOneTime(g, start_node, walk_length)
        seqs.append(w)
    return seqs


def deepwalk(g, dimensions=10, walk_length=80, num_walks=10, min_count=3):
    seqs = getDeepwalkSeqs(g, walk_length=walk_length, num_walks=num_walks)
    model = word2vec.Word2Vec(seqs, vector_size=dimensions, min_count=min_count)
    return model


if __name__ == '__main__':
    g = nx.fast_gnp_random_graph(n=100, p=0.5, directed=True)  # 快速随机生成一个有向图
    model = deepwalk(g, dimensions=10, walk_length=20, num_walks=100, min_count=3)

    print(model.wv.most_similar('2', topn=3))  # 观察与节点2最相近的三个节点
    model.wv.save_word2vec_format('e.emd')  # 可以把emd储存下来以便下游任务使用
    model.save('m.model')  # 可以把模型储存下来以便下游任务使用
