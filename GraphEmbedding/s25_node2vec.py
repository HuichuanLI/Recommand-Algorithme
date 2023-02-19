import networkx as nx
from node2vec import Node2Vec

graph = nx.fast_gnp_random_graph(n=100, p=0.5)#快速随机生成一个无向图
node2vec = Node2Vec ( graph, dimensions=64, walk_length=30, num_walks=100, p=0.3,q=0.7,workers=4)#初始化模型
model = node2vec.fit()#训练模型
print(model.wv.most_similar('2',topn=3))# 观察与节点2最相近的三个节点

