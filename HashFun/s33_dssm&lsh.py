import numpy as np
import dataloader
from data_set import filepaths as fp
from torch.utils.data import DataLoader
from torch import nn
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score
import s32_lsh as lsh


class DSSM(nn.Module):

    def __init__(self, n_users, n_items, dim):
        super(DSSM, self).__init__()
        '''
        :param n_users: 用户数量
        :param n_items: 物品数量
        :param dim: 向量维度
        '''
        self.dim = dim
        self.n_users = n_users
        self.n_items = n_items
        # 随机初始化用户的向量, 将向量约束在L2范数为1以内
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        # 随机初始化物品的向量, 将向量约束在L2范数为1以内
        self.items = nn.Embedding(n_items, dim, max_norm=1)

        self.user_tower = self.tower()
        self.item_tower = self.tower()

    def tower(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim // 3),
            nn.ReLU(),
            nn.Linear(self.dim // 3, self.dim // 4),
        )

    # 前向传播
    def forward(self, u, v):
        '''
         :param u: 用户索引id shape:[batch_size]
         :param i: 用户索引id shape:[batch_size]
         :return: 用户向量与物品向量的内积 shape:[batch_size]
         '''
        u, v = self.towerForward(u, v)
        uv = torch.sum(u * v, axis=1)
        logit = torch.sigmoid(uv)
        return logit

    # “塔”的传播
    def towerForward(self, u, v):
        u = self.users(u)
        u = self.user_tower(u)
        v = self.items(v)
        v = self.item_tower(v)
        return u, v

    # 该方法返回的是“塔”最后一层的用户物品embedding
    def getEmbeddings(self):
        u = torch.LongTensor(range(self.n_users))
        v = torch.LongTensor(range(self.n_items))
        u, v = self.towerForward(u, v)
        return u, v


def doEva(net, d):
    net.eval()
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=10, batchSize=1024, lr=0.01, dim=256, eva_per_epochs=1):
    '''
    :param epochs: 迭代次数
    :param batchSize: 一批次的数量
    :param lr: 学习率
    :param dim: 用户物品向量的维度
    :param eva_per_epochs: 设定每几次进行一次验证
    '''
    # 读取数据
    user_set, item_set, train_set, test_set = \
        dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    # 初始化DSSM模型
    net = DSSM(len(user_set), len(item_set), dim)
    # 定义优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 开始迭代
    for e in range(epochs):
        net.train()
        all_lose = 0
        # 每一批次地读取数据
        for u, i, r in DataLoader(train_set, batch_size=batchSize, shuffle=True):
            optimizer.zero_grad()
            r = torch.FloatTensor(r.detach().numpy())
            result = net(u, i)
            loss = criterion(result, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {}, avg_loss = {:.4f}'.format(e, all_lose / (len(train_set) // batchSize)))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))

    return net


def doRecall():
    # 训练模型
    net = train()
    # 得到最后一层的用户，物品向量
    user_embs, item_embs = net.getEmbeddings()
    # 初始化LSH模型
    lsh_net = lsh.LSH(w=4, rows=32, bands=6)
    # 传入用户物品向量进行哈希分桶
    recall_dict = lsh_net.getRecalls(user_embs, item_embs)

    return recall_dict


if __name__ == '__main__':
    print(doRecall())
