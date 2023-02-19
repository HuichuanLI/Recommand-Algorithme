import numpy as np
from data_set import filepaths as fp
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score
from basic_sim import dataloader


class ALS_MLP(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super(ALS_MLP, self).__init__()
        '''
        :param n_users: 用户数量
        :param n_items: 物品数量
        :param dim: 向量维度
        '''
        # 随机初始化用户的向量,
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        # 随机初始化物品的向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)

        # 初始化用户向量的隐层
        self.u_hidden_layer1 = self.dense_layer(dim, dim // 2)
        self.u_hidden_layer2 = self.dense_layer(dim // 2, dim // 4)

        # 初始化物品向量的隐层
        self.i_hidden_layer1 = self.dense_layer(dim, dim // 2)
        self.i_hidden_layer2 = self.dense_layer(dim // 2, dim // 4)

        self.sigmoid = nn.Sigmoid()

    def dense_layer(self, in_features, out_features):
        # 每一个mlp单元包含一个线性层和激活层，当前代码中激活层采取Tanh双曲正切函数。
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

    def forward(self, u, v, isTrain=True):
        '''
         :param u: 用户索引id shape:[batch_size]
         :param i: 用户索引id shape:[batch_size]
         :return: 用户向量与物品向量的内积 shape:[batch_size]
         '''
        u = self.users(u)
        v = self.items(v)

        u = self.u_hidden_layer1(u)
        u = self.u_hidden_layer2(u)

        v = self.i_hidden_layer1(v)
        v = self.i_hidden_layer2(v)

        # 训练时采取dropout来防止过拟合
        if isTrain:
            u = F.dropout(u)
            v = F.dropout(v)

        uv = torch.sum(u * v, axis=1)
        logit = self.sigmoid(uv * 3)

        return logit


def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i, False)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=10, batchSize=1024, lr=0.001, dim=256, eva_per_epochs=1):
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
    # 初始化ALS模型
    net = ALS_MLP(len(user_set), len(item_set), dim)
    # 定义优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.2)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 开始迭代
    for e in range(epochs):
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


if __name__ == '__main__':
    train()
