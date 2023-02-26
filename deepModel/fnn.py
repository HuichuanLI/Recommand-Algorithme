import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4ml100kIndexs
from torch.utils.data import DataLoader
import torch
from torch import nn


class FNN_plus(nn.Module):

    def __init__(self, n_features, user_df, item_df, dim=128):
        super(FNN_plus, self).__init__()
        # 随机初始化所有特征的特征向量
        self.features = nn.Embedding(n_features, dim, max_norm=1)
        self.mlp_layer = self.__mlp(dim)
        # 记录好用户和物品的特征索引
        self.user_df = user_df
        self.item_df = item_df

    def __mlp(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def FMaggregator(self, feature_embs):
        # feature_embs:[batch_size, n_features, dim]
        # [batch_size, dim]
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        # [batch_size, dim]
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        # [batch_size, dim]
        output = square_of_sum - sum_of_square
        return output

    # 把用户和物品的特征合并起来
    def __getAllFeatures(self, u, i):
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        all = torch.cat([users, items], dim=1)
        return all

    def forward(self, u, i):
        # 得到用户与物品组合起来后的特征索引
        all_feature_index = self.__getAllFeatures(u, i)
        # 取出特征向量
        all_feature_embs = self.features(all_feature_index)
        # [batch_size, dim]
        out = self.FMaggregator(all_feature_embs)
        # [batch_size, 1]
        out = self.mlp_layer(out)
        # [batch_size]
        out = torch.squeeze(out)
        return out


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    precision = precision_score(r, y_pred)
    recall = recall_score(r, y_pred)
    acc = accuracy_score(r, y_pred)
    return precision, recall, acc


def train(epochs=20, batchSize=1024, lr=0.02, dim=128, eva_per_epochs=1, need_eva=True):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_features = \
        dataloader4ml100kIndexs.read_data()
    # 初始化模型
    net = FNN_plus(n_features, user_df, item_df, dim)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for u, i, r in DataLoader(train_triples, batch_size=batchSize, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logits = net(u, i)
            loss = criterion(logits, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(train_triples) // batchSize)))

        # 评估模型
        if e % eva_per_epochs == 0 and need_eva:
            p, r, acc = doEva(net, train_triples)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_triples)
            print('test:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))

    return net


if __name__ == '__main__':
    train()
