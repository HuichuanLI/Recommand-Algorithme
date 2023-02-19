import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4ml100kIndexs
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn
import sys


class embedding_CNN(nn.Module):

    def __init__(self, n_user_features, n_item_features, user_df, item_df, dim=128):
        super(embedding_CNN, self).__init__()
        # 随机初始化所有特征的特征向量
        self.user_features = nn.Embedding(n_user_features, dim, max_norm=1)
        self.item_features = nn.Embedding(n_item_features, dim, max_norm=1)
        # 记录好用户和物品的特征所以
        self.user_df = user_df
        self.item_df = item_df

        # 得到用户和物品特征的数量的和
        total_neigbours = user_df.shape[1] + item_df.shape[1]

        self.Conv = nn.Conv1d(in_channels=total_neigbours, out_channels=1, kernel_size=3)

        # 定义MLP传播的全连接层
        self.dense1 = self.dense_layer(dim - 2, dim // 2)
        self.dense2 = self.dense_layer(dim // 2, 1)

        self.sigmoid = nn.Sigmoid()

    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

    def forward(self, u, i, isTrain=True):
        user_ids = torch.LongTensor(self.user_df.loc[u].values)
        item_ids = torch.LongTensor(self.item_df.loc[i].values)
        # print(user_ids)
        # [batch_size, user_neibours, dim]
        user_features = self.user_features(user_ids)
        # print(user_features.shape)

        # [batch_size, item_neibours, dim]
        item_features = self.item_features(item_ids)

        # 将用户和物品特征向量拼接起来
        # [batch_size, total_neigbours, dim]
        uv = torch.cat([user_features, item_features], dim=1)

        # [batch_size, 1, dim+1-kernel_size]
        uv = self.Conv(uv)
        # [batch_size, dim+1-kernel_size]
        uv = torch.squeeze(uv)

        # 开始MLP的传播
        # [batch_size, dim//2]
        uv = self.dense1(uv)
        # 训练时采取dropout来防止过拟合
        if isTrain: uv = F.dropout(uv)
        # [batch_size, 1]
        uv = self.dense2(uv)
        # [batch_size]
        uv = torch.squeeze(uv)
        logit = self.sigmoid(uv)
        return logit


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


def train(epochs=10, batchSize=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_user_features, n_item_features = \
        dataloader4ml100kIndexs.read_data_user_item_df()

    # 初始化模型
    net = embedding_CNN(n_user_features, n_item_features, user_df, item_df, dim)

    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
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
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_triples)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_triples)
            print('test:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))


if __name__ == '__main__':
    train()
