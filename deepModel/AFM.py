import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from chapter2 import dataloader4ml100kIndexs
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import Parameter, init


class AFM(nn.Module):

    def __init__(self, n_features, user_df, item_df, k, t):
        super(AFM, self).__init__()
        # 随机初始化所有特征的特征向量
        self.features = nn.Embedding(n_features, k, max_norm=1)
        # 注意力计算中的线性层
        self.attention_liner = nn.Linear(k, t)
        # AFM公式中的h
        self.h = init.xavier_uniform_(Parameter(torch.empty(t, 1)))
        # AFM公式中的p
        self.p = init.xavier_uniform_(Parameter(torch.empty(k, 1)))
        # 记录好用户和物品的特征索引
        self.user_df = user_df
        self.item_df = item_df

    # FM聚合
    def FMaggregator(self, feature_embs):
        # feature_embs:[ batch_size, n_features, k ]
        # [ batch_size, k ]
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        # [ batch_size, k ]
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        # [ batch_size, k ]
        output = square_of_sum - sum_of_square
        return output

    # 注意力计算
    def attention(self, embs):
        # embs: [ batch_size, k ]
        # [ batch_size, t ]
        embs = self.attention_liner(embs)
        # [ batch_size, t ]
        embs = torch.relu(embs)
        # [ batch_size, 1 ]
        embs = torch.matmul(embs, self.h)
        # [ batch_size, 1 ]
        atts = torch.softmax(embs, dim=1)
        return atts

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
        # 经过FM层得到输出
        embs = self.FMaggregator(all_feature_embs)
        # 得到注意力
        atts = self.attention(embs)
        # [ batch_size, 1 ]
        outs = torch.matmul(atts * embs, self.p)
        # [ batch_size ]
        outs = torch.squeeze(outs)
        # [ batch_size ]
        logit = torch.sigmoid(outs)
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


def train(epochs=20, batchSize=1024, lr=0.02, k=128, t=64, eva_per_epochs=1, need_eva=True):
    # 读取数据
    train_triples, test_triples, user_df, item_df, n_features = \
        dataloader4ml100kIndexs.read_data()
    # 初始化模型
    net = AFM(n_features, user_df, item_df, k, t)
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
