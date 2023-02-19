import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import s14_RNN_data_prepare as dp
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn
import sys
from data_set import filepaths as fp


class RNN_rec(nn.Module):

    def __init__(self, n_items, hidden_size=64, dim=128):
        super(RNN_rec, self).__init__()
        # 随机初始化所有物品向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        self.rnn = nn.RNN(dim, hidden_size, batch_first=True)
        self.dense = self.dense_layer(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()

    # 全连接层
    def dense_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid())

    def forward(self, x, isTrain=True):
        # [batch_size, len_seqs, dim]
        item_embs = self.items(x)
        # [1, batch_size, hidden_size]
        _, h = self.rnn(item_embs)
        # [batch_size, hidden_size]
        h = torch.squeeze(h)
        # 训练时采取dropout来防止过拟合
        if isTrain: h = F.dropout(h)
        # [batch_size, 1]
        out = self.dense(h)
        # [batch_size]
        out = torch.squeeze(out)
        # logit = self.sigmoid(out)
        return out


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    x = d[:, :-1]
    y = d[:, -1].float()
    with torch.no_grad():
        out = net(x)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return precision, recall, acc


def train(epochs=10, batchSize=1024, lr=0.001, rnn_hidden_size=64, dim=128, eva_per_epochs=1):
    # 读取数据
    train, test, allItems = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    net = RNN_rec(max(allItems) + 1, rnn_hidden_size, dim)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for seq in DataLoader(train, batch_size=batchSize, shuffle=True, ):
            x = torch.LongTensor(seq[:, :-1].detach().numpy())
            y = torch.FloatTensor(seq[:, -1].detach().numpy())
            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(train) // batchSize)))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test)
            print('test:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))


if __name__ == '__main__':
    train()
