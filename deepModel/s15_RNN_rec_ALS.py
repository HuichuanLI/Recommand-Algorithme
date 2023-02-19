import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import s14_RNN_data_prepare as dp
from torch.utils.data import DataLoader
import torch
from torch import nn
from data_set import filepaths as fp


class RNN_ALS_rec(nn.Module):

    def __init__(self, n_items, dim=128):
        super(RNN_ALS_rec, self).__init__()
        # 随机初始化所有物品的特征向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        # 因为要进行向量点积运算，所以rnn层的输出向量维度也需与物品向量一致
        self.rnn = nn.RNN(dim, dim, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, item):
        # [batch_size, len_seqs, dim]
        item_embs = self.items(x)
        # [1, batch_size, dim]
        _, h = self.rnn(item_embs)
        # [batch_size, dim]
        h = torch.squeeze(h)
        # [batch_size, dim]
        one_item = self.items(item)
        # [batch_size]
        out = torch.sum(h * one_item, dim=1)
        logit = self.sigmoid(out)
        return logit


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    x = d[:, :-2]
    item = d[:, -2]
    y = d[:, -1].float()

    with torch.no_grad():
        out = net(x, item)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return precision, recall, acc


def train(epochs=10, batchSize=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train, test, allItems = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    net = RNN_ALS_rec(max(allItems) + 1, dim)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for seq in DataLoader(train, batch_size=batchSize, shuffle=True, ):
            x = torch.LongTensor(seq[:, :-2].detach().numpy())
            item = torch.LongTensor(seq[:, -2].detach().numpy())
            y = torch.FloatTensor(seq[:, -1].detach().numpy())
            optimizer.zero_grad()
            logits = net(x, item)
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
