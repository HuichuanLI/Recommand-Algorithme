import torch
from torch import nn
from torch.nn import Parameter, init
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dataloader4ml100kOneHot


class FM(nn.Module):
    def __init__(self, n_features, dim):
        '''
        :param n_features: 特征数量
        :param dim: 隐向量维度
        '''
        super(FM, self).__init__()
        self.w0 = init.xavier_uniform_(Parameter(torch.empty(1, 1)))
        self.w1 = init.xavier_uniform_(Parameter(torch.empty(n_features, 1)))
        self.w2 = init.xavier_uniform_(Parameter(torch.empty(n_features, dim)))

    # FM交叉相乘
    def FMcross(self, x):
        # [batch_size, dim]
        square_of_sum = torch.matmul(x, self.w2) ** 2
        # [batch_size, dim]
        sum_of_square = torch.matmul(x ** 2, self.w2 ** 2)

        output = square_of_sum - sum_of_square
        output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output

    def forward(self, x):
        lr_out = self.w0 + torch.matmul(x, self.w1)
        cross_out = self.FMcross(x)
        logits = torch.sigmoid(lr_out + cross_out)
        return logits


# 做评估
def doEva(net, x, y):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    with torch.no_grad():
        out = net(x)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = y
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc


def train(epochs=20, batchSize=1024, lr=0.01, dim=128, eva_per_epochs=1):
    # 读取数据
    x_train, x_test, y_train, y_test = dataloader4ml100kOneHot.read_data()
    # 得到特征数量
    features = len(x_train[0])
    # 初始化模型
    net = FM(features, dim)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 初始化数据迭代器
    dataIter = dataloader4ml100kOneHot.DataIter(x_train, y_train)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for datas in dataIter.iter(batchSize=batchSize):
            Xs = torch.FloatTensor([d[0] for d in datas])
            labels = torch.FloatTensor([d[1] for d in datas])
            optimizer.zero_grad()
            logits = net(Xs)
            loss = criterion(logits, labels)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(y_train) // batchSize)))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, x_train, y_train)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, x_test, y_test)
            print('test:p:{:.4f}, r:{:.4f}, acc:{:.4f}'.format(p, r, acc))


if __name__ == '__main__':
    train()
