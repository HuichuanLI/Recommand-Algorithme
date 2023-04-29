import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import s14_RNN_data_prepare as dp

from torch.utils.data import DataLoader
import torch
from torch import nn
from data_set import filepaths as fp
from tqdm import tqdm

import s47_transformerOnlyEncoder as TE


class BST(nn.Module):

    def __init__(self, n_items, all_seq_lens, e_dim=128, n_heads=3, n_layers=2):
        '''
        :param n_items: 总物品数量
        :param all_seq_lens: 序列总长度，包含历史物品序列及目标物品
        :param e_dim: 向量维度
        :param n_heads: Transformer中多头注意力层的头目数
        :param n_layers: Transformer中的encoder_layer层数
        '''
        super(BST, self).__init__()
        self.items = nn.Embedding(n_items, e_dim, max_norm=1)
        self.transformer_encoder = TE.TransformerEncoder(e_dim, e_dim // 2, n_heads, n_layers)
        self.mlp = self.__MLP(e_dim * all_seq_lens)

    def __MLP(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(dim // 2, dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid())

    def forward(self, x, target_item):
        # [ batch_size, seqs_len, dim ]
        item_embs = self.items(x)
        # [ batch_size, 1, dim ]
        one_item = torch.unsqueeze(self.items(target_item), dim=1)
        # [ batch_size, all_seqs_len, dim ]
        all_item_embs = torch.cat([item_embs, one_item], dim=1)
        # [ batch_size, all_seqs_len, dim ]
        all_item_embs = self.transformer_encoder(all_item_embs)
        # [ batch_size, all_seqs_len * dim ]
        all_item_embs = torch.flatten(all_item_embs, start_dim=1)
        # [ batch_size, 1 ]
        logit = self.mlp(all_item_embs)
        # [ batch_size ]
        logit = torch.squeeze(logit)
        return logit


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    x = d[:, :-2]
    item = d[:, -2]
    y = torch.FloatTensor(d[:, -1].detach().numpy())
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
    all_seq_lens = len(train[0]) - 1
    # 初始化模型
    net = BST(max(allItems) + 1, all_seq_lens, dim)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for seq in tqdm(DataLoader(train, batch_size=batchSize, shuffle=True)):
            x = torch.LongTensor(seq[:, :-2].detach().numpy())
            target_item = torch.LongTensor(seq[:, -2].detach().numpy())
            y = torch.FloatTensor(seq[:, -1].detach().numpy())
            optimizer.zero_grad()
            logits = net(x, target_item)
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
