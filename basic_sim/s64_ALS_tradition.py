from data_set import filepaths as fp
from tqdm import tqdm
from utils import evaluate
import dataloader
import numpy as np


class ALS():

    def __init__(self, n_users, n_items, dim):
        '''
        :param n_users: 用户数量
        :param n_items: 物品数量
        :param dim: 隐因子数量或者称隐向量维度
        '''
        # 首先初始化用户矩阵，物品矩阵，用户偏置项及物品偏置项
        self.p = np.random.uniform(size=(n_users, dim))
        self.q = np.random.uniform(size=(n_items, dim))
        self.bu = np.random.uniform(size=(n_users, 1))
        self.bi = np.random.uniform(size=(n_items, 1))

    def forward(self, u, i):
        '''
        :param u: 用户id shape:[batch_size]
        :param i: 物品id shape:[batch_size]
        :return: 预测的评分 shape:[batch_size,1]
        '''
        return np.sum(self.p[u] * self.q[i], axis=1, keepdims=True) + self.bu[u] + self.bi[i]

    def backword(self, r, r_pred, u, i, lr, lamda):
        '''
        反向传播方法，根据梯度下降的方法迭代模型参数
        :param r: 真实评分 shape:[batch_size, 1]
        :param r_pred: 预测评分 shape:[batch_size, 1]
        :param u: 用户id shape:[batch_size]
        :param i: 物品id shape:[batch_size]
        :param lr: 学习率
        :param lamda: 正则项系数
        '''
        loss = r - r_pred
        self.p[u] += lr * (loss * self.q[i] - lamda * self.p[u])
        self.q[i] += lr * (loss * self.p[u] - lamda * self.q[i])
        self.bu[u] += lr * (loss - lamda * self.bu[u])
        self.bi[i] += lr * (loss - lamda * self.bi[i])


# 测试RMSE
def evaluateRMSE(testSet, als):
    testSet = np.array(testSet)
    u = testSet[:, 0]
    i = testSet[:, 1]
    r = testSet[:, 2]
    hat = als.forward(u, i)
    hat = hat.reshape(-1)
    return evaluate.RMSE(r, hat)


# 一个数据迭代器
class DataIter():

    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    # 每次返回batch_size个数据
    def iter(self, batch_size):
        for _ in range(len(self.dataset) // batch_size):
            np.random.shuffle(self.dataset)
            yield self.dataset[:batch_size]


def train(epochs=20, batchSize=1024, lr=0.01, lamda=0.1, factors_dim=64):
    '''
    :param epochs: 迭代次数
    :param batchSize: 一批次的数量
    :param lr: 学习率
    :param lamda: 正则系数
    :param factors_dim: 隐因子数量
    :return:
    '''
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING5, test_ratio=0.1)

    # 初始化ALS模型
    als = ALS(len(user_set), len(item_set), factors_dim)
    # 初始化批量提出数据的迭代器
    dataIter = DataIter(train_set)

    for e in range(epochs):
        for batch in tqdm(dataIter.iter(batchSize)):
            # 将用户id,物品id,评分从三元组中拆出
            u = batch[:, 0]
            i = batch[:, 1]
            r = batch[:, 2].reshape(-1, 1)  # 形状变一变是为了方便等会广播计算
            # 得到预测评分
            r_pred = als.forward(u, i)
            # 根据梯度下降迭代
            als.backword(r, r_pred, u, i, lr, lamda)

        print("TrainSet: Epoch {} | RMSE {:.4f} ".format(e, evaluateRMSE(train_set, als)))
        print("TestSet: Epoch {} | RMSE {:.4f} ".format(e, evaluateRMSE(test_set, als)))


if __name__ == '__main__':
    train()
