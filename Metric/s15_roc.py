from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import random
import numpy as np


def getRandomTrueList(n):
    return [random.choice([0, 1]) for _ in range(n)]


def getRandomPredList(trues, adjust=0.5):
    '''
    :param trues: 模拟真实的坐标
    :param adjust: 调整系数，0-1，越靠近1则预测值更接近真实
    :return: 模拟预测的坐标
    '''
    adjust = adjust - 0.5
    return [random.random() + (adjust if t == 1 else -adjust)
            for t in trues]


def getRocCurveSK(t, p):
    fpr, tpr, thresholds = roc_curve(t, p, pos_label=1)
    return fpr, tpr


def __sepPreds(p, threshold):
    preds = [1 if i >= threshold else 0 for i in p]
    return preds


def __getTP(t, p):
    t[t == 0] = 2
    p[p == 0] = 3
    TP = sum(t == p)
    return TP


def __getFP(t, p):
    t[t == 1] = 2
    p[p == 0] = 3
    p[p == 1] = 0
    FP = sum(t == p)
    return FP


def __getTPR(t, p):
    t = np.array(t.copy())
    p = np.array(p.copy())
    TP = __getTP(t, p)
    return TP / sum(t == 1)


def __getFPR(t, p):
    t = np.array(t.copy())
    p = np.array(p.copy())
    FP = __getFP(t, p)
    return FP / sum(t == 0)


def getRocCurve(t, p):
    '''
    :param t: 真实标注
    :param p: 预测分数
    :return: TPR列表与FPR列表
    '''

    # 将所有预测分数从大到小排序作为阈值集
    thresholds = sorted(p, reverse=True)
    # 加入一个高阈值在首位是为了产生一个 ( 0, 0 )坐标的点
    thresholds.insert(0, 1 + thresholds[0])
    tprs, fprs = [], []
    for th in thresholds:
        # 根据阈值将预测分数切分成0或1的列表
        preds = __sepPreds(p, th)
        FPR = __getFPR(t, preds)
        TPR = __getTPR(t, preds)
        fprs.append(FPR)
        tprs.append(TPR)
    return fprs, tprs


def drawRoc(fprs, tprs):
    plt.figure()
    plt.plot(fprs, tprs, 'black', marker='*', label='ROC')
    # 中间蓝线的坐标

    middle_x = np.linspace(0, 1, len(fprs))
    middle_y = np.linspace(0, 1, len(fprs))
    plt.plot(middle_x, middle_y, 'black', marker='.', label='Diagonal')
    plt.legend(fontsize=12, loc='upper left')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    trues = getRandomTrueList(100)
    preds = getRandomPredList(trues, 0.8)

    fprs, tprs = getRocCurve(trues, preds)
    drawRoc(fprs, tprs)

    fprs, tprs = getRocCurveSK(trues, preds)
    drawRoc(fprs, tprs)
