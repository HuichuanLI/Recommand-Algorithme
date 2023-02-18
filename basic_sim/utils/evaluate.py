from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import numpy as np


def precision( y_true, y_pred ):
    return precision_score( y_true, y_pred )

def precision4Set( test_pos_set, test_neg_set, pred_set ):
    '''
    :param test_pos_set: 真实的用户喜爱的物品集合{iid1,iid2,iid3}
    :param test_neg_set: 真实的用户不喜爱的物品集合{iid1,iid2,iid3}
    :param pred_set: 预测的推荐集合{iid2,iid3,iid4}
    :return: 精确率
    '''
    TP = len( pred_set & test_pos_set )
    FP = len( pred_set & test_neg_set )
    # 若推荐列表和真实的正负例样本均无交集，则返回none
    p = TP / (TP + FP) if TP + FP > 0 else None
    # p = TP/len(pred_set) #若对模型严格一点可这么去算精确度
    return p

def recall( y_true, y_pred ):
    return recall_score( y_true, y_pred )

def recall4Set( test_set, pred_set ):
    '''
    :param test_set:真实的用户喜爱的物品集合{iid1,iid2,iid3}
    :param pred_set: 预测的推荐集合{iid2,iid3,iid4}
    :return: 召回率
    '''
    #计算它们的交集数量 除以 测试集的数量 即可
    return len(pred_set & test_set)/(len(test_set))

def auc(y_true,y_scores):
    return roc_auc_score(y_true,y_scores)

def accuracy(y_true,y_scores):
    return accuracy_score(y_true,y_scores)

def MSE(y_true, y_pred):
    return np.average((np.array(y_true) - np.array(y_pred)) ** 2)

def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred) ** 0.5

def MAE(y_true,y_pred):
    return np.average(abs(np.array(y_true) - np.array(y_pred)))


