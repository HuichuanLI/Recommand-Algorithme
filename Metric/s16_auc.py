import s15_roc
from sklearn.metrics import auc, roc_auc_score
import numpy as np


def getRocAuc(t, p):
    fpr, tpr = s15_roc.getRocCurveSK(t, p)
    return getAuc(fpr, tpr)


# 根据fprs,tprs获得Auc
def getAuc(fprs, tprs):
    dx = np.diff(fprs)
    auc = sum(dx * tprs[1:])
    return auc


if __name__ == '__main__':
    t = s15_roc.getRandomTrueList(100)
    p = s15_roc.getRandomPredList(t)

    fprs, tprs = s15_roc.getRocCurveSK(t, p)
    print(getAuc(fprs, tprs))
    print(auc(fprs, tprs))

    print(getRocAuc(t, p))
    print(roc_auc_score(t, p))
