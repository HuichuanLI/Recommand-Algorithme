import numpy as np
from data_set import filepaths as fp
import pandas as pd


def doSeq(x, seqs):
    single = []
    for _, i in x.iterrows():
        u, i, r, t = i
        if len(single) == 5:
            single.append(i)
            single.append(r)
            seqs.append(single)
            single = single[1:5]
        if r == 1:
            single.append(i)


def genrateRecSeq(inPath, outPath):
    '''
    生成序列，序列会由6个物品id 加 1个标注组成
    例如：
    [1973,5995,560,5550,6517,4620,1],
    [5995,560,5550,6517,4620,4563,1],
    [560,5550,6517,4620,4563,1314,1],
    [2439,1600,7999,1743,8282,8204,0]
    前5个物品id代表用户最近点击的物品id,第6个物品id代表用户当前观看的物品，
    第7位的标注即代表用户真实点击情况，1为点击，0为未点击。
    '''
    df = pd.read_csv(inPath, sep='\t', header=None)
    df = df.sort_values(by=[0], axis=0)
    seqs = []
    df.groupby(0).apply(lambda x: doSeq(x, seqs))
    seqs = np.array(seqs)
    np.save(outPath, seqs)


def doSeqWithNeg(x, seqs):
    singleSeq, singleTarg = [], []
    for _, i in x.iterrows():
        u, i, r, t = i
        if len(singleSeq) == 6:
            single = [singleSeq, singleTarg]
            seqs.append(single)
            singleSeq = singleSeq[1:]
            singleTarg = singleTarg[1:]
        singleSeq.append(i)
        singleTarg.append(r)


def genrateRecSeqWithNeg(inPath, outPath):
    '''
    生成物品id序列和标注序列
    物品id序列是由6个物品id组成，标注序列由1和0组成，1为正例，代表用户喜欢对应位置的物品。
    0为负例，代表用户不喜欢对应位置的物品。数据样例如下：
    [[1973,5995,560,5550,6517,4620],[1,1,1,1,0,1]]
    [5995,560,5550,6517,4620,4563],[0,1,1,1,1,1]]
    [560,5550,6517,4620,4563,1314], [1,1,1,0,1,0]]
    [2439,1600,7999,1743,8282,8204],[1,0,1,1,0,1]
    '''
    df = pd.read_csv(inPath, sep='\t', header=None)
    df = df.sort_values(by=[0, 3], axis=0)
    seqs = []
    df.groupby(0).apply(lambda x: doSeqWithNeg(x, seqs))
    seqs = np.array(seqs)
    np.save(outPath, seqs)


def getTrainAndTestSeqs(inPath, test_ratio=0.1):
    seqs = np.load(inPath)

    allItems = set()
    for seq in seqs:
        allItems |= set(seq[:-1])

    np.random.shuffle(seqs)
    split_number = int(len(seqs) * test_ratio)
    test = seqs[:split_number]
    train = seqs[split_number:]
    return train, test, allItems


def getTrainAndTestSeqsWithNeg(seqInPath, test_ratio=0.1):
    seqs = np.load(seqInPath)

    allItems = set()
    for seq in seqs:
        allItems |= set(seq[0])

    np.random.shuffle(seqs)
    split_number = int(len(seqs) * test_ratio)
    test = seqs[:split_number]
    train = seqs[split_number:]
    return train, test, allItems


if __name__ == '__main__':
    # genrateRecSeq(fp.Ml_latest_small.RATING_TS,fp.Ml_latest_small.SEQS)
    # print(getTrainAndTestSeqs(fp.Ml_latest_small.SEQS))
    #
    # genrateRecSeqWithNeg(fp.Ml_latest_small.RATING_TS,fp.Ml_latest_small.SEQS_NEG)

    train, test, allItems = getTrainAndTestSeqsWithNeg(fp.Ml_latest_small.SEQS_NEG)
    print(train)
