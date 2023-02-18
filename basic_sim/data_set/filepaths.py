import os

ROOT = os.path.split(os.path.realpath(__file__))[0]
Model_Dir = os.path.join(ROOT,'model')

class Ml_100K():
    #下载地址：https://github.com/rexrex9/kb4recMovielensDataProcess
    __BASE = os.path.join(ROOT, 'ml-100k')
    ORGINAL_DIR = os.path.join(ROOT,'ml-100k-orginal')
    USER_DF = os.path.join(ORGINAL_DIR,'user_df.csv')
    ITEM_DF = os.path.join(ORGINAL_DIR,'item_df.csv')
    ITEM_DF_0 = os.path.join(ORGINAL_DIR, 'item_df_0.csv')

    KG=os.path.join(__BASE,'kg_index.tsv')
    RATING = os.path.join(__BASE,'rating_index.tsv')
    RATING5 = os.path.join(__BASE, 'rating_index_5.tsv')

class Ml_latest_small():
    __BASE = os.path.join(ROOT,'ml-latest-small')
    RATING_TS = os.path.join(__BASE,'rating_index_ts.tsv')
    SEQS = os.path.join(__BASE, 'seqs.npy')

    SEQS_NEG = os.path.join(__BASE, 'seqsWithNeg.npy')


