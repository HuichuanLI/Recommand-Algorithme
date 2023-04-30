from data_set import filepaths as fp
from torch.utils.data import DataLoader
import torch
import os
import random
import numpy as np
from utils import topKevaluate as tke
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,roc_auc_score
from chapter3 import s22_FNN_plus as fnnTrain,s24_DeepFM as deepfmTrain,s25_AFM as afmTrain
from chapter2 import dataloader4ml100kIndexs
import collections

random.seed(2)

afm_model_path = os.path.join( fp.Model_Dir, 'afm.model' )
fnn_model_path = os.path.join( fp.Model_Dir, 'fnn.model' )
deepfm_model_path = os.path.join( fp.Model_Dir, 'deepfm.model' )

model_paths = {
    'afm' : afm_model_path,
    'fnn' : fnn_model_path,
    'deepfm' : deepfm_model_path
}
modelsTrain = {
    'afm': afmTrain,
    'fnn': fnnTrain,
    'deepfm': deepfmTrain
}

# 模型训练
def train( ):
    for model in modelsTrain:
        print( 'train {}'.format( model ) )
        net = modelsTrain[ model ].train( need_eva = False )
        torch.save( net, model_paths[ model ] )

# 进行普通的模型评估
def doModelEva():
    _, test_set, _, _, _ = \
        dataloader4ml100kIndexs.read_data( )
    test_set = torch.LongTensor( test_set )
    for model in model_paths:
        net =  torch.load( model_paths[ model ] )
        u, i, r = test_set[:, 0], test_set[:, 1], test_set[:, 2]
        y_true = r.detach( ).numpy( )
        net.eval( )
        with torch.no_grad( ):
            out = net( u, i )
        auc = roc_auc_score( y_true, out )
        y_pred = np.array( [ 1 if i >= 0.5 else 0 for i in out ] )
        p = precision_score( y_true, y_pred )
        r = recall_score( y_true, y_pred )
        print('model:{}, p:{:.4f}, r:{:.4f}, auc:{:.4f}'.format( model, p, r, auc ) )

#进行topK评估
def doTopKEva( ks = [ 1,2,5,10,20,50,100 ] ):
    _, test_set, _, _, _ = \
        dataloader4ml100kIndexs.read_data( )
    pos_dct, neg_dct, all_items = tke.fromTripleToSetDict( test_set )
    pairs, pos_set, neg_set = tke.fromSetDictToBocastTestPair(pos_dct, neg_dct, all_items)
    all_dict = {}
    for model in model_paths:
        net = torch.load( model_paths[ model ] )
        all_preds = collections.defaultdict( list )
        for u, i in tqdm( DataLoader( pairs, batch_size = len( all_items ), shuffle = False )):
            out = net( u, i )
            outs = [[ round( float( r ), 4 ), i ] for r, i in zip( out.detach().numpy(), all_items )]
            outs = sorted(outs, reverse=True)
            for k in ks:
                preds = [ o[1] for o in outs[:k] ]
                all_preds[k].append( preds )
        return_dict = tke.doTopKEva( all_preds, pos_set, neg_set )
        all_dict[ model ] = return_dict

    tke.drawPlot( all_dict, ks )
    return all_dict

if __name__ == '__main__':
    #train( )
    #doModelEva( )
    doTopKEva( )

