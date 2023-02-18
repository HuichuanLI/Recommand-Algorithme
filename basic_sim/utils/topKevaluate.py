import collections
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import os


#------------------调整数据部分----------------------------#

# 从三元组数据变形为正负例的字典并顺便返回所有物品集
def fromTripleToSetDict( triples ):
    '''
    :param triples: 三元组数据[[0,1,1],[1,2,0],[1,3,1]...]
    :return: pos_dct: 正例集字典 {uid:{i1,i2,i3}..}
             net_dct: 负例集字典 {uid:{i1,i2,i3}..}
             all_items : 所有物品列表 [i1,i2,i3,i4...]
    '''
    pos_dct, neg_dct = collections.defaultdict( set ), collections.defaultdict( set )
    all_items = set( )
    for u, i, r in triples:
        all_items.add( i )
        if r == 1:
            pos_dct[u].add( i )
        else:
            neg_dct[u].add( i )
    return pos_dct, neg_dct, list( all_items )

# 通过正负例集及所有物品广播生成新的用户物品对
def fromSetDictToBocastTestPair( pos_dct, neg_dct, all_items ):
    pair = [ ]
    pos_set = [ ]
    neg_set = [ ]
    for u in pos_dct:
        pair.extend( [ [ u, i ] for i in all_items ] )
        pos_set.append( pos_dct[ u ] )
        neg_set.append( neg_dct[ u ] )
    return pair, pos_set, neg_set


# 通过正负例集及所有物品广播生成新的用户物品三元组
def fromSetDictToBocastTestTriples( pos_dct, neg_dct, all_items ):
    triples = [ ]
    pos_set = [ ]
    neg_set = [ ]
    for u in pos_dct:
        triples.extend( [ [ u, i , 1 if i in pos_dct[u] else 0 ] for i in all_items ] )
        pos_set.append( pos_dct[ u ] )
        neg_set.append( neg_dct[ u ] )
    return triples, pos_set, neg_set

#---------------------------------------------------#

#-------------------评估指标部分-----------------------#

# 精确率
def precision( pred, pos, t_neg):
    TP = len(set(pred) & set(pos))
    FP = len(set(pred) & set(t_neg))
    p = TP/(TP+FP) if TP+FP > 0 else None
    p_full = TP / len(pred)
    return p, p_full

# 召回率
def recall( pred, pos ):
    TP = len( set( pred ) & set( pos ) )
    r = TP/len( pos )
    return r

# Average Precision
def AP( pred, pos ):
    hits = 0
    sum_precs = 0
    for n in range( len( pred ) ):
        if pred[n] in pos:
            hits += 1
            sum_precs += hits / ( n + 1.0 )
    return sum_precs / len( pos )

# Mean Average Precision
def MAP( preds, poss ):
    ap = 0
    for pred, pos in zip( preds, poss ):
        ap += AP( pred, pos )
    return ap / len( preds )

#item HR
def hit_ratio_for_item( all_preds, all_pos ):
    '''
    :param all_preds: 全部的预测集
    :param all_pos:  全部的正例集
    '''
    return len( all_preds&all_pos )/len( all_pos )

# User HR
def hit_ratio_for_user( pred, pos ):
    '''
    :param pred: 单个用户的预测集
    :param pos:  单个用户的正例集
    '''
    return 1 if len(set( pred )&set( pos )) > 0 else  0

# Reciprocal Rank
def RR( pred, pos ):
    for n in range( len( pred ) ):
        if pred[n] in pos:
            return 1/( n + 1 )
    else:
        return 0

# Mean Reciprocal Rank
def MRR( preds, poss ):
    rr = 0
    for pred, pos in zip( preds, poss ):
        rr += RR( pred, pos )
    return rr / len( preds )

# Discounted Cumulative Gain
def DCG( scores ):
    return np.sum(
        np.divide( np.array( scores ),
                   np.log2( np.arange ( len( scores ) ) + 2 ) ) )

# Normalized Discounted Cumulative Gain
def NDCG( pred, pos ):
    dcg = DCG( [ 1 if i in pos else 0  for i in pred ] )
    idcg = DCG( [ 1 for _ in pred ] )
    return dcg / idcg

#---------------------------------------------------#

#-----------------评估测量部分--------------------------#

#做TopK的测量,精确率，全负精确率，召回率
def doTopKEva( all_preds_dict, poss, negs ):
    return_dict = {
        'precision':[],
        'precision_full': [],
        'recall':[]
    }
    for k in all_preds_dict:
        all_p, all_p_full, all_r = 0, 0, 0
        p_count = 0
        for pred, pos, neg in zip( all_preds_dict[k], poss, negs ):
            p, p_full = precision( pred, pos, neg )
            r = recall( pred, pos )
            if p:
                all_p += p
                p_count += 1
            all_p_full += p_full
            all_r += r
        p = all_p/p_count if p_count>0 else 0
        p_full = all_p_full/len( all_preds_dict[k] )
        r = all_r/len( all_preds_dict[k] )
        print( 'top {}, p:{:.4f}, p_full:{:.4f}, r:{:.4f}'.format( k, p, p_full, r ) )
        return_dict['precision'].append( p )
        return_dict['precision_full'].append( p_full )
        return_dict['recall'].append( r )
    return return_dict


# 评估方法的枚举
class EvaTarget( Enum ):
    precision = ('precision',precision)
    precision_full = ('precision_full',precision)
    recall = ('recall',recall)
    map = ('map',AP)
    mrr = ('mrr',RR)
    ndcg = ('ndcg',NDCG)
    user_hr = ('user_hr',hit_ratio_for_user)
    item_hr = ('item_hr',hit_ratio_for_item)


#做TopK的测量,可选本书所涉及的所有评估指标
def doTopKEvaAll( all_preds_dict, poss, negs,
                  evaMethods= [ EvaTarget.precision,
                                EvaTarget.precision_full,
                                EvaTarget.recall,
                                EvaTarget.map,
                                EvaTarget.mrr,
                                EvaTarget.ndcg,
                                EvaTarget.user_hr,
                                EvaTarget.item_hr
                                ],needPrint=True):

    sp_evas = [ EvaTarget.precision, EvaTarget.precision_full, EvaTarget.item_hr ]
    names = [et.value[0] for et in evaMethods if et not in sp_evas]
    return_dict = collections.defaultdict(list)
    for k in all_preds_dict:
        begin_values = np.zeros(len([et for et in evaMethods if et not in sp_evas]))
        if EvaTarget.item_hr in evaMethods:
            all_preds,all_pos = set(),set()
        if EvaTarget.precision in evaMethods or EvaTarget.precision_full in evaMethods:
            p_count, all_p, all_p_full = 0, 0, 0
        for pred, pos, neg in zip( all_preds_dict[k], poss, negs ):
            if EvaTarget.precision in evaMethods or EvaTarget.precision_full in evaMethods:
                p, p_full = EvaTarget.precision.value[1]( pred, pos, neg )
                if p and EvaTarget.precision in evaMethods:
                    all_p += p
                    p_count += 1
                if EvaTarget.precision_full in evaMethods:
                    all_p_full += p_full
            begin_values += np.array([ et.value[1](pred,pos) for et in evaMethods if et not in sp_evas])
            if EvaTarget.item_hr in evaMethods:
                all_preds|=set(pred)
                all_pos|=set(pos)

        begin_values /= len( all_preds_dict[k] )

        s = 'top {}:\t'.format( k )
        if EvaTarget.precision in evaMethods:
            p = all_p/p_count if p_count > 0 else 0
            return_dict[EvaTarget.precision.value[0]].append(p)
            s += '{}:{:.4f}\t'.format(EvaTarget.precision.value[0], p)
        if EvaTarget.precision_full in evaMethods:
            p_full = all_p_full/len( all_preds_dict[k] )
            return_dict[EvaTarget.precision_full.value[0]].append(p_full)
            s += '{}:{:.4f}\t'.format(EvaTarget.precision_full.value[0], p_full)

        for e,v in zip( names, begin_values ):
            return_dict[ e ].append( v )
            s += '{}:{:.4f}\t'.format( e, v )

        if EvaTarget.item_hr in evaMethods:
            item_hr = EvaTarget.item_hr.value[1]( all_preds, all_pos )
            return_dict[EvaTarget.item_hr.value[0]].append(item_hr)
            s += '{}:{:.4f}\t'.format(EvaTarget.item_hr.value[0], item_hr)

        if needPrint:
            print(s)

    return return_dict

# 调整一下字典结构以方便画图
def __fromModelDct2EvaDct( dct ):
    eva_dct = collections.defaultdict(dict)
    for model in dct:
        for eva in dct[ model ]:
            eva_dct[eva][model] = dct[model][eva]
    return eva_dct

# 画出topK函数图
def drawPlot( dct, ks, saveDirPath=None,needShow=True ):
    eva_dct = __fromModelDct2EvaDct(dct)
    colors = ['red', 'blue', 'green', 'yellow', 'brown', 'hotpink', 'gold','aqua','purple','limegreen','beige']
    #colors = ['black','black','black','black','black','black','black','black','black','black','black']
    markers=['o','*','^','x','P','+','1','s','v','p','X']
    for eva in eva_dct:
        plt.figure()
        index = 0
        for model in eva_dct[eva]:
            y = eva_dct[eva][model]
            plt.plot( ks, y, color = colors[index], marker=markers[index], label=model )
            index+=1
        plt.xlabel('K', fontsize = 14 )
        plt.ylabel('{}@K'.format(eva), fontsize = 14 )
        plt.legend(fontsize = 16,loc = 'upper left')
        plt.grid()
        if saveDirPath:
            path_name = '{}.png'.format(eva)
            plt.savefig(os.path.join(saveDirPath,path_name ))
        if needShow:
            plt.show()

#---------------------------------------------------#




if __name__ =='__main__':
    pred=[1,2,3,4,5,6]
    pos=[6,7,8,9,10,11]

    p = np.array(pred)
    pos = np.array(pos)

    print(p+pos)
    import math
    print(math.log2(2))
    #withShowTime(pred,pos,RR)
    #withShowTime(pred, pos, RR_quick)
