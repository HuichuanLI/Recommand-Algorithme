# -*-coding:utf8-*-
"""
author:huichuan
date:2019****
train gbdt model
"""
import xgboost as xgb

from get_feature_num import get_feature_num
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix


def get_train_data(train_file, feature_num_file):
    """
    get train data and label for training
    """
    total_feature_num = get_feature_num(feature_num_file)
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=-1)
    feature_list = range(total_feature_num)
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=feature_list)
    return train_feature, train_label


def train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate):
    """
    Args:
        train_mat:train data and label
        tree_depth:
        tree_num:total tree num
        learning_rate: step_size
    Return:Booster
    """
    para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
    bst = xgb.train(para_dict, train_mat, tree_num)
    # grid_search(train_mat)
    return bst


def choose_parameter():
    """
    Return:
         list: such as [(tree_depth, tree_num, step_size),...]
    """
    result_list = []
    tree_depth_list = [4, 5, 6]
    tree_num_list = [10, 50, 100]
    learning_rate_list = [0.3, 0.5, 0.7]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                result_list.append((ele_tree_depth, ele_tree_num, ele_learning_rate))
    return result_list


def grid_search(train_mat):
    """
    Args:
        train_mat: train data and train label
    select the best parameter for training model
    """
    para_list = choose_parameter()
    for ele in para_list:
        (tree_depth, tree_num, learning_rate) = ele
        para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
        res = xgb.cv(para_dict, train_mat, tree_num, nfold=5, metrics={'auc'})
        auc_score = res.loc[tree_num - 1, ['test-auc-mean']].values[0]
        print("tree_depth:%s,tree_num:%s, learning_rate:%s, auc:%f" \
              % (tree_depth, tree_num, learning_rate, auc_score))


def train_tree_model(train_file, feature_num_file, tree_model_file):
    """
    Args:
        train_file: data for train model
        tree_model_file: file to store model
        feature_num_file:file to record feature total num
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    tree_num = 10
    tree_depth = 6
    learning_rate = 0.3
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(tree_model_file)


def get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth):
    """
    Args:
        tree_leaf: prediction of the tree model
        tree_num:total_tree_num
        tree_depth:total_tree_depth
    Return:
         Sparse Matrix to record total train feature for lr part of mixed model
    """
    total_node_num = 2 ** (tree_depth + 1) - 1
    yezi_num = 2 ** tree_depth
    feiyezi_num = total_node_num - yezi_num
    total_col_num = yezi_num * tree_num
    total_row_num = len(tree_leaf)
    col = []
    row = []
    data = []
    base_row_index = 0
    for one_result in tree_leaf:
        base_col_index = 0
        for fix_index in one_result:
            yezi_index = fix_index - feiyezi_num
            yezi_index = yezi_index if yezi_index >= 0 else 0
            col.append(base_col_index + yezi_index)
            row.append(base_row_index)
            data.append(1)
            base_col_index += yezi_num
        base_row_index += 1

    total_feature_list = coo_matrix((data, (row, col)), shape=(total_row_num, total_col_num))
    return total_feature_list


def train_tree_and_lr_model(train_file, feature_num_file, mix_tree_model_file, mix_lr_model_file):
    """
    Args:
        train_file:file for training model
        feature_num_file:file to store total feature len
        mix_tree_model_file: tree part of the mix model
        mix_lr_model_file:lr part of the mix model
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    # 实战中最好数据num:维度1：100
    (tree_depth, tree_num, learning_rate) = 6, 10, 0.3
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(mix_tree_model_file)
    tree_leaf = bst.predict(train_mat, pred_leaf=True)
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth)
    lr_clf = LRCV(Cs=[1.0], penalty='l2', dual=False, tol=0.0001, max_iter=500, scoring='roc_auc', cv=5).fit(
        total_feature_list, train_label)
    scores = lr_clf.scores_[1]
    print("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis=0)])))
    print("Accuracy:%s (+-%0.2f)" % (scores.mean(), scores.std() * 2))
    fw = open(mix_lr_model_file, "w+")
    coef = lr_clf.coef_[0]
    fw.write(','.join([str(ele) for ele in coef]))


if __name__ == "__main__":
    train_tree_and_lr_model("../data/train_file", "../data/feature_num", "../data/xgb_mix_model",
                            "../data/lr_coef_mix_model")
