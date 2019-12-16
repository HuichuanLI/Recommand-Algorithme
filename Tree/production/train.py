# -*-coding:utf8-*-
"""
author:david
date:2019****
train gbdt model
"""
import xgboost as xgb

from get_feature_num import get_feature_num
import numpy as np


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
    print(xgb.cv(para_dict, train_mat, tree_num, nfold=5, metrics={'auc'}))
    return bst


def train_tree_model(train_file, feature_num_file, tree_model_file):
    """
    Args:
        train_file: data for train model
        tree_model_file: file to store model
        feature_num_file:file to record feature total num
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    tree_num = 5
    tree_depth = 4
    learning_rate = 0.3
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)


if __name__ == "__main__":
    train_tree_model("../data/train_file", "../data/feature_num", "../data/xgb.model")
