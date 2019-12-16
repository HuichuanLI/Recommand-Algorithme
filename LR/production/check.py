# -*-coding:utf8-*-
"""
author:huichuan
date:2019****
use lr model to check the performance in test file
"""

# 1 0.9
# 1 0.8
# 1 0.3
# 0 0.2
# 0 0.4
#
# auc = 2+2+1/3*2
import numpy as np
from sklearn.externals import joblib
import math


def get_test_data(test_file, feature_num_file):
    """
    Args:
        test_file:file to check performance
        feature_num_file: the file record total num of feature
    Return:
         two np array: test _feature, test_label
    """
    total_feature_num = 118
    test_label = np.genfromtxt(test_file, dtype=np.float32, delimiter=",", usecols=-1)
    feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype=np.float32, delimiter=",", usecols=feature_list)
    return test_feature, test_label


def predict_by_lr_model(test_feature, lr_model):
    """
    predict by lr_model
    """
    result_list = []
    prob_list = lr_model.predict_proba(test_feature)
    for index in range(len(prob_list)):
        result_list.append(prob_list[index][1])
    return result_list


def sigmoid(x):
    """
    sigmoid function
    """
    return 1 / (1 + math.exp(-x))


def predict_by_lr_coef(test_feature, lr_coef):
    """
    predict by lr_coef
    universe functions 每个函数
    """
    sigmoid_func = np.frompyfunc(sigmoid, 1, 1)
    return sigmoid_func(np.dot(test_feature, lr_coef))


def get_auc(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of  test data
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num
    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    sorted_total_list = sorted(total_list, key=lambda ele: ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label, predict_score = zuhe
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index += count
        count += 1
    auc_score = (total_pos_index - (pos_num) * (pos_num + 1) / 2) / (pos_num * neg_num)
    print("auc:%.5f" % (auc_score))


def get_accuary(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of test data
    """
    score_thr = 0.5
    right_num = 0
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[index]:
            right_num += 1
    total_num = len(predict_list)
    accuary_score = right_num / total_num
    print("accuary:%.5f" % (accuary_score))


def run_check_core(test_feature, test_label, model, score_func):
    """
    Args:
        test_feature:
        test_label:
        model: lr_coef, lr_model
        score_func: use different model to predict
    """
    predict_list = score_func(test_feature, model)
    get_auc(predict_list, test_label)
    get_accuary(predict_list, test_label)


def run_check(test_file, lr_coef_file, lr_model_file, feature_num_file):
    """
    Args:
        test_file: file to check performace
        lr_coef_file: w1,w2
        lr_model_file: dump file
        feature_num_file: file to record num of feature
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    lr_coef = np.genfromtxt(lr_coef_file, dtype=np.float32, delimiter=",")
    lr_model = joblib.load(lr_model_file)
    run_check_core(test_feature, test_label, lr_model, predict_by_lr_model)
    run_check_core(test_feature, test_label, lr_coef, predict_by_lr_coef)


if __name__ == "__main__":
    run_check("../data/test_file", "../data/lr_coef", "../data/lr_model_file", "../data/feature_num")
