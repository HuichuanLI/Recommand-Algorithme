# -*-coding:utf8-*-
"""
author:Huichuan
date:2019****
feature selection and data selection
"""

import pandas as pd
import numpy as np


def get_input(input_train_file, input_test_file):
    """
    Args:
        input_train_file:
        input_test_file:
    Return:
         pd.DataFrame train_data
         pd.DataFrame test_data
    """
    dtype_dict = {"age": np.int32,
                  "education-num": np.int32,
                  "capital - gain": np.int32,
                  "capital - loss": np.int32,
                  "hours - per - week": np.int32}
    use_list = [i for i in range(15)]
    use_list.pop(2)
    train_data_df = pd.read_csv(input_train_file, sep=",", header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    train_data_df = train_data_df.dropna(axis=0, how="any")
    test_data_df = pd.read_csv(input_test_file, sep=",", header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    test_data_df = test_data_df.dropna(axis=0, how="any")
    return train_data_df, test_data_df


def ana_train_data(input_train_data, input_test_data, out_train_file, out_test_file, feature_num_file):
    """
    Args:
        input_train_data:
        input_test_data:
        out_train_file:
        out_test_file:
        feature_num_file:
    """
    train_data_df, test_data_df = get_input(input_train_data, input_test_data)
    print(train_data_df.columns)
    process_dis_feature("workclass", train_data_df, test_data_df)

    return train_data_df, test_data_df


def label_trans(x):
    """
    Args:
        x: each element in fix col of df
    """
    if x == "<=50K":
        return "0"
    if x == ">50K":
        return "1"
    return "0"


def process_label_feature(lable_feature_str, df_in):
    """
    Args:
        lable_feature_str:"label"
        df_in:DataFrameIn
    """
    df_in.loc[:, lable_feature_str] = df_in.loc[:, lable_feature_str].apply(label_trans)
    process_label_feature("label", train_data_df)
    process_label_feature("label", test_data_df)


def dict_trans(dict_in):
    """
    Args:
        dict_in: key str, value int
    Return:
        a dict, key str, value index for example 0,1,2
    """
    output_dict = {}
    index = 0
    for zuhe in sorted(dict_in.items(), key=lambda x: x[1], reverse=True):
        output_dict[zuhe[0]] = index
        index += 1
    return output_dict


def dis_to_feature(x, feature_dict):
    """
    Args:
        x: element
        feature_dict: pos dict
    Return:
        a str as "0,1,0"
    """
    output_list = [0] * len(feature_dict)
    if x not in feature_dict:
        return ",".join([str(ele) for ele in output_list])
    else:
        index = feature_dict[x]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])


def process_dis_feature(feature_str, df_train, df_test):
    """
        Args:
        feature_str: feature_str
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of the feature output
    process dis feature for lr train
    """
    origin_dict = df_train.loc[:, feature_str].value_counts().to_dict()
    feature_dict = dict_trans(origin_dict)
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(dis_to_feature, args=(feature_dict,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(dis_to_feature, args=(feature_dict,))


if __name__ == "__main__":
    train_data_df, test_data_df = ana_train_data("../data/train.txt", "../data/test.txt", "", "", "")
    # process_label_feature("")
