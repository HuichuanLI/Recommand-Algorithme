import numpy as np
from os.path import exists
import re

def read_rating(path,data_name, num_users, num_items,num_total_ratings, a, b, test_fold,random_seed):

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    R = np.zeros((num_users,num_items))
    mask_R = np.zeros((num_users, num_items))
    C = np.ones((num_users, num_items)) * b

    train_R = np.zeros((num_users, num_items))
    test_R = np.zeros((num_users, num_items))

    train_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    if (data_name == 'politic_new') or (data_name == 'politic_old'):
        num_train_ratings = 0
        num_test_ratings = 0

        train_file_name = 'Train_ratings_fold_' + str(test_fold)
        test_file_name = 'Test_ratings_fold_' + str(test_fold)

        ''' load train fold '''
        with open(path + train_file_name) as f1:
            lines = f1.readlines()
            for line in lines:
                user, item, voting = line.split("\t")
                user = int(user)
                item = int(item)
                voting = int(voting)
                if voting == -1:
                    voting = 0

                ''' Total '''
                R[user, item] = voting
                mask_R[user, item] = 1

                ''' Train '''
                train_R[user, item] = int(voting)
                train_mask_R[user, item] = 1
                C[user, item] = a

                user_train_set.add(user)
                item_train_set.add(item)
                num_train_ratings = num_train_ratings + 1

        ''' load test fold '''
        with open(path + test_file_name) as f2:
            lines = f2.readlines()
            for line in lines:
                user, item, voting = line.split("\t")
                user = int(user)
                item = int(item)
                voting = int(voting)
                if voting == -1:
                    voting = 0

                ''' Total '''
                R[user, item] = voting
                mask_R[user, item] = 1

                ''' Test '''
                test_R[user, item] = int(voting)
                test_mask_R[user, item] = 1

                user_test_set.add(user)
                item_test_set.add(item)

                num_test_ratings = num_test_ratings + 1

    assert num_train_ratings == np.sum(train_mask_R)
    assert num_test_ratings == np.sum(test_mask_R)
    assert num_total_ratings == num_train_ratings + num_test_ratings

    return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set

def read_trust(path,data_name, num_users):
    if (data_name == 'politic_new') or (data_name == 'politic_old'):
        T = np.load(path + "user_user_matrix.npy")
    else:
        raise NotImplementedError("ERROR")
    return T

def read_bill_term(path,data_name,num_items,num_voca):
    file = path + 'Bill_Term'
    X_dw = np.zeros((num_items,num_voca))
    with open(file,'r') as f:
        contents = f.readlines()
        for line in contents:
            elements = line.split('\t')
            d = int(elements[0])
            w = int(elements[1])
            frequency = int(elements[2])
            X_dw[d,w] = frequency
    return X_dw
