import os
import numpy as np
from data_set import filepaths as fp
import random

base_path = fp.Ml_100K.ORGINAL_DIR
train_path = os.path.join(base_path, 'ua.base')
test_path = os.path.join(base_path, 'ua.test')
user_path = os.path.join(base_path, 'u.user')
item_path = os.path.join(base_path, 'u.item')
occupation_path = os.path.join(base_path, 'u.occupation')


def get1or0(r):
    return 1.0 if r > 3 else 0.0


def __read_rating_data(path):
    dataSet = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('\t')
            dataSet[(int(d[0]), int(d[1]))] = [get1or0(int(d[2]))]
    return dataSet


def __read_item_hot():
    items = {}
    with open(item_path, 'r', encoding='ISO-8859-1') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            items[int(d[0])] = np.array(d[5:], dtype='float64')
    return items


def __read_age_hot():
    age_levels = set()
    with open(user_path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            age_level = int(d[1]) // 10
            age_levels.add(age_level)
    age_level_sort = sorted(list(age_levels))
    age_level_dict = {}
    for i in range(len(age_level_sort)):
        l = np.zeros(len(age_level_sort), dtype='float64')
        l[i] = 1
        age_level_dict[age_level_sort[i]] = l
    return age_level_dict


def __read_occupation_hot():
    occupations = {}
    with open(occupation_path, 'r') as f:
        names = f.read().strip().split('\n')
    length = len(names)
    for i in range(length):
        l = np.zeros(length, dtype='float64')
        l[i] = 1
        occupations[names[i]] = l
    return occupations


def __read_user_hot():
    users = {}
    gender_dict = {'M': 1, 'F': 0}
    age_dict = __read_age_hot()
    occupation_dict = __read_occupation_hot()

    with open(user_path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            ages = age_dict[int(d[1]) // 10]
            a = np.append(ages, gender_dict[d[2]])
            users[int(d[0])] = np.append(a, occupation_dict[d[3]])
    return users


def read_dataSet(user_dict, item_dict, path):
    X, Y = [], []
    ratings = __read_rating_data(path)
    for k in ratings:
        X.append(list(np.append(user_dict[k[0]], item_dict[k[1]])))
        Y.append(ratings[k])
    return X, Y


def read_data():
    user_dict = __read_user_hot()
    item_dict = __read_item_hot()
    x_train, y_train = read_dataSet(user_dict, item_dict, train_path)
    x_test, y_test = read_dataSet(user_dict, item_dict, test_path)

    return x_train, x_test, y_train, y_test


class DataIter():

    def __init__(self, trainX, trainY):
        self.trainSet = self.__getMixTrainSet(trainX, trainY)

    def __getMixTrainSet(self, trainX, trainY):
        trainSet = [[list(x), y] for x, y in zip(trainX, trainY)]
        return trainSet

    def iter(self, batchSize):
        for i in range(len(self.trainSet) // batchSize):
            dataSet = random.sample(self.trainSet, batchSize)
            yield dataSet


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = read_data()

    print(x_train[:5])
    print(y_train[:5])

    print(len(x_train[0]))
    print(len(x_train))
    print(len(x_test))
