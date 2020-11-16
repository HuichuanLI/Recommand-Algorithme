import numpy as np
from pandas import read_csv

def read_rating(file_path, has_header=True):
    rating_mat = list()
    with open(file_path) as fp:
        if has_header is True:
            fp.readline()
        for line in fp:
            line = line.split(',')
            user, item, rating = line[0], line[1], line[2]
            rating_mat.append( [user, item, rating] )
    return np.array(rating_mat).astype('float32')

def read_feature(file_path):
    feat_mat = read_csv(file_path, sep=',')
    assert( np.all(feat_mat['id'] == feat_mat.index) )
    return feat_mat.drop('id', 1).as_matrix()