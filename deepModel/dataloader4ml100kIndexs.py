import os
import numpy as np
from data_set import filepaths as fp
import pandas as pd

base_path = fp.Ml_100K.ORGINAL_DIR
train_path = os.path.join(base_path,'ua.base')
test_path = os.path.join(base_path,'ua.test')
user_path = os.path.join(base_path,'u.user')
item_path = os.path.join(base_path,'u.item')
occupation_path = os.path.join(base_path,'u.occupation')


def __read_age_index():
    age_levels = set()
    with open(user_path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            age_level = int(d[1])//10
            age_levels.add(age_level)
    return len(age_levels)

def __read_occupation_index(begin):
    occupations = {}
    with open(occupation_path,'r') as f:
        names = f.read().strip().split('\n')
    for name in names:
        occupations[name]=begin
        begin+=1
    return occupations,begin

def generate_user_df():
    begin = __read_age_index()
    gender_dict = { 'M':begin, 'F':begin+1 }
    begin += 2
    occupation_dict,begin = __read_occupation_index(begin)
    uids = []
    all_users = []

    with open(user_path,'r') as f:
        for line in f.readlines():
            user_indexs=[]
            d = line.strip().split('|')
            age = int(d[1])//10
            uids.append(d[0])
            user_indexs.append(age)
            user_indexs.append(gender_dict[d[2]])
            user_indexs.append(occupation_dict[d[3]])
            all_users.append(user_indexs)

    df = pd.DataFrame(all_users,index=uids,columns=['age', 'gender', 'occupation'])
    df.to_csv(fp.Ml_100K.USER_DF)
    return begin

def __get_year_index(begin):
    years = set()
    with open(item_path, 'r', encoding = 'ISO-8859-1') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            year = d[2].split('-')
            if len(year)>2:
                years.add(int(year[2]))
    years.add(0)
    years = sorted(years)
    print(years)
    return {k:v+begin for v,k in enumerate(years)},len(years)

def generate_item_df(begin,out):
    items = {}
    years_dict, begin = __get_year_index(begin)
    max_n_neibour = 0
    all_items = []
    iids = []
    with open( item_path, 'r', encoding = 'ISO-8859-1' ) as f:
        for line in f.readlines():
            item_index = []
            d = line.strip().split('|')
            iids.append(int(d[0]))
            year = d[2].split('-')
            if len(year) > 2:
                item_index.append(years_dict[int(year[2])])
            else:
                item_index.append(0)

            subjects = d[5:]
            if begin == 0:
                begin = len(subjects)
            for i in range(len(subjects)):
                if int(subjects[i]) == 1:
                    item_index.append( begin+i )
            all_items.append( item_index )
            if len(item_index) > max_n_neibour:
                max_n_neibour = len(item_index)
    n_all=[]
    for item in all_items:
        n_all.append( np.random.choice( item, size = max_n_neibour, replace = True ) )

    df = pd.DataFrame( n_all, index = iids )
    df.to_csv(out )

    #print( all_items, max_n_neibour )
    return items

def get1or0(r):
    return 1.0 if r>3 else 0.0


def __read_rating_data(path):
    triples=[]
    with open(path,'r') as f:
        for line in f.readlines():
            d=line.strip().split('\t')
            triples.append([int(d[0]),int(d[1]),get1or0(int(d[2]))])
    return triples

def read_data_user_item_df():
    user_df = pd.read_csv( fp.Ml_100K.USER_DF, index_col = 0 )
    item_df = pd.read_csv( fp.Ml_100K.ITEM_DF_0, index_col = 0 )
    train_triples = __read_rating_data(train_path)
    test_triples= __read_rating_data(test_path)
    return train_triples, test_triples, user_df, item_df, max(user_df.max())+1, max(item_df.max())+1


def read_data():
    user_df = pd.read_csv( fp.Ml_100K.USER_DF, index_col = 0 )
    item_df = pd.read_csv( fp.Ml_100K.ITEM_DF, index_col = 0 )
    train_triples = __read_rating_data(train_path)
    test_triples= __read_rating_data(test_path)
    return train_triples, test_triples, user_df, item_df,max(item_df.max())+1


if __name__ == '__main__':
    item_df = generate_item_df(0, fp.Ml_100K.ITEM_DF_0)
    #print(item_df)

    train_triples, test_triples, user_df, item_df,lenitems = read_data()
    print(user_df)
    print(item_df)

