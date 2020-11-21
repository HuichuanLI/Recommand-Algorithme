# 分析MovieLens 电影分类中的频繁项集和关联规则
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 数据加载
movies = pd.read_csv('./movie_actors.csv')
#print(movies.head())
# 将genres进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
movies_hot_encoded = movies.drop('actors',1).join(movies.actors.str.get_dummies('/'))
pd.options.display.max_columns=100
print(movies_hot_encoded.head())

# 将movieId, title设置为index
movies_hot_encoded.set_index(['title'],inplace=True)
#print(movies_hot_encoded.head())
# 挖掘频繁项集，最小支持度为0.02
itemsets = apriori(movies_hot_encoded,use_colnames=True, min_support=0.05)
# 按照支持度从大到小进行时候粗
itemsets = itemsets.sort_values(by="support" , ascending=False) 
print('-'*20, '频繁项集', '-'*20)
print(itemsets)
# 根据频繁项集计算关联规则，设置最小提升度为2
rules =  association_rules(itemsets, metric='lift', min_threshold=2)
# 按照提升度从大到小进行排序
rules = rules.sort_values(by="lift" , ascending=False) 
#rules.to_csv('./rules.csv')
print('-'*20, '关联规则', '-'*20)
print(rules)

