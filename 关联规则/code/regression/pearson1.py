import pandas as pd

def compute(x):
    return 2*x*x+1
x=[i for i in range(100)]
y=[compute(i) for i in x]
data = pd.DataFrame({'x':x,'y':y})
# 查看pearson系数
print(data.corr())
print(data.corr(method='spearman'))
print(data.corr(method='kendall'))
