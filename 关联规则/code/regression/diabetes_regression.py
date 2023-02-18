"""
	使用sklearn自带的糖尿病数据集，进行回归分析
	Diabetes：包含442个患者的10个生理特征（年龄，性别、体重、血压）和一年以后疾病级数指标
"""
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error

# 加载数据
diabetes = datasets.load_diabetes()
data = diabetes.data
#import numpy as np
#print(np.max(data))
#data.to_csv('diabetes.csv')
# 数据探索
print(data.shape)
print(data[0])

# 训练集 70%，测试集30%
train_x, test_x, train_y, test_y = train_test_split(diabetes.data, diabetes.target, test_size=0.3, random_state=14)
print(len(train_x))

#回归训练及预测
clf = linear_model.LinearRegression()
clf.fit(train_x, train_y)

print(clf.coef_)
#print(train_x.shape)
#print(clf.score(test_x, test_y))
pred_y = clf.predict(test_x)
print(mean_squared_error(test_y, pred_y))
r_sq = clf.score(train_x, train_y) #确定系数
print('r_sq:', r_sq)


