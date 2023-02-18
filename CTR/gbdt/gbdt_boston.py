import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# 波士顿房价回归数据集
boston = load_boston()  
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.1, random_state=0)  
clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,max_depth=4,min_samples_split=2,loss='ls')
clf.fit(X_train, y_train)
print('GBDT回归MSE：',mean_squared_error(y_test, clf.predict(X_test)))
#print('每次训练的得分记录：',clf.train_score_)
print('各特征的重要程度：',clf.feature_importances_)
# 每次训练，增加新的CART树，带来的训练得分变化
# train_score_:表示在样本集上每次迭代以后的对应的损失函数值。
plt.plot(np.arange(500), clf.train_score_, 'b-')  
#print(clf.train_score_)
plt.show()
