import numpy as np
from sklearn.linear_model import LinearRegression

# 一元线性回归
print('一元线性回归')
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y) #确定系数
print('评分结果:', r_sq)
print('系数:', model.coef_) # 系数
print('截距:', model.intercept_) #截距
y_pred = model.predict(x)
print('预测结果:', y_pred, sep='\n')

# 多元线性回归
print('\n多元线性回归')
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('评分结果:', r_sq)
print('系数:', model.coef_)
print('截距:', model.intercept_)
y_pred = model.predict(x)
print('预测结果:', y_pred, sep='\n')

# 多项式回归
from sklearn.preprocessing import PolynomialFeatures
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
transformer = PolynomialFeatures(degree=2, include_bias=False)
x_ = transformer.fit_transform(x)
print(x_)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)
print('评分结果:', r_sq)
print('系数:', model.coef_)
print('截距:', model.intercept_)
y_pred = model.predict(x_)
print('预测结果:', y_pred, sep='\n')