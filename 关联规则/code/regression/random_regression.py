# 回归分析
import random
from sklearn import linear_model
reg = linear_model.LinearRegression()

def generate(x):
	y = 2*x+10+random.random()
	return y

train_x = []
train_y = []
for x in range(1000):
	train_x.append([x])
	y = generate(x)
	train_y.append([y])

reg.fit (train_x, train_y)
# coef_ 保存线性模型的系数w
print(reg.coef_)
print(reg.intercept_)