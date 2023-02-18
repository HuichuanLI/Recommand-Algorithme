from datasketch import MinHash, MinHashLSH, MinHashLSHForest
data1 = ['这个', '程序', '代码', '太乱', '那个', '代码', '规范']
data2 = ['这个', '程序', '代码', '不', '规范', '那个', '更', '规范']
data3 = ['这个', '程序', '代码', '不', '规范', '那个', '规范', '些']

# 创建MinHash对象
m1 = MinHash()
m2 = MinHash()
m3 = MinHash()
for d in data1:
	m1.update(d.encode('utf8'))
for d in data2:
	m2.update(d.encode('utf8'))
for d in data3:
	m3.update(d.encode('utf8'))
# 创建LSH Forest
forest = MinHashLSHForest()
forest.add("m2", m2)
forest.add("m3", m3)
# 在检索前，需要使用index
forest.index()
# 判断forest是否存在m2, m3
print("m2" in forest)
print("m3" in forest)
# 查询forest中与m1相似的Top-K个邻居
result = forest.query(m1, 2)
print("Top 2 邻居", result)
