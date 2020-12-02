from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble
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
# 创建LSH Ensemble
lshensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128)
# Index takes an iterable of (key, minhash, size)
lshensemble.index([("m2", m2, len(data2)), ("m3", m3, len(data3))])
# 判断lshensemble是否存在m2, m3
print("m2" in lshensemble)
print("m3" in lshensemble)
# 查询与m1相似度大于0.8的集合
print("与m1相似度大于0.8的集合：")
for key in lshensemble.query(m1, len(data1)):
    print(key)
