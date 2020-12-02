

from datasketch import MinHash, MinHashLSH
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
# 创建LSH
lsh = MinHashLSH(threshold=0.5, num_perm=128)
lsh.insert("m2", m2)
lsh.insert("m3", m3)
result = lsh.query(m1)
print("近似的邻居（Jaccard相似度>0.5）", result)
