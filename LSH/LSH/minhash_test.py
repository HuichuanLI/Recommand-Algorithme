from datasketch import MinHash
data1 = ['这个', '程序', '代码', '太乱', '那个', '代码', '规范']
data2 = ['这个', '程序', '代码', '不', '规范', '那个', '更', '规范']


m1 = MinHash()
m2 = MinHash()
for d in data1:
	m1.update(d.encode('utf8'))
for d in data2:
    m2.update(d.encode('utf8'))
print("使用MinHash预估的Jaccard相似度", m1.jaccard(m2))

s1 = set(data1)
s2 = set(data2)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Jaccard相似度实际值", actual_jaccard)
print(s1.intersection(s2))
print(len(s1.intersection(s2)))
print(len(s1.union(s2)))
print(s1.union(s2))
