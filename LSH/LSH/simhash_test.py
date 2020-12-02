# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from simhash import Simhash, SimhashIndex

#print(Simhash('这个程序代码太乱').value)
#print(Simhash('1').distance(Simhash('2')))
#sh1 = Simhash(u'这个程序代码太乱,那个代码规范')
#sh2 = Simhash(u'这个程序代码不规范,那个更规范')
#print(sh1.value)
#print(sh2.value)
#print(sh1.distance(sh2))

data = [
    '这个程序代码太乱,那个代码规范',
    '这个程序代码不规范,那个更规范',
    '我是佩奇，这是我的弟弟乔治'
]

data = [
    '这个 程序 代码 太乱 那个 代码 规范',
    '这个 程序 代码 不 规范 那个 更 规范',
    '我 是 佩奇 这 是 我的 弟弟 乔治'
]

vec = TfidfVectorizer()
D = vec.fit_transform(data)
voc = dict((i, w) for w, i in vec.vocabulary_.items())
#print(voc)

# 生成Simhash
sh_list = []
for i in range(D.shape[0]):
    Di = D.getrow(i)
    #print(Di.indices)
    #print(Di.data)
    # features表示 (token, weight)元祖形式的列表
    features = zip([voc[j] for j in Di.indices], Di.data)
    sh_list.append(Simhash(features))
print(sh_list[0].distance(sh_list[1]))
print(sh_list[0].distance(sh_list[2]))
print(sh_list[1].distance(sh_list[2]))
