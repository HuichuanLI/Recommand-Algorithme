# -*- coding: utf-8 -*-
import jieba
from jieba import del_word, add_word

# 分词
# 精确模式，试图将句子最精确地切开，适合文本分析；
# 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
# 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
# paddle模式，利用PaddlePaddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词。同时支持词性标注.

seg_list = jieba.cut('我来到北京清华大学', cut_all=True)
print('全模式分词结果: ' + '/ '.join(seg_list))

seg_list = jieba.cut('我来到北京清华大学', cut_all=False)
print('精确模式分词结果: ' + '/ '.join(seg_list))

seg_list = jieba.cut('他来到了网易杭研大厦')
print(', '.join(seg_list))

seg_list = jieba.cut_for_search('小明硕士毕业于中国科学院计算所，后在日本京都大学深造')
print('搜索引擎模式分词结果: ' + ', '.join(seg_list))

# 载入词典 载入文件格式一个词占一行，每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒，词频省略时使用自动计算的能保证分出该词的词频。
jieba.load_userdict('vocabularity.txt')
# 使用 add_word(word, freq=None, tag=None) 和 del_word(word) 可在程序中动态修改词典
# del_word('杭研大厦')
add_word('杭研大厦')
# 比如医疗，旅游中有一些词分不出来，可以加大这些词的权重，这样就可以分出来了
seg_list = jieba.cut('他来到了网易杭研大厦', cut_all=False)
print('载入词典分词结果: ' + '/ '.join(seg_list))
