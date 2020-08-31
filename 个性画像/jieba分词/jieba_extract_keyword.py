# -*- coding: utf-8 -*-
from jieba.analyse import extract_tags

# 基于 TF-IDF 算法的关键词抽取

# jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
#   sentence 为待提取的文本
#   topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
#   withWeight 为是否一并返回关键词权重值，默认值为 False
#   allowPOS 仅包括指定词性的词，默认值为空，即不筛选
# jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
sentence = "本科及以上学历，计算机、数学等相关专业重点学校在校生(硕士为佳)-- 至少掌握一门编程语言，包括SQL。熟悉Linux；"
keywords = extract_tags(sentence, topK=20, withWeight=True, allowPOS=('n'))
for item in keywords:
    print('TF-IDF： ', item[0], item[1])

# 词表

# 关键词提取所使用逆向文件频率（IDF）文本语料库可以切换成自定义语料库的路径
# jieba.analyse.set_idf_path('word_frequency.txt')
# keywords = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
# for item in keywords:
#     print('TF-IDF加载逆向文件频率： ', item[0], item[1])

# 关键词提取所使用停止词（StopWords）文本语料库可以切换成自定义语料库的路径
# jieba.analyse.set_stop_words('stop_words.txt')
# keywords = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
# for item in keywords:
#     print('TF-IDF加载停用词文件频率： ', item[0], item[1])

# 基于 TextRank 算法的关键词抽取

# jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
#   sentence 为待提取的文本
#   topK 为返回几个 textrank 权重最大的关键词，默认值为 20
#   withWeight 为是否一并返回关键词权重值，默认值为 False
#   allowPOS 仅包括指定词性的词，默认值为空，即不筛选
# jieba.analyse.TextRank() 新建自定义 TextRank 实例
# keywords = jieba.analyse.textrank(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
# for item in keywords:
#     print('TextRank：', item[0], item[1])
