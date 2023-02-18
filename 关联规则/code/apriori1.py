from efficient_apriori import apriori
# 设置数据集
transactions = [('牛奶','面包','尿布'),
		('可乐','面包', '尿布', '啤酒'),
		('牛奶','尿布', '啤酒', '鸡蛋'),
		('面包', '牛奶', '尿布', '啤酒'),
		('面包', '牛奶', '尿布', '可乐')]
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)
print("频繁项集：", itemsets)
print("关联规则：", rules)
