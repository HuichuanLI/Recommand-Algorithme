from surprise import KNNWithMeans
from surprise import Dataset, Reader

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})
algo.fit(trainset)

uid = str(196)
iid = str(302)

pred = algo.predict(uid, iid)
print(pred)
