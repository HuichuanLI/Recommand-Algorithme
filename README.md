# 推荐算法
主要在工业界使用了推荐算法在我们的系统中，为此总结了一下工业界使用的推荐算法和实现
## 图神经网络相关实现 [实现](https://github.com/HuichuanLI/GraphNeuralNetWork)

## 召回算法
### itemcf 物品协同过滤 [实现](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/CF)
由此产生了基于物品的协同过滤（itemCF）给用户推荐和他们之前喜欢的物品相似的物品。不过ItemCF算法不是根据物品内容属性计算物品之间相似度，它主要通过分析用户的行为记录来计算物品之间的相似度。

基于物品的协同过滤算法主要分为两步：
    1：计算物品之间的相似度
    2：根据物品之间相似度和用户的历史行为给用户生产推荐列表。

### LFM [实现](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/LFM)
LFM算法是属于隐含语义模型的算法，不同于基于邻域的推荐算法。
[link](https://blog.csdn.net/weixin_41843918/article/details/90216729)
### Personal rank [实现](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/PersonalRank)
基于图的推荐算法，类似于Page Rank 算法
[link](https://blog.csdn.net/bbbeoy/article/details/78646635)
### item2vector [实现](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/Item2vec)
    顺序性时序性缺失
    item 是无区分性
    从log获取行为序列
    word2Vector item embedding
    计算item sim

    Word2Vctor --- Cbow/Skip Gram 
### 基于内容ContentBase算法 [实现](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/ContentBased)
    优点：
    1. 思想简单
    2. 独立性
    3。 流行度比较高
    缺点：
    1。必须积累一定的时间
    2。无法跨领域的推荐
    
    item_profile
    user Profile
        genre/Topic
        time Decay
    online Recommendation
        Find top k Genre/Topic

## 总结召回算法        
### 基于领域的
PR,userCF,itermCF,LFM,ALS[pyspark实现](https://github.com/HuichuanLI/Spark-in-machine-learning/blob/master/02%20%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AE%9E%E6%88%98/pyspark_mlib_als.ipynb)
### 基于内容的
ContentedBase
### 基于neural network的
item2Vector
![](./photo/1.png)


### 离线评价
![](./photo/2.png)

训练集：周一到周五
测试集:周六到周天
### 在线评价收益
![](./photo/3.png)


### 排序算法
召回其实就是天花板，排序其实就是逼近这个天花班
![](./photo/4.png)

![](./photo/5.png)

### 分类
    单一浅层模型
        LR，FM
    浅层模型的组合
        Tree
    深度学习
        Tensorflow

![](./photo/6.png)
### 逻辑回归 [完成](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/LR)

1.易于理解，计算代价小
2。容易欠拟合，需要特征工程


#### Loss Function

![](./photo/7.png)

#### 梯度
![](./photo/8.png)

![](./photo/9.png)


![](./photo/10.png)


#### 特征的统计和分析
1.覆盖率(1%)
2.成本
3.准确率
### 特征的处理
    1.缺失值的处理：中位数，众数，平均数
    2.特征的归一化，最大归一化，排序归一化
    3.特征的离散化: 分位数
    
    
### Tree Model[完成](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/Tree)
    1.完成GBDT模型
    2.完成GDBT+LR混合模型
#### CART 算法，误差和gini系数


![](./photo/11.png)
### CART 算法
![](./photo/12.png)



![](./photo/13.png)
### 分类树
![](./photo/14.png)
### GINI 求解问题
![](./photo/15.png)

### Boosting 算法

    1. 如何改变权重？
    上一轮错的权重增加
    2. 如何组合模型？
    每个提升树都权重一样Gradient，ada 是不一样的分类器误差率小的权重增大


![](./photo/16.png)
![](./photo/17.png)
![](./photo/18.png)

### XGboost 算法

![](./photo/19.png)

![](./photo/20.png)

![](./photo/21.png)

#### Q 为节点的个数

![](./photo/22.png)

![](./photo/23.png)

![](./photo/24.png)

![](./photo/25.png)


#### GBDT 和 LR 混合模型

![](./photo/26.png)

优缺点总结 
    优点：利用树模型做特征转化   
    缺点：两个模型单独训练不是联合训练


### Wide and Deep Model[完成](https://github.com/HuichuanLI/Recommand-Algorithme/tree/master/WD)

![](./photo/27.png)

![](./photo/28.png)
#### 因为zj 会有 k之和

##### Generalization and Memorization
![](./photo/29.png)

    Wide 其实就是Memorization  一半离散特征和特征组合
    Deep 其实就是Generalization 一半连续特征
![](./photo/30.png)

![](./photo/31.png)

![](./photo/32.png)

![](./photo/33.png)
 
 
 ### 总结
 
 ![](./photo/34.png)

User特征：基础特征
Item特征：基础特征
Context:几点
UIRelation: 点击率。。。
Static supplement:上架商品的

特征数目
1：100 和数据最好
