# 推荐算法

## 召回算法
### itemcf 物品协同过滤 【实现】
### LFM 【实现】
### Personal rank 【实现】
### item2vector 【实现】
    顺序性时序性缺失
    item 是无区分性
    从log获取行为序列
    word2Vector item embedding
    计算item sim

    Word2Vctor --- Cbow
    Skip Gram 
### 基于内容ContentBase算法 [实现]
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
PR,userCF,itermCF,LFM,ALS
### 基于内容的
ContentedBase
### 基于neural network的
item2Vector
![](1.png)


### 离线评价
![](2.png)

训练集：周一到周五
测试集:周六到周天
### 在线评价收益
![](3.png)


### 排序算法
召回其实就是天花板，排序其实就是逼近这个天花班
![](4.png)

![](5.png)

### 分类
    单一浅层模型
        LR，FM
    浅层模型的组合
        Tree
    深度学习
        Tensorflow

![](6.png)
### 逻辑回归

1.易于理解，计算代价小
2。容易欠拟合，需要特征工程
