import numpy as np

# 物品特征向量
item_features = {
    'A': np.array([1, 0, 0]),
    'B': np.array([0, 1, 0]),
    'C': np.array([0, 0, 1])
}

# 用户偏好
user_preferences = {
    'A': 5,
    'B': 3,
    'C': 1
}


# 定义价值迭代算法函数
def value_iteration(item_features, user_preferences, num_iterations, discount_factor):
    # 初始化值函数
    values = {item: 0 for item in item_features}

    for _ in range(num_iterations):
        # 迭代更新值函数
        for item in item_features:
            best_value = 0
            for next_item in item_features:
                # 计算当前动作的奖励
                reward = user_preferences[next_item]
                # 根据马尔可夫决策过程的Bellman方程更新值函数
                value = reward + discount_factor * values[next_item]
                if value > best_value:
                    best_value = value
            values[item] = best_value

    return values


# 使用价值迭代算法求解最优推荐策略
optimal_values = value_iteration(item_features, user_preferences, num_iterations=100, discount_factor=0.9)

# 根据最优值函数进行推荐
best_item = max(optimal_values, key=optimal_values.get)
print("最优推荐物品:", best_item)
