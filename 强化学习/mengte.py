import numpy as np

# 物品列表
items = ['A', 'B', 'C']

# 用户行为模拟函数
def simulate_user_action(item):
    # 模拟用户行为，返回对物品的评分
    return np.random.randint(1, 6)

# 蒙特卡洛方法评估推荐策略
def evaluate_policy(policy, num_episodes):
    total_reward = 0

    for _ in range(num_episodes):
        # 随机选择一个物品
        item = np.random.choice(items)
        # 模拟用户行为
        reward = simulate_user_action(item)
        # 根据策略计算推荐得分
        recommendation = policy(item)
        # 累积总奖励
        total_reward += reward * recommendation

    average_reward = total_reward / num_episodes
    return average_reward

# 随机推荐策略
def random_policy(item):
    # 随机返回一个推荐得分
    return np.random.randint(0, 2)

# 评估随机推荐策略
num_episodes = 1000
average_reward = evaluate_policy(random_policy, num_episodes)
print("随机推荐策略的平均得分:", average_reward)
