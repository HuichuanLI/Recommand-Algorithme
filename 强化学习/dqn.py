import numpy as np
import tensorflow as tf


# 构建深度 Q 网络模型
class DQNModel(tf.keras.Model):
    def __init__(self, num_items):
        super(DQNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_items, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output


# DQN 推荐系统类
class DQNRecommender:
    def __init__(self, num_items, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, discount_factor=0.99,
                 learning_rate=0.001):
        self.num_items = num_items
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.model = DQNModel(num_items)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 随机选择一个动作
            return np.random.randint(self.num_items)
        else:
            # 根据模型预测选择最优动作
            q_values = self.model.predict(state)
            return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q_values = self.model.predict(next_state)[0]
            target += self.discount_factor * np.max(next_q_values)
        target_q_values = self.model.predict(state)
        target_q_values[0][action] = target

        with tf.GradientTape() as tape:
            q_values = self.model(state)
            loss = tf.keras.losses.MSE(target_q_values, q_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 创建一个简单的推荐系统环境
class RecommendationEnvironment:
    def __init__(self, num_items):
        self.num_items = num_items

    def get_state(self):
        # 返回当前状态（可以是用户历史行为的特征表示）
        return np.zeros((1, self.num_items))

    def take_action(self, action):
        # 执行动作，返回奖励
        return np.random.randint(0, 10)

    def is_done(self):
        # 判断是否结束
        return np.random.rand() < 0.1


# 定义训练参数
num_items = 10
num_episodes = 1000

# 创建推荐系统实例
recommender = DQNRecommender(num_items)

# 创建环境实例
env = RecommendationEnvironment(num_items)

# 开始训练
for episode in range(num_episodes):
    state = env.get_state()
    done = False

    while not done:
        action = recommender.get_action(state)
        reward = env.take_action(action)
        next_state = env.get_state()
        done = env.is_done()
        recommender.train(state, action, reward, next_state, done)
        state = next_state

    # 打印每个回合的总奖励
    print("Episode:", episode, "Total Reward:", reward)
