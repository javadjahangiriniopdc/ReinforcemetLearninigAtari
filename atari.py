import os

import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

# خاموش کردن هشدار oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# پارامترها
learning_rate = 0.001
gamma = 0.95  # ضریب تخفیف
epsilon = 1.0  # مقدار اولیه اکتشاف
epsilon_decay = 0.995  # کاهش تدریجی اکتشاف
epsilon_min = 0.01
batch_size = 64
memory_size = 2000

# تعریف محیط بازی
env = gym.make('ALE/Riverraid-v5', render_mode='human')
state_shape = env.observation_space.shape
action_size = env.action_space.n


# ساخت مدل Q-Network
def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=state_shape))  # فلت کردن ورودی‌ها
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


# کلاس DQN
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = build_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([s[0] for s in minibatch])
        targets = self.model.predict(states, verbose=0)

        next_states = np.vstack([s[3] for s in minibatch])
        next_q_values = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward if done else reward + self.gamma * np.amax(next_q_values[i])
            targets[i][action] = target

        # نمایش خطا برای هر مرحله از آموزش
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]  # دسترسی به مقدار خطا
        print(f"Training loss: {loss}")  # نمایش خطا

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ایجاد عامل
agent = DQNAgent(state_shape, action_size)

# آموزش عامل
episodes = 1000
for e in range(episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1] + list(state_shape))
    score = 0
    for time in range(500):
        if e % 10 == 0:  # نمایش محیط هر 10 اپیزود یک‌بار
            env.render()
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        score += reward
        next_state = np.reshape(next_state, [1] + list(state_shape))
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done or truncated:
            print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.4f}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

env.close()
