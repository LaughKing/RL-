# 关于gymnasium-Taxi-v3的学习记录

首先 打开了gymnasium库的官网

简单读了读 发现读不懂

去csdn上找到一些入门简介

终于懂了。开始跑第一个测试代码

发现box[2d]的install一直出问题

上网查了不少资料

都试了下，又问了gpt

最后是少了个关键的库没安装，终于好了

开始看taxi的文档

好多专属名词，discreate，action_space，observation_space ...看不懂

又去读env的操作文档了



读完了，开始写自己的第一份代码了。（v1.1）

```python
import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v3",render_mode = 'human')
q_table = np.zeros([env.observation_space.n,env.action_space.n])

alpha = 0.1
gamma = 0.9
epsilon = 0.1
training_episodes = 1
display_episodes = 20

# 训练智能体
for i in range(training_episodes):
    observasion,info = env.reset()
    reward,penalty = 0,0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observasion])
        
        next_observasion,reward,done,truncated,info = env.step(action)
        old_q = q_table[observasion,action]
        max_q = np.max(q_table[next_observasion])

        next_q = old_q + alpha*(reward+gamma*max_q - old_q)

        if reward == -10:
            penalty += 1
        
        observasion = next_observasion

print("训练完成")


```



```python
total_epochs,total_penalties = 0,0
for _ in range(display_episodes):
    observasion,info = env.reset()
    epochs,penalties,reward = 0,0,0
    done = False

    while not done:
        action = np.argmax(q_table[observasion])
        observasion,reward,done,truncated,info = env.step(action)
        if reward == -10:
            penalties += 1

        epochs += 1

        env.render("human")
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.15)  # Sleep so the user can see the 

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / display_episodes}")
print(f"Average penalties per episode: {total_penalties / display_episodes}")

```

出现问题：

训练速度非常非常慢感觉跑不动的感觉。

经常性卡住不动。试一下epsilon的收敛算法。

```python
import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v3",render_mode = 'human')
q_table = np.zeros([env.observation_space.n,env.action_space.n])

alpha = 0.1 #学习率
gamma = 0.9 #折扣因子
epsilon = 1.0 #探索
epsilon_decay = 0.99
training_episodes = 5000
display_episodes = 20
max_steps = 100


# 训练智能体
for i in range(training_episodes):
    observasion,info = env.reset()
    reward,penalty = 0,0
    done = False
    steps = 0

    while not done and steps < max_steps:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observasion])
        
        next_observasion,reward,done,truncated,info = env.step(action)
        old_q = q_table[observasion,action]
        max_q = np.max(q_table[next_observasion])

        next_q = old_q + alpha*(reward+gamma*max_q - old_q)

        if reward == -10:
            penalty += 1
        
        steps += 1
        observasion = next_observasion
    epsilon = max{0.1,epsilon*epsilon_decay}

    if i % 1000 == 0:
        print(f"Episode:{i}")

print("训练完成")

total_epochs,total_penalties = 0,0
for _ in range(display_episodes):
    observasion,info = env.reset()
    epochs,penalties,reward = 0,0,0
    done = False

    while not done:
        action = np.argmax(q_table[observasion])
        observasion,reward,done,truncated,info = env.step(action)
        if reward == -10:
            penalties += 1

        epochs += 1

        env.render("human")
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.1)  

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"平均步数: {total_epochs / display_episodes}")
print(f"平均惩罚: {total_penalties / display_episodes}")

```

还是有点卡

我决定去先尝试下可视化并去除了渲染，看看问题出在哪里

增加了以下代码

```python
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
q_table = np.zeros([env.observation_space.n,env.action_space.n])

alpha = 0.1 #学习率
gamma = 0.9 #折扣因子
epsilon = 1.0 #探索
epsilon_decay = 0.99

#训练5000轮
training_episodes = 5000
display_episodes = 20

episode_rewards = []
episode_lengths = []


# 训练智能体
for i in range(training_episodes):
    observasion,info = env.reset()
    reward,penalty = 0,0
    done = False
    steps = 0
    total_rewards = 0

    while not done:
        #epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observasion])
        
        next_observasion,reward,done,truncated,info = env.step(action)
        old_q = q_table[observasion,action]
        max_q = np.max(q_table[next_observasion])

        next_q = old_q + alpha*(reward+gamma*max_q - old_q)

        if reward == -10:
            penalty += 1
        
        steps += 1
        total_rewards += reward
        observasion = next_observasion
        epsilon = max(0.1 , epsilon*epsilon_decay)
    
    episode_rewards.append(total_rewards)
    episode_lengths.append(steps)
    


    if i % 1000 == 0:
        print(f"Episode:{i}")

print("训练完成")

# 训练过程可视化
#累积奖励
plt.figure(figsize=(8,5))
plt.plot(episode_rewards)
plt.title("Episode-rewards")
plt.xlabel("Episodes")
plt.ylabel("Total rewards")
plt.show()

#累积步数
plt.figure(figsize=(8,5))
plt.plot(episode_lengths)
plt.title("Episode-lengths")
plt.xlabel("Episode")
plt.ylabel("Total steps")
plt.show()


total_epochs,total_penalties = 0,0
for _ in range(display_episodes):
    observasion,info = env.reset()
    epochs,penalties,reward = 0,0,0
    done = False

    while not done:
        action = np.argmax(q_table[observasion])
        observasion,reward,done,truncated,info = env.step(action)
        if reward == -10:
            penalties += 1

        epochs += 1

        env.render()
        print(f"Timestep: {epochs}")
        print(f"Obs: {observasion}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.1)  

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"平均步数: {total_epochs / display_episodes}")
print(f"平均惩罚: {total_penalties / display_episodes}")

```

终于发下问题了：q_table 的 更新出现了问题。

修改之后，正确的跑出了代码

一次次训练太麻烦了，经过search，我发现了可以通过保存Q-table的方法保存训练结果。并进行了可视化

```python
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 1.0  # 探索率
epsilon_decay = 0.99

# 训练5000轮
training_episodes = 5000
display_episodes = 20

episode_rewards = []
episode_lengths = []

# 定义 Reward Shaping 相关的辅助奖励
def shaped_reward(old_state, new_state, reward):
    # 如果新的状态比旧的状态更接近目标，就给予额外的奖励
    # 比如：状态编号大的值可能意味着更接近目标
    proximity_bonus = 0
    if new_state > old_state:  # 简单的接近性条件
        proximity_bonus = 1  # 正奖励
    
    # 维持主要的环境奖励，同时增加额外的奖励
    return reward + proximity_bonus

# 训练智能体
for i in range(training_episodes):
    observasion, info = env.reset()
    done = False
    steps = 0
    total_rewards = 0
    
    while not done:
        # epsilon-greedy 策略选择动作
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observasion])

        next_observasion, reward, done, truncated, info = env.step(action)
        
        # 使用 Reward Shaping 技巧
        shaped_r = shaped_reward(observasion, next_observasion, reward)
        
        # Q-learning 更新公式
        old_q = q_table[observasion, action]
        max_q = np.max(q_table[next_observasion])
        q_table[observasion, action] = old_q + alpha * (shaped_r + gamma * max_q - old_q)

        total_rewards += shaped_r
        steps += 1
        observasion = next_observasion

        # 减少 epsilon
        epsilon = max(0.1, epsilon * epsilon_decay)
    
    episode_rewards.append(total_rewards)
    episode_lengths.append(steps)
    
    if i % 1000 == 0:
        print(f"Episode: {i}, Total Reward: {total_rewards}")

print("训练完成")

# 训练过程可视化
# 累积奖励
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards)
plt.title("Episode-rewards with Reward Shaping")
plt.xlabel("Episodes")
plt.ylabel("Total rewards")
plt.show()

# 累积步数
plt.figure(figsize=(8, 5))
plt.plot(episode_lengths)
plt.title("Episode-lengths with Reward Shaping")
plt.xlabel("Episodes")
plt.ylabel("Total steps")
plt.show()


total_epochs,total_penalties = 0,0
for _ in range(display_episodes):
    env = gym.make('Taxi-v3',render_mode = 'human')
    observasion,info = env.reset()
    epochs,penalties,reward = 0,0,0
    done = False

    while not done:
        action = np.argmax(q_table[observasion])
        observasion,reward,done,truncated,info = env.step(action)
        if reward == -10:
            penalties += 1

        epochs += 1

        
        print(f"Timestep: {epochs}")
        print(f"Obs: {observasion}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"平均步数: {total_epochs / display_episodes}")
print(f"平均惩罚: {total_penalties / display_episodes}")

```

最后，简单的试了下 reward-shaping

```python
def shaped_reward(old_state, new_state, reward):
    # 如果新的状态比旧的状态更接近目标，就给予额外的奖励
    # 比如：状态编号大的值可能意味着更接近目标
    proximity_bonus = 0
    if new_state > old_state:  # 简单的接近性条件
        proximity_bonus = 1  # 正奖励
    
    # 维持主要的环境奖励，同时增加额外的奖励
    return reward + proximity_bonus

```

