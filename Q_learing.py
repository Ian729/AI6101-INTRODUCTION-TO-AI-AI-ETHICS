from environment import CliffBoxPushingBase
from collections import defaultdict
import numpy as np
import random

import matplotlib.pyplot as plt

class QAgent(object):
    def __init__(self):
        self.action_space = [1,2,3,4]
#         self.V = []
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.discount_factor=0.99
        self.alpha=0.5
        self.epsilon=0.01

    def take_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[state])]
        return action

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        self.Q[state][action-1] = self.Q[state][action-1] + self.alpha * (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state][action-1])


if __name__ == '__main__':
    episode_rewards = []
    env = CliffBoxPushingBase()
    # you can implement other algorithms
    agent = QAgent()
    teminated = False
    rewards = []
    time_step = 0
    num_iterations = 1000
    for i in range(num_iterations):
        env.reset()
        while not teminated:
            state = env.get_state()
            action = agent.take_action(state)
    #         print(action)
            reward, teminated, _ = env.step([action])
            next_state = env.get_state()
            rewards.append(reward)
        #     print(f'step: {time_step}, actions: {action}, reward: {reward}')
            time_step += 1
            agent.train(state, action, next_state, reward)
        print(f'rewards: {sum(rewards)}')
        episode_rewards.append(sum(rewards))
        print(f'print the historical actions: {env.episode_actions}')
        teminated = False
        rewards = []
    # f = open("Q_table.csv","w")
    # for key in sorted(agent.Q.keys()):
        # print(f"{key}:{agent.Q[key][0]:.2f};{agent.Q[key][1]:.2f};{agent.Q[key][2]:.2f};{agent.Q[key][3]:.2f}")
        # f.write(f"{key}:{agent.Q[key][0]:.2f};{agent.Q[key][1]:.2f};{agent.Q[key][2]:.2f};{agent.Q[key][3]:.2f}\n")
    # f.close()
    times = np.array(list(range(num_iterations)))   
    
    m,b = np.polyfit(times,episode_rewards,1)
    
    plt.plot(times, episode_rewards, 'o')
    plt.plot(times, m*times+b)

    plt.title("Episode Rewards vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Episode Rewards")

    plt.ylim(min(episode_rewards)-200, 0)
    plt.show()