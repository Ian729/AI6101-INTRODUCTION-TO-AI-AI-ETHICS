from environment import CliffBoxPushingBase
from collections import defaultdict
import numpy as np
import random

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
        pass


if __name__ == '__main__':
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
    #     print(f'print the historical actions: {env.episode_actions}')
        teminated = False
        rewards = []
