import gym
import random
import torch
import numpy as np
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
plt.ion()
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../action_state_generation/model')
from dqn import DQN
from dqn_agent import Agent, ReplayBuffer

env = gym.make('LunarLander-v2')
# env = gym.make('CartPole-v0')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


agent = DQN(env.observation_space.shape[0], 2,
            64, env.action_space.n, True, 0.99, int(1e5), 64)
agent.memory = ReplayBuffer(env.action_space.n, int(1e5), 64, 0)
agent.env = env
agent.t_step = 0
agent.optimizer = optim.Adam(agent.parameters())
# agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)


def train(n_episodes=20000, max_t=1000, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        temp_exp = []
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            temp_exp.append((state, action, reward, next_state, done))
            state = next_state
            score += reward
            if done:
                break
        for state, action, reward, next_state, done in temp_exp:
            agent.memory.add(state, action, reward, next_state, done)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                  .format(i_episode - 100,
                          np.mean(scores_window)))
            break
    return scores


scores = train()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()