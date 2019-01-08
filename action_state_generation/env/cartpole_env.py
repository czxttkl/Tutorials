"""
An open AI gym environment "CartPole-v0"
State space is 4-dimensional continuous space. Action space is 2 discrete actions.
"""
import numpy as np
import torch
import gym
from env.lunar_env import LunarEnv


class CartPoleEnv(LunarEnv):

    def __init__(self, device):
        self.env = gym.make("CartPole-v0")
        self.device = device
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.cur_state = self.env.reset()

    def test_step_limit(self):
        """ test step limit. If test steps beyond this limit, terminate the episode """
        return 1001
