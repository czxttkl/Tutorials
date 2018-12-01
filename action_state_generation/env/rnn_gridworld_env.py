"""
Rule of reward: after reaching R cell,
get action history passing RNN - num of actions
"""
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


W = 1  # Walls
S = 2  # Starting position
G = 3  # Goal position

L = 0
R = 1
U = 2
D = 3


class GridWorldEnv:

    def __init__(self, device):
        # x, y
        self.agent_pos = [0, 0]
        # record how many steps an agent has gone
        self.num_step = 0
        # record action sequences so far
        self.act_seq = []
        self.lstm_hidden_dim = 20
        self.lstm_num_layer = 2

        self.grid = np.array(
            [
                [S, 0, 0, G],
            ]
        )

        # self.grid = np.array(
        #     [
        #         [S, 0, 0, 0],
        #         [0, W, W, 0],
        #         [0, W, G, 0],
        #         [0, 0, 0, 0],
        #     ]
        # )

        # self.grid = np.array(
        #     [
        #         [S, 0, 0],
        #         [W, 0, 0],
        #         [G, 0, 0],
        #     ]
        # )

        # self.grid = np.array(
        #     [
        #         [S, 0],
        #         [0, W],
        #         [0, G],
        #     ]
        # )

        self.device = device
        self.action_dim = 4
        # exclude the last dim of the state, which records # of back-steps
        self.state_dim = self.grid.shape[0] * self.grid.shape[1]

    def init_lstm(self):
        self.lstm = nn.LSTM(input_size=self.action_dim,
                            hidden_size=self.lstm_hidden_dim,
                            batch_first=True,
                            num_layers=self.lstm_num_layer)
        self.fc = nn.Linear(self.lstm_hidden_dim, self.action_dim)

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        self.hidden = torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim), \
                      torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim)

    def init_weight(self):
        # relatively large std such that RNN reward's scale is
        # comparable with other kinds of reward
        for p in self.lstm.params():
            nn.init.normal_(0, 10)

    def cur(self):
        x, y = self.agent_pos
        # w, h = self.grid.shape[1], self.grid.shape[0]
        # out = np.zeros(w * h + 1)
        # out[x * w + y] = 1
        # out[-1] = self.back_step
        # out_torch = torch.from_numpy(out).float()
        # return out_torch.unsqueeze(0).to(self.device)
        return x, y, self.num_step, self.act_seq

    def state_to_x_y(self, state):
        return state[0], state[1]

    def reset(self):
        self.agent_pos = [0, 0]
        self.num_step = 0
        self.act_seq = []

    def step(self, action):
        x, y = self.agent_pos
        w, h = self.grid.shape[1], self.grid.shape[0]
        if action == L:
            if y > 0 and self.grid[x, y - 1] != W:
                self.agent_pos[1] -= 1
            else:
                raise Exception
        elif action == R:
            if y < w - 1 and self.grid[x, y + 1] != W:
                self.agent_pos[1] += 1
            else:
                raise Exception
        elif action == U:
            if x > 0 and self.grid[x - 1, y] != W:
                self.agent_pos[0] -= 1
            else:
                raise Exception
        elif action == D:
            if x < h - 1 and self.grid[x + 1, y] != W:
                self.agent_pos[0] += 1
            else:
                raise Exception

        done = self.grid[self.agent_pos[0], self.agent_pos[1]] == G
        if done:
            reward1 = self.cal_rnn_reward()
            reward2 = self.num_step
            reward = reward1 - reward2
        # reward = torch.tensor([reward], device=self.device).float()
        self.num_step += 1
        self.act_seq.append(action)
        return self.cur(), reward, done, None

    def invalid_actions(self):
        x, y = self.agent_pos
        return self.invalid_actions_by_x_y(x, y)

    def invalid_actions_by_x_y(self, x, y):
        w, h = self.grid.shape[1], self.grid.shape[0]
        ia = []
        if y == 0 or self.grid[x, y - 1] == W:
            ia.append(L)
        if y == w - 1 or self.grid[x, y + 1] == W:
            ia.append(R)
        if x == 0 or self.grid[x - 1, y] == W:
            ia.append(U)
        if x == h - 1 or self.grid[x + 1, y] == W:
            ia.append(D)
        return ia

    def invalid_actions_by_state(self, state):
        x, y = self.state_to_x_y(state)
        invalid_actions = self.invalid_actions_by_x_y(x, y)
        return invalid_actions
