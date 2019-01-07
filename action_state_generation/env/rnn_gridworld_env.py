"""
A grid world environment
Rule of reward: after reaching R cell,
10 + get action history passing RNN - num of actions
"""
import numpy as np
import torch
import random

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

    class RNNGridWorld(nn.Module):

        def __init__(self, lstm_input_dim, lstm_num_layer, lstm_hidden_dim):
            super(GridWorldEnv.RNNGridWorld, self).__init__()
            self.lstm_input_dim = lstm_input_dim
            self.lstm_num_layer = lstm_num_layer
            self.lstm_hidden_dim = lstm_hidden_dim
            self.init_lstm()
            self.init_weight()
            self.init_hidden()

        def init_lstm(self):
            self.lstm = nn.LSTM(input_size=self.lstm_input_dim,
                                hidden_size=self.lstm_hidden_dim,
                                batch_first=True,
                                num_layers=self.lstm_num_layer)
            # output scalar reward
            self.fc = nn.Linear(self.lstm_hidden_dim, 1)

        def init_hidden(self, batch_size=1):
            # Before we've done anything, we dont have any hidden state.
            self.hidden = torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim), \
                          torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim)

        def init_weight(self):
            torch.manual_seed(3212)
            # relatively large std such that RNN reward's scale is
            # comparable with other kinds of reward
            for n, p in self.lstm.named_parameters():
                nn.init.normal_(p, 0, 5)
            for n, p in self.fc.named_parameters():
                nn.init.normal_(p, 0, 5)
            torch.manual_seed(random.randint(0, 1000))

        def forward(self, X):
            # X_shape: (1, act_seq_len, lstm_input_dim)
            X, self.hidden = self.lstm(X, self.hidden)
            Y = self.fc(X)
            # return a scalar
            return Y[0, -1, 0]

    def __init__(self, device):
        # x, y
        self.agent_pos = [0, 0]
        # record how many steps an agent has gone
        self.num_step = 0
        # record action sequences so far
        self.act_seq = []

        # self.grid = np.array(
        #     [
        #         [S, 0, 0, G],
        #     ]
        # )

        # self.grid = np.array(
        #     [
        #         [S, 0, 0, 0],
        #         [0, W, W, 0],
        #         [0, W, G, 0],
        #         [0, 0, 0, 0],
        #     ]
        # )

        self.grid = np.array(
            [
                [S, 0, 0],
                [W, 0, 0],
                [G, 0, 0],
            ]
        )

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
        self.rnn_gridworld = GridWorldEnv.RNNGridWorld(lstm_input_dim=self.action_dim,
                                                       lstm_hidden_dim=20,
                                                       lstm_num_layer=2)

    def test_step_limit(self):
        """ test step limit. If test steps beyond this limit, terminate the episode """
        return 30

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

    def action_to_action_vec_lstm(self, action):
        tensor_action = torch.zeros((1, 1, self.action_dim))
        tensor_action[0, 0, action] = 1
        return tensor_action

    def state_to_state_vec_dqn(self, state):
        """ from state representation to state vector used in dqn """
        x, y, _, _ = state
        w, h = self.grid.shape[1], self.grid.shape[0]
        out = np.zeros((1, w * h))
        out[0, x * w + y] = 1
        out_torch = torch.from_numpy(out).float()
        return out_torch

    def reset(self):
        self.agent_pos = [0, 0]
        self.num_step = 0
        self.act_seq = []
        self.rnn_gridworld.init_hidden()

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

        self.num_step += 1
        self.act_seq.append(action)
        done = self.grid[self.agent_pos[0], self.agent_pos[1]] == G or self.num_step > 15
        if done:
            if self.num_step > 15:
                reward0 = 0
            else:
                reward0 = 10
            reward1 = self.cal_rnn_reward()
            reward2 = self.num_step
            reward = reward0 + reward1 - reward2
        else:
            reward = 0
        reward = torch.tensor([reward], device=self.device).float()
        return self.cur(), self.invalid_actions(), reward, done, None

    def cal_rnn_reward(self):
        tensor_action = torch.zeros((1, len(self.act_seq), self.action_dim))
        for i, act in enumerate(self.act_seq):
            tensor_action[0, i, act] = 1

        self.rnn_gridworld.init_hidden()
        reward = self.rnn_gridworld(tensor_action)
        return reward

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

    def print_memory(self, net, i_episode, state, action, invalid_actions,
                     next_state, reward, last_output, next_invalid_actions,
                     epsilon, verbose):
        state_x, state_y = self.state_to_x_y(state)
        text_act = ['L', 'R', 'U', 'D'][action]
        if next_state is None:
            text_next_state = '     G,  '
        else:
            next_state_x, next_state_y = self.state_to_x_y(next_state)
            text_next_state = (next_state_x, next_state_y), ", "
        # print if verbose=True or verbose=False && terminal state or verbose=False && test
        if verbose or next_state is None or i_episode == 'test':
            print('episode',
                  i_episode,
                  'push to mem:',
                  (state_x, state_y),
                  text_next_state,
                  text_act, "_", action,
                  ', reward:', reward.numpy()[0],
                  ', eps:', epsilon,
                  ', invalid actions', invalid_actions,
                  'mem size:', len(net.memory))

            last_output_np = last_output.detach().numpy()[0]
            last_output_np[next_invalid_actions] = 0
            print("next output", last_output_np)
