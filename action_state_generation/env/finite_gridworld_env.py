"""
A grid world environment
Rule of reward: after reaching R cell, get 10
"""
import numpy as np
import torch


W = 1  # Walls
S = 2  # Starting position
G = 3  # Goal position

L = 0
R = 1
U = 2
D = 3


class GridWorldEnv:

    def __init__(self, device, grid=None):
        # x, y
        self.agent_pos = [0, 0]
        # an episodes terminals when back_steps > 100
        self.back_steps = 0
        # self.grid = np.array(
        #     [
        #         [S, 0, 0, G],
        #     ]
        # ) if grid is None else grid

        # self.grid = np.array(
        #     [
        #         [S, 0, 0, 0],
        #         [0, W, W, 0],
        #         [0, W, G, 0],
        #         [0, 0, 0, 0],
        #     ]
        # ) if grid is None else grid

        self.grid = np.array(
            [
                [S, 0, 0],
                [W, 0, 0],
                [G, 0, 0],
            ]
        ) if grid is None else grid

        # self.grid = np.array(
        #     [
        #         [S, 0],
        #         [0, W],
        #         [0, G],
        #     ]
        # ) if grid is None else grid

        self.device = device
        self.action_dim = 4
        self.state_dim = self.grid.shape[0] * self.grid.shape[1]

    def test_step_limit(self):
        """ test step limit. If test steps beyond this limit, terminate the episode """
        return 30

    def cur(self):
        x, y = self.agent_pos
        return x, y

    def state_to_x_y(self, state):
        x, y = state
        return x, y

    def action_to_action_vec_lstm(self, action):
        tensor_action = torch.zeros((1, 1, self.action_dim))
        tensor_action[0, 0, action] = 1
        return tensor_action

    def state_to_state_vec_dqn(self, state):
        """ from state representation to state vector used in dqn """
        x, y = state
        w, h = self.grid.shape[1], self.grid.shape[0]
        out = np.zeros((1, w * h))
        out[0, x * w + y] = 1
        out_torch = torch.from_numpy(out).float()
        return out_torch

    def reset(self):
        self.agent_pos = [0, 0]
        self.back_steps = 0

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

        done = (self.grid[self.agent_pos[0], self.agent_pos[1]] == G) or (self.back_steps > 100)
        self.back_steps += 1
        reward = 10 if (self.grid[self.agent_pos[0], self.agent_pos[1]] == G) else 0
        reward = torch.tensor([reward], device=self.device).float()
        return self.cur(), self.invalid_actions(), reward, done, None

    def invalid_actions(self):
        x, y = self.cur()
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
        x, y = state
        invalid_actions = self.invalid_actions_by_x_y(x, y)
        return invalid_actions

    def print_memory(self, net, i_episode, state, action, invalid_actions,
                     next_state, reward, done, last_output, next_invalid_actions,
                     epsilon, verbose):
        state_x, state_y = self.state_to_x_y(state)
        text_act = ['L', 'R', 'U', 'D'][action]
        if done:
            text_next_state = '     G,  '
        else:
            next_state_x, next_state_y = self.state_to_x_y(next_state)
            text_next_state = (next_state_x, next_state_y), ", "
        # print if verbose=True or verbose=False && terminal state or verbose=False && test
        if verbose or done or i_episode == 'test':
            print('episode',
                  i_episode,
                  'push to mem:',
                  (state_x, state_y),
                  text_next_state,
                  text_act, "_", action,
                  ', reward:', reward.numpy()[0],
                  ', invalid actions', invalid_actions,
                  ', epsi:', epsilon,
                  'mem size:', len(net.memory))

            last_output_np = last_output.detach().numpy()[0]
            last_output_np[next_invalid_actions] = 0
            print("next output", last_output_np)
