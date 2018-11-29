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

    def __init__(self, device):
        # x, y
        self.agent_pos = [0, 0]
        self.back_step = 0
        # can only go back (L or U) twice
        self.back_step_thres = 400000

        # self.grid = np.array(
        #     [
        #         [S, 0, 0, G],
        #     ]
        # )

        self.grid = np.array(
            [
                [S, 0, 0, 0],
                [0, W, W, 0],
                [0, W, G, 0],
                [0, 0, 0, 0],
            ]
        )

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

    def cur(self):
        x, y = self.agent_pos
        w, h = self.grid.shape[1], self.grid.shape[0]
        out = np.zeros(w * h + 1)
        out[x * w + y] = 1
        out[-1] = self.back_step
        out_torch = torch.from_numpy(out).float()
        return out_torch.unsqueeze(0).to(self.device)

    def state_vec_from_x_y_backstep(self, x, y, backstep):
        w, h = self.grid.shape[1], self.grid.shape[0]
        out = np.zeros(w * h + 1)
        out[x * w + y] = 1
        out[-1] = backstep
        out_torch = torch.from_numpy(out).float()
        return out_torch.unsqueeze(0).to(self.device)

    def state_vec_to_x_y_backstep(self, state):
        pos = (state[0, :-1].squeeze() == 1).nonzero().item()
        w, h = self.grid.shape[1], self.grid.shape[0]
        x, y = pos // w, pos % w
        backstep = state[0, -1].item()
        return x, y, backstep

    def reset(self):
        self.agent_pos = [0, 0]
        self.back_step = 0

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
        reward = 10 if done else 0
        reward = torch.tensor([reward], device=self.device).float()
        if action == L or action == U:
            self.back_step += 1
        return self.cur(), reward, done, None

    def invalid_actions(self):
        x, y = self.agent_pos
        return self.invalid_actions_by_x_y_backstep(x, y, self.back_step)

    def invalid_actions_by_x_y_backstep(self, x, y, bs):
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
        if bs >= self.back_step_thres and len(ia) < 3:
            if L not in ia:
                ia.append(L)
            if U not in ia:
                ia.append(U)
        return ia

    def invalid_actions_by_state(self, state):
        x, y, bs = self.state_vec_to_x_y_backstep(state)
        invalid_actions = self.invalid_actions_by_x_y_backstep(x, y, bs)
        return invalid_actions
