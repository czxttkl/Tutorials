import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, transitions):
            """ Saves transitions of an episode """
            for transition in transitions:
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = transition
                self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            # sample with replacement
            return random.choices(self.memory, k=batch_size)

        def __len__(self):
            return len(self.memory)

    def __init__(self, dqn_input_dim, dqn_num_layer, dqn_hidden_dim, dqn_output_dim,
                 gamma, replay_memory_size, training_batch_size):
        super(DQN, self).__init__()
        h_sizes = [dqn_input_dim] + [dqn_hidden_dim] * dqn_num_layer + [dqn_output_dim]
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        self.memory = DQN.ReplayMemory(replay_memory_size)

        self.dqn_input_dim = dqn_input_dim
        self.dqn_num_layer = dqn_num_layer
        self.dqn_hidden_dim = dqn_hidden_dim
        self.dqn_output_dim = dqn_output_dim

        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.training_batch_size = training_batch_size

    def forward(self, x):
        for i in range(self.dqn_num_layer + 1):
            if i == self.dqn_num_layer:
                x = self.hidden[i](x)
            else:
                x = F.relu(self.hidden[i](x))
        return x

    def init_hidden(self, batch_size):
        pass

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def select_action(self, step, state, last_output, invalid_actions, action_dim, eps_thres):
        sample = random.random()
        # the first action is randomly selected
        if step == 0:
            action = random.randrange(4)
            while action in invalid_actions:
                action = random.randrange(4)
            return torch.tensor([[action]], dtype=torch.long), action
        # greedy-epsilon
        elif sample > eps_thres:
            with torch.no_grad():
                action_output = self(state[:, :-1])
                action_output[0, invalid_actions] = -999999.9
                action_vec = action_output.max(1)[1].detach().view(1, 1)
                return action_vec, action_vec.item()
        else:
            action = random.randrange(4)
            while action in invalid_actions:
                action = random.randrange(4)
            return torch.tensor([[action]], dtype=torch.long), action

    def optimize_model(self, env):
        BATCH_SIZE = self.training_batch_size

        transitions = self.memory.sample(BATCH_SIZE)

        # shape: (batch_size, state_dim)
        state_batch = torch.cat([t.state[:, :-1] for t in transitions])
        action_batch = torch.cat([t.action for t in transitions])
        reward_batch = torch.cat([t.reward for t in transitions])

        next_state_batch = torch.zeros([BATCH_SIZE, state_batch.size()[1]], dtype=torch.float)
        non_final_mask = torch.zeros([BATCH_SIZE, self.dqn_output_dim], dtype=torch.float)
        valid_action_masks = torch.ones([BATCH_SIZE, self.dqn_output_dim])

        for i, t in enumerate(transitions):
            if t.next_state is not None:
                non_final_mask[i, :] = 1
                next_state_batch[i, :] = t.next_state[:, :-1]
                inv_acts = env.invalid_actions_by_state(t.next_state)
                valid_action_masks[i, inv_acts] = 0

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_full_values = self(next_state_batch)
        next_state_full_values *= non_final_mask
        next_state_full_values *= valid_action_masks
        next_state_max_values = next_state_full_values.max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_max_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        print(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def output(self, state, action, action_env, next_state):
        """ output Q(s',a) for all a """
        return self(next_state[:, :-1])

    def print_memory(self, env, i_episode, state, action, action_env, invalid_actions,
                     next_state, reward, last_output):
        state_indx = np.argwhere(state.detach().numpy() == 1)[0, 1]
        w, h = env.grid.shape[1], env.grid.shape[0]
        text_act = ['L', 'R', 'U', 'D'][action_env]
        if next_state is None:
            text_next_state = '     G,  '
        else:
            next_state_indx = np.argwhere(next_state.detach().numpy() == 1)[0, 1]
            text_next_state = (next_state_indx // w, next_state_indx % w), ", "
        print('episode',
              i_episode,
              'push to mem:',
              (state_indx // w, state_indx % w),
              text_next_state,
              text_act, "_", action_env,
              ', reward:', reward.numpy()[0],
              ', invalid actions', invalid_actions,
              'mem size:', len(self.memory))
        print("last output", last_output.detach().numpy()[0])
        print()
