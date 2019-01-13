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
            print("push an episode of {} transitions, mem size {}\n"
                  .format(len(transitions), len(self.memory)))

        def sample(self, batch_size):
            # sample with replacement
            return random.choices(self.memory, k=batch_size)

        def __len__(self):
            return len(self.memory)

    def __init__(self, dqn_input_dim, dqn_num_layer, dqn_hidden_dim, dqn_output_dim,
                 parametric, gamma, replay_memory_size, training_batch_size):
        super(DQN, self).__init__()
        h_sizes = [dqn_input_dim] + [dqn_hidden_dim] * dqn_num_layer + [dqn_output_dim]
        self.layers = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.layers.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        self.memory = DQN.ReplayMemory(replay_memory_size)

        self.dqn_input_dim = dqn_input_dim
        self.dqn_num_layer = dqn_num_layer
        self.dqn_hidden_dim = dqn_hidden_dim
        self.dqn_output_dim = dqn_output_dim

        self.parametric = parametric
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.training_batch_size = training_batch_size

    def forward(self, x):
        for i in range(self.dqn_num_layer + 1):
            if i == self.dqn_num_layer:
                x = self.layers[i](x)
            else:
                x = F.relu(self.layers[i](x))
        return x

    def init_hidden(self, batch_size):
        pass

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def select_action(self, env, step, state, last_output, invalid_actions, action_dim, eps_thres):
        sample = random.random()
        if sample > eps_thres:
            self.eval()
            with torch.no_grad():
                action_output = self(env.state_to_state_vec_dqn(state))
                action_output[0, invalid_actions] = -999999.9
                action = action_output.max(1)[1].detach().item()
            self.train()
        else:
            action = random.randrange(action_dim)
            while action in invalid_actions:
                action = random.randrange(action_dim)
        return action

    def optimize_model(self, env, target_net=None):
        BATCH_SIZE = self.training_batch_size

        transitions = self.memory.sample(BATCH_SIZE)

        # shape: (batch_size, state_dim)
        state_batch = torch.cat([env.state_to_state_vec_dqn(t.state) for t in transitions])
        action_batch = torch.cat([torch.tensor([[t.action]], dtype=torch.long) for t in transitions])
        reward_batch = torch.cat([t.reward for t in transitions])
        done_batch = torch.cat([torch.tensor([t.done], dtype=torch.float) for t in transitions])

        next_state_batch = torch.zeros([BATCH_SIZE, state_batch.size()[1]], dtype=torch.float)
        valid_action_masks = torch.ones([BATCH_SIZE, self.dqn_output_dim])

        for i, t in enumerate(transitions):
            if t.done != 1:
                next_state_batch[i, :] = env.state_to_state_vec_dqn(t.next_state)
                inv_acts = env.invalid_actions_by_state(t.next_state)
                valid_action_masks[i, inv_acts] = 0

        # Compute Q(s_{t+1}, a)
        if target_net:
            next_state_full_values = target_net(next_state_batch).detach()
        else:
            next_state_full_values = self(next_state_batch).detach()
        next_state_full_values *= valid_action_masks
        next_state_max_values = next_state_full_values.max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_max_values * self.gamma * (1 - done_batch)) + reward_batch
        # expected_state_action_values = expected_state_action_values.detach().numpy()
        # expected_state_action_values = torch.from_numpy(expected_state_action_values)

        # if parametric is True, only fit the q-values of the states/actions that have been visited
        if self.parametric:
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            state_action_values = self(state_batch).gather(1, action_batch)
            loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        else:
            state_full_values = self(state_batch)
            state_full_values_to_fit = state_full_values.clone().detach()
            state_full_values_to_fit[list(range(BATCH_SIZE)), action_batch.squeeze()] = expected_state_action_values
            # Compute MSE loss
            loss = F.mse_loss(state_full_values, state_full_values_to_fit)

        # Optimize the model
        print('\rloss:', loss.item(), end='')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def output(self, env, action, next_state):
        """ output Q(s',a) for all a """
        # return self(next_state[:, :-1])
        next_state_vec = env.state_to_state_vec_dqn(next_state)
        self.eval()
        with torch.no_grad():
            output = self(next_state_vec)
        self.train()
        return output

    # use for test
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).unsqueeze(0).float()
        sample = random.random()
        if sample > eps:
            self.eval()
            with torch.no_grad():
                action_output = self(state)
                action_output[0, []] = -999999.9
                action = action_output.max(1)[1].detach().item()
            self.train()
        else:
            action = random.randrange(self.dqn_output_dim)
            while action in []:
                action = random.randrange(self.dqn_output_dim)
        return action

    def step(self, state, action, reward, next_state, done):
        SOFT_UPDATE_EVERY = 2
        BATCH_SIZE = 64
        GAMMA = 0.99
        TAU = 1
        # Learn every UPDATE_EVERY time steps.
        # self.t_step = (self.t_step + 1) % LEARN_EVERY
        # if self.t_step == 0:
        self.t_step += 1
        if done:
            for _ in range(self.t_step // SOFT_UPDATE_EVERY):
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    # self.learn_DDQN(experiences, GAMMA)
                    self.learn(experiences, GAMMA)
            self.t_step = 0

    def learn(self, experiences, gamma):
        BATCH_SIZE = 64
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = experiences
        valid_action_masks = torch.ones([BATCH_SIZE, self.dqn_output_dim])

        for i, next_state in enumerate(done_batch):
            inv_acts = []
            valid_action_masks[i, inv_acts] = 0

        next_state_full_values = self(next_state_batch).detach()
        next_state_full_values *= valid_action_masks
        next_state_max_values = next_state_full_values.max(1)[0].detach().unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_max_values * self.gamma * (1 - done_batch)) + reward_batch
        # expected_state_action_values = expected_state_action_values.detach().numpy()
        # expected_state_action_values = torch.from_numpy(expected_state_action_values)

        # if parametric is True, only fit the q-values of the states/actions that have been visited
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self(state_batch).gather(1, action_batch)
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        # Optimize the model
        print('loss:', loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
