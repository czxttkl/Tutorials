import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LSTM(nn.Module):

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0
            self.tran_len = []

        def push(self, transitions):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
                self.tran_len.append(None)
            self.memory[self.position] = transitions
            self.tran_len[self.position] = len(transitions)
            self.position = (self.position + 1) % self.capacity
            print("push an episode of {} transitions, mem size {}, ave mem size {}"
                  .format(len(transitions), len(self.memory), np.average(self.tran_len)))

        def sample(self, batch_size):
            # sample with replacement
            return random.choices(self.memory, k=batch_size)

        def __len__(self):
            return len(self.memory)

    def __init__(self, lstm_input_dim, lstm_num_layer, lstm_hidden_dim, lstm_output_dim,
                 gamma, replay_memory_size, training_batch_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True,
                            num_layers=lstm_num_layer)
        self.fc = nn.Linear(lstm_hidden_dim, lstm_output_dim)
        self.memory = LSTM.ReplayMemory(replay_memory_size)
        self.lstm_input_dim = lstm_input_dim
        self.lstm_num_layer = lstm_num_layer
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.training_batch_size = training_batch_size

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (minibatch_size, num_layers, hidden_dim)
        self.hidden = torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim), \
                      torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim)

    def forward(self, X, X_lengths):
        # X shape: (batch_size, lstm_seq_max_length, lstm_input_dim)
        # X_lengths: (batch_size), lengths of each seq (not including padding)

        # 1. Run through RNN
        # Dim transformation: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, lstm_hidden_dim)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)
        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # 2. Run through actual linear layer
        # shape (batch_size, lstm_seq_max_length, lstm_output_dim)
        # Note, zero-padded elements from the last step will still become non-zero after passing
        # the fully connected layer
        X = self.fc(X)

        return X

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def select_action(self, step, last_lstm_output, invalid_actions, action_dim, eps_thres):
        """ return action vector and action type accepted by env"""
        sample = random.random()
        # the first action is randomly selected
        if step == 0:
            action = random.randrange(4)
            while action in invalid_actions:
                action = random.randrange(4)
        # greedy-epsilon
        elif sample > eps_thres:
            with torch.no_grad():
                last_lstm_output[0, invalid_actions] = -999999.9
                action = last_lstm_output.max(1)[1].detach().item()
        else:
            action = random.randrange(4)
            while action in invalid_actions:
                action = random.randrange(4)

        tensor_action = torch.zeros((1, 1, action_dim))
        tensor_action[0, 0, action] = 1

        return tensor_action, tensor_action.squeeze().nonzero().item()

    def optimize_model(self, env):
        BATCH_SIZE = self.training_batch_size

        if len(self.memory) < BATCH_SIZE * 2:
            return

        # refresh policy_net hidden state
        self.init_hidden(batch_size=BATCH_SIZE)

        transitions = self.memory.sample(BATCH_SIZE)
        transitions_len = list(map(lambda x: len(x), transitions))
        max_seq_len = max(transitions_len)

        # lstm requires transitions sorted by lens
        dec_order_by_rand_len = np.argsort(transitions_len)[::-1]
        transitions = [transitions[i] for i in dec_order_by_rand_len]
        transitions_len = [transitions_len[i] for i in dec_order_by_rand_len]

        i = 0
        action_vecs = torch.zeros([BATCH_SIZE, max_seq_len, self.lstm_input_dim], dtype=torch.float)
        action_indexs = torch.zeros([BATCH_SIZE, max_seq_len, 1], dtype=torch.long)
        last_rewards = torch.zeros([BATCH_SIZE, max_seq_len], dtype=torch.float)
        non_final_masks = torch.zeros([BATCH_SIZE, max_seq_len - 1], dtype=torch.float)
        valid_masks = torch.zeros([BATCH_SIZE, max_seq_len - 1], dtype=torch.float)
        valid_action_masks = torch.ones([BATCH_SIZE, max_seq_len, self.lstm_output_dim])

        for tran_len, transition in zip(transitions_len, transitions):
            transition_actions = list(map(lambda x: x.action, transition))
            transition_actions = torch.cat(transition_actions).squeeze(1)
            action_vecs[i, :tran_len, :] = transition_actions
            transition_action_indexs = list(map(lambda x: x.action.squeeze().nonzero(), transition))
            action_indexs[i, :tran_len, :] = torch.cat(transition_action_indexs)
            transition_rewards = torch.tensor(list(map(lambda x: x.reward, transition)))
            last_rewards[i, :tran_len] = transition_rewards
            non_final_masks[i, :tran_len - 2] = 1
            valid_masks[i, :tran_len - 1] = 1
            for j, t in enumerate(transition[1:]):
                valid_action_masks[i, j, t.invalid_actions] = 0
            i += 1

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        # shape: (batch_size, max_seq_len, lstm_output_dim)
        state_action_values_full = self(action_vecs, transitions_len)
        # shape: (batch_size, max_seq_len - 1, lstm_output_dim)
        cur_state_action_values_full = state_action_values_full[:, :-1, :]
        # shape: (batch_size, max_seq_len - 1)
        cur_state_action_values = cur_state_action_values_full.gather(2, action_indexs[:, 1:, :]).squeeze(2)
        # shape: (batch_size, max_seq_len - 1)
        cur_state_action_values *= valid_masks

        # Compute Q(s_{t+1}, a_{t+1}) for all a_{t+1}.
        with torch.no_grad():
            # shape: (batch_size, max_seq_len - 1, lstm_output_dim)
            next_state_action_values_full = state_action_values_full[:, 1:, :].detach()
            # shape: (batch_size, max_seq_len - 1, lstm_output_dim)
            next_state_action_values_full *= valid_action_masks[:, 1:, :]
            # shape: (batch_size, max_seq_len - 1)
            next_state_max_action_values = next_state_action_values_full.max(2)[0]
            next_state_max_action_values *= non_final_masks
            expected_state_action_values = (next_state_max_action_values * self.gamma) + last_rewards[:, 1:]
            expected_state_action_values *= valid_masks

        # Compute Huber loss
        loss = F.smooth_l1_loss(cur_state_action_values, expected_state_action_values.detach())
        print(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def output(self, state, action):
        """ output Q(s,a) """
        assert len(action) == 1
        # shape: (1, lstm_output_dim)
        return self(action, [1])[0, :, :]

    def print_memory(self, env, i_episode, state, action, action_env, invalid_actions, next_state, reward, last_output):
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
