# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from finite_gridworld_env import GridWorldEnv


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'invalid_actions', 'next_state', 'reward'))


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


action_dim = lstm_input_dim = lstm_output_dim = 4
lstm_hidden_dim = 50
lstm_num_layer = 2


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True,
                            num_layers=lstm_num_layer)
        self.fc = nn.Linear(lstm_hidden_dim, lstm_output_dim)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (minibatch_size, num_layers, hidden_dim)
        self.hidden = torch.zeros(lstm_num_layer, batch_size, lstm_hidden_dim), \
                      torch.zeros(lstm_num_layer, batch_size, lstm_hidden_dim)

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


BATCH_SIZE = 4
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TEST_EVERY_EPISODE = 10
EPS_THRES = 0.4


def select_action(env, step, last_lstm_output, eps_thres):
    sample = random.random()
    iv_act = env.invalid_actions()
    # the first action is randomly selected
    if step == 0:
        action = random.randrange(4)
        while action in iv_act:
            action = random.randrange(4)
        tensor_action = torch.zeros((1, 1, action_dim))
        tensor_action[0, 0, action] = 1
        return tensor_action
    # greedy-epsilon
    elif sample > eps_thres:
        with torch.no_grad():
            last_lstm_output[0, iv_act] = -999999.9
            action = last_lstm_output.max(1)[1].detach().item()
            tensor_action = torch.zeros((1, 1, action_dim))
            tensor_action[0, 0, action] = 1
            return tensor_action
    else:
        action = random.randrange(4)
        while action in iv_act:
            action = random.randrange(4)
        tensor_action = torch.zeros((1, 1, action_dim))
        tensor_action[0, 0, action] = 1
        return tensor_action


def optimize_model(env, memory, policy_net, optimizer):
    if len(memory) < BATCH_SIZE * 2:
        return

    # refresh policy_net hidden state
    policy_net.init_hidden(batch_size=BATCH_SIZE)

    transitions = memory.sample(BATCH_SIZE)
    transitions_len = list(map(lambda x: len(x), transitions))
    max_seq_len = max(transitions_len)

    # lstm requires transitions sorted by lens
    dec_order_by_rand_len = np.argsort(transitions_len)[::-1]
    transitions = [transitions[i] for i in dec_order_by_rand_len]
    transitions_len = [transitions_len[i] for i in dec_order_by_rand_len]

    i = 0
    action_seqs = torch.zeros([BATCH_SIZE, max_seq_len, lstm_input_dim], dtype=torch.float)
    last_actions = torch.zeros([BATCH_SIZE, max_seq_len - 1, 1], dtype=torch.long)
    last_rewards = torch.zeros([BATCH_SIZE, max_seq_len - 1], dtype=torch.float)
    non_final_masks = torch.zeros([BATCH_SIZE, max_seq_len - 1], dtype=torch.float)
    non_valid_masks = torch.zeros([BATCH_SIZE, max_seq_len - 1], dtype=torch.float)
    invalid_action_masks = torch.ones([BATCH_SIZE, max_seq_len - 1, lstm_output_dim])

    for tran_len, transition in zip(transitions_len, transitions):
        transition_actions = list(map(lambda x: x.action, transition))
        transition_actions = torch.cat(transition_actions).squeeze(1)
        action_seqs[i, :tran_len, :] = transition_actions
        # last_actions[i, 0] = transition[rand_len].action
        transition_rewards = torch.tensor(list(map(lambda x: x.reward, transition)))
        last_rewards[i, :tran_len-1] = transition_rewards[1:]
        non_final_masks[i, :tran_len - 2] = 1
        non_valid_masks[i, :tran_len - 1] = 1
        transition_action_indexs = list(map(lambda x: x.action.squeeze().nonzero(), transition))
        last_actions[i, :tran_len - 1, :] = torch.cat(transition_action_indexs)[1:]
        for j, t in enumerate(transition[1:]):
            invalid_action_masks[i, j, t.invalid_actions] = 0
        i += 1

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    # shape: (batch_size, max_seq_len, lstm_output_dim)
    state_action_values_full = policy_net(action_seqs, transitions_len)
    # shape: (batch_size, max_seq_len - 1, lstm_output_dim)
    cur_state_action_values_full = state_action_values_full[:, :-1, :]
    # shape: (batch_size, max_seq_len - 1)
    cur_state_action_values = cur_state_action_values_full.gather(2, last_actions).squeeze(2)
    # shape: (batch_size, max_seq_len - 1)
    cur_state_action_values *= non_valid_masks

    # Compute Q(s_{t+1}, a_{t+1}) for all a_{t+1}.
    with torch.no_grad():
        # shape: (batch_size, max_seq_len - 1, lstm_output_dim)
        next_state_action_values_full = state_action_values_full[:, 1:, :]
        # shape: (batch_size, max_seq_len - 1, lstm_output_dim)
        next_state_action_values_full  *= invalid_action_masks
        # shape: (batch_size, max_seq_len - 1)
        next_state_max_action_values = next_state_action_values_full.max(2)[0]
        next_state_max_action_values *= non_final_masks
        expected_state_action_values = (next_state_max_action_values * GAMMA) + last_rewards
        expected_state_action_values *= non_valid_masks

    # Compute Huber loss
    loss = F.smooth_l1_loss(cur_state_action_values, expected_state_action_values.detach())
    print(loss.item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def print_memory(env, i_episode, state, action, invalid_actions, next_state, reward, last_lstm_output, memory):
    action = action.squeeze().nonzero().item()
    state_indx = np.argwhere(state.detach().numpy() == 1)[0, 1]
    w, h = env.grid.shape[1], env.grid.shape[0]
    text_act = ['L', 'R', 'U', 'D'][action]
    if next_state is None:
        print('episode', i_episode, 'push to mem:', (state_indx // w, state_indx % w), '     G,  ', text_act, "_", action, ', reward:', reward.numpy()[0], ', invalid actions', invalid_actions, 'mem size:', len(memory))
    else:
        next_state_indx = np.argwhere(next_state.detach().numpy() == 1)[0, 1]
        print('episode', i_episode, 'push to mem:', (state_indx // w, state_indx % w), (next_state_indx // w, next_state_indx % w), ", ", text_act, "_", action, ', reward:', reward.numpy()[0], ', invalid actions', invalid_actions, 'mem size:', len(memory))
    print("last lstm output", last_lstm_output.detach().numpy()[0])
    print()


def test(env, i_train_episode, test_episode_durations, policy_net, memory):
    print("-------------test at {} train episode------------".format(i_train_episode))
    # Initialize the environment and state
    env.reset()
    state = env.cur()
    last_lstm_output = None
    # reset the LSTM hidden state. Must be done before you run a new episode. Otherwise the LSTM will treat
    # a new episode as a continuation of the last episode
    policy_net.init_hidden(batch_size=1)

    for t in count():
        invalid_actions = env.invalid_actions()
        # Select and perform an action
        action = select_action(env, t, last_lstm_output, 0)
        next_state, reward, done, _ = env.step(action.squeeze().nonzero().item())

        last_lstm_output = policy_net(action, [1])[0, :, :]

        print_memory(env, 0, state, action, invalid_actions, next_state, reward, last_lstm_output, memory)

        # Move to the next state
        state = next_state

        if done:
            test_episode_durations.append(t + 1)
            break

        if t > 30:
            test_episode_durations.append(30)
            break

    print('Complete Testing')
    print(test_episode_durations)
    print("-------------test at {} train episode------------".format(i_train_episode))


######################################################
#                       Train                        #
######################################################

def train():
    num_episodes = 1501
    episode_durations = []
    test_episode_durations = []
    env = GridWorldEnv(device)

    policy_net = LSTM().to(device)
    print(policy_net)
    print("trainable param num:", sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
    optimizer = optim.RMSprop(policy_net.parameters())
    replay_capacity = 1000
    memory = ReplayMemory(replay_capacity)

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = env.cur()
        episode_memory = []
        last_lstm_output = None
        # reset the LSTM hidden state. Must be done before you run a new episode. Otherwise the LSTM will treat
        # a new episode as a continuation of the last episode
        policy_net.init_hidden(batch_size=1)

        for t in count():
            # Select and perform an action
            invalid_actions = env.invalid_actions()
            action = select_action(env, t, last_lstm_output, EPS_THRES)
            next_state, reward, done, _ = env.step(action.squeeze().nonzero().item())

            last_lstm_output = policy_net(action, [1])[0, :, :]

            # Observe new state
            if done:
                next_state = None

            print_memory(env, i_episode, state, action, invalid_actions, next_state, reward, last_lstm_output, memory)

            step_transition = Transition(state=state, action=action, invalid_actions=invalid_actions, next_state=next_state, reward=reward)
            episode_memory.append(step_transition)

            # Move to the next state
            state = next_state

            # plot if done
            if done:
                episode_durations.append(t + 1)
                break

        # Store the episode transitions in memory
        memory.push(episode_memory)

        if i_episode % TEST_EVERY_EPISODE == 0:
            test(env, i_episode, test_episode_durations, policy_net, memory)

        # Perform one step of the optimization
        optimize_model(env, memory, policy_net, optimizer)

    print('Complete Training')
    print(np.average(episode_durations))
    print(episode_durations)
    return test_episode_durations


def main():
    test_episode_durations = []
    for i in range(5):
        t = train()
        test_episode_durations.append(t)

    test_episode_durations = np.vstack(test_episode_durations)
    test_episode_durations_ave = np.average(test_episode_durations, axis=0)
    test_episode_durations_std = np.std(test_episode_durations, axis=0)
    for i, (ave, std) in enumerate(zip(test_episode_durations_ave, test_episode_durations_std)):
        print("Episode {}: {:.2f} +- {:.2f}".format(i*TEST_EVERY_EPISODE, ave, std))


main()



