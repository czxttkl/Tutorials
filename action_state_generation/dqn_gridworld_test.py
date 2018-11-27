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

    def push(self, transitions):
        """Saves a transition."""
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


dqn_hidden_num = 1500
dqn_output_num = action_dim = 4


class DQN(nn.Module):

    def __init__(self, env):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(env.grid.shape[0] * env.grid.shape[1], dqn_hidden_num)
        self.fc2 = nn.Linear(dqn_hidden_num, dqn_output_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x


BATCH_SIZE = 4
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TEST_EVERY_EPISODE = 10
EPS_THRES = 0.4


def select_action(env, step, state, eps_thres, policy_net):
    sample = random.random()
    # the first action is randomly selected
    iv_act = env.invalid_actions()
    if step == 0:
        action = random.randrange(4)
        while action in iv_act:
            action = random.randrange(4)
        return torch.tensor([[action]], device=device, dtype=torch.long)
    # greedy-epsilon
    elif sample > eps_thres:
        with torch.no_grad():
            action_output = policy_net(state[:, :-1])
            action_output[0, iv_act] = -999999.9
            action = action_output.max(1)[1].detach().view(1, 1)
            return action
    else:
        action = random.randrange(4)
        while action in iv_act:
            action = random.randrange(4)
        return torch.tensor([[action]], device=device, dtype=torch.long)


def optimize_model(env, memory, policy_net, optimizer):
    if len(memory) < BATCH_SIZE * 2:
        return

    transitions = memory.sample(BATCH_SIZE)

    # shape: (batch_size, state_dim)
    state_batch = torch.cat([t.state[:, :-1] for t in transitions])
    action_batch = torch.cat([t.action for t in transitions])
    reward_batch = torch.cat([t.reward for t in transitions])

    next_state_batch = torch.zeros([BATCH_SIZE, state_batch.size()[1]], device=device, dtype=torch.float)
    non_final_mask = torch.zeros([BATCH_SIZE, dqn_output_num], device=device, dtype=torch.float)
    invalid_action_masks = torch.ones([BATCH_SIZE, dqn_output_num])

    for i, t in enumerate(transitions):
        if t.next_state is not None:
            non_final_mask[i, :] = 1
            next_state_batch[i, :] = t.next_state[:, :-1]
            inv_acts = env.invalid_actions_by_state(t.next_state)
            invalid_action_masks[i, inv_acts] = 0

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_full_values = policy_net(next_state_batch)
    next_state_full_values *= non_final_mask
    next_state_full_values *= invalid_action_masks
    next_state_max_values = next_state_full_values.max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_max_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def print_memory(env, i_episode, state, action, invalid_actions, next_state, reward, memory):
    state_indx = np.argwhere(state.detach().numpy() == 1)[0, 1]
    w, h = env.grid.shape[1], env.grid.shape[0]
    text_act = ['L', 'R', 'U', 'D'][action]
    if next_state is None:
        print(i_episode, 'push to mem:', (state_indx // w, state_indx % w), '     G,  ', text_act, ", ", action.numpy()[0, 0], reward, ', invalid actions', invalid_actions, 'mem size:', len(memory))
    else:
        next_state_indx = np.argwhere(next_state.detach().numpy() == 1)[0, 1]
        print(i_episode, 'push to mem:', (state_indx // w, state_indx % w), (next_state_indx // w, next_state_indx % w), ", ", text_act, ",", action.numpy()[0, 0], reward, ', invalid actions', invalid_actions, 'mem size:', len(memory))


def test(env, i_train_episode, test_episode_durations, policy_net, memory):
    print("-------------test at {} train episode------------".format(i_train_episode))
    action_map = np.chararray(env.grid.shape, unicode=True)
    for y in range(env.grid.shape[1]):
        for x in range(env.grid.shape[0]):
            dummy_state = env.state_vec_from_x_y_backstep(x, y, 0)
            output = policy_net(dummy_state[:, :-1]).detach().numpy()[0]
            action = ['L', 'R', 'U', 'D'][np.argmax(output)]
            action_map[x, y] = action
            print("x={}, y={}, act={}, q value: L={:.3f}, R={:.3f}, U={:.3f}, D={:.3f}"
                  .format(x, y, action, output[0], output[1], output[2], output[3]))
    print(action_map)

    # Initialize the environment and state
    env.reset()
    state = env.cur()

    for t in count():
        invalid_actions = env.invalid_actions()
        # Select and perform an action
        action = select_action(env, t, state, 0, policy_net)
        next_state, reward, done, _ = env.step(action.item())

        print_memory(env, 0, state, action, invalid_actions, next_state, reward, memory)

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

    policy_net = DQN(env).to(device)
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

        for t in count():
            # Select and perform an action
            invalid_actions = env.invalid_actions()
            action = select_action(env, t, state, EPS_THRES, policy_net)
            next_state, reward, done, _ = env.step(action.item())

            # Observe new state
            if done:
                next_state = None

            print_memory(env, i_episode, state, action, invalid_actions, next_state, reward, memory)

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
        print()

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


if __name__ == '__main__':
    main()



