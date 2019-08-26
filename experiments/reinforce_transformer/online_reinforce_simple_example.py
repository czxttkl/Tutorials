import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn
from torch import optim
from reinforce_transformer_classes import ReinforceLossCompute


class PolicyEstimator(nn.Module):
    def __init__(self, env):
        super(PolicyEstimator, self).__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def forward(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


class Baseline(nn.Module):
    def __init__(self, env):
        super(Baseline, self).__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = 1

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs))

    def forward(self, state):
        baseline_reward = self.network(torch.FloatTensor(state))
        return baseline_reward


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    baseline = Baseline(env)
    optimizer = optim.Adam(policy_estimator.parameters(), lr=0.01)
    baseline_optimizer = optim.Adam(baseline.parameters())
    reinforce_trainer = ReinforceLossCompute(True, optimizer, baseline_optimizer)

    action_space = np.arange(env.action_space.n)
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # my reinforce
                # If batch is complete, update network
                if batch_counter == batch_size:
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)
                    # Calculate loss
                    logprob = torch.log(policy_estimator(state_tensor))
                    logprob = logprob[np.arange(len(action_tensor)), action_tensor]
                    neg_reward_tensor = -reward_tensor
                    reinforce_trainer(logprob, neg_reward_tensor, baseline, state_tensor)

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                # original reinfoce
                # If batch is complete, update network
                # if batch_counter == batch_size:
                #     optimizer.zero_grad()
                #     state_tensor = torch.FloatTensor(batch_states)
                #     reward_tensor = torch.FloatTensor(batch_rewards)
                #     # Actions are used as indices, must be LongTensor
                #     action_tensor = torch.LongTensor(batch_actions)
                #
                #     # Calculate loss
                #     logprob = torch.log(
                #         policy_estimator(state_tensor))
                #
                #     selected_logprobs = reward_tensor * \
                #                         logprob[np.arange(len(action_tensor)), action_tensor]
                #     loss = -selected_logprobs.mean()
                #
                #     # Calculate gradients
                #     loss.backward()
                #     # Apply gradients
                #     optimizer.step()
                #
                #     batch_rewards = []
                #     batch_actions = []
                #     batch_states = []
                #     batch_counter = 1

                # Print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")

    return total_rewards




env = gym.make('CartPole-v0')
s = env.reset()
pe = PolicyEstimator(env)
print(pe(s))
print(pe(torch.FloatTensor(s)))

rewards = reinforce(env, pe)
window = 10
smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window
                    else np.mean(rewards[:i+1]) for i in range(len(rewards))]
print(rewards)

plt.figure(figsize=(12,8))
plt.plot(rewards)  # blue lines
plt.plot(smoothed_rewards)   # orange lines
plt.ylabel('Total Rewards')
plt.xlabel('Episodes')
plt.savefig("aaa.png")

