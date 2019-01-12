"""
An open AI gym environment "LunarLander-v2"
State space is 8-dimensional continuous space. Action space is 4 discrete actions.
"""
import numpy as np
import torch
import gym


class LunarEnv:

    def __init__(self, device):
        self.env = gym.make("LunarLander-v2")
        self.device = device
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.cur_state = self.env.reset()

    def test_step_limit(self):
        """ test step limit. If test steps beyond this limit, terminate the episode """
        return 1001

    def cur(self):
        return self.cur_state

    def action_to_action_vec_lstm(self, action):
        tensor_action = torch.zeros((1, 1, self.action_dim))
        tensor_action[0, 0, action] = 1
        return tensor_action

    def state_to_state_vec_dqn(self, state):
        """ from state representation to state vector used in dqn """
        out_torch = torch.from_numpy(state).unsqueeze(0).float()
        return out_torch

    def reset(self):
        self.cur_state = self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.cur_state = observation
        reward = torch.tensor([reward], device=self.device).float()
        return self.cur(), self.invalid_actions(), reward, done, None

    def invalid_actions(self):
        # always no invalid actions
        return []

    def invalid_actions_by_state(self, state):
        return []

    def print_memory(self, net, i_episode, state, action, invalid_actions,
                     next_state, reward, done, last_output, next_invalid_actions,
                     epsilon, verbose):
        np.set_printoptions(precision=2)
        text_act = 'ACT' + str(action)
        if done:
            text_next_state = '   G'
        else:

            text_next_state = str(next_state)
        # print if verbose=True or verbose=False && terminal state or verbose=False && test (i_episode == 'test')
        if verbose or done:
            print("episode {} push to mem: {}, next_state: {}, {}, reward: {}, eps: {}, mem size: {}"
                  .format(i_episode,
                          state,
                          text_next_state,
                          text_act,
                          reward.numpy()[0],
                          epsilon,
                          len(net.memory)
                          )
                  )
            last_output_np = last_output.detach().numpy()[0]
            last_output_np[next_invalid_actions] = 0
            print("next output", last_output_np)

