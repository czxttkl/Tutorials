import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.optim as optim

from env.finite_gridworld_env import GridWorldEnv
from model.lstm import LSTM
from model.dqn import DQN

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'invalid_actions', 'next_state', 'reward'))


def test(env, i_train_episode, test_episode_durations, policy_net):
    print("-------------test at {} train episode------------".format(i_train_episode))
    # Initialize the environment and state
    env.reset()
    state = env.cur()
    last_output = None
    # reset the LSTM hidden state. Must be done before you run a new episode. Otherwise the LSTM will treat
    # a new episode as a continuation of the last episode
    # for other models, this function will do nothing
    policy_net.init_hidden(batch_size=1)

    for t in count():
        # Select and perform an action
        invalid_actions = env.invalid_actions()
        action_dim = env.action_dim
        action = policy_net.select_action(
            t, state, last_output, invalid_actions, action_dim, 0)

        # env step
        next_state, reward, done, _ = env.step(action)

        # needs to compute Q(s', a) for all a, which will be used in the next iteration
        last_output = policy_net.output(state, action, next_state)

        policy_net.print_memory(
            env, 'test',
            state, action,
            invalid_actions, next_state, reward,
            last_output
        )

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


def train(model_str):
    episode_durations = []
    test_episode_durations = []
    env = GridWorldEnv(device)

    if model_str == 'lstm':
        lstm_input_dim = lstm_output_dim = env.action_dim
        lstm_hidden_dim = 50
        lstm_num_layer = 2
        policy_net = LSTM(lstm_input_dim, lstm_num_layer,
                          lstm_hidden_dim, lstm_output_dim,
                          GAMMA, REPLAY_MEMORY_SIZE, BATCH_SIZE)\
                     .to(device)
    elif model_str == 'dqn':
        dqn_input_dim = env.grid.shape[0] * env.grid.shape[1]
        dqn_hidden_dim = 168
        dqn_num_layer = 2
        dqn_output_dim = env.action_dim
        policy_net = DQN(dqn_input_dim, dqn_num_layer,
                         dqn_hidden_dim, dqn_output_dim,
                         GAMMA, REPLAY_MEMORY_SIZE, BATCH_SIZE)\
                     .to(device)
    else:
        raise Exception

    print(policy_net)
    print("trainable param num:", policy_net.num_of_params())
    policy_net.optimizer = optim.RMSprop(policy_net.parameters())

    for i_episode in range(NUM_EPISODES):
        # Initialize the environment and state
        env.reset()
        state = env.cur()
        episode_memory = []
        last_output = None
        # reset the LSTM hidden state. Must be done before you run a new episode. Otherwise the LSTM will treat
        # a new episode as a continuation of the last episode
        # for other models, this function will do nothing
        policy_net.init_hidden(batch_size=1)

        for t in count():
            # Select and perform an action
            invalid_actions = env.invalid_actions()
            action_dim = env.action_dim
            action = policy_net.select_action(
                t, state, last_output, invalid_actions, action_dim, EPS_THRES)

            # env step
            next_state, reward, done, _ = env.step(action)

            # needs to compute Q(s',a) for all a, which will be used in the next iteration
            last_output = policy_net.output(state, action, next_state)

            if done:
                next_state = None

            policy_net.print_memory(
                env, i_episode,
                state, action,
                invalid_actions, next_state, reward,
                last_output
            )

            step_transition = Transition(state=state, action=action, invalid_actions=invalid_actions,
                                         next_state=next_state, reward=reward)
            episode_memory.append(step_transition)

            # Move to the next state
            state = next_state

            if done:
                episode_durations.append(t + 1)
                break

        # Store the episode transitions in memory
        policy_net.memory.push(episode_memory)

        if i_episode % TEST_EVERY_EPISODE == 0:
            test(env, i_episode, test_episode_durations, policy_net)

        # Perform one step of the optimization
        if i_episode > LEARNING_START_EPISODES:
            policy_net.optimize_model(env)

    print('Complete Training')
    # print(np.average(episode_durations))
    # print(episode_durations)
    return test_episode_durations


def main(model_str):
    test_episode_durations = []
    for i in range(TRAIN_TIMES):
        t = train(model_str)
        test_episode_durations.append(t)

    test_episode_durations = np.vstack(test_episode_durations)
    test_episode_durations_ave = np.average(test_episode_durations, axis=0)
    test_episode_durations_std = np.std(test_episode_durations, axis=0)
    for i, (ave, std) in enumerate(zip(test_episode_durations_ave, test_episode_durations_std)):
        print("Episode {}: {:.2f} +- {:.2f}".format(i * TEST_EVERY_EPISODE, ave, std))


if __name__ == '__main__':
    TRAIN_TIMES = 1
    BATCH_SIZE = 4
    GAMMA = 0.9
    TEST_EVERY_EPISODE = 10
    EPS_THRES = 0.4
    REPLAY_MEMORY_SIZE = 1000
    NUM_EPISODES = 2001
    LEARNING_START_EPISODES = 500

    model_str = 'lstm'

    main(model_str)



