import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.optim as optim
import pandas as pd
from plot_helper.plot_helper import plot_seaborn

from model.lstm import LSTM
from model.dqn import DQN
from helper.helper import(
    get_model, adjust_epsilon, soft_update, get_env
)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'invalid_actions', 'next_state', 'reward'))


def test(env, gamma, i_train_episode, policy_net, verbose):
    print("-------------test at {} train episode------------".format(i_train_episode))
    # Initialize the environment and state
    env.reset()
    state = env.cur()
    invalid_actions = env.invalid_actions()
    last_output = None
    final_accumulated_reward = 0
    # reset the LSTM hidden state. Must be done before you run a new episode. Otherwise the LSTM will treat
    # a new episode as a continuation of the last episode
    # for other models, this function will do nothing
    policy_net.init_hidden(batch_size=1)

    for t in count():
        # Select and perform an action
        action_dim = env.action_dim
        action = policy_net.select_action(
            env, t, state, last_output, invalid_actions, action_dim, 0)

        # env step
        next_state, next_invalid_actions, reward, done, _ = env.step(action)

        # needs to compute Q(s', a) for all a, which will be used in the next iteration
        last_output = policy_net.output(env, action, next_state)

        env.print_memory(
            policy_net,
            'test',
            state,
            action,
            invalid_actions,
            next_state,
            reward,
            done,
            last_output,
            next_invalid_actions,
            0,  # epsilon
            verbose,
        )

        # final_accumulated_reward += np.power(gamma, t) * reward.numpy()[0]
        # final reward (no discounted):
        final_accumulated_reward += reward.numpy()[0]
        # final reward (no accumulated):
        # final_accumulated_reward = reward.numpy()[0]

        # Move to the next state
        state = next_state
        invalid_actions = next_invalid_actions

        if done:
            duration = t + 1
            break

        if t > env.test_step_limit():
            duration = env.test_step_limit()
            break

    return duration, np.round(final_accumulated_reward, 2)


def train(model_str, env_str, test_times, batch_size, gamma, test_every_episode,
          replay_memory_size, num_episodes, learning_start_episodes,
          target_update_every_episode, epsilon_thres, verbose):
    episode_durations = []
    test_episode_durations = []
    test_episode_rewards = []

    env = get_env(env_str, device)
    policy_net, target_net = get_model(model_str, env, gamma, replay_memory_size, batch_size, device)

    print(policy_net)
    print("trainable param num:", policy_net.num_of_params())
    policy_net.optimizer = optim.Adam(policy_net.parameters())

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = env.cur()
        invalid_actions = env.invalid_actions()
        episode_memory = []
        last_output = None
        epsilon = adjust_epsilon(i_episode, num_episodes, learning_start_episodes, epsilon_thres)
        # reset the LSTM hidden state. Must be done before you run a new episode. Otherwise the LSTM will treat
        # a new episode as a continuation of the last episode
        # for other models, this function will do nothing
        policy_net.init_hidden(batch_size=1)

        for t in count():
            # Select and perform an action
            action_dim = env.action_dim
            action = policy_net.select_action(
                env, t, state, last_output, invalid_actions, action_dim, epsilon
            )

            # env step
            next_state, next_invalid_actions, reward, done, _ = env.step(action)

            # needs to compute Q(s',a) for all a, which will be used in the next iteration
            last_output = policy_net.output(env, action, next_state)

            if done:
                next_state = None

            env.print_memory(
                policy_net,
                i_episode,
                state,
                action,
                invalid_actions,
                next_state,
                reward,
                done,
                last_output,
                next_invalid_actions,
                epsilon,
                verbose,
            )

            step_transition = Transition(state=state, action=action, invalid_actions=invalid_actions,
                                         next_state=next_state, reward=reward)
            episode_memory.append(step_transition)

            # Move to the next state
            state = next_state
            invalid_actions = next_invalid_actions

            if done:
                episode_durations.append(t + 1)
                break

        # Store the episode transitions in memory
        policy_net.memory.push(episode_memory)

        # perform test
        if i_episode % test_every_episode == 0:
            results = [test(env, gamma, i_episode, policy_net, verbose) for _ in range(test_times)]
            test_duration, final_test_reward = np.mean([r[0] for r in results]), np.mean([r[1] for r in results])
            test_episode_durations.append(test_duration)
            test_episode_rewards.append(final_test_reward)
            print('Complete Testing')
            print(test_episode_durations)
            print(test_episode_rewards)
            print("-------------test at {} train episode------------".format(i_episode))

        # perform target network update
        if i_episode % target_update_every_episode == 0 and target_net:
            soft_update(policy_net, target_net, 1)

        # Perform one step of the optimization
        if i_episode > learning_start_episodes:
            for _ in range(t // target_update_every_episode):
                policy_net.optimize_model(env, target_net)

    print('Complete Training')
    return test_episode_durations, test_episode_rewards


def train_main(model_str, env_str, train_times, test_times,
               batch_size, gamma,
               test_every_episode, replay_memory_size,
               num_episodes, learning_start_episodes,
               epsilon_thres, target_update_every_episode, verbose, plot):
    test_durations = []
    test_rewards = []
    for i in range(train_times):
        duration, reward = train(
            model_str, env_str, test_times, batch_size, gamma,
            test_every_episode, replay_memory_size, num_episodes,
            learning_start_episodes, target_update_every_episode,
            epsilon_thres, verbose)
        test_durations.append(duration)
        test_rewards.append(reward)

    test_durations = np.vstack(test_durations)
    test_durations_ave = np.average(test_durations, axis=0)
    test_durations_std = np.std(test_durations, axis=0)
    print("------------- Test Steps --------------")
    for i, (ave, std) in enumerate(zip(test_durations_ave, test_durations_std)):
        print("Episode {}: {:.2f} +- {:.2f}".format(i * test_every_episode, ave, std))

    test_rewards = np.vstack(test_rewards)
    test_rewards_ave = np.average(test_rewards, axis=0)
    test_rewards_std = np.std(test_rewards, axis=0)

    print()

    print("------------- Test Rewards --------------")
    for i, (ave, std) in enumerate(zip(test_rewards_ave, test_rewards_std)):
        print("Episode {}: {:.2f} +- {:.2f}".format(i * test_every_episode, ave, std))

    if plot:
        reward_df = pd.DataFrame(
            {
                'reward': np.hstack(test_rewards),
                'epoch': np.tile(
                    np.arange(len(test_rewards_ave)) * test_every_episode,
                    len(test_rewards)
                )
            }
        )
        plot_seaborn(reward_df, xaxis='epoch', yaxis='reward',
                     title='reward_plot_{}_{}_tt{}_eps{}'.format(model_str, env_str, train_times, num_episodes),
                     file='reward_plot_{}_{}_tt{}_eps{}.png'.format(model_str, env_str, train_times, num_episodes),
                     show=False)
    return test_rewards_ave, test_durations_ave


if __name__ == '__main__':
    TRAIN_TIMES = 1
    TEST_TIMES = 100
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_THRES = 0.05
    TEST_EVERY_EPISODE = 100
    TARGET_UPDATE_EVERY_EPISODE = 2
    REPLAY_MEMORY_SIZE = int(1e5)
    NUM_EPISODES = 20001
    LEARNING_START_EPISODES = 0
    VERBOSE = False
    PLOT = True

    # model_str = 'lstm'
    model_str = 'dqn'
    # env_str = 'finite'
    # env_str = 'rnn'
    env_str = "lunar"
    # env_str = "cartpole"

    train_main(
        model_str,
        env_str,
        TRAIN_TIMES,
        TEST_TIMES,
        BATCH_SIZE,
        GAMMA,
        TEST_EVERY_EPISODE,
        REPLAY_MEMORY_SIZE,
        NUM_EPISODES,
        LEARNING_START_EPISODES,
        EPSILON_THRES,
        TARGET_UPDATE_EVERY_EPISODE,
        VERBOSE,
        PLOT,
    )



