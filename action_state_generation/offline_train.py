import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.optim as optim
import pandas as pd
from plot_helper.plot_helper import plot_seaborn

from helper.helper import(
    get_model, soft_update, get_env
)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'invalid_actions', 'next_state', 'reward', 'done')
)


def evaluate(env, gamma, i_train_epoch, policy_net, verbose):
    print("-------------test at {} train episode------------".format(i_train_epoch))
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


def train(model_str, env_str, test_times, batch_size, gamma,
          replay_memory_size, num_epochs,
          target_update_every_batch, verbose):
    episode_durations = []
    test_episode_durations = []
    test_episode_rewards = []

    env = get_env(env_str, device)
    policy_net, target_net = get_model(model_str, env, gamma, replay_memory_size, batch_size, device)
    print(policy_net)
    print("trainable param num:", policy_net.num_of_params())
    policy_net.optimizer = optim.Adam(policy_net.parameters())

    for i_episode in range(10000000):
        # Initialize the environment and state
        env.reset()
        state = env.cur()
        invalid_actions = env.invalid_actions()
        episode_memory = []
        last_output = None
        # reset the LSTM hidden state. Must be done before you run a new episode. Otherwise the LSTM will treat
        # a new episode as a continuation of the last episode
        # for other models, this function will do nothing
        policy_net.init_hidden(batch_size=1)

        for t in count():
            # Select and perform an action
            action_dim = env.action_dim
            # set epsilon to 1 so every time a random action is selected
            action = policy_net.select_action(
                env, t, state, last_output, invalid_actions, action_dim, 1.0
            )
            # env step
            next_state, next_invalid_actions, reward, done, _ = env.step(action)
            # needs to compute Q(s',a) for all a, which will be used in the next iteration
            last_output = None
            step_transition = Transition(state=state, action=action, invalid_actions=invalid_actions,
                                         next_state=next_state, reward=reward, done=done)
            episode_memory.append(step_transition)
            # Move to the next state
            state = next_state
            invalid_actions = next_invalid_actions
            if done:
                episode_durations.append(t + 1)
                break

        # Store the episode transitions in memory
        policy_net.memory.push(episode_memory)
        if len(policy_net.memory) >= replay_memory_size:
            break

    for i_epoch in range(num_epochs):
        # perform test before each epoch
        results = [evaluate(env, gamma, i_epoch, policy_net, verbose) for _ in range(test_times)]
        test_duration, final_test_reward = np.mean([r[0] for r in results]), np.mean([r[1] for r in results])
        test_episode_durations.append(test_duration)
        test_episode_rewards.append(final_test_reward)
        print('Complete Testing')
        print(test_episode_durations)
        print(test_episode_rewards)
        print("-------------test at {} train epoch------------".format(i_epoch))

        for i_batch in range(len(policy_net.memory) // batch_size):
            # Perform one step of the optimization
            policy_net.optimize_model(env, target_net)
            # perform target network update
            if i_batch % target_update_every_batch == 0 and target_net:
                print('epoch {} batch {} soft update'.format(i_epoch, i_batch))
                soft_update(policy_net, target_net, 1.0)

    print('Complete Training')
    return test_episode_durations, test_episode_rewards


def train_main(model_str, env_str, train_times, test_times,
               batch_size, gamma, replay_memory_size,
               num_epochs, target_update_every_batch,
               verbose, plot):
    test_durations = []
    test_rewards = []
    for i in range(train_times):
        duration, reward = train(
            model_str, env_str, test_times, batch_size, gamma,
            replay_memory_size, num_epochs,
            target_update_every_batch, verbose)
        test_durations.append(duration)
        test_rewards.append(reward)

    test_durations = np.vstack(test_durations)
    test_durations_ave = np.average(test_durations, axis=0)
    test_durations_std = np.std(test_durations, axis=0)
    print("------------- Test Steps --------------")
    for i, (ave, std) in enumerate(zip(test_durations_ave, test_durations_std)):
        print("Epoch {}: {:.2f} +- {:.2f}".format(i, ave, std))

    test_rewards = np.vstack(test_rewards)
    test_rewards_ave = np.average(test_rewards, axis=0)
    test_rewards_std = np.std(test_rewards, axis=0)

    print()

    print("------------- Test Rewards --------------")
    for i, (ave, std) in enumerate(zip(test_rewards_ave, test_rewards_std)):
        print("Epoch {}: {:.2f} +- {:.2f}".format(i, ave, std))

    if plot:
        reward_df = pd.DataFrame(
            {
                'reward': np.hstack(test_rewards),
                'epoch': np.tile(
                    np.arange(len(test_rewards_ave)),
                    len(test_rewards)
                )
            }
        )
        plot_seaborn(reward_df, xaxis='epoch', yaxis='reward',
                     title='reward_plot_{}_{}_tt{}_mem{}'.format(model_str, env_str, train_times, replay_memory_size),
                     file='offline_reward_plot_{}_{}_tt{}_mem{}.png'.format(model_str, env_str, train_times, replay_memory_size),
                     show=False)
    return test_rewards_ave, test_durations_ave


if __name__ == '__main__':
    TRAIN_TIMES = 1
    TEST_TIMES = 20
    BATCH_SIZE = 64
    GAMMA = 0.99
    TARGET_UPDATE_EVERY_BATCH = 10
    REPLAY_MEMORY_SIZE = 100000
    NUM_EPOCHS = 10
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
        REPLAY_MEMORY_SIZE,
        NUM_EPOCHS,
        TARGET_UPDATE_EVERY_BATCH,
        VERBOSE,
        PLOT,
    )



