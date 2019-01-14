import sys
import os
import numpy as np
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from online_train import train_main


def test_online_train_dqn_gridworld():
    TRAIN_TIMES = 1
    TEST_TIMES = 1
    BATCH_SIZE = 4
    GAMMA = 0.9
    EPSILON_THRES = 0.4
    TEST_EVERY_EPISODE = 10
    REPLAY_MEMORY_SIZE = 20000
    NUM_EPISODES = 2001
    LEARNING_START_EPISODES = 500
    TARGET_UPDATE_EVERY_EPISODE = 2
    VERBOSE = False
    PLOT = False

    model_str = 'dqn'
    env_str = "finite"

    test_rewards_ave, test_durations_ave = train_main(
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
    # at least 5 in last 10 tests are optimal
    assert np.sum(test_rewards_ave[-10:] == 10) > 5
    assert np.sum(test_durations_ave[-10:] == 4) > 5


def test_online_train_lstm_gridworld():
    TRAIN_TIMES = 1
    TEST_TIMES = 1
    BATCH_SIZE = 4
    GAMMA = 0.9
    EPSILON_THRES = 0.4
    TEST_EVERY_EPISODE = 10
    REPLAY_MEMORY_SIZE = 20000
    NUM_EPISODES = 4001
    LEARNING_START_EPISODES = 500
    TARGET_UPDATE_EVERY_EPISODE = 2
    VERBOSE = False
    PLOT = False

    model_str = 'lstm'
    env_str = "finite"

    test_rewards_ave, test_durations_ave = train_main(
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
    # at least 5 in last 10 tests are optimal
    assert np.sum(test_rewards_ave[-10:] == 10) > 5
    assert np.sum(test_durations_ave[-10:] == 4) > 5


def test_online_train_lstm_rnn_gridworld():
    TRAIN_TIMES = 1
    TEST_TIMES = 1
    BATCH_SIZE = 4
    GAMMA = 0.9
    EPSILON_THRES = 0.4
    TEST_EVERY_EPISODE = 10
    REPLAY_MEMORY_SIZE = 20000
    NUM_EPISODES = 5001
    LEARNING_START_EPISODES = 500
    TARGET_UPDATE_EVERY_EPISODE = 2
    VERBOSE = True
    PLOT = False

    model_str = 'lstm'
    env_str = "rnn"

    test_rewards_ave, test_durations_ave = train_main(
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
    # at least 5 in last 10 tests are optimal
    assert np.sum(test_rewards_ave[-10:] > 28) > 5


def test_online_train_dqn_rnn_gridworld():
    TRAIN_TIMES = 1
    TEST_TIMES = 1
    BATCH_SIZE = 4
    GAMMA = 0.9
    EPSILON_THRES = 0.4
    TEST_EVERY_EPISODE = 10
    REPLAY_MEMORY_SIZE = 20000
    NUM_EPISODES = 5001
    LEARNING_START_EPISODES = 500
    TARGET_UPDATE_EVERY_EPISODE = 2
    VERBOSE = False
    PLOT = False

    model_str = 'dqn'
    env_str = "rnn"

    test_rewards_ave, test_durations_ave = train_main(
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
    # not reach optimal
    assert np.sum(test_rewards_ave[-10:] > 31) == 0


def test_online_train_dqn_cartpole():
    TRAIN_TIMES = 1
    TEST_TIMES = 20
    BATCH_SIZE = 64
    GAMMA = 0.9
    EPSILON_THRES = 0.05
    TEST_EVERY_EPISODE = 20
    REPLAY_MEMORY_SIZE = 20000
    NUM_EPISODES = 1001
    LEARNING_START_EPISODES = 500
    TARGET_UPDATE_EVERY_EPISODE = 2
    VERBOSE = False
    PLOT = False

    model_str = 'dqn'
    env_str = "cartpole"

    test_rewards_ave, test_durations_ave = train_main(
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
    # cartpole is a little special, we only need at least 1 test optimal
    assert np.sum(np.array(test_rewards_ave) > 195) > 0


def test_online_train_dqn_lunar():
    TRAIN_TIMES = 1
    TEST_TIMES = 10
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_THRES = 0.05
    TEST_EVERY_EPISODE = 100
    REPLAY_MEMORY_SIZE = 100000
    NUM_EPISODES = 901
    LEARNING_START_EPISODES = 10
    TARGET_UPDATE_EVERY_EPISODE = 2
    VERBOSE = False
    PLOT = False

    model_str = 'dqn'
    env_str = "lunar"

    test_rewards_ave, test_durations_ave = train_main(
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
    assert np.sum(np.array(test_rewards_ave) > 195) > 0


if __name__ == "__main__":
    # test_online_train_dqn_gridworld()
    # test_online_train_lstm_gridworld()
    # test_online_train_lstm_rnn_gridworld()
    # test_online_train_dqn_rnn_gridworld()
    # test_online_train_dqn_cartpole()
    test_online_train_dqn_lunar()
