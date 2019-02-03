import sys
import os
import numpy as np
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from offline_train import train_main


def test_offline_train_dqn_gridworld():
    TRAIN_TIMES = 1
    TEST_TIMES = 20
    BATCH_SIZE = 64
    GAMMA = 0.99
    TARGET_UPDATE_EVERY_BATCH = 10
    REPLAY_MEMORY_SIZE = 100000
    NUM_EPOCHS = 10
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
        REPLAY_MEMORY_SIZE,
        NUM_EPOCHS,
        TARGET_UPDATE_EVERY_BATCH,
        VERBOSE,
        PLOT,
    )

    # at least 5 in last 10 tests are optimal
    assert np.sum(test_rewards_ave[-10:] == 10) > 5
    assert np.sum(test_durations_ave[-10:] == 4) > 5


def test_offline_train_dqn_cartpole():
    TRAIN_TIMES = 1
    TEST_TIMES = 20
    BATCH_SIZE = 64
    GAMMA = 0.99
    TARGET_UPDATE_EVERY_BATCH = 10
    REPLAY_MEMORY_SIZE = 100000
    NUM_EPOCHS = 6
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
        REPLAY_MEMORY_SIZE,
        NUM_EPOCHS,
        TARGET_UPDATE_EVERY_BATCH,
        VERBOSE,
        PLOT,
    )

    # at least 3 in last 10 tests are okay
    assert np.sum(test_rewards_ave[-10:] > 150) > 3


if __name__ == "__main__":
    # test_offline_train_dqn_cartpole()
    test_offline_train_dqn_gridworld()
