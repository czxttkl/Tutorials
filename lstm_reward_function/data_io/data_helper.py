from collections import defaultdict
import numpy as np
import random
from scipy import stats


def generate_raw_data(presto_data, feature_index, action_index, reward_metric, earliest_seq_num):
    # collect history of state, reward, action, and time_diff for each user
    X_state, X_reward, X_action, X_time_diff, Y = \
        defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    for i, s in presto_data.iterrows():
        state_feature = np.zeros(len(feature_index))
        for key, val in s['state_features'].items():
            state_feature[feature_index[key]] = val
        X_state[s['mdp_id']].append(state_feature)

        reward_feature = np.zeros(1)
        reward_feature[0] = s['metrics'][reward_metric]
        X_reward[s['mdp_id']].append(reward_feature)

        action_feature = np.zeros(len(action_index))
        action_feature[action_index[s['action']]] = 1
        X_action[s['mdp_id']].append(action_feature)

        time_diff_feature = np.zeros(1)
        time_diff_feature[0] = s['sequence_number']
        X_time_diff[s['mdp_id']].append(time_diff_feature)

        Y[s['mdp_id']].append(int(s['metrics'][reward_metric]))

    for k, tdf in X_time_diff.items():
        new_time_diff_feature = np.zeros_like(tdf)
        for i, v in enumerate(tdf):
            if i == 0:
                # time to the earliest record time
                new_time_diff_feature[i] = v - earliest_seq_num
            else:
                # time to last record
                new_time_diff_feature[i] = tdf[i] - tdf[i-1]
        X_time_diff[k] = new_time_diff_feature

    # process train data and test data
    data_users = sorted(list(X_state.keys()))
    # random.shuffle(data_users)
    return X_state, X_reward, X_action, X_time_diff, Y, data_users


def generate_nn_data(X_state,
                     X_reward,
                     X_action,
                     X_time_diff,
                     Y,
                     data_users,
                     test_denominator):
    feature_data = []
    for data in [X_time_diff, X_action, X_state, X_reward]:
        if data is not None:
            feature_data.append(data)
    assert len(feature_data) >= 1

    X_nn_train = [np.hstack(
                    tuple(data[mdp_id][-1] for data in feature_data)
                  )
                  for mdp_id in data_users[len(data_users)//test_denominator:]]

    X_nn_test = [np.hstack(
                    tuple(data[mdp_id][-1] for data in feature_data)
                  )
                  for mdp_id in data_users[:len(data_users)//test_denominator]]

    Y_nn_train = [Y[mdp_id][-1] for mdp_id in data_users[len(data_users)//test_denominator:]]

    Y_nn_test = [Y[mdp_id][-1] for mdp_id in data_users[:len(data_users)//test_denominator]]

    X_nn_train = np.array(X_nn_train)
    X_nn_test = np.array(X_nn_test)
    Y_nn_train = np.array(Y_nn_train)
    Y_nn_test = np.array(Y_nn_test)

    print('\nbefore normalization\n')
    print(stats.describe(X_nn_train))
    print("X_nn_train (shape: {})".format(X_nn_train.shape))
    print(X_nn_train)
    print("\nX_nn_test (shape: {})".format(X_nn_test.shape))
    print(X_nn_test)
    print("\nY_nn_train (shape: {})".format(Y_nn_train.shape))
    print(Y_nn_train)
    print("\nY_nn_test (shape: {})".format(Y_nn_test.shape))
    print(Y_nn_test)

    # calculate mean using every sequence's last element
    X_nn_mean = np.squeeze(
        np.mean(X_nn_train, axis=0)
    )
    for i, v in enumerate(X_nn_mean):
        X_nn_train[:, i] = X_nn_train[:, i] - v
        X_nn_test[:, i] = X_nn_test[:, i] - v

    # calculate std using every sequence's last element
    X_nn_std = np.squeeze(
        np.std(X_nn_train, axis=0)
    )
    for i, v in enumerate(X_nn_std):
        if v != 0:
            X_nn_train[:, i] = X_nn_train[:, i] / v
            X_nn_test[:, i] = X_nn_test[:, i] / v
    print('\nnormalization mean:', X_nn_mean)
    print('normalization std:', X_nn_std)

    print('\nafter normalization\nn')
    print(stats.describe(X_nn_train))
    print("X_nn_train (shape: {})".format(X_nn_train.shape))
    print(X_nn_train)
    print("\nX_nn_test (shape: {})".format(X_nn_test.shape))
    print(X_nn_test)

    return X_nn_train, X_nn_test, Y_nn_train, Y_nn_test


def generate_lstm_data(X_state,
                       X_reward,
                       X_action,
                       X_time_diff,
                       Y,
                       data_users,
                       max_seq_len,
                       lstm_feature_size,
                       test_denominator):
    feature_data = []
    for data in [X_time_diff, X_action, X_state, X_reward]:
        if data is not None:
            feature_data.append(data)
    assert len(feature_data) >= 1

    X_nn_train_lens = np.zeros(len(data_users) - len(data_users) // test_denominator, dtype=int)
    X_nn_test_lens = np.zeros(len(data_users) // test_denominator, dtype=int)
    X_nn_train = np.zeros((len(data_users) - len(data_users) // test_denominator,
                           max_seq_len, lstm_feature_size))
    X_nn_test = np.zeros((len(data_users) // test_denominator,
                          max_seq_len, lstm_feature_size))

    for i, mdp_id in enumerate(data_users[len(data_users) // test_denominator:]):
        for j in range(len(X_state[mdp_id])):
            X_nn_train[i, j, :] = np.hstack(
                tuple(data[mdp_id][j] for data in feature_data)
            )
        X_nn_train_lens[i] = len(feature_data[0][mdp_id])

    for i, mdp_id in enumerate(data_users[:len(data_users) // test_denominator]):
        for j in range(len(X_state[mdp_id])):
            X_nn_test[i, j, :] = np.hstack(
                tuple(data[mdp_id][j] for data in feature_data)
            )
        X_nn_test_lens[i] = len(feature_data[0][mdp_id])

    Y_nn_train = np.array([Y[mdp_id][-1] for mdp_id in data_users[len(data_users) // test_denominator:]])

    Y_nn_test = np.array([Y[mdp_id][-1] for mdp_id in data_users[:len(data_users) // test_denominator]])

    print('\nbefore normalization')
    print("X_nn_train (shape: {})".format(X_nn_train.shape))
    print(X_nn_train)
    print("X_nn_train_lens (shape: {})".format(X_nn_train_lens.shape))
    print(X_nn_train_lens)
    print("\nX_nn_test (shape: {})".format(X_nn_test.shape))
    print(X_nn_test)
    print("X_nn_test_lens (shape: {})".format(X_nn_test_lens.shape))
    print(X_nn_test_lens)
    print("\nY_nn_train (shape: {})".format(Y_nn_train.shape))
    print(Y_nn_train)
    print("\nY_nn_test (shape: {})".format(Y_nn_test.shape))
    print(Y_nn_test)

    # calculate mean using every sequence's last element
    X_nn_mean = np.squeeze(
        np.mean(X_nn_train[:, X_nn_train_lens - 1, :], axis=0)
    )
    for i, v in enumerate(X_nn_mean):
        for j in range(len(X_nn_train)):
            X_nn_train[j, :X_nn_train_lens[j], i] = X_nn_train[j, :X_nn_train_lens[j], i] - v
        for j in range(len(X_nn_test)):
            X_nn_test[j, :X_nn_test_lens[j], i] = X_nn_test[j, :X_nn_test_lens[j], i] - v

    # calculate std using every sequence's last element
    X_nn_std = np.squeeze(
        np.std(X_nn_train[:, X_nn_train_lens - 1, :], axis=0)
    )
    for i, v in enumerate(X_nn_std):
        if v != 0:
            for j in range(len(X_nn_train)):
                X_nn_train[j, :X_nn_train_lens[j], i] = X_nn_train[j, :X_nn_train_lens[j], i] / v
            for j in range(len(X_nn_test)):
                X_nn_test[j, :X_nn_test_lens[j], i] = X_nn_test[j, :X_nn_test_lens[j], i] / v

    print('\nnormalization mean:', X_nn_mean)
    print('normalization std:', X_nn_std)

    print('\nafter normalization')
    print("X_nn_train (shape: {})".format(X_nn_train.shape))
    print(X_nn_train)
    print("\nX_nn_test (shape: {})".format(X_nn_test.shape))
    print(X_nn_test)

    return X_nn_train, X_nn_test, Y_nn_train, Y_nn_test, X_nn_train_lens, X_nn_test_lens


def balance_nn_labels(X_nn_train, Y_nn_train, X_nn_test, Y_nn_test):
    print(
        "Before label balance: \nY_train=1: {}, \nY_train=0: {}, \nY_test=1: {}, \nY_test=0: {}".format(
            np.sum(Y_nn_train == 1),
            np.sum(Y_nn_train == 0),
            np.sum(Y_nn_test == 1),
            np.sum(Y_nn_test == 0),
        )
    )

    one_index_train = np.where(Y_nn_train == 1)[0]
    zero_index_train = np.where(Y_nn_train == 0)[0]
    one_index_test = np.where(Y_nn_test == 1)[0]
    zero_index_test = np.where(Y_nn_test == 0)[0]
    if len(one_index_test) > len(zero_index_test):
        index_test_to_remove = one_index_test[:(len(one_index_test) - len(zero_index_test))]
    else:
        index_test_to_remove = zero_index_test[:(len(zero_index_test) - len(one_index_test))]

    if len(one_index_train) > len(zero_index_train):
        index_train_to_remove = one_index_train[:(len(one_index_train) - len(zero_index_train))]
    else:
        index_train_to_remove = zero_index_train[:(len(zero_index_train) - len(one_index_train))]

    X_nn_train = np.delete(X_nn_train, index_train_to_remove, axis=0)
    Y_nn_train = np.delete(Y_nn_train, index_train_to_remove, axis=0)
    X_nn_test = np.delete(X_nn_test, index_test_to_remove, axis=0)
    Y_nn_test = np.delete(Y_nn_test, index_test_to_remove, axis=0)

    print(
        "\nAfter label balance: \nY_train=1: {}, \nY_train=0: {}, \nY_test=1: {}, \nY_test=0: {}".format(
            np.sum(Y_nn_train == 1),
            np.sum(Y_nn_train == 0),
            np.sum(Y_nn_test == 1),
            np.sum(Y_nn_test == 0),
        )
    )

    return X_nn_train, Y_nn_train, X_nn_test, Y_nn_test
