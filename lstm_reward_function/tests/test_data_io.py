import pandas as pd
import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from data_io.helper import generate_raw_data, generate_nn_data


def test_data_io():
    df_dict = {
                'mdp_id': ['123', '123', '123',
                           '234', '234'],
                'sequence_number': [1, 2, 4,
                                    5, 10],
                'state_features': [
                                    {'553': 0, '554': 1},
                                    {'553': 1, '554': 2},
                                    {'553': 2, '554': 3},
                                    {'553': 2, '554': 3},
                                    {'553': 3, '554': 4},
                                  ],
                'action': ['video_action_2', 'video_action_1', 'video_action_2',
                           'video_action_1', 'video_action_2'],
                'metrics': [
                            {'a': 3, 'dap': 1},
                            {'b': 0, 'dap': 0},
                            {'c': 1, 'dap': 1},
                            {'d': 2, 'dap': 1},
                            {'e': 1, 'dap': 0}
                           ]
    }
    df = pd.DataFrame(df_dict)
    feature_index = {'553': 0, '554': 1}
    action_index = {'video_action_1':0, 'video_action_2': 1}
    reward_metric = 'dap'
    earliest_seq_num = 0

    X_state, X_reward, X_action, X_time_diff, Y, data_users = \
        generate_raw_data(df, feature_index, action_index, reward_metric, earliest_seq_num)

    assert np.array_equal(X_state['123'][0], [0, 1])
    assert np.array_equal(X_state['123'][1], [1, 2])
    assert np.array_equal(X_state['123'][2], [2, 3])
    assert np.array_equal(X_state['234'][0], [2, 3])
    assert np.array_equal(X_state['234'][1], [3, 4])

    assert np.array_equal(X_action['123'][0], [0, 1])
    assert np.array_equal(X_action['123'][1], [1, 0])
    assert np.array_equal(X_action['123'][2], [0, 1])
    assert np.array_equal(X_action['234'][0], [1, 0])
    assert np.array_equal(X_action['234'][1], [0, 1])

    assert np.array_equal(X_reward['123'][0], [1])
    assert np.array_equal(X_reward['123'][1], [0])
    assert np.array_equal(X_reward['123'][2], [1])
    assert np.array_equal(X_reward['234'][0], [1])
    assert np.array_equal(X_reward['234'][1], [0])

    assert np.array_equal(X_time_diff['123'][0], [1])
    assert np.array_equal(X_time_diff['123'][1], [1])
    assert np.array_equal(X_time_diff['123'][2], [2])
    assert np.array_equal(X_time_diff['234'][0], [5])
    assert np.array_equal(X_time_diff['234'][1], [5])

    assert np.array_equal(Y['123'], [1, 0, 1])
    assert np.array_equal(Y['234'], [1, 0])

    X_nn_train, X_nn_test, Y_nn_train, Y_nn_test = \
        generate_nn_data(X_state, X_reward, X_action, X_time_diff, Y, data_users)


if __name__ == '__main__':
    test_data_io()
