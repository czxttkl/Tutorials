import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from lstm_train import LSTM


def test_lstm_classify():
    batch_size = 8
    epoch_num = 5
    max_seq_len = 7
    feature_size = 1379
    hidden_dim = 100
    train_size = 400
    test_size = 100
    output_dim = 2

    X_nn_train = np.random.normal(0, 5, (train_size, max_seq_len, feature_size))
    X_nn_test = np.random.normal(0, 5, (test_size, max_seq_len, feature_size))
    X_nn_train_lens = np.random.randint(1, max_seq_len + 1, train_size)
    X_nn_test_lens = np.random.randint(1, max_seq_len + 1, test_size)
    Y_nn_train = np.random.randint(0, 2, train_size)
    Y_nn_test = np.random.randint(0, 2, test_size)
    net = LSTM(
        lstm_input_dim=feature_size,
        lstm_num_hidden_layer=1,
        lstm_hidden_dim=hidden_dim,
        lstm_output_dim=output_dim,
        regress=False,
    )
    print('number of params:', net.num_of_params())
    LSTM.train_loop(net, epoch_num, batch_size, X_nn_train, X_nn_train_lens, Y_nn_train)

    print('Training statistics')
    print('train data size:', len(Y_nn_train))
    acc_train = net.accuracy(X_nn_train, X_nn_train_lens, Y_nn_train)
    class_acc_train = net.accuracy_per_class(X_nn_train, X_nn_train_lens, Y_nn_train)
    assert acc_train > 0.9
    for cat in class_acc_train:
        assert cat > 0.9
    print()
    print('Testing statistics')
    print('test data size:', len(Y_nn_test))
    acc_test = net.accuracy(X_nn_test, X_nn_test_lens, Y_nn_test)
    class_acc_test = net.accuracy_per_class(X_nn_test, X_nn_test_lens, Y_nn_test)
    assert acc_test < 0.7
    for cat in class_acc_test:
        assert cat < 0.7
    print()


def test_lstm_regress():
    batch_size = 8
    epoch_num = 5
    max_seq_len = 7
    feature_size = 1379
    hidden_dim = 100
    train_size = 400
    test_size = 100
    output_dim = 1

    X_nn_train = np.random.normal(0, 5, (train_size, max_seq_len, feature_size))
    X_nn_test = np.random.normal(0, 5, (test_size, max_seq_len, feature_size))
    X_nn_train_lens = np.random.randint(1, max_seq_len + 1, train_size)
    X_nn_test_lens = np.random.randint(1, max_seq_len + 1, test_size)
    # less bipolar rewards wil favor low MSE for label 0
    # Y_nn_train = np.random.randint(0, 2, train_size)
    # Y_nn_test = np.random.randint(0, 2, test_size)
    # more bipolar rewards
    Y_nn_train = np.random.randint(0, 2, train_size) * 10 - 5
    Y_nn_test = np.random.randint(0, 2, test_size) * 10 - 5
    net = LSTM(
        lstm_input_dim=feature_size,
        lstm_num_hidden_layer=1,
        lstm_hidden_dim=hidden_dim,
        lstm_output_dim=output_dim,
        regress=True,
    )
    print('number of params:', net.num_of_params())
    LSTM.train_loop(net, epoch_num, batch_size, X_nn_train, X_nn_train_lens, Y_nn_train)

    print('Training statistics')
    print('train data size:', len(Y_nn_train))
    mse_train = net.mse(X_nn_train, X_nn_train_lens, Y_nn_train)
    class_mse_train = net.mse_per_class(X_nn_train, X_nn_train_lens, Y_nn_train)
    assert mse_train < 2
    for cmt in class_mse_train:
        assert cmt < 2
    print()
    print('Testing statistics')
    print('test data size:', len(Y_nn_test))
    mse_test = net.mse(X_nn_test, X_nn_test_lens, Y_nn_test)
    class_mse_test = net.mse_per_class(X_nn_test, X_nn_test_lens, Y_nn_test)
    assert mse_test > 18
    for cmt in class_mse_test:
        assert cmt > 18


if __name__ == '__main__':
    test_lstm_classify()
    test_lstm_regress()
