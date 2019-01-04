import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from nn_train import NeuralNetwork


def test_nn_classify():
    batch_size = 8
    epoch_num = 5
    feature_size = 1379
    hidden_dim = 100
    train_size = 400
    test_size = 100
    output_dim = 2

    X_nn_train = np.random.normal(0, 5, (train_size, feature_size))
    X_nn_test = np.random.normal(0, 5, (test_size, feature_size))
    Y_nn_train = np.random.randint(0, 2, train_size)
    Y_nn_test = np.random.randint(0, 2, test_size)
    net = NeuralNetwork(
        nn_input_dim=feature_size,
        nn_num_hidden_layer=1,
        nn_hidden_dim=hidden_dim,
        nn_output_dim=output_dim,
        regress=False,
    )
    print('number of params:', net.num_of_params())

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0
        batch_start = 0
        batch_end = batch_start + batch_size
        while batch_start < X_nn_train.shape[0]:
            inputs, labels = X_nn_train[batch_start:batch_end], Y_nn_train[batch_start:batch_end]
            loss = net.optimize_model(inputs, labels)
            batch_start += batch_size
            batch_end += batch_size
            running_loss += loss
            batch_num += 1
        print('epoch [%d] loss: %.3f' %
              (epoch, running_loss / batch_num))

    print('Finished Training\n')
    print('Training statistics')
    print('train data size:', len(Y_nn_train))
    net.accuracy(X_nn_train, Y_nn_train)
    net.accuracy_per_class(X_nn_train, Y_nn_train)
    print()
    print('Testing statistics')
    print('test data size:', len(Y_nn_test))
    net.accuracy(X_nn_test, Y_nn_test)
    net.accuracy_per_class(X_nn_test, Y_nn_test)
    print()


def test_nn_regress():
    batch_size = 8
    epoch_num = 5
    feature_size = 1379
    hidden_dim = 100
    train_size = 400
    test_size = 100
    output_dim = 1

    X_nn_train = np.random.normal(0, 5, (train_size, feature_size))
    X_nn_test = np.random.normal(0, 5, (test_size, feature_size))
    Y_nn_train = np.random.randint(0, 2, train_size) * 10 - 5
    Y_nn_test = np.random.randint(0, 2, test_size) * 10 - 5
    net = NeuralNetwork(
        nn_input_dim=feature_size,
        nn_num_hidden_layer=1,
        nn_hidden_dim=hidden_dim,
        nn_output_dim=output_dim,
        regress=True,
    )
    print('number of params:', net.num_of_params())

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0
        batch_start = 0
        batch_end = batch_start + batch_size
        while batch_start < X_nn_train.shape[0]:
            inputs, labels = X_nn_train[batch_start:batch_end], Y_nn_train[batch_start:batch_end]
            loss = net.optimize_model(inputs, labels)
            batch_start += batch_size
            batch_end += batch_size
            running_loss += loss
            batch_num += 1
        print('epoch [%d] loss: %.3f' %
              (epoch, running_loss / batch_num))

    print('Finished Training\n')
    print('Training statistics')
    print('train data size:', len(Y_nn_train))
    net.mse(X_nn_train, Y_nn_train)
    net.mse_per_class(X_nn_train, Y_nn_train)
    print()
    print('Testing statistics')
    print('test data size:', len(Y_nn_test))
    net.mse(X_nn_test, Y_nn_test)
    net.mse_per_class(X_nn_test, Y_nn_test)
