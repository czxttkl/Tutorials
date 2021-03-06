import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, lstm_input_dim, lstm_num_hidden_layer, lstm_hidden_dim, lstm_output_dim, regress=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True,
                            num_layers=lstm_num_hidden_layer)
        self.fc = nn.Linear(lstm_hidden_dim, lstm_output_dim)
        self.lstm_input_dim = lstm_input_dim
        self.lstm_num_layer = lstm_num_hidden_layer
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.regress = regress
        self.optimizer = optim.Adam(self.parameters())
        if self.regress:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim), \
                      torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim)

    def forward(self, X, X_lengths):
        # X shape: (batch_size, lstm_seq_max_length, lstm_input_dim)
        # X_lengths: (batch_size), lengths of each seq (not including padding)

        # 1. Run through RNN
        # Dim transformation: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, lstm_hidden_dim)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)
        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # 2. Run through actual linear layer
        # shape (batch_size, lstm_seq_max_length, lstm_output_dim)
        # Note, zero-padded elements from the last step will still become non-zero after passing
        # the fully connected layer
        X = self.fc(X)

        # 3. Keep the last output for each seq
        # Dim transformation: (batch_size, max_seq_len, lstm_hidden_dim) -> (batch_size, lstm_hidden_dim)
        idx = (torch.LongTensor(X_lengths) - 1) \
            .view(-1, 1) \
            .expand(
            len(X_lengths),
            X.size(2)
        )
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if X.is_cuda:
            idx = idx.cuda(X.data.get_device())
        # Shape: (batch_size, lstm_hidden_dim)
        X = X.gather(
            time_dimension, Variable(idx)
        ).squeeze(time_dimension)

        return X

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _process_data(self, inputs, inputs_lens, labels):
        # needs to sort input according to their lengths
        order = np.argsort(inputs_lens)[::-1]
        inputs = inputs[order]
        inputs_lens = inputs_lens[order]
        labels = labels[order]
        inputs = self.from_data_numpy_to_tensor(inputs)
        labels = self.from_label_numpy_to_tensor(labels)
        return inputs, inputs_lens, labels

    def optimize_model(self, inputs, inputs_lens, labels):
        inputs, inputs_lens, labels = self._process_data(inputs, inputs_lens, labels)
        BATCH_SIZE = inputs.size()[0]
        # refresh lstm hidden state
        self.init_hidden(batch_size=BATCH_SIZE)
        self.optimizer.zero_grad()
        outputs = self(inputs, inputs_lens)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def from_data_numpy_to_tensor(self, x):
        return torch.tensor(x).float()

    def from_label_numpy_to_tensor(self, y):
        if self.regress:
            return torch.tensor(y.tolist()).unsqueeze(1).float()
        else:
            return torch.tensor(y.tolist()).long()

    def accuracy(self, inputs, inputs_lens, labels):
        assert not self.regress
        inputs, inputs_lens, labels = self._process_data(inputs, inputs_lens, labels)
        BATCH_SIZE = inputs.size()[0]
        # refresh lstm hidden state
        self.init_hidden(batch_size=BATCH_SIZE)

        with torch.no_grad():
            outputs = self(inputs, inputs_lens)
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
        if total == 0:
            acc = 0
        else:
            acc = correct / total
        print('Accuracy ({}/{}): {}'.format(correct, total, acc))
        return acc

    def mse(self, inputs, inputs_lens, labels):
        assert self.regress
        inputs, inputs_lens, labels = self._process_data(inputs, inputs_lens, labels)
        BATCH_SIZE = inputs.size()[0]
        # refresh lstm hidden state
        self.init_hidden(batch_size=BATCH_SIZE)

        with torch.no_grad():
            outputs = self(inputs, inputs_lens)
            total = labels.size(0)
            m = (outputs - labels) ** 2
            m = m.sum().item()
        if total == 0:
            m_ave = 0
        else:
            m_ave = m / total
        print('MSE ({}/{}): {}'.format(m, total, m_ave))
        return m_ave

    def accuracy_per_class(self, inputs, inputs_lens, labels):
        assert not self.regress
        distinct_labels = np.unique(labels)
        distinct_labels_num = len(distinct_labels)
        inputs, inputs_lens, labels = self._process_data(inputs, inputs_lens, labels)
        BATCH_SIZE = inputs.size()[0]
        # refresh policy_net hidden state
        self.init_hidden(batch_size=BATCH_SIZE)

        class_correct = list(0. for i in range(self.lstm_output_dim))
        class_total = list(0. for i in range(self.lstm_output_dim))
        class_acc = list(0. for i in range(self.lstm_output_dim))
        with torch.no_grad():
            outputs = self(inputs, inputs_lens)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(BATCH_SIZE):
                label = labels[i].long().item()
                label_index = np.where(distinct_labels == label)[0][0]
                class_correct[label_index] += c[i].item()
                class_total[label_index] += 1

        for i in range(distinct_labels_num):
            if class_total[i] == 0:
                class_acc[i] = 0
            else:
                class_acc[i] = class_correct[i] / class_total[i]
            print('Accuracy of class {} (label={}) ({}/{}): {}'
                  .format(i, distinct_labels[i], class_correct[i], class_total[i], class_acc[i]))
        return class_acc

    def mse_per_class(self, inputs, inputs_lens, labels):
        assert self.regress
        distinct_labels = np.unique(labels)
        distinct_labels_num = len(distinct_labels)
        inputs, inputs_lens, labels = self._process_data(inputs, inputs_lens, labels)
        BATCH_SIZE = inputs.size()[0]
        # refresh policy_net hidden state
        self.init_hidden(batch_size=BATCH_SIZE)

        class_mse = list(0. for i in range(distinct_labels_num))
        class_total = list(0. for i in range(distinct_labels_num))
        with torch.no_grad():
            outputs = self(inputs, inputs_lens)
            m = (outputs - labels) ** 2
            for i in range(BATCH_SIZE):
                label = labels[i].long().item()
                label_index = np.where(distinct_labels == label)[0][0]
                class_mse[label_index] += m[i].item()
                class_total[label_index] += 1

        for i in range(distinct_labels_num):
            class_mse_temp = class_mse[i]
            if class_total[i] == 0:
                class_mse[i] = 999
            else:
                class_mse[i] /= class_total[i]
            print('MSE of class {} (label={}) ({}/{}): {}'
                  .format(i, distinct_labels[i], class_mse_temp, class_total[i], class_mse[i]))
        return class_mse

    @staticmethod
    def train_loop(net, epoch_num, batch_size, X_nn_train, X_nn_train_lens, Y_nn_train):
        ave_loss = 999
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            running_loss = 0.0
            batch_num = 0
            batch_start = 0
            batch_end = batch_start + batch_size
            while batch_start < X_nn_train.shape[0]:
                inputs, inputs_lens, labels = \
                    X_nn_train[batch_start:batch_end], \
                    X_nn_train_lens[batch_start:batch_end], \
                    Y_nn_train[batch_start:batch_end]
                loss = net.optimize_model(inputs, inputs_lens, labels)
                batch_start += batch_size
                batch_end += batch_size
                running_loss += loss
                batch_num += 1
            print('epoch [%d] loss: %.3f' %
                  (epoch, running_loss / batch_num))
            ave_loss = running_loss / batch_num

        print('Finished Training\n')
        return ave_loss


if __name__ == '__main__':
    #############################################################
    #                      Classification                       #
    #############################################################
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
    net.accuracy(X_nn_train, X_nn_train_lens, Y_nn_train)
    net.accuracy_per_class(X_nn_train, X_nn_train_lens, Y_nn_train)
    print()
    print('Testing statistics')
    print('test data size:', len(Y_nn_test))
    net.accuracy(X_nn_test, X_nn_test_lens, Y_nn_test)
    net.accuracy_per_class(X_nn_test, X_nn_test_lens, Y_nn_test)
    print()

    #############################################################
    #                       Regression                          #
    #############################################################
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
    net.mse(X_nn_train, X_nn_train_lens, Y_nn_train)
    net.mse_per_class(X_nn_train, X_nn_train_lens, Y_nn_train)
    print()
    print('Testing statistics')
    print('test data size:', len(Y_nn_test))
    net.mse(X_nn_test, X_nn_test_lens, Y_nn_test)
    net.mse_per_class(X_nn_test, X_nn_test_lens, Y_nn_test)
