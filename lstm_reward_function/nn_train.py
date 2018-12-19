import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NeuralNetwork(nn.Module):

    def __init__(self, nn_input_dim, nn_num_hidden_layer, nn_hidden_dim, nn_output_dim):
        super(NeuralNetwork, self).__init__()
        h_sizes = [nn_input_dim] + [nn_hidden_dim] * nn_num_hidden_layer + [nn_output_dim]
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.nn_input_dim = nn_input_dim
        self.nn_num_layer = nn_num_hidden_layer
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        for i in range(self.nn_num_layer + 1):
            if i == self.nn_num_layer:
                x = self.hidden[i](x)
            else:
                x = F.relu(self.hidden[i](x))
        return x

    def init_hidden(self, batch_size):
        pass

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def optimize_model(self, inputs, labels):
        inputs = self.from_data_numpy_to_tensor(inputs)
        labels = self.from_label_numpy_to_tensor(labels)
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def from_data_numpy_to_tensor(self, x):
        return torch.tensor(x).float()

    def from_label_numpy_to_tensor(self, y):
        return torch.tensor(y).long()

    def accuracy(self, inputs, labels):
        inputs = self.from_data_numpy_to_tensor(inputs)
        labels = self.from_label_numpy_to_tensor(labels)
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = self(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if total == 0:
            acc = 0
        else:
            acc = correct / total
        print('Accuracy ({}/{}): {}'.format(correct, total, acc))

    def accuracy_per_class(self, inputs, labels):
        inputs = self.from_data_numpy_to_tensor(inputs)
        labels = self.from_label_numpy_to_tensor(labels)
        class_correct = list(0. for i in range(self.nn_output_dim))
        class_total = list(0. for i in range(self.nn_output_dim))
        with torch.no_grad():
            outputs = self(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(labels.size()[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(self.nn_output_dim):
            if class_total[i] == 0:
                acc = 0
            else:
                acc = class_correct[i] / class_total[i]
            print('Accuracy of class {} ({}/{}): {}'.format(i, class_correct[i], class_total[i], acc))


if __name__ == '__main__':
    batch_size = 8
    epoch_num = 3
    feature_size = 1379
    hidden_dim = 100
    train_size = 400
    test_size = 100
    output_dim = 2

    X_nn_train = np.random.normal(0, 5, (train_size, feature_size))
    X_nn_test = np.random.normal(0, 5, (test_size, feature_size))
    Y_nn_train = np.random.randint(0, 2, train_size)
    Y_nn_test = np.random.randint(0, 2, test_size)
    print(Y_nn_test)
    net = NeuralNetwork(nn_input_dim=feature_size, nn_num_hidden_layer=1,
                        nn_hidden_dim=hidden_dim, nn_output_dim=output_dim)
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
