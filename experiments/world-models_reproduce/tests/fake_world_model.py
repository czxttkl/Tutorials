import numpy as np
import torch
import random

import torch
import torch.nn as nn
import torch.nn.functional as f


class SimulatedWorldModel(nn.Module):
    """ A world model used for simulation. Underlying is an RNN with fixed parameters. """

    def __init__(self, action_dim, state_dim, num_gaussian, lstm_num_layer, lstm_hidden_dim):
        super(SimulatedWorldModel, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_gaussian = num_gaussian
        self.lstm_num_layer = lstm_num_layer
        self.lstm_hidden_dim = lstm_hidden_dim
        self.init_lstm()
        self.init_weight()
        self.init_hidden()
        self.eval()

    def init_lstm(self):
        self.lstm = nn.LSTM(
            input_size=self.action_dim + self.state_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layer
        )
        # output mus for each guassian, and reward
        self.gmm_linear = nn.Linear(self.lstm_hidden_dim, self.state_dim * self.num_gaussian + 1)

    def init_hidden(self, batch_size=1):
        # (num_layers * num_directions, batch, hidden_size)
        self.hidden = torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim), \
                      torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim)

    def init_weight(self):
        torch.manual_seed(3212)
        for n, p in self.lstm.named_parameters():
            nn.init.normal_(p, 0, 1)
        for n, p in self.gmm_linear.named_parameters():
            nn.init.normal_(p, 0, 1)
        torch.manual_seed(random.randint(0, 1000))

    def forward(self, actions, cur_states):
        seq_len, batch_size = actions.size(0), actions.size(1)

        # actions: (SEQ_LEN, BATCH_SIZE, ACTION_SIZE)
        # cur_states: (SEQ_LEN, BATCH_SIZE, FEATURE_SIZE)
        X = torch.cat([actions, cur_states], dim=-1)
        # X_shape: (1, act_seq_len, lstm_input_dim)
        Y, self.hidden = self.lstm(X, self.hidden)
        gmm_outs = self.gmm_linear(Y)

        mus = gmm_outs[:, :, :-1]
        mus = mus.view(seq_len, batch_size, self.num_gaussian, self.state_dim)
        rs = gmm_outs[:, :, -1]

        return mus, rs