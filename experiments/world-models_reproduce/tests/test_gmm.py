""" Test gmm loss """
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from models.mdrnn import MDRNN, gmm_loss
from torch.distributions.normal import Normal
from utils.learning import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tests.fake_world_model import SimulatedWorldModel
import numpy


class TestGMM(unittest.TestCase):
    def test_gmm_loss_my(self):
        # seq_len x batch_size x gaussian_size x feature_size
        # 1 x 1 x 2 x 2
        mus = torch.Tensor([
            [
                [
                    [0., 0.],
                    [6., 6.]
                ],
            ]
        ])
        stds = torch.Tensor([
            [
                [
                    [2., 2.],
                    [2., 2.]
                ],
            ]
        ])
        # seq_len x batch_size x gaussian_size
        pi = torch.Tensor([[[.5, .5]]])
        logpi = torch.log(pi)

        # seq_len x batch_size x feature_size
        batch = torch.Tensor([
                [
                    [3., 3.]
                ]
        ])
        gl = gmm_loss(batch, mus, stds, logpi)

        # first component, first dimension
        n11 = Normal(mus[0, 0, 0, 0], stds[0, 0, 0, 0])
        # first component, second dimension
        n12 = Normal(mus[0, 0, 0, 1], stds[0, 0, 0, 1])
        p1 = pi[0, 0, 0] * torch.exp(n11.log_prob(batch[0, 0, 0])) * torch.exp(n12.log_prob(batch[0, 0, 1]))
        # second component, first dimension
        n21 = Normal(mus[0, 0, 1, 0], stds[0, 0, 1, 0])
        # second component, second dimension
        n22 = Normal(mus[0, 0, 1, 1], stds[0, 0, 1, 1])
        p2 = pi[0, 0, 1] * torch.exp(n21.log_prob(batch[0, 0, 0])) * torch.exp(n22.log_prob(batch[0, 0, 1]))

        print("gmm loss={}, p1={}, p2={}, p1+p2={}, -log(p1+p2)={}".format(gl, p1, p2, p1+p2, -torch.log(p1 + p2)))
        assert -torch.log(p1 + p2) == gl
        print()

    def transpose(self, *args):
        res = []
        for arg in args:
            res.append(arg.transpose(1, 0))
        return res

    def get_loss(self, obs, action, reward, terminal, next_obs, state_dim, mdrnn):
        """ Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :args obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args terminal: (BSIZE, SEQ_LEN) torch tensor
        :args next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        obs, action, reward, terminal, next_obs = self.transpose(obs, action, reward, terminal, next_obs)
        mus, sigmas, logpi, rs, ds = mdrnn(action, obs)
        gmm = gmm_loss(next_obs, mus, sigmas, logpi)
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
        mse = f.mse_loss(rs, reward)
        # loss = (gmm + bce + mse) / (state_dim + 2)
        # loss = gmm
        loss = mse + bce + gmm
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

    def test_mdrnn_learning(self):
        num_epochs = 60000
        num_episodes = 20000
        batch_size = 2000
        action_dim = 2
        seq_len = 1
        state_dim = 5
        simulated_num_gaussian = 2
        mdrnn_num_gaussian = 2
        simulated_hidden_size = 3
        mdrnn_hidden_size = 300
        mdrnn_hidden_layer = 1
        cur_state_mem = numpy.zeros((num_episodes, seq_len, state_dim))
        next_state_mem = numpy.zeros((num_episodes, seq_len, state_dim))
        action_mem = numpy.zeros((num_episodes, seq_len, action_dim))
        reward_mem = numpy.zeros((num_episodes, seq_len))
        terminal_mem = numpy.zeros((num_episodes, seq_len))
        next_mus_mem = numpy.zeros((num_episodes, seq_len, simulated_num_gaussian, state_dim))

        swm = SimulatedWorldModel(
            action_dim=action_dim,
            state_dim=state_dim,
            num_gaussian=simulated_num_gaussian,
            lstm_num_layer=1,
            lstm_hidden_dim=simulated_hidden_size,
        )

        init_cur_state = torch.randn((1, 1, state_dim))
        actions = torch.eye(action_dim)
        for e in range(num_episodes):
            swm.init_hidden(batch_size=1)
            next_state = init_cur_state
            # next_state = torch.randn((1, 1, state_dim))
            for s in range(seq_len):
                cur_state = next_state

                action = torch.zeros((1, 1, action_dim))
                # action[0, 0, :] = actions[numpy.random.randint(action_dim)]
                action[0, 0, :] = actions[0]
                next_mus, reward = swm(action, cur_state)
                if s == seq_len - 1:
                    terminal = 1
                else:
                    terminal = 0

                next_pi = torch.ones(simulated_num_gaussian) / simulated_num_gaussian
                # index = Categorical(next_pi).sample((1,)).long().item()
                index = e % simulated_num_gaussian
                next_state = torch.zeros_like(cur_state)
                next_state[0, 0, :] = next_mus[0, 0, index]

                print("{} cur_state: {}, action: {}, next_state: {}, reward: {}, terminal: {}"
                      .format(e, cur_state, action, next_state, reward, terminal))
                print("next_pi: {}, sampled index: {}".format(next_pi, index))
                print("next_mus:", next_mus, "\n")

                cur_state_mem[e, s, :] = cur_state.detach().numpy()
                action_mem[e, s, :] = action.numpy()
                reward_mem[e, s] = reward.detach().numpy()
                terminal_mem[e, s] = terminal
                next_state_mem[e, s, :] = next_state.detach().numpy()
                next_mus_mem[e, s, :, :] = next_mus.detach().numpy()

        mdrnn = MDRNN(
            latents=state_dim,
            actions=action_dim,
            gaussians=mdrnn_num_gaussian,
            hiddens=mdrnn_hidden_size,
            layers=mdrnn_hidden_layer,
        )
        mdrnn.train()
        optimizer = torch.optim.Adam(mdrnn.parameters(), lr=0.01)
        earlystopping = EarlyStopping('min', patience=30)

        cum_loss = []
        cum_gmm = []
        cum_bce = []
        cum_mse = []
        num_batch = num_episodes // batch_size
        for e in range(num_epochs):
            for i in range(0, num_batch):
                mdrnn.init_hidden(batch_size=batch_size)
                optimizer.zero_grad()
                sample_indices = numpy.random.randint(num_episodes, size=batch_size)

                obs, action, reward, terminal, next_obs = \
                    cur_state_mem[sample_indices], \
                    action_mem[sample_indices], \
                    reward_mem[sample_indices], \
                    terminal_mem[sample_indices], \
                    next_state_mem[sample_indices]
                obs, action, reward, terminal, next_obs = \
                    torch.tensor(obs, dtype=torch.float), \
                    torch.tensor(action, dtype=torch.float), \
                    torch.tensor(reward, dtype=torch.float), \
                    torch.tensor(terminal, dtype=torch.float), \
                    torch.tensor(next_obs, dtype=torch.float)

                print("learning at epoch {} step {} best score {} counter {}".format(e, i, earlystopping.best, earlystopping.num_bad_epochs))
                losses = self.get_loss(obs, action, reward, terminal, next_obs, state_dim, mdrnn)
                losses['loss'].backward()
                optimizer.step()

                cum_loss += [losses['loss'].item()]
                cum_gmm += [losses['gmm'].item()]
                cum_bce += [losses['bce'].item()]
                cum_mse += [losses['mse'].item()]
                # print("loss={loss:10.6f} bce={bce:10.6f} gmm={gmm:10.6f} mse={mse:10.6f}"
                #       .format(
                #           loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                #           gmm=cum_gmm / state_dim / (i + 1), mse=cum_mse / (i + 1)))
                print("loss={loss:10.6f} bce={bce:10.6f} gmm={gmm:10.6f} mse={mse:10.6f}"
                    .format(
                    loss=losses['loss'],
                    bce=losses['bce'],
                    gmm=losses['gmm'],
                    mse=losses['mse'],
                )
                )
                print("cum loss={loss:10.6f} cum bce={bce:10.6f} cum gmm={gmm:10.6f} cum mse={mse:10.6f}"
                    .format(
                    loss=numpy.mean(cum_loss),
                    bce=numpy.mean(cum_bce),
                    gmm=numpy.mean(cum_gmm),
                    mse=numpy.mean(cum_mse),
                )
                )

                print()

            earlystopping.step(numpy.mean(cum_loss[-num_batch:]))
            if numpy.mean(cum_loss[-num_batch:]) < -5:
                break

        sample_indices = [0, 1]
        mdrnn.init_hidden(batch_size=len(sample_indices))
        mdrnn.eval()
        obs, action, reward, terminal, next_obs = \
            cur_state_mem[sample_indices], \
            action_mem[sample_indices], \
            reward_mem[sample_indices], \
            terminal_mem[sample_indices], \
            next_state_mem[sample_indices]
        obs, action, reward, terminal, next_obs = \
            torch.tensor(obs, dtype=torch.float), \
            torch.tensor(action, dtype=torch.float), \
            torch.tensor(reward, dtype=torch.float), \
            torch.tensor(terminal, dtype=torch.float), \
            torch.tensor(next_obs, dtype=torch.float)
        transpose_obs, transpose_action, transpose_reward, transpose_terminal, transpose_next_obs = \
            self.transpose(obs, action, reward, terminal, next_obs)
        mus, sigmas, logpi, rs, ds = mdrnn(transpose_action, transpose_obs)
        pi = torch.exp(logpi)
        gl = gmm_loss(transpose_next_obs, mus, sigmas, logpi)
        print(gl)

        print()


    """ Test GMMs """
    def test_gmm_loss(self):
        """ Test case 1 """
        n_samples = 10000

        means = torch.Tensor([[0., 0.],
                              [1., 1.],
                              [-1., 1.]])
        stds = torch.Tensor([[.03, .05],
                             [.02, .1],
                             [.1, .03]])
        pi = torch.Tensor([.2, .3, .5])

        cat_dist = Categorical(pi)
        indices = cat_dist.sample((n_samples,)).long()
        rands = torch.randn(n_samples, 2)

        samples = means[indices] + rands * stds[indices]

        class _model(nn.Module):
            def __init__(self, gaussians):
                super().__init__()
                self.means = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pre_stds = nn.Parameter(torch.Tensor(1, gaussians, 2).normal_())
                self.pi = nn.Parameter(torch.Tensor(1, gaussians).normal_())

            def forward(self, *inputs):
                return self.means.data, torch.exp(self.pre_stds), f.softmax(self.pi, dim=1)

        model = _model(3)
        optimizer = torch.optim.Adam(model.parameters())

        iterations = 100000
        log_step = iterations // 10
        pbar = tqdm(total=iterations)
        cum_loss = 0
        for i in range(iterations):
            batch = samples[torch.LongTensor(128).random_(0, n_samples)]
            m, s, p = model.forward()
            loss = gmm_loss(batch, m, s, p)
            cum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str("avg_loss={:10.6f}".format(
                cum_loss / (i + 1)))
            pbar.update(1)
            if i % log_step == log_step - 1:
                print(m)
                print(s)
                print(p)

if __name__ == "__main__":
    # TestGMM().test_gmm_loss_my()
    TestGMM().test_mdrnn_learning()
