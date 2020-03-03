import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import itertools
from abc import ABC, abstractmethod
from typing import Optional
from torch.distributions import Gumbel


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

env_config = {
  'batch_size': 1024,
  'num_batches': 1024,
  'slate_size': 5,
  'user_dim': 6,
  'candidate_dim': 7,
}


def order_comp(order1, order2):
    return torch.sum((order1 == order2).float(), dim=-1)


def random_init(model, requires_grad):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.uniform_(p, -1, 1)
        p.requires_grad = requires_grad


def select_indices(scores: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if len(actions.shape) > 1:
        num_rows = scores.size(0)
        row_indices = torch.arange(num_rows).unsqueeze(0).T
        return scores[row_indices, actions].T
    else:
        return scores[actions]


class FrechetOrderSampler:
    def __init__(
        self,
        shape: float = 1,
        upto: Optional[int] = None
    ):
        """Samples an ordering

        Args:
            shape: 1/scale parameter of Gumbel distribution
                At lower temperatures, order is closer to argmax(scores)
            upto [Default: None]: If provided, position upto which log_prob is computed

        Returns:

        An action sampler which produces ordering
        """
        self.shape = shape
        self.upto = upto
        self.gumbel_noise = Gumbel(0, 1.0 / shape)

    def sample(
        self,
        scores: torch.Tensor
    ):
        """Sample an ordering given scores"""
        perturbed = torch.log(scores) + self.gumbel_noise.sample((len(scores),))
        return torch.argsort(-perturbed.detach())

    def log_prob(self, scores : torch.Tensor, permutations):
        """Compute log probability given scores and an action (a permutation).
        The formula uses the equivalence of sorting with Gumbel noise and
        Plackett-Luce model (See Yellot 1977)

        Args:
            scores: scores of different items
            action: prescribed (permutation) order of the items
        """
        s = torch.log(select_indices(scores, permutations))
        n = len(scores)
        p = self.upto if self.upto is not None else n - 1
        return -sum(
            torch.log(torch.exp((s[k:] - s[k]) * self.shape).sum(dim=0))
            for k in range(p))


class SlateTruncatePolicy:
    def __init__(
            self,
            perm_policy,
            slate_length: int
    ):
        super(SlateTruncatePolicy, self).__init__()
        self.perm_policy = perm_policy
        self.slate_length = slate_length

    def act(self, observation):
        action, prob = self.perm_policy.act(observation)
        return action[:self.slate_length], prob


class RecsimEnvironment:
    def __init__(self, batch_size):
        self.slate_size = env_config["slate_size"]
        self.user_dim = env_config["user_dim"]
        self.candidate_dim = env_config["candidate_dim"]
        self.batch_size = batch_size

        self.user_transition = MLP(
            in_dims=self.user_dim,
            nb_classes=self.user_dim,
            layer_dims=[16, 16],
        )
        random_init(self.user_transition, requires_grad=False)

        self.ground_truth_scorer = MLP(
            in_dims=self.user_dim + self.candidate_dim,
            nb_classes=1,
            layer_dims=[16, 16],
        )
        random_init(self.ground_truth_scorer, requires_grad=False)

    def reset(self):
        self.user = torch.randn(self.user_dim)
        self.items = torch.randn(self.slate_size, self.candidate_dim)
        return {'user': self.user, 'items': self.items}

    def step(self, action):
        true_scores = self.ground_truth_scorer(
            torch.cat(
                (
                    self.user.repeat(1, self.slate_size).reshape(
                        self.batch_size, self.slate_size, self.user_dim
                    ),
                    self.items,
                ),
                dim=-1
            )
        ).squeeze()
        true_order = torch.argsort(true_scores, dim=-1, descending=True)
        reward = order_comp(true_order, action)

        done = torch.tensor([True]).repeat(self.batch_size)  # one step env
        return None, reward, done, None


class MLP(nn.Module):
    """Simple MLP used to map features to cluster labels.  It's a generic
    MLP but the architecture is hard-coded.

    Arguments:
        in_dims: dimensionality of 1-D features to be mapped
        nb_classes: number of cluster labels to support for the multi-class
            classification.
        layer_dims: List of integers, each specifying the dimensionality
            of the hidden layers.  For example, [128,256,512] would be
            specifying 3 hidden layers: the first would have 128 outputs,
            the second would have 256 outputs, and so on.
    """
    def __init__(self, in_dims, nb_classes, layer_dims):
        super(MLP, self).__init__()

        # Mix Linear layers with ReLU layers, except for the last one.
        inputs = [in_dims] + layer_dims
        outputs = layer_dims + [nb_classes]
        fc_layers = [nn.Linear(ind, outd) for ind, outd in zip(inputs, outputs)]
        relu_layers = [nn.ReLU(inplace=True)] * len(fc_layers)
        all_layers = list(
            itertools.chain.from_iterable(zip(fc_layers, relu_layers))
        )[:-1]  # drop last relu layer
        self.mlp_model = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.mlp_model(x)


class UserItemScorer(nn.Module):
    def __init__(
        self,
        user_dim: int,
        candidate_dim: int,
        slate_size: int,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.candidate_dim = candidate_dim
        self.slate_size = slate_size
        self.mlp = MLP(
            in_dims=self.user_dim,
            nb_classes=self.user_dim,
            layer_dims=[64, 64],
        )

    def forward(self, batch):
        batch_size = batch["user"].shape[0]
        batch_feat = torch.cat(
            (
                batch["user"].repeat(1, self.slate_size).reshape(
                    batch_size, self.slate_size, self.user_dim
                ),
                batch["items"],
            ),
            dim=-1
        )
        scores = self.mlp(batch_feat).squeeze()
        return scores


class Actor:
    def __init__(
        self,
        sampler,
        scorer,
    ):
        self.scorer = scorer
        self.sampler = sampler

    def act(self, observation):
        action_scores = None
        if self.scorer is not None:
            obs = observation.unsqueeze(0)
            action_scores = self.scorer(obs).squeeze()
        action = self.sampler.sample(action_scores)
        log_prob = self.sampler.log_prob(action_scores, action)
        return action, torch.exp(log_prob)


fpg = Actor(
    scorer=UserItemScorer(
        user_dim=env_config["user_dim"],
        candidate_dim=env_config["candidate_dim"],
        slate_size=env_config["slate_size"]
    ),
    sampler=FrechetOrderSampler(shape=2, upto=env_config["slate_size"] - 1)
)

