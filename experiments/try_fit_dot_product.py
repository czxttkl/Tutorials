import torch
import torch.nn as nn
import numpy as np
import dataclasses
from dataclasses import asdict, dataclass
from collections import deque


@dataclass
class PreprocessedRankingInput:
    state_features: torch.Tensor
    tgt_out_seq: torch.Tensor
    y: torch.Tensor


class ValueModel(nn.Module):
    """
    Generate ground-truth VM coefficients based on user features + candidate distribution
    """

    def __init__(self, state_feat_dim, candidate_feat_dim, hidden_size):
        super(ValueModel, self).__init__()
        self.state_feat_dim = state_feat_dim
        self.candidate_feat_dim = candidate_feat_dim
        self.hidden_size = hidden_size
        # self.layer1 = nn.Linear(state_feat_dim + candidate_feat_dim, 1)
        self.layer1 = nn.Linear(state_feat_dim, candidate_feat_dim)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # model will be called with fixed parameters
            p.requires_grad = False

    def forward(
        self,
        state_features,
        tgt_out_seq,
    ):
        vm_coef = self.layer1(state_features).unsqueeze(2)
        # tgt_out_seq shape: batch-size x max_tgt_seq_len x candidate_feat_dim
        # vm_coef shape: batch_size x candidate_feat_dim x 1
        # return shape: batch_size x max_tgt_seq_len
        pointwise_score = torch.bmm(tgt_out_seq, vm_coef).squeeze()
        return pointwise_score


def dataset_gen(
    batch_size,
    state_feat_dim,
    candidate_feat_dim,
    max_tgt_seq_len,
    vm,
):
    for _ in range(10000000):
        state_features = torch.randn(batch_size, state_feat_dim)
        tgt_out_seq = torch.randn(batch_size, max_tgt_seq_len, candidate_feat_dim)
        y = vm(state_features, tgt_out_seq)
        yield PreprocessedRankingInput(
            state_features=state_features,
            tgt_out_seq=tgt_out_seq,
            y=y,
        )


def create_nn(
    input_dim,
    d_model=512,
):
    feat_embed = nn.Linear(input_dim, d_model)
    scorer = nn.Linear(d_model, 1)
    final_nn = nn.Sequential(
        feat_embed,
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        scorer,
    )
    return final_nn


def batch_to_score(model, batch):
    batch_size, tgt_seq_len, candidate_feat_dim = batch.tgt_out_seq.shape
    state_feat_dim = batch.state_features.shape[1]

    # shape: batch_size, tgt_seq_len, candidate_feat_dim + state_feat_dim
    concat_feat_vec = torch.cat(
        (
            batch.state_features.repeat(1, tgt_seq_len).reshape(
                batch_size, tgt_seq_len, state_feat_dim
            ),
            batch.tgt_out_seq,
        ),
        dim=2,
    )

    # shape: batch_size * tgt_seq_len
    model_output = model(concat_feat_vec).squeeze()
    return model_output


def train(model, batch, optimizer):
    # shape: batch_size, tgt_seq_len
    output = batch_to_score(model, batch)

    mse_loss = nn.MSELoss()
    assert output.shape == batch.y.shape
    loss = mse_loss(output, batch.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy()



def main(
    create_model_func,
    state_feat_dim,
    candidate_feat_dim,
    batch_size,
    max_tgt_seq_len,
):
    vm = ValueModel(state_feat_dim, candidate_feat_dim, 3)
    dataset = dataset_gen(
        batch_size,
        state_feat_dim,
        candidate_feat_dim,
        max_tgt_seq_len,
        vm,
    )

    model = create_model_func(
        input_dim=state_feat_dim + candidate_feat_dim
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, amsgrad=True,
    )

    epoch_loss = deque([], maxlen=100)
    i = 0
    for batch in dataset:
        i += 1
        loss = train(model, batch, optimizer)
        epoch_loss.append(loss)
        print(f"batch {i} loss {loss} average loss: {np.mean(epoch_loss)}")



if __name__ == "__main__":
    state_feat_dim = 3
    candidate_feat_dim = 4
    batch_size = 1024
    max_tgt_seq_len = 5
    main(
        create_nn,
        state_feat_dim,
        candidate_feat_dim,
        batch_size,
        max_tgt_seq_len,
    )