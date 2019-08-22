import datetime
import logging
import math
import os
import random
from typing import List, NamedTuple, Optional
import time
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


"""
Note on dimensions:
S: input set cardinality
N: batch size
L: output sequence length
E: embedding cardinality
"""


@torch.jit.script
def make_positional_mask(top_indices, num_inputs):
    # type: (Tensor, int) -> Tensor
    # shape: top_indices = K, N
    mask = torch.zeros(top_indices.shape[0], top_indices.shape[1], num_inputs)

    for i in range(top_indices.shape[0]):
        # TorchScript doesn't accept start index
        # conitnue is not support either
        if i != 0:
            # Copy from previous position
            mask[i, :, :] = mask[i - 1, :, :]

            for j in range(top_indices.shape[1]):
                for k in range(num_inputs):
                    # Set the used index to -inf to exclude it from loss calculation
                    if k == top_indices[i - 1][j]:
                        mask[i][j][k] = float("-inf")

    return mask


def collate_fn(batch):
    assert len(batch) == 1
    return tuple(torch.cat(ts, dim=0) for ts in zip(*batch))


class DataGenerator(Dataset):
    def __init__(self, locs, scales, weights, length, batch_size, num_items, k):
        self.num_input_dim = len(weights)
        assert len(locs) == len(scales) and len(locs) == len(weights)
        self.dist = torch.distributions.Normal(torch.tensor(locs), torch.tensor(scales))
        self.weights = torch.tensor(weights).unsqueeze(dim=1)
        self.length = length
        self.batch_size = batch_size
        self.num_items = num_items
        self.k = k

    def __getitem__(self, index):
        x, top_x, top_indices = self.gen_training_data(
            self.batch_size, self.num_items, self.k
        )
        positional_mask = make_positional_mask(top_indices, self.num_items)
        return x, top_x, top_indices, positional_mask

    def __len__(self):
        return self.length

    @torch.no_grad()
    def gen_data(self, n, m, k):
        x = self.dist.sample([n, m])
        scores = x @ self.weights
        top_scores, top_indices = torch.topk(scores.squeeze(dim=2), k, dim=1)
        return x, top_indices

    @torch.no_grad()
    def gen_training_data(self, n, m, k):
        """
        Prepare data in (sequence_length, batch_size, embedding_dim) format
        """
        x, top_k_indices = self.gen_data(n, m, k)
        gather_indices = torch.repeat_interleave(
            top_k_indices, self.num_input_dim, dim=1
        ).reshape(n, k, self.num_input_dim)
        top_k_x = torch.gather(x, 1, gather_indices)
        return (
            x.transpose(0, 1).contiguous(),
            top_k_x.transpose(0, 1).contiguous(),
            top_k_indices.transpose(0, 1).contiguous(),
        )


class LinearResLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    # @torch.jit.script_method
    def forward(self, x):
        return self.layer_norm(self.linear(x) + x)


# TODO: convert this to ScriptModule
class SelfAttentionResLayerNorm(nn.Module):
    def __init__(self, embed_dim, num_heads, mask_right=False):
        super().__init__()
        # FIXME: This is throwing error when converting to ScriptModule
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mask_right = mask_right
        if mask_right:
            self.register_buffer("attn_mask", torch.zeros(1, 1))

    def _reset_attn_mask(self, seq_length):
        self.attn_mask.resize_(seq_length, seq_length)
        for i in range(seq_length):
            for j in range(seq_length):
                self.attn_mask[i][j] = float("-inf") if j > i else 0.0

    def forward(self, x):
        if self.mask_right:
            seq_length = x.shape[0]
            if seq_length != self.attn_mask.shape[0]:
                self._reset_attn_mask(seq_length)
            attn_mask = self.attn_mask
        else:
            attn_mask = None
        y, _y_weights = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        return self.layer_norm(y + x)


class EncoderUnit(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attention = SelfAttentionResLayerNorm(embed_dim, num_heads)
        self.linear_res_layer_norm = LinearResLayerNorm(embed_dim)

    # @torch.jit.script_method
    def forward(self, input):
        return self.linear_res_layer_norm(self.self_attention(input))


class Encoder(nn.Module):
    __constants__ = ["units"]

    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.units = nn.ModuleList(
            [EncoderUnit(embed_dim, num_heads) for i in range(depth)]
        )

    # @torch.jit.script_method
    def forward(self, input):
        # type: (Tensor) -> List[Tensor]
        outputs = []
        x = input
        for unit in self.units:
            outputs.append(unit(x))
            x = outputs[-1]
        return outputs


class InputAttentionResLayerNorm(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # FIXME: This throws error when converting to TorchScript
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, input):
        y, _y_weights = self.attention(x, input, input, need_weights=False)
        return self.layer_norm(y + x)


class DecoderUnit(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attention = SelfAttentionResLayerNorm(
            embed_dim, num_heads, mask_right=True
        )
        self.input_attention = InputAttentionResLayerNorm(embed_dim, num_heads)
        self.linear_res_layer_norm = LinearResLayerNorm(embed_dim)

    # @torch.jit.script_method
    def forward(self, output, input):
        x = self.self_attention(output)
        y = self.input_attention(x, input)
        return self.linear_res_layer_norm(y)


class Decoder(nn.Module):
    __constants__ = ["units"]

    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.units = nn.ModuleList(
            [DecoderUnit(embed_dim, num_heads) for i in range(depth)]
        )

    # @torch.jit.script_method
    def forward(self, output, inputs):
        # type: (Tensor, List[Tensor]) -> Tensor
        x = output
        i = 0
        # TorchScript doesn't support enumerate() or zip() here
        for unit in self.units:
            x = unit(x, inputs[-1])
            i += 1
        return x


class Transformer(nn.Module):
    __constants__ = ["embed_dim", "scaling"]

    def __init__(
            self,
            embed_dim,
            num_heads,
            depth,
            num_input_dim,
            output_positional_embedding=True,
    ):
        super().__init__()
        self.encoder = Encoder(embed_dim, num_heads, depth)
        self.decoder = Decoder(embed_dim, num_heads, depth)
        self.input_linear = nn.Linear(num_input_dim, embed_dim)
        self.embed_dim = embed_dim
        self.register_buffer("zero_output_embed", torch.zeros(1, 1, embed_dim))
        self.scaling = embed_dim ** -0.5
        if output_positional_embedding:
            self.register_buffer(
                "output_positional_embedding", torch.zeros(0, 0, embed_dim)
            )
        else:
            self.output_positional_embedding = None

    # @torch.jit.script_method
    def _reset_zero_output_embed(self, batch_size):
        # type: (int) -> None
        self.zero_output_embed.resize_(1, batch_size, self.embed_dim)
        self.zero_output_embed.fill_(0.0)

    def _reset_output_positional_embedding(self, output_length):
        # type: (int) -> None
        self.output_positional_embedding.resize_(output_length, 1, self.embed_dim)
        for pos in range(output_length):
            for i in range(self.embed_dim):
                if i % 2 == 0:
                    self.output_positional_embedding[pos][0][i] = math.sin(
                        float(pos) / (10000 ** (float(i) / self.embed_dim))
                    )
                else:
                    self.output_positional_embedding[pos][0][i] = math.cos(
                        float(pos) / (10000 ** (float(i - 1) / self.embed_dim))
                    )

    # @torch.jit.script_method
    def forward(self, input, output):
        input_embed = self.input_linear(input)
        # shape: (S, N, E)
        encoded_inputs = self.encoder(input_embed)

        output = output[:-1, :, :]  # Removing the last element
        output_embed = self.input_linear(output)

        batch_size = output_embed.shape[1]
        if batch_size != self.zero_output_embed.shape[1]:
            self._reset_zero_output_embed(batch_size)

        decoder_input = torch.cat((self.zero_output_embed, output_embed), dim=0)

        if self.output_positional_embedding is not None:
            output_length = decoder_input.shape[0]
            if output_length != self.output_positional_embedding.shape[0]:
                self._reset_output_positional_embedding(output_length)
            decoder_input = decoder_input + self.output_positional_embedding

        # shape: (L, N, E)
        decoder_output = self.decoder(decoder_input, encoded_inputs) * self.scaling

        return torch.bmm(
            decoder_output.transpose(0, 1),  # (N, L, E)
            input_embed.transpose(0, 1).transpose(1, 2),  # (N, E, S)
        ).transpose(
            0, 1
        )  # (L, N, S)


class TrainingConfig(NamedTuple):
    embed_dim: int = 64
    num_heads: int = 4
    depth: int = 2
    lr: float = 1e-5
    batch_size: int = 128
    num_inputs: int = 7
    k: int = 5
    epoch_length: int = 1000
    num_epoch: int = 1
    report_interval: int = 10
    locs: List[float] = [0.3, 0.5, 0.1]
    scales: List[float] = [0.1, 0.1, 0.1]
    weights: List[float] = [0.2, 0.3, 0.5]
    output_positional_embedding: bool = True


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cpu")
    config = TrainingConfig()

    transformer = Transformer(
        config.embed_dim,
        config.num_heads,
        config.depth,
        len(config.weights),
        output_positional_embedding=config.output_positional_embedding,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), amsgrad=True)

    for epoch in range(config.num_epoch):
        dataset = DataGenerator(
            config.locs,
            config.scales,
            config.weights,
            config.epoch_length,
            config.batch_size,
            config.num_inputs,
            config.k,
        )
        dataloader = DataLoader(
            dataset, num_workers=0, pin_memory=True, collate_fn=collate_fn
        )

        start_time = time.time()
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()
            x, top_x, top_indices, positional_mask = [t.to(device) for t in batch]
            y = transformer(x, top_x)
            y = y + positional_mask

            loss = loss_fn(
                y.contiguous().view(-1, config.num_inputs), top_indices.view(-1)
            )
            loss.backward()
            if i % config.report_interval == 0:
                current_loss = loss.cpu().item()
                print(f"Epoch {epoch} iteration {i} loss {current_loss} elapsed {time.time() - start_time}")
                start_time = time.time()
            optimizer.step()

        for i, batch in enumerate(dataloader):
            x, top_x, top_indices, positional_mask = [t.to(device) for t in batch]




