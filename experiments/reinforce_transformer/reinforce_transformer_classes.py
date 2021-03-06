import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional
import scipy.stats as stats
from itertools import combinations
from common import START_SYMBOL, PADDING_SYMBOL


def eval_function_high_reward_prob(log_probs, rewards):
    highest_possible_reward = np.max(rewards)
    # what's the log prob of sequences whose rewards are the highest
    eval1 = np.mean(log_probs[rewards == highest_possible_reward])
    # kendall tau test. High rewards should correlate to high log probs
    eval2 = stats.kendalltau(log_probs, rewards).correlation
    return round(eval1, 5), round(eval2, 5)


def reward_function_pairwise(user_feature, vocab_feature, tgt_out_idx, true_tgt_out_idx):
    if len(tgt_out_idx) == 1:
        return float(tgt_out_idx[0] == true_tgt_out_idx[0])
    truth_pairs = set(combinations(true_tgt_out_idx, 2))
    tgt_pairs = set(combinations(tgt_out_idx, 2))
    return float(len(truth_pairs & tgt_pairs))


def embedding(idx, table):
    """ numpy version of embedding look up """
    new_shape = (*idx.shape, -1)
    return table[idx.flatten()].reshape(new_shape)


def subsequent_mask(size):
    """
    Mask out subsequent positions. Mainly used in the decoding process,
    in which an item should not attend subsequent items.
    """
    attn_shape = (1, size, size)
    subsequent_mask = (1 - torch.triu(torch.ones(*attn_shape), diagonal=1)).type(
        torch.int8
    )
    return subsequent_mask


def subsequent_and_padding_mask(tgt_in_idx):
    """ Create a mask to hide padding and future items """
    # tgt_in_idx shape: batch_size, seq_len

    # tgt_tgt_mask shape: batch_size, 1, seq_len
    tgt_tgt_mask = (tgt_in_idx != PADDING_SYMBOL).unsqueeze(-2).type(torch.int8)
    # subseq_mask shape: 1, seq_len, seq_len
    subseq_mask = subsequent_mask(tgt_in_idx.size(-1))
    # tgt_tgt_mask shape: batch_size, seq_len, seq_len
    tgt_tgt_mask = tgt_tgt_mask & subseq_mask
    return tgt_tgt_mask


def clones(module, N):
    """
    Produce N identical layers.

    :param module: nn.Module class
    :param N: number of copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask):
    """ Scaled Dot Product Attention """
    # mask shape: batch_size x 1 x seq_len x seq_len

    d_k = query.size(-1)
    # scores shape: batch_size x num_heads x seq_len x seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn shape: batch_size x num_heads x seq_len x seq_len
    p_attn = F.softmax(scores, dim=-1)
    # attn shape: batch_size x num_heads x seq_len x d_k
    attn = torch.matmul(p_attn, value)
    return attn, p_attn


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(
        self,
        user_features,
        src_src_mask,
        tgt_idx_with_start_sym,
        truth_idx,
        src_features,
        src_in_idx,
        tgt_features_with_start_sym,
        rewards,
        tgt_probs,
    ):
        # user_features shape: batch_size, user_dim
        # src_src_mask shape: batch_size, seq_len, seq_len
        # tgt_idx_with_start_sym shape: batch_size, tgt_seq_len + 1
        # truth_idx shape: batch_size, tgt_seq_len
        # src_features shape: batch_size, seq_len, vocab_dim
        # src_in_idx shape: batch_size, seq_len
        # tgt_features_with_start_sym shape: batch_size, tgt_seq_len + 1, vocab_dim (including the feature of starting symbol)
        # rewards shape: batch_size
        # tgt_probs shape: batch_size

        self.src_in_idx = src_in_idx
        self.src_src_mask = src_src_mask
        self.user_features = user_features
        self.rewards = rewards
        self.tgt_probs = tgt_probs
        self.src_features = src_features
        self.truth_idx = truth_idx

        if tgt_idx_with_start_sym is not None:
            # igt_idx shape: batch_size, tgt_seq_len
            self.tgt_in_idx = tgt_idx_with_start_sym[:, :-1]
            # tgt_out_idx shape: batch_size, tgt_seq_len
            self.tgt_out_idx = tgt_idx_with_start_sym[:, 1:]
            # tgt_tgt_mask shape: batch_size, tgt_seq_len, tgt_seq_len
            self.tgt_tgt_mask = subsequent_and_padding_mask(self.tgt_in_idx)
            # tgt_features shape: batch_size x seq_len x vocab_dim
            self.tgt_features = tgt_features_with_start_sym[:, :-1, :]
            # ntoken shape: batch_size * seq_len
            self.ntokens = (self.tgt_out_idx != PADDING_SYMBOL).data.sum()
        else:
            self.tgt_in_idx = None
            self.tgt_out_idx = None
            self.tgt_tgt_mask = None
            self.tgt_features = None
            self.ntokens = None



class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture
    """

    def __init__(
        self,
        encoder,
        decoder,
        vocab_embedder,
        user_embedder,
        generator,
        positional_encoding,
        dim_model,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_embedder = vocab_embedder
        self.user_embedder = user_embedder
        self.generator = generator
        self.positional_encoding = positional_encoding
        self.dim_model = dim_model
        self.padding_mask_param = torch.nn.Parameter(torch.zeros(dim_model))
        self.decoder_start_mask_param = torch.nn.Parameter(torch.zeros(dim_model))

    def forward(self, batch: Batch, mode: str, tgt_seq_len: Optional[int] = None, greedy: Optional[bool] = None):
        if mode == "log_probs":
            return self._log_probs(
                batch.user_features,
                batch.src_features,
                batch.tgt_features,
                batch.src_src_mask,
                batch.tgt_tgt_mask,
                batch.src_in_idx,
                batch.tgt_in_idx,
                batch.tgt_out_idx,
            )
        elif mode == "rank":
            assert tgt_seq_len is not None and greedy is not None
            return self._rank(batch.user_features, batch.src_features, batch.src_src_mask, batch.src_in_idx, tgt_seq_len, greedy)

    def _rank(self, user_features, src_features, src_src_mask, src_in_idx, tgt_seq_len, greedy):
        """ Decode sequences based on given inputs """
        device = src_features.device
        batch_size, src_seq_len, vocab_dim = src_features.shape
        vocab_size = src_seq_len + 2
        # vocab_features is used as look-up table for vocab features.
        # the second dim is src_seq_len + 2 because we also want to include features of start symbol and padding symbol
        vocab_features = torch.zeros(batch_size, src_seq_len + 2, vocab_dim).to(device)
        vocab_features[:, 2:, :] = src_features

        tgt_in_idx = torch.ones(batch_size, 1).fill_(START_SYMBOL).type(torch.long).to(device)
        decoder_probs = torch.zeros(batch_size, tgt_seq_len, vocab_size).to(device)

        memory = self.encode(user_features, src_features, src_in_idx, src_src_mask)

        for l in range(tgt_seq_len):
            tgt_features = vocab_features[
                torch.arange(batch_size).repeat_interleave(l + 1),
                tgt_in_idx.flatten(),
            ].view(batch_size, l + 1, -1).to(device)
            tgt_src_mask = src_src_mask[:, :l + 1, :]
            out = self.decode(
                memory=memory,
                user_features=user_features,
                tgt_src_mask=tgt_src_mask,
                tgt_features=tgt_features,
                tgt_tgt_mask=subsequent_mask(l + 1).to(device),
                tgt_seq_len=l+1,
            )
            # next word shape: batch_size, 1
            # prob shape: batch_size, vocab_size
            next_word, prob = self.generator(
                mode="decode_one_step",
                decoder_output=out,
                tgt_in_idx=tgt_in_idx,
                greedy=greedy
            )
            decoder_probs[:, l, :] = prob
            tgt_in_idx = torch.cat(
                [tgt_in_idx, next_word],
                dim=1,
            ).to(device)
        # remove the starting symbol
        # shape: batch_size, tgt_seq_len
        tgt_in_idx = tgt_in_idx[:, 1:]
        return decoder_probs, tgt_in_idx

    def _log_probs(
        self, user_features, src_features, tgt_features, src_src_mask, tgt_tgt_mask, src_in_idx, tgt_in_idx, tgt_out_idx
    ):
        """
        Compute log of generative probabilities of given tgt sequences (used for REINFORCE training)
        """
        # encoder_output shape: batch_size, seq_len + 1, dim_model
        encoder_output = self.encode(user_features, src_features, src_in_idx, src_src_mask)

        tgt_seq_len = tgt_features.shape[1]
        src_seq_len = src_features.shape[1]
        assert tgt_seq_len <= src_seq_len

        # tgt_src_mask shape: batch_size, seq_len, seq_len
        tgt_src_mask = src_src_mask[:, :tgt_seq_len, :]

        # decoder_output shape: batch_size, seq_len, dim_model
        decoder_output = self.decode(
            memory=encoder_output,
            user_features=user_features,
            tgt_src_mask=tgt_src_mask,
            tgt_features=tgt_features,
            tgt_tgt_mask=tgt_tgt_mask,
            tgt_seq_len=tgt_seq_len,
        )
        # log_probs shape: batch_size
        log_probs = self._output_to_log_prob(decoder_output, tgt_in_idx, tgt_out_idx)

        return log_probs

    def _output_to_log_prob(self, decoder_output, tgt_in_idx, tgt_out_idx):
        # decoder_output: the output from the decoder. Shape: batch_size, seq_len, dim_model
        # tgt_in_idx: input idx to the decoder, the first symbol is always the starting symbol
        # tgt_in_idx shape: batch_size, seq_len
        # tgt_out_idx: output idx of the decoder, shape: batch_size, seq_len

        # log probs: log probability distribution of each symbol
        # shape: batch_size, seq_len, vocab_size
        raw_log_probs = self.generator(mode="log_probs", decoder_output=decoder_output, tgt_in_idx=tgt_in_idx)
        batch_size, seq_len, vocab_size = raw_log_probs.shape

        # log_probs: each symbol of the label sequence's generative log probability
        # shape: batch_size, seq_len
        log_probs = raw_log_probs.view(-1, vocab_size)[
            np.arange(batch_size * seq_len), tgt_out_idx.flatten()
        ].view(batch_size, seq_len)

        # shape: batch_size
        return log_probs.sum(dim=1)

    def encode(self, user_features, src_features, src_in_idx, src_src_mask):
        # user_features: batch_size, dim_user
        # src_features: batch_size, seq_len, dim_vocab
        # src_src_mask shape: batch_size, seq_len, seq_len
        batch_size, seq_len, _ = src_features.shape

        # vocab_embed: batch_size, seq_len, dim_model/2
        vocab_embed = self.vocab_embedder(src_features)
        # user_embed: batch_size, dim_model/2
        user_embed = self.user_embedder(user_features)
        # user_embed: batch_size, seq_len, dim_model/2
        user_embed = user_embed.repeat(1, seq_len).reshape(batch_size, seq_len, -1)

        concat_embed = torch.cat((user_embed, vocab_embed), dim=-1)

        # since all sequences are of the same length for now, there shouldn't be any
        # padding symbol
        # concat_embed[src_in_idx == PADDING_SYMBOL] = self.padding_mask_param
        # assert not torch.any(src_in_idx == PADDING_SYMBOL)

        # src_embed shape: batch_size, seq_len, dim_model
        src_embed = self.positional_encoding(concat_embed, seq_len)

        # encoder_output shape: batch_size, seq_len + 1, dim_model
        return self.encoder(src_embed, src_src_mask)

    def decode(self, memory, user_features, tgt_src_mask, tgt_features, tgt_tgt_mask, tgt_seq_len):
        # memory is the output of the encoder, the attention of each input symbol
        # memory shape: batch_size, seq_len, dim_model
        # tgt_src_mask shape: batch_size, tgt_seq_len, seq_len
        # tgt_features shape: batch_size, tgt_seq_len, dim_vocab
        # tgt_tgt_mask shape: batch_size, tgt_seq_len, tgt_seq_len
        batch_size = tgt_features.shape[0]

        # vocab_embed shape: batch_size, seq_len, dim_model/2
        vocab_embed = self.vocab_embedder(tgt_features)
        # user_embed: batch_size, dim_model/2
        user_embed = self.user_embedder(user_features)
        # user_embed: batch_size, seq_len, dim_model/2
        user_embed = user_embed.repeat(1, tgt_seq_len).reshape(batch_size, tgt_seq_len, -1)

        concat_embed = torch.cat((user_embed, vocab_embed), dim=-1)
        # not using decoder_start_mask_param because it slows down the training
        # concat_embed[:, 0] = self.decoder_start_mask_param

        # tgt_embed: batch_size, seq_len, dim_model
        tgt_embed = self.positional_encoding(concat_embed, tgt_seq_len)

        # output of decoder will be later transformed into probabilities over symbols.
        # shape: batch_size, seq_len, dim_model
        return self.decoder(tgt_embed, memory, tgt_src_mask, tgt_tgt_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of num_layers layers"

    def __init__(self, layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.dim_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, dim_model, vocab_size):
        super(Generator, self).__init__()
        self.dim_model = dim_model
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, mode, decoder_output=None, tgt_in_idx=None, greedy=None):
        if mode == "log_probs":
            return self._log_probs(decoder_output, tgt_in_idx)
        elif mode == "decode_one_step":
            assert greedy is not None
            return self._decode_one_step(decoder_output, tgt_in_idx, greedy)

    def _log_probs(self, x, tgt_in_idx):
        # x: the output of decoder. Shape: batch_size, seq_len, dim_model
        # tgt_in_idx: input to the decoder, the first symbol is always the starting symbol
        # Shape: batch_size, seq_len

        # logits: the probability distribution of each symbol
        # batch_size, seq_len, vocab_size
        logits = self.proj(x)
        # the first two symbols are reserved for padding and decoder-starting symbols
        # so they should never be a possible output label
        logits[:, :, :2] = float("-inf")
        batch_size, seq_len = tgt_in_idx.shape
        mask_indices = torch.tril(tgt_in_idx.repeat(1, seq_len).reshape(batch_size, seq_len, seq_len), diagonal=0)
        logits.scatter_(2, mask_indices, float("-inf"))
        # log_probs shape: batch_size, seq_len, vocab_size
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def _decode_one_step(self, x, tgt_in_idx, greedy):
        # decode one-step
        # x: the output ofthe decoder. Shape: batch_size, seq_len, dim_model
        # tgt_in_idx: input to the decoder, the first symbol is always the starting symbol
        # Shape: batch_size, seq_len
        # greedy: whether to greedily pick or sample the next symbol

        # get the last step of decoder output
        last_step_x = x[:, -1, :]

        batch_size = x.shape[0]
        logits = self.proj(last_step_x)
        # invalidate the padding symbol and decoder-starting symbol
        logits[:, :2] = float("-inf")
        # invalidate symbols already appeared in decoded sequences
        logits.scatter_(1, tgt_in_idx, float("-inf"))
        prob = F.softmax(logits, dim=-1)
        if greedy:
            _, next_word = torch.max(prob, dim=1)
        else:
            next_word = torch.multinomial(prob, num_samples=1, replacement=False)
        next_word = next_word.reshape(batch_size, 1)

        # next_word: the decoded symbols for the latest step
        # shape: batch_size x 1
        # prob: generative probabilities of the latest step
        # shape: batch_size x vocab_size
        return next_word, prob


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, dim_model):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, dim_model, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(dim_model), 2)
        self.dim_model = dim_model

    def forward(self, src_embed, src_mask):
        # src_embed shape: batch_size, seq_len, dim_model
        # src_src_mask shape: batch_size, seq_len, seq_len

        def self_attn_layer(x):
            return self.self_attn(x, x, x, src_mask)

        # attn_output shape: batch_size, seq_len, dim_model
        attn_output = self.sublayer[0](src_embed, self_attn_layer)
        # return shape: batch_size, seq_len, dim_model
        return self.sublayer[1](attn_output, self.feed_forward)


class Decoder(nn.Module):
    "Generic num_layers layer decoder with masking."

    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, tgt_src_mask, tgt_mask):
        # each layer is one DecoderLayer
        for layer in self.layers:
            x = layer(x, memory, tgt_src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 3)

    def forward(self, x, memory, tgt_src_mask, tgt_mask):
        # x shape: batch_size, seq_len, dim_model
        # x is target embedding or the output of previous decoder layer
        # memory shape: batch_size, seq_len, dim_model
        # memory is the output of the last encoder layer
        # tgt_src_mask shape: batch_size, seq_len, seq_len + 1
        # tgt_mask shape: batch_size, seq_len, seq_len
        m = memory

        def self_attn_layer_tgt(x):
            return self.self_attn(query=x, key=x, value=x, mask=tgt_mask)

        def self_attn_layer_src(x):
            return self.self_attn(query=x, key=m, value=m, mask=tgt_src_mask)

        x = self.sublayer[0](x, self_attn_layer_tgt)
        x = self.sublayer[1](x, self_attn_layer_src)
        # return shape: batch_size, seq_len, dim_model
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim_model):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert dim_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = dim_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all num_heads heads.
            # mask shape: batch_size, 1, seq_len, seq_len
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from dim_model => num_heads x d_k
        # self.linear[0, 1, 2] is query weight matrix, key weight matrix, and value weight matrix, respectively
        # l(x) represents the transformed query matrix, key matrix and value matrix
        # l(x) has shape batch_size, seq_len, dim_model. You can think l(x) as the matrices from a one-head attention
        # or after view() and transpose(), it has shape batch_size, num_heads, seq_len, d_k, so that you
        # can think l(x) as the matrices of num_heads attentions, each attention has d_k dimension.
        query, key, value = [
            l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: batch_size, num_heads, seq_len, d_k
        x, self.attn = attention(query, key, value, mask)

        # 3) "Concat" using a view and apply a final linear.
        # each attention's output is d_k dimension. Concat num_heads attention's outputs
        # x shape: batch_size, seq_len, dim_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, dim_model, dim_feedforward):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, dim_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class VocabEmbedder(nn.Module):
    def __init__(self, dim_vocab, dim_model):
        super(VocabEmbedder, self).__init__()
        self.dim_model = dim_model // 2
        self.dim_vocab = dim_vocab
        self.linear = nn.Linear(self.dim_vocab, self.dim_model)

    def forward(self, x):
        # x: raw input features. Shape: batch_size, seq_len, dim_vocab
        output = self.linear(x) * math.sqrt(self.dim_model)
        # output shape: batch_size, seq_len, dim_model / 2
        return output


class UserEmbedder(nn.Module):
    def __init__(self, dim_user, dim_model):
        super(UserEmbedder, self).__init__()
        self.dim_model = dim_model // 2
        self.dim_user = dim_user
        self.linear = nn.Linear(self.dim_user, self.dim_model)

    def forward(self, x):
        # x: raw input features. Shape: batch_size, seq_len, dim_user
        output = self.linear(x) * math.sqrt(self.dim_model)
        # output shape: batch_size, seq_len, dim_model / 2
        return output


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, dim_model, 2) * -(math.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe shape: 1, max_len, dim_model
        self.register_buffer("pe", pe)

    def forward(self, x, seq_len):
        x = x + self.pe[:, :seq_len]
        return x


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, dim_model, factor, warmup, optimizer, constant_rate=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.dim_model = dim_model
        self._rate = 0
        # if True, use constant rate
        self.constant_rate = constant_rate

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if self.constant_rate:
            return self.constant_rate
        if step is None:
            step = self._step
        return self.factor * (
            self.dim_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


# # Test LabelSmoothing
# crit = LabelSmoothing(tgt_vocab_size=5, padding_idx=0, smoothing=0.0)
# pred = torch.tensor([[0, 0.9, 0.1, 0, 0]], dtype=torch.float32)
# label = torch.tensor([1], dtype=torch.long)
# print(crit(pred, label), crit.true_dist)
# pred = torch.tensor([[0, 0.7, 0.1, 0.1, 0.1]], dtype=torch.float32)
# label = torch.tensor([1], dtype=torch.long)
# print(crit(pred, label), crit.true_dist)
#
# crit = LabelSmoothing(tgt_vocab_size=5, padding_idx=0, smoothing=0.3)
# pred = torch.tensor([[0, 0.7, 0.1, 0.1, 0.1]], dtype=torch.float32)
# label = torch.tensor([1], dtype=torch.long)
# print(crit(pred, label), crit.true_dist)


class BaselineNN(nn.Module):

    def __init__(self, dim_model, user_dim, num_layers):
        super(BaselineNN, self).__init__()
        h_sizes = [user_dim] + [dim_model] * num_layers + [1]
        self.num_layers = num_layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

    def forward(self, x):
        for i in range(self.num_layers + 1):
            if i == self.num_layers:
                return self.hidden[i](x)
            else:
                x = F.relu(self.hidden[i](x))


class ReinforceLossCompute:

    def __init__(self, on_policy, rl_opt=None, baseline_opt=None):
        self.on_policy = on_policy
        self.rl_opt = rl_opt
        self.baseline_opt = baseline_opt

    def __call__(self, log_probs, reward, baseline, user_features, tgt_probs):
        # log_probs: the generative probability of each sequence
        # Shape: batch_size
        # reward: reward associated with each sequence
        # reward shape: batch_size
        # baseline: the baseline model
        # user_features: the user feature associated with each sequence. Used to compute baseline
        # shape: batch_size x user_dim
        # tgt_probs: tgt sequence probs, used for off policy learning
        # shape: batch_size
        batch_size = log_probs.shape[0]

        b = baseline(user_features).squeeze()
        assert b.requires_grad
        baseline_loss = 1. / batch_size * torch.sum((b - reward) ** 2)
        baseline_loss.backward()
        if self.baseline_opt:
            self.baseline_opt.step()
            self.baseline_opt.zero_grad()

        b = b.detach()

        assert b.shape == reward.shape == log_probs.shape
        assert not b.requires_grad
        assert not reward.requires_grad
        assert log_probs.requires_grad

        # importance sampling
        if not self.on_policy:
            importance_sampling = torch.exp(log_probs.detach()) / tgt_probs
        else:
            importance_sampling = 1
        # add negative sign because we take gradient descent but we want to maximize rewards
        # rl_loss = - 1. / batch_size * torch.sum(log_probs * (reward - b))
        batch_loss = -importance_sampling * log_probs * (reward - b)
        rl_loss = 1. / batch_size * torch.sum(batch_loss)
        rl_loss.backward()
        if self.rl_opt:
            self.rl_opt.step()
            self.rl_opt.zero_grad()

        return rl_loss.cpu().detach().numpy(), baseline_loss.cpu().detach().numpy()

