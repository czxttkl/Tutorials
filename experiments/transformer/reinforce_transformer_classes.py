import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def embedding(idx, table):
    new_shape = (*idx.shape, -1)
    return table[idx.flatten()].reshape(new_shape)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    attn = torch.matmul(p_attn, value)
    # scores shape: batch_size x num_heads x seq_len x seq_len
    # p_attn shape: batch_size x num_heads x seq_len x seq_len
    # attn shape: batch_size x num_heads x seq_len x d_k
    # mask shape: batch_size x 1 x seq_len x seq_len
    return attn, p_attn


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
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, vocab_embedder, user_embedder, generator, positional_encoding):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_embedder = vocab_embedder
        self.user_embedder = user_embedder
        self.generator = generator
        self.positional_encoding = positional_encoding

    def forward(self, user_features, src_features, decoder_input_features, src_mask, decoder_input_mask):
        "Take in and process masked src and target sequences."

        # encoder_output shape: batch_size, seq_len + 1, dim_model
        encoder_output = self.encode(user_features, src_features, src_mask)

        # tgt_src_mask shape: batch_size, seq_len, seq_len + 1
        tgt_seq_len = decoder_input_features.shape[1]
        src_seq_len = src_features.shape[1]
        assert tgt_seq_len <= src_seq_len
        tgt_src_mask = src_mask[:, :tgt_seq_len, :]

        # decoder_output shape: batch_size, seq_len, dim_model
        decoder_output = self.decode(
            encoder_output, user_features, tgt_src_mask, decoder_input_features, decoder_input_mask
        )
        return decoder_output

    def encode(self, user_features, src_features, src_mask):
        # user_features: batch_size, dim_user
        # src_features: batch_size, seq_len, dim_vocab
        # src_src_mask shape: batch_size, seq_len + 1, seq_len + 1
        batch_size, seq_len, _ = src_features.shape

        # vocab_embed: batch_size, seq_len, dim_model/2
        vocab_embed = self.vocab_embedder(src_features)
        # user_embed: batch_size, dim_model/2
        user_embed = self.user_embedder(user_features)
        # user_embed: batch_size, seq_len, dim_model/2
        user_embed = user_embed.repeat(1, seq_len).reshape(batch_size, seq_len, -1)

        # src_embed shape: batch_size, seq_len, dim_model
        src_embed = self.positional_encoding(
            torch.cat((user_embed, vocab_embed), dim=-1)
        )

        # encoder_output shape: batch_size, seq_len + 1, dim_model
        return self.encoder(src_embed, src_mask)

    def decode(self, memory, user_features, tgt_src_mask, decoder_input_features, decoder_input_mask):
        # memory is the output of the encoder, the attention of each input symbol
        # memory shape: batch_size, seq_len, dim_model
        # tgt_src_mask shape: batch_size, seq_len, seq_len + 1
        # decoder_input_features shape: batch_size, seq_len, dim_vocab
        # decoder_input_mask shape: batch_size, seq_len, seq_len
        batch_size, seq_len, _ = decoder_input_features.shape

        # decoder_input_embed shape: batch_size, seq_len, dim_model/2
        decoder_input_embed = self.vocab_embedder(decoder_input_features)
        # user_embed: batch_size, dim_model/2
        user_embed = self.user_embedder(user_features)
        # user_embed: batch_size, seq_len, dim_model/2
        user_embed = user_embed.repeat(1, seq_len).reshape(batch_size, seq_len, -1)

        # decoder_input_embed: batch_size, seq_len, dim_model
        decoder_input_embed = self.positional_encoding(
            torch.cat((user_embed, decoder_input_embed), dim=-1)
        )

        # return shape: batch_size, seq_len, dim_model
        return self.decoder(decoder_input_embed, memory, tgt_src_mask, decoder_input_mask)


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

    def forward(self, x, decoder_input_idx):
        # generator receives the attention x from the decoder. Shape: batch_size, seq_len, dim_model
        # decoder_input_idx: input to the decoder, the first symbol is always the starting symbol
        # Shape: batch_size, seq_len

        # logits: the probability distribution of each symbol
        # batch_size, seq_len, vocab_size
        logits = self.proj(x)
        # the first two symbols are reserved for padding and decoder-starting symbols
        # so they should never be a possible output label
        logits[:, :, :2] = float("-inf")
        batch_size, seq_len = decoder_input_idx.shape
        mask_indices = torch.tril(decoder_input_idx.repeat(1, seq_len).reshape(batch_size, seq_len, seq_len), diagonal=0)
        logits.scatter_(2, mask_indices, float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def greedy_decode(self, x, y_decoder):
        # x is the attention of the latest step from the decoder. Shape: batch_size, dim_model
        # y_decoder: input to the decoder, the first symbol is always the starting symbol
        # Shape: batch_size, seq_len
        # the first two symbols are reserved for padding and decoder-starting symbols so
        # they should never be a possible output label
        assert len(x.shape) == 2 and x.shape[1] == self.dim_model
        logits = self.proj(x)
        # invalidate the padding symbol and decoder-starting symbol
        logits[:, :2] = float("-inf")
        logits.scatter_(1, y_decoder, float("-inf"))
        return F.log_softmax(logits, dim=-1)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dim_model):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim_model)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
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
        x, self.attn = attention(query, key, value, mask=mask)

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

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
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


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, user_features, src_mask, trg_idx, src_features, tgt_features, padding_symbol):
        # user_features shape: batch_size, user_dim
        # src_src_mask shape: batch_size, seq_len + 1, seq_len + 1
        # tgt_in_idx shape: batch_size, seq_len + 1
        # src_embed shape: batch_size, seq_len, vocab_dim
        # tgt_embed shape: batch_size, seq_len + 1, vocab_dim

        batch_size, seq_len, _ = src_mask.shape
        self.src_mask = src_mask
        self.user_features = user_features

        # decoder_input_idx shape: batch_size, seq_len
        self.decoder_input_idx = trg_idx[:, :-1]
        # target_label_idx shape: batch_size, seq_len
        self.target_label_idx = trg_idx[:, 1:]
        # trg_mask shape: batch_size, seq_len, seq_len
        self.trg_mask = self.make_std_mask(self.decoder_input_idx, padding_symbol)
        # ntoken shape: batch_size * seq_len
        self.ntokens = (self.target_label_idx != padding_symbol).data.sum()

        # src_embed shape: batch_size x seq_len x vocab_dim
        self.src_features = src_features
        # decoder_input_embed shape: batch_size x seq_len x vocab_dim
        self.decoder_input_features = tgt_features[:, :-1, :]

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # tgt shape: batch_size, seq_len
        # tgt_mask shape: batch_size, 1, seq_len
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # subseq_mask shape: 1, seq_len, seq_len
        subseq_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        # tgt_mask shape: batch_size, seq_len, seq_len
        tgt_mask = tgt_mask & subseq_mask
        return tgt_mask.type(torch.int8)


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


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, decoder_input_idx, y_label, norm):
        # x: the attention x from the decoder. Shape: batch_size, seq_len, dim_model
        # decoder_input_idx: input to the decoder, the first symbol is always the starting symbol
        # y_label shape: batch_size, seq_len

        # log probs: log probability distribution of each symbol
        # shape: batch_size, seq_len, vocab_size
        log_probs = self.generator(x, decoder_input_idx)
        loss = (
            self.criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)),
                y_label.contiguous().view(-1)
            )
            / norm
        )
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss * norm


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, tgt_vocab_size, padding_idx, starting_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.starting_idx = starting_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.true_dist = None

    def forward(self, x, target):
        # x shape: flatten_batch_size x tgt_vocab_size
        # target shape: flatten_batch_size
        # flatten_batch_size = batch_size x seq_len
        assert x.size(1) == self.tgt_vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.tgt_vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        true_dist[:, self.starting_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)