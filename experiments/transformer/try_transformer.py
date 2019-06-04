import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


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

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        # encode_output shape: batch_size, input_dim, dim_model
        encode_output = self.encode(src, src_mask)
        return self.decode(encode_output, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # src shape: batch_size, input_dim
        # src_mask shape: batch_size, 1, input_dim
        # src_embed: batch_size, input_dim, dim_model
        src_embed = self.src_embed(src)
        return self.encoder(src_embed, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory shape: batch_size, input_dim, dim_model
        # src_mask shape: batch_size, 1, input_dim
        # tgt shape: batch_size, input_dim - 1
        # tgt_mask shape: batch_size, input_dim-1, input_dim-1
        # tgt_embed shape: batch_size, input_dim-1, dim_model
        tgt_embed = self.tgt_embed(tgt)
        return self.decoder(tgt_embed, memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.dim_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dim_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, dim_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(dim_model, dropout), 2)
        self.dim_model = dim_model

    def forward(self, src_embed, src_mask):
        "Follow Figure 1 (left) for connections."
        # attn_output = self.self_attn(x, x, x, src_mask)
        src_embed = self.sublayer[0](src_embed, lambda x: self.self_attn(x, x, x, src_mask))
        return self.sublayer[1](src_embed, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, dim_model, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


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


def get_std_opt(model):
    return NoamOpt(
        model.src_embed[0].d_model,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )


def make_model(
    src_vocab_size,
    tgt_vocab_size,
    num_stacked_layers=6,
    dim_model=512,
    dim_feedforward=512,
    num_heads=8,
    dropout=0.1,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_feedforward, dropout)
    position = PositionalEncoding(dim_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), num_stacked_layers),
        Decoder(
            DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout),
            num_stacked_layers,
        ),
        nn.Sequential(Embeddings(dim_model, src_vocab_size), c(position)),
        nn.Sequential(Embeddings(dim_model, tgt_vocab_size), c(position)),
        Generator(dim_model, tgt_vocab_size),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        # src shape: batch_size, input_dim
        # tgt shape: batch_size, input_dim
        # src_mask shape: batch_size, 1, input_dim
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # trg shape: batch_size, input_dim - 1
            self.trg = trg[:, :-1]
            # trg_y shape: batch_size, input_dim - 1
            self.trg_y = trg[:, 1:]
            # trg_mask shape: batch_size, input_dim-1, input-dim-1
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # ntoken shape: batch_size * (input_dim - 1)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # tgt shape: batch_size, input_dim-1
        # tgt_mask shape: batch_size, 1, input_dim-1
        tgt_mask = (tgt != pad).unsqueeze(-2)
        subseq_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        # tgt_mask shape: batch_size, input_dim-1, input_dim-1
        tgt_mask = tgt_mask & Variable(subseq_mask)
        return tgt_mask


def run_epoch(epoch, data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # out shape: batch_size, input_dim-1, dim_model
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss.detach().numpy()
        total_tokens += batch.ntokens.numpy()
        tokens += batch.ntokens.numpy()
        if i and i % 10 == 9:
            elapsed = time.time() - start
            print(
                "Epoch %d Step: %d Loss: %f Tokens per Sec: %f"
                % (epoch, i, loss.detach().numpy() / batch.ntokens.numpy(), tokens / elapsed)
            )
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def data_gen(vocab_size, batch_size, num_batches):
    "Generate random data for a src-tgt copy task."
    for _ in range(num_batches):
        data = torch.from_numpy(np.random.randint(1, vocab_size, size=(batch_size, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, pad=0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss * norm


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, tgt_vocab_size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.true_dist = None

    def forward(self, x, target):
        # x shape: batch_size x tgt_vocab_size
        # target shape: batch_size
        assert x.size(1) == self.tgt_vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.tgt_vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


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


# src vocab size equals to tgt vocab size
VOCAB_SIZE = 11
EPOCH_NUM = 6
DIM_MODEL = 512
DIM_FEEDFORWARD = 256
NUM_STACKED_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 30
NUM_TRAIN_BATCHES = 50
NUM_EVAL_BATCHES = 5

criterion = LabelSmoothing(tgt_vocab_size=VOCAB_SIZE, padding_idx=0, smoothing=0.0)
model = make_model(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    num_stacked_layers=NUM_STACKED_LAYERS,
    dim_model=DIM_MODEL,
    dim_feedforward=DIM_FEEDFORWARD,
    num_heads=NUM_HEADS,
)
model_opt = NoamOpt(
    dim_model=DIM_MODEL,
    factor=1,
    warmup=400,
    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
)

for epoch in range(EPOCH_NUM):
    model.train()
    run_epoch(
        epoch,
        data_gen(vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE, num_batches=NUM_TRAIN_BATCHES),
        model,
        SimpleLossCompute(model.generator, criterion, model_opt),
    )
    model.eval()
    print(
        "eval loss:",
        run_epoch(
            epoch,
            data_gen(vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE, num_batches=NUM_EVAL_BATCHES),
            model,
            SimpleLossCompute(model.generator, criterion, None),
        ),
    )


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory,
            src_mask,
            Variable(ys),
            Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


model.eval()
src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
