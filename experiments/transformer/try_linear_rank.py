import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformer_classes import (
    clones,
    subsequent_mask,
    Embeddings,
    PositionalEncoding,
    PositionwiseFeedForward,
    MultiHeadedAttention,
    EncoderDecoder,
    EncoderLayer,
    Encoder,
    SimpleLossCompute,
    LabelSmoothing,
    Generator,
    DecoderLayer,
    Decoder,
    Batch
)


def make_model(
    src_vocab_size,
    tgt_vocab_size,
    num_stacked_layers=6,
    dim_model=512,
    dim_feedforward=512,
    num_heads=8,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_feedforward)
    position = PositionalEncoding(dim_model)
    embedding = nn.Sequential(Embeddings(dim_model, src_vocab_size), c(position))
    model = EncoderDecoder(
        Encoder(EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers),
        Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers),
        src_embed=embedding,
        tgt_embed=embedding,
        generator=Generator(dim_model, tgt_vocab_size),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_epoch(epoch, data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # out shape: batch_size, seq_len-1, dim_model
        loss = loss_compute(out, batch.trg, batch.trg_y, batch.ntokens)
        total_loss += loss.detach().numpy()
        total_tokens += batch.ntokens.numpy()
        tokens += batch.ntokens.numpy()
        avg_loss = loss.detach().numpy() / batch.ntokens.numpy()
        if i and i % 10 == 9:
            elapsed = time.time() - start
            print(
                "Epoch %d Step: %d Loss: %f Tokens %f Elapse: %f Tokens per Sec: %f"
                % (
                    epoch,
                    i,
                    avg_loss,
                    tokens,
                    elapsed,
                    tokens / elapsed,
                )
            )
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def data_gen(vocab_size, batch_size, num_batches, seq_len, start_symbol):
    """
    Generate random data for a src-tgt copy task.
    """
    for _ in range(num_batches):
        data = torch.zeros(batch_size, seq_len + 1)
        for i in range(batch_size):
            # symbol 0 is used for padding and symbol 1 is used for starting symbol. So we generate symbols between [2, vocab_size)
            data[i] = (torch.randperm(vocab_size-2)+2)[:seq_len+1]

        # the first column is starting symbol, used to kick off the decoder
        # the last seq_len columns are real sequence data in shape: batch_size, seq_len
        data[:, 0] = start_symbol
        data = data.long()
        # src shape: batch_size x seq_len
        # tgt shape: batch_size x (seq_len + 1)
        src = Variable(data[:, 1:], requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        # tgt will be further separated into trg (first seq_len columns, including the starting symbol)
        # and trg_y (last seq_len columns, not including the starting symbol) in Batch constructor
        # trg is used to generate target masks and embeddings, trg_y is used as labels
        yield Batch(src, tgt, pad=0)



# src vocab size equals to tgt vocab size
# vocab symbol includes padding symbol (0) and sequence starting symbol (1)
PADDING_SYMBOL = 0
START_SYMBOL = 1
VOCAB_SIZE = 10 + 1 + 1
SEQ_LEN = 7
EPOCH_NUM = 1
DIM_MODEL = 512
DIM_FEEDFORWARD = 256
NUM_STACKED_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 128
NUM_TRAIN_BATCHES = 1500
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
# model_opt = NoamOpt(
#     dim_model=DIM_MODEL,
#     factor=1,
#     warmup=400,
#     optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
# )
model_opt = torch.optim.Adam(model.parameters())

for epoch in range(EPOCH_NUM):
    model.train()
    run_epoch(
        epoch,
        data_gen(
            vocab_size=VOCAB_SIZE,
            batch_size=BATCH_SIZE,
            num_batches=NUM_TRAIN_BATCHES,
            seq_len=SEQ_LEN,
            start_symbol=START_SYMBOL,
        ),
        model,
        SimpleLossCompute(model.generator, criterion, model_opt),
    )
    model.eval()
    print(
        "eval loss:",
        run_epoch(
            epoch,
            data_gen(
                vocab_size=VOCAB_SIZE,
                batch_size=BATCH_SIZE,
                num_batches=NUM_EVAL_BATCHES,
                seq_len=SEQ_LEN,
                start_symbol=START_SYMBOL,
            ),
            model,
            SimpleLossCompute(model.generator, criterion, None),
        ),
    )


def greedy_decode(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(START_SYMBOL).type_as(src.data)
    for _ in range(max_len):
        out = model.decode(
            memory=memory,
            src_mask=src_mask,
            tgt=Variable(ys),
            tgt_mask=Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
        )
        prob = model.generator.greedy_decode(out[:, -1, :], Variable(ys))
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


model.eval()
src = Variable(torch.LongTensor([[3, 2, 4, 5, 6, 7, 8]]))
src_mask = Variable(torch.ones(1, 1, SEQ_LEN))
output_tgt = greedy_decode(model, src, src_mask, max_len=SEQ_LEN)
print(f"input seq: {src}")
print(f"output seq: {output_tgt}")
