import copy
import math
import time

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reinforce_transformer_classes import (
    clones,
    subsequent_mask,
    VocabEmbedder,
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
    vocab_size,
    max_seq_len,
    num_stacked_layers=6,
    vocab_dim=16,
    dim_model=512,
    dim_feedforward=512,
    num_heads=8,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_feedforward)
    position = PositionalEncoding(dim_model, max_len=max_seq_len)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers),
        decoder=Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers),
        vocab_embedder=VocabEmbedder(vocab_dim, dim_model, position),
        generator=Generator(dim_model, vocab_size),
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
        out = model.forward(
            batch.src_idx,
            batch.decoder_input_idx,
            batch.src_embed,
            batch.decoder_input_embed,
            batch.src_mask,
            batch.trg_mask
        )
        # out shape: batch_size, seq_len, dim_model
        loss = loss_compute(out, batch.decoder_input_idx, batch.target_label_idx, batch.ntokens)
        total_loss += loss.detach().numpy()
        total_tokens += batch.ntokens.numpy()
        tokens += batch.ntokens.numpy()
        if i and i % 10 == 9:
            elapsed = time.time() - start
            print(
                "Epoch %d Step: %d Loss: %f Tokens per Sec: %f"
                % (
                    epoch,
                    i,
                    loss.detach().numpy() / batch.ntokens.numpy(),
                    tokens / elapsed,
                )
            )
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def data_gen(vocab_size, vocab_dim, vocab_embed, batch_size, num_batches, max_seq_len, start_symbol, padding_symbol):
    """
    Generate random data for a src-tgt copy task.
    """
    for _ in range(num_batches):
        indices = torch.zeros(batch_size, max_seq_len + 1).fill_(padding_symbol)
        for i in range(batch_size):
            random_seq_len = (i % max_seq_len) + 1
            # random_seq_len = 7
            # symbol 0 is used for padding and symbol 1 is used for starting symbol.
            # So we generate symbols between [2, vocab_size)
            indices[i, 1:random_seq_len+1] = (torch.randperm(vocab_size-2)+2)[:random_seq_len]

        # the first column is starting symbol, used to kick off the decoder
        # the last seq_len columns are real sequence data in shape: batch_size, seq_len
        indices[:, 0] = start_symbol
        indices = indices.long()
        # src shape: batch_size x seq_len
        # tgt shape: batch_size x (seq_len + 1)
        src_idx = indices[:, 1:]
        tgt_idx = indices
        # tgt will be further separated into trg (first seq_len columns, including the starting symbol)
        # and trg_y (last seq_len columns, not including the starting symbol) in Batch constructor
        # trg is used to generate target masks and embeddings, trg_y is used as labels

        # src_embed shape: batch_size x seq_len x vocab_dim
        # tgt_embed shape: batch_size x (seq_len + 1) x vocab_dim
        src_embed = F.embedding(src_idx, vocab_embed)
        tgt_embed = F.embedding(tgt_idx, vocab_embed)
        yield Batch(
            src_idx=src_idx,
            trg_idx=tgt_idx,
            src_embed=src_embed,
            tgt_embed=tgt_embed,
            padding_symbol=padding_symbol
        )



# src vocab size equals to tgt vocab size
# vocab symbol includes padding symbol (0) and sequence starting symbol (1)
PADDING_SYMBOL = 0
START_SYMBOL = 1
VOCAB_SIZE = 10 + 1 + 1
VOCAB_DIM = 16
MAX_SEQ_LEN = 7
EPOCH_NUM = 1
DIM_MODEL = 512
DIM_FEEDFORWARD = 256
NUM_STACKED_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 128
NUM_TRAIN_BATCHES = 220
NUM_EVAL_BATCHES = 5

criterion = LabelSmoothing(tgt_vocab_size=VOCAB_SIZE, padding_idx=PADDING_SYMBOL, starting_idx=START_SYMBOL, smoothing=0.0)
model = make_model(
    vocab_size=VOCAB_SIZE,
    vocab_dim=VOCAB_DIM,
    max_seq_len=MAX_SEQ_LEN,
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
vocab_features = torch.randn(VOCAB_SIZE, VOCAB_DIM)

for epoch in range(EPOCH_NUM):
    model.train()
    run_epoch(
        epoch,
        data_gen(
            vocab_size=VOCAB_SIZE,
            vocab_dim=VOCAB_DIM,
            vocab_embed=vocab_features,
            batch_size=BATCH_SIZE,
            num_batches=NUM_TRAIN_BATCHES,
            max_seq_len=MAX_SEQ_LEN,
            start_symbol=START_SYMBOL,
            padding_symbol=PADDING_SYMBOL,
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
                vocab_dim=VOCAB_DIM,
                vocab_embed=vocab_features,
                batch_size=BATCH_SIZE,
                num_batches=NUM_EVAL_BATCHES,
                max_seq_len=MAX_SEQ_LEN,
                start_symbol=START_SYMBOL,
                padding_symbol=PADDING_SYMBOL,
            ),
            model,
            SimpleLossCompute(model.generator, criterion, None),
        ),
    )


def greedy_decode(model, vocab_embed, src_idx, src_embed, src_mask, max_seq_len):
    memory = model.encode(src_idx, src_embed, src_mask)
    decoder_input_idx = torch.ones(1, 1).fill_(START_SYMBOL).type_as(src_idx.data)
    for _ in range(max_seq_len):
        out = model.decode(
            memory=memory,
            src_mask=src_mask,
            decoder_input_idx=decoder_input_idx,
            decoder_input_embed=F.embedding(decoder_input_idx, vocab_embed),
            decoder_input_mask=subsequent_mask(decoder_input_idx.size(1)).type_as(src_idx.data),
        )
        prob = model.generator.greedy_decode(out[:, -1, :], decoder_input_idx)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        decoder_input_idx = torch.cat([decoder_input_idx, torch.ones(1, 1).type_as(src_idx.data).fill_(next_word)], dim=1)
    return decoder_input_idx


model.eval()
src_idx = torch.LongTensor([[3, 2, 4, 5, 6, 7, 8]])
src_embed = F.embedding(src_idx, vocab_features)
src_mask = torch.ones(1, 1, MAX_SEQ_LEN)
output_tgt = greedy_decode(model, vocab_features, src_idx, src_embed, src_mask, max_seq_len=MAX_SEQ_LEN)
print(f"input seq: {src_idx}")
print(f"output seq: {output_tgt}")
