import copy
import math
import time

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reinforce_transformer_classes import (
    aug_user_features_mask,
    subsequent_mask,
    UserVocabEmbedder,
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
    dim_user=20,
    dim_model=512,
    dim_feedforward=512,
    num_heads=8,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_feedforward)
    position = PositionalEncoding(dim_model, max_len=2*max_seq_len)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers),
        decoder=Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers),
        user_vocab_embedder=UserVocabEmbedder(dim_user, vocab_dim, dim_model, position),
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
            batch.user_features,
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
        if avg_loss < 0.01:
            break
    return total_loss / total_tokens


def data_gen(dim_user, vocab_size, vocab_dim, vocab_embed, batch_size, num_batches, max_seq_len, start_symbol, padding_symbol):
    """
    Generate random data for a src-tgt copy task.
    """
    for _ in range(num_batches):
        user_features = torch.randn(batch_size, dim_user)

        src_idx = torch.zeros(batch_size, max_seq_len).fill_(padding_symbol).long()
        # the first column is starting symbol, used to kick off the decoder
        # the last seq_len columns are real sequence data in shape: batch_size, seq_len
        tgt_idx = torch.zeros(batch_size, max_seq_len + 1).fill_(padding_symbol).long()
        tgt_idx[:, 0] = start_symbol

        for i in range(batch_size):
            # random_seq_len = (i % max_seq_len) + 1
            random_seq_len = 7
            # symbol 0 is used for padding and symbol 1 is used for starting symbol.
            # So we generate symbols between [2, vocab_size)
            indices = (torch.randperm(vocab_size-2)+2)[:random_seq_len]
            src_idx[i, :random_seq_len] = indices
            if torch.sum(user_features[i]) > 0:
                tgt_idx[i, 1:random_seq_len+1] = indices.clone()
            else:
                tgt_idx[i, 1:random_seq_len+1] = torch.from_numpy(np.array(indices.numpy()[::-1]))

        # src_Idx shape: batch_size x seq_len
        # tgt_tgt shape: batch_size x (seq_len + 1)

        # tgt will be further separated into trg (first seq_len columns, including the starting symbol)
        # and trg_y (last seq_len columns, not including the starting symbol) in Batch constructor
        # trg is used to generate target masks and embeddings, trg_y is used as labels

        # src_embed shape: batch_size x seq_len x vocab_dim
        # tgt_embed shape: batch_size x (seq_len + 1) x vocab_dim
        src_embed = F.embedding(src_idx, vocab_embed)
        tgt_embed = F.embedding(tgt_idx, vocab_embed)
        yield Batch(
            user_features=user_features,
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
DIM_VOCAB = 16
DIM_USER = 20
MAX_SEQ_LEN = 7
EPOCH_NUM = 1
DIM_MODEL = 512
DIM_FEEDFORWARD = 256
NUM_STACKED_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 128
NUM_TRAIN_BATCHES = 10000
NUM_EVAL_BATCHES = 5

criterion = LabelSmoothing(tgt_vocab_size=VOCAB_SIZE, padding_idx=PADDING_SYMBOL, starting_idx=START_SYMBOL, smoothing=0.0)
model = make_model(
    vocab_size=VOCAB_SIZE,
    vocab_dim=DIM_VOCAB,
    max_seq_len=MAX_SEQ_LEN,
    num_stacked_layers=NUM_STACKED_LAYERS,
    dim_user=DIM_USER,
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
model_opt = torch.optim.Adam(model.parameters(), amsgrad=True)
vocab_features = torch.randn(VOCAB_SIZE, DIM_VOCAB)

for epoch in range(EPOCH_NUM):
    model.train()
    run_epoch(
        epoch,
        data_gen(
            dim_user=DIM_USER,
            vocab_size=VOCAB_SIZE,
            vocab_dim=DIM_VOCAB,
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
                dim_user=DIM_USER,
                vocab_size=VOCAB_SIZE,
                vocab_dim=DIM_VOCAB,
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


def greedy_decode(model, user_features, vocab_features, src_vocab_idx, src_vocab_embed, src_vocab_mask, max_seq_len):
    batch_size = src_vocab_idx.shape[0]
    src_mask = aug_user_features_mask(src_vocab_mask)

    memory = model.encode(user_features, src_vocab_idx, src_vocab_embed, src_mask)
    decoder_input_idx = torch.ones(batch_size, 1).fill_(START_SYMBOL).type_as(src_vocab_idx.data)
    for i in range(max_seq_len):
        tgt_src_mask = src_mask[:, :i+1, :]
        out = model.decode(
            memory=memory,
            tgt_src_mask=tgt_src_mask,
            decoder_input_idx=decoder_input_idx,
            decoder_input_embed=F.embedding(decoder_input_idx, vocab_features),
            decoder_input_mask=subsequent_mask(decoder_input_idx.size(1)).type_as(src_vocab_idx.data),
        )
        prob = model.generator.greedy_decode(out[:, -1, :], decoder_input_idx)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.clone().detach().reshape(batch_size, 1)
        decoder_input_idx = torch.cat(
            [decoder_input_idx, next_word],
            dim=1
        )
    return decoder_input_idx


model.eval()
user_features = torch.randn(2, DIM_USER)
user_features[0] = -0.1
user_features[1] = 0.1
src_vocab_idx = torch.LongTensor([
    [3, 2, 4, 5, 6, 7, 8],
    [8, 10, 9, 5, 11, PADDING_SYMBOL, PADDING_SYMBOL],
])
src_vocab_embed = F.embedding(src_vocab_idx, vocab_features)
src_vocab_mask = torch.ones(2, MAX_SEQ_LEN, MAX_SEQ_LEN)
output_tgt = greedy_decode(
    model, user_features, vocab_features, src_vocab_idx, src_vocab_embed, src_vocab_mask, max_seq_len=MAX_SEQ_LEN
)
print(f"input seq:\n{src_vocab_idx}")
print(f"output seq:\n{output_tgt}")
