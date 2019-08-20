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
    embedding,
    subsequent_mask,
    VocabEmbedder,
    UserEmbedder,
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
    num_stacked_layers,
    vocab_dim,
    user_dim,
    dim_model,
    dim_feedforward,
    num_heads,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_feedforward)
    position = PositionalEncoding(dim_model, max_len=2 * max_seq_len)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers),
        decoder=Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers),
        vocab_embedder=VocabEmbedder(vocab_dim, dim_model),
        user_embedder=UserEmbedder(user_dim, dim_model),
        generator=Generator(dim_model, vocab_size),
        positional_encoding=c(position),
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
            batch.src_features,
            batch.decoder_input_features,
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
            total_elapsed = time.time() - total_start_time
            print(
                "Epoch %d Step: %d Loss: %f Tokens per Sec: %d Elapse: %.3f, %.3f"
                % (
                    epoch,
                    i,
                    avg_loss,
                    tokens / elapsed,
                    elapsed,
                    total_elapsed,
                )
            )
            start = time.time()
            tokens = 0
        if avg_loss < 0.05:
            break

    return total_loss / total_tokens


def data_gen(vocab_size, user_dim, vocab_dim, batch_size, num_batches, max_seq_len, start_symbol, padding_symbol):
    """
    Generate random data for a src-tgt copy task.
    """
    for _ in range(num_batches):
        user_features = np.random.randn(batch_size, user_dim).astype(np.float32)

        # src_idx shape: batch_size x seq_len
        src_idx = np.full((batch_size, max_seq_len), padding_symbol).astype(np.long)
        # src_mask shape: batch_size x seq_len x seq_len
        src_mask = np.ones((batch_size, max_seq_len, max_seq_len)).astype(np.int8)
        # tgt_idx shape: batch_size x (seq_len + 1)
        tgt_idx = np.full((batch_size, max_seq_len + 1), padding_symbol).astype(np.long)
        # the first column is starting symbol, used to kick off the decoder
        # the last seq_len columns are real sequence data in shape: batch_size, seq_len
        tgt_idx[:, 0] = start_symbol

        # src_features shape: batch_size x seq_len x vocab_dim
        # tgt_features shape: batch_size x (seq_len + 1) x vocab_dim
        src_features = np.zeros((batch_size, max_seq_len, vocab_dim)).astype(np.float32)
        tgt_features = np.zeros((batch_size, max_seq_len + 1, vocab_dim)).astype(np.float32)

        for i in range(batch_size):
            vocab_features = np.random.randn(VOCAB_SIZE, VOCAB_DIM).astype(np.float32)
            vocab_features[padding_symbol] = 0.0
            vocab_features[start_symbol] = 0.0

            # random_seq_len = (i % max_seq_len) + 1
            random_seq_len = max_seq_len

            # symbol 0 is used for padding and symbol 1 is used for starting symbol.
            src_idx[i] = np.arange(VOCAB_SIZE)[2:]
            src_idx[i, random_seq_len:] = padding_symbol
            src_mask[i] = (src_idx[i] != padding_symbol).repeat(max_seq_len).reshape((max_seq_len, max_seq_len))

            # order = 1. if np.sum(user_features[i]) > 0 else -1.
            order = 1. if np.sum(vocab_features[3]) > 0 else -1.
            sort_idx = np.argsort(np.sum(vocab_features[2:2+random_seq_len], axis=1) * order) + 2
            tgt_idx[i, 1:1+random_seq_len] = sort_idx
            src_features[i] = embedding(src_idx[i], vocab_features)
            tgt_features[i] = embedding(tgt_idx[i], vocab_features)

        # tgt will be further separated into trg (first seq_len columns, including the starting symbol)
        # and trg_y (last seq_len columns, not including the starting symbol) in Batch constructor
        # trg is used to generate target masks and embeddings, trg_y is used as labels

        # we also need to attend user features, which will be placed at the beginning of each sequence
        # src_mask shape: batch_size x (seq_len + 1) x (seq_len + 1)
        src_mask = aug_user_features_mask(src_mask)

        yield Batch(
            user_features=torch.from_numpy(user_features),
            src_mask=torch.from_numpy(src_mask),
            trg_idx=torch.from_numpy(tgt_idx),
            src_features=torch.from_numpy(src_features),
            tgt_features=torch.from_numpy(tgt_features),
            padding_symbol=padding_symbol
        )



# src vocab size equals to tgt vocab size
# vocab symbol includes padding symbol (0) and sequence starting symbol (1)
PADDING_SYMBOL = 0
START_SYMBOL = 1
DIM_USER = 16
VOCAB_DIM = 16
MAX_SEQ_LEN = 5
VOCAB_SIZE = MAX_SEQ_LEN + 2
EPOCH_NUM = 1
DIM_MODEL = 64
DIM_FEEDFORWARD = 512
NUM_STACKED_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 128
NUM_TRAIN_BATCHES = 10000
NUM_EVAL_BATCHES = 5

criterion = LabelSmoothing(tgt_vocab_size=VOCAB_SIZE, padding_idx=PADDING_SYMBOL, starting_idx=START_SYMBOL, smoothing=0.0)
model = make_model(
    vocab_size=VOCAB_SIZE,
    vocab_dim=VOCAB_DIM,
    user_dim=DIM_USER,
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
model_opt = torch.optim.Adam(model.parameters(), amsgrad=True)
total_start_time = time.time()
for epoch in range(EPOCH_NUM):
    model.train()
    run_epoch(
        epoch,
        data_gen(
            vocab_size=VOCAB_SIZE,
            user_dim=DIM_USER,
            vocab_dim=VOCAB_DIM,
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
                user_dim=DIM_USER,
                vocab_dim=VOCAB_DIM,
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
total_elapse_time = time.time() - total_start_time
print(f"Total time: {total_elapse_time}")

def greedy_decode(
    model, user_features, vocab_features, src_features, src_mask, max_seq_len
):
    batch_size = src_features.shape[0]
    memory = model.encode(user_features, src_features, src_mask)
    decoder_input_idx = torch.ones(batch_size, 1).fill_(START_SYMBOL).type(torch.long)
    for l in range(max_seq_len):
        decoder_input_features = torch.tensor(
            [embedding(decoder_input_idx[i], vocab_features[i]) for i in range(batch_size)]
        )
        tgt_src_mask = src_mask[:, :l + 1, :]
        out = model.decode(
            memory=memory,
            tgt_src_mask=tgt_src_mask,
            decoder_input_features=decoder_input_features,
            decoder_input_mask=subsequent_mask(decoder_input_idx.size(1)).type(torch.long),
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
test_batch_size = 2
vocab_features1 = np.random.randn(VOCAB_SIZE, VOCAB_DIM).astype(np.float32)
vocab_features2 = np.random.randn(VOCAB_SIZE, VOCAB_DIM).astype(np.float32)
vocab_features = np.array([vocab_features1, vocab_features2])
vocab_features[:, :2, :] = 0.0
print("correct order1", np.argsort(np.sum(vocab_features1[2:], axis=1)) + 2)
print("correct order1", np.argsort(np.sum(vocab_features2[2:], axis=1)) + 2)
user_features = torch.randn(test_batch_size, DIM_USER)
user_features[0] = -0.1
user_features[1] = 0.1
src_embed = torch.from_numpy(vocab_features[:, 2:, :])
src_mask = torch.from_numpy(
    aug_user_features_mask(np.ones((test_batch_size, MAX_SEQ_LEN, MAX_SEQ_LEN)))
)
output_tgt = greedy_decode(model, user_features, vocab_features, src_embed, src_mask, max_seq_len=MAX_SEQ_LEN)
print(f"output seq:\n{output_tgt}")


