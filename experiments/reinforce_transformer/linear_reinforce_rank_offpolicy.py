import copy
import math
import time

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_reinforce_rank_onpolicy import decode
from common import START_SYMBOL, PADDING_SYMBOL
from reinforce_transformer_classes import (
    eval_function_high_reward_prob,
    eval_function_corr,
    reward_function_f1,
    reward_function_pairwise,
    embedding,
    subsequent_mask,
    ReinforceLossCompute,
    BaselineNN,
    VocabEmbedder,
    UserEmbedder,
    PositionalEncoding,
    PositionwiseFeedForward,
    MultiHeadedAttention,
    EncoderDecoder,
    EncoderLayer,
    Encoder,
    Generator,
    DecoderLayer,
    Decoder,
    Batch
)


def make_baseline(
    baseline_dim_model,
    user_dim,
    num_layers,
    device,
):
    return BaselineNN(baseline_dim_model, user_dim, num_layers).to(device)


def make_model(
    vocab_size,
    max_seq_len,
    num_stacked_layers,
    vocab_dim,
    user_dim,
    dim_model,
    dim_feedforward,
    num_heads,
    device,
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
    ).to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_epoch(epoch, data_iter, model, baseline, log_prob_compute, loss_compute, eval_function):
    "Standard Training and Logging Function"
    start = time.time()
    total_rl_loss = 0.
    total_baseline_loss = 0.
    total_eval_res = []
    tmp_tokens = 0

    for i, batch in enumerate(data_iter):
        # out shape: batch_size, seq_len, dim_model
        out = model.forward(
            batch.user_features,
            batch.src_features,
            batch.tgt_features,
            batch.src_src_mask,
            batch.tgt_tgt_mask
        )
        # log_probs shape: batch_size
        log_probs = log_prob_compute(out, batch.tgt_idx, batch.label_idx)
        rl_loss, baseline_loss = loss_compute(log_probs, batch.rewards, baseline, batch.user_features)
        eval_res = eval_function(log_probs.cpu().detach().numpy(), batch.rewards.cpu().detach().numpy())

        ntokens = batch.ntokens.cpu().numpy()
        total_eval_res.append(eval_res)
        total_rl_loss += rl_loss
        total_baseline_loss += baseline_loss
        tmp_tokens += ntokens
        avg_rl_loss = rl_loss
        avg_baseline_loss = baseline_loss

        if i and i % 10 == 9:
            elapsed = time.time() - start
            total_elapsed = time.time() - total_start_time
            print(
                "Epoch %d, Step: %d, RL Loss: %f, Bsl Loss: %f, Eval Res: %f, Tokens per Sec: %d, Elapse: %.3f, %.3f"
                % (
                    epoch,
                    i,
                    avg_rl_loss,
                    avg_baseline_loss,
                    eval_res,
                    tmp_tokens / elapsed,
                    elapsed,
                    total_elapsed,
                )
            )
            start = time.time()
            tmp_tokens = 0

        # off-policy learning is not stable. So once eval_res is below -0.1, it is a good sign of
        # successful learning
        if eval_res > -0.1:
            break

    return total_rl_loss / (i + 1), total_baseline_loss / (i + 1), np.mean(total_eval_res)


def data_gen(
    user_dim, vocab_dim, batch_size, num_batches, max_seq_len, tgt_seq_len, start_symbol, padding_symbol, reward_function, device
):
    """
    Generate random data for a src-tgt copy task.
    """
    for _ in range(num_batches):
        user_features = np.random.randn(batch_size, user_dim).astype(np.float32)
        vocab_features = np.random.randn(batch_size, VOCAB_SIZE, VOCAB_DIM).astype(np.float32)
        vocab_features[:, padding_symbol] = 0.0
        vocab_features[:, start_symbol] = 0.0
        # the last dim is the sum of all other dims
        vocab_features[:, :, -1] = np.sum(vocab_features[:, :, :-1], axis=-1)

        # rewards shape: batch_size
        rewards = np.zeros(batch_size).astype(np.float32)
        # truth_idx shape: batch_size
        truth_idx = np.zeros((batch_size, tgt_seq_len)).astype(np.long)
        # src_idx shape: batch_size x seq_len
        src_idx = np.full((batch_size, max_seq_len), padding_symbol).astype(np.long)
        # src_src_mask shape: batch_size x seq_len x seq_len
        src_src_mask = np.ones((batch_size, max_seq_len, max_seq_len)).astype(np.int8)
        # tgt_idx shape: batch_size x (seq_len + 1)
        tgt_idx_with_start_sym = np.full((batch_size, tgt_seq_len + 1), padding_symbol).astype(np.long)
        # the first column is starting symbol, used to kick off the decoder
        # the last seq_len columns are real sequence data in shape: batch_size, seq_len
        tgt_idx_with_start_sym[:, 0] = start_symbol

        # src_features shape: batch_size x seq_len x vocab_dim
        # tgt_features shape: batch_size x (seq_len + 1) x vocab_dim
        src_features = np.zeros((batch_size, max_seq_len, vocab_dim)).astype(np.float32)
        tgt_features = np.zeros((batch_size, tgt_seq_len + 1, vocab_dim)).astype(np.float32)

        for i in range(batch_size):
            # random_seq_len = (i % max_seq_len) + 1
            random_seq_len = max_seq_len

            # symbol 0 is used for padding and symbol 1 is used for starting symbol.
            src_idx[i] = np.arange(VOCAB_SIZE)[2:]
            src_idx[i, random_seq_len:] = padding_symbol
            src_src_mask[i] = np.tile(src_idx[i] != padding_symbol, (max_seq_len, 1))
            src_features[i] = embedding(src_idx[i], vocab_features[i])

            order = 1. if np.sum(user_features[i]) > 0 else -1.
            sort_idx = np.argsort(np.sum(vocab_features[i, 2:2 + random_seq_len, :-1], axis=-1) * order) + 2
            truth_idx[i] = sort_idx[:tgt_seq_len]

            tgt_idx_with_start_sym[i, 1:1 + tgt_seq_len] = np.random.permutation(sort_idx)[:tgt_seq_len]
            tgt_features[i] = embedding(tgt_idx_with_start_sym[i], vocab_features[i])
            rewards[i] = reward_function(user_features[i], vocab_features[i], tgt_idx_with_start_sym[i, 1:], truth_idx[i])

        # tgt will be further separated into trg (first seq_len columns, including the starting symbol)
        # and trg_y (last seq_len columns, not including the starting symbol) in Batch constructor
        # trg is used to generate target masks and embeddings, trg_y is used as labels

        yield Batch(
            user_features=torch.from_numpy(user_features).to(device),
            src_src_mask=torch.from_numpy(src_src_mask).to(device),
            tgt_idx_with_start_sym=torch.from_numpy(tgt_idx_with_start_sym).to(device),
            truth_idx=torch.from_numpy(truth_idx).to(device),
            src_features=torch.from_numpy(src_features).to(device),
            tgt_features_with_start_sym=torch.from_numpy(tgt_features).to(device),
            rewards=torch.from_numpy(rewards).to(device),
            padding_symbol=padding_symbol,
        )


# vocab symbol includes padding symbol (0) and sequence starting symbol (1)
DIM_USER = 4
VOCAB_DIM = 4
MAX_SEQ_LEN = 3
TARGET_SEQ_LEN = 2
VOCAB_SIZE = MAX_SEQ_LEN + 2
EPOCH_NUM = 1
DIM_MODEL = 32
DIM_FEEDFORWARD = 512
NUM_STACKED_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 1280
NUM_TRAIN_BATCHES = 100000

BASELINE_DIM_MODEL = DIM_FEEDFORWARD
BASELINE_LAYERS = 2

# reward_function = reward_function_f1
reward_function = reward_function_pairwise
# eval_function = eval_function_corr
eval_function = eval_function_high_reward_prob
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = make_model(
    vocab_size=VOCAB_SIZE,
    vocab_dim=VOCAB_DIM,
    user_dim=DIM_USER,
    max_seq_len=MAX_SEQ_LEN,
    num_stacked_layers=NUM_STACKED_LAYERS,
    dim_model=DIM_MODEL,
    dim_feedforward=DIM_FEEDFORWARD,
    num_heads=NUM_HEADS,
    device=device,
)
model.register_hook(lambda x: print("w", x))
baseline = make_baseline(
    baseline_dim_model=BASELINE_DIM_MODEL,
    user_dim=DIM_USER,
    num_layers=BASELINE_LAYERS,
    device=device,
)
# model_opt = NoamOpt(
#     dim_model=DIM_MODEL,
#     factor=1,
#     warmup=400,
#     optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
# )
# off-policy learning is sensitive to learning rate. lr=1e-3 performs worse
model_opt = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
baseline_opt = torch.optim.Adam(baseline.parameters(), lr=1e-4, amsgrad=True)

total_start_time = time.time()
for epoch in range(EPOCH_NUM):
    model.train()
    run_epoch(
        epoch,
        data_gen(
            user_dim=DIM_USER,
            vocab_dim=VOCAB_DIM,
            batch_size=BATCH_SIZE,
            num_batches=NUM_TRAIN_BATCHES,
            max_seq_len=MAX_SEQ_LEN,
            tgt_seq_len=TARGET_SEQ_LEN,
            start_symbol=START_SYMBOL,
            padding_symbol=PADDING_SYMBOL,
            reward_function=reward_function,
            device=device,
        ),
        model,
        baseline,
        LogProbCompute(model.generator),
        ReinforceLossCompute(on_policy=False, rl_opt=model_opt, baseline_opt=baseline_opt),
        eval_function,
    )

total_elapse_time = time.time() - total_start_time
print(f"Total time: {total_elapse_time}")


model.eval()
test_batch_size = 5
user_features = np.random.randn(test_batch_size, DIM_USER).astype(np.float32)
vocab_features = np.random.randn(test_batch_size, VOCAB_SIZE, VOCAB_DIM).astype(np.float32)
vocab_features[:, :2, :] = 0.0
vocab_features[:, :, -1] = np.sum(vocab_features[:, :, :-1], axis=-1)
for i in range(test_batch_size):
    order = 1. if np.sum(user_features[i]) > 0 else -1.
    print(f"{i}-th correct order ({order})", (np.argsort(np.sum(vocab_features[i, 2:] * order, axis=1)) + 2)[:TARGET_SEQ_LEN])

user_features = torch.from_numpy(user_features).to(device)
src_features = torch.from_numpy(vocab_features[:, 2:, :]).to(device)
src_mask = torch.from_numpy(
    np.ones((test_batch_size, MAX_SEQ_LEN, MAX_SEQ_LEN))
).to(device)
decoder_probs, output_tgt = decode(model, user_features, vocab_features, src_features, src_mask, TARGET_SEQ_LEN, greedy=True)
print(f"output seq:\n{output_tgt}")
print(f"decoder probs:\n{decoder_probs}")


