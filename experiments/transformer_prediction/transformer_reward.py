import numpy as np
import torch
from functools import reduce
from itertools import combinations
import time
import torch.nn.functional as F
from transformer_class import Seq2SlateRewardModel, RewardNetTrainer, Batch


def embedding(idx, table):
    """ numpy version of embedding look up """
    new_shape = (*idx.shape, -1)
    return table[idx.flatten()].reshape(new_shape)


def reward_function_pairwise(user_feature, vocab_feature, tgt_out_idx, true_tgt_out_idx):
    if len(tgt_out_idx) == 1:
        return float(tgt_out_idx[0] == true_tgt_out_idx[0])
    truth_pairs = set(combinations(true_tgt_out_idx, 2))
    tgt_pairs = set(combinations(tgt_out_idx, 2))
    return float(len(truth_pairs & tgt_pairs))


def data_gen(
    user_dim, vocab_dim, batch_size, num_batches, max_seq_len, tgt_seq_len, device
):
    """
    Generate random data for a src-tgt copy task.
    """
    for _ in range(num_batches):
        user_features = np.random.randn(batch_size, user_dim).astype(np.float32)
        vocab_features = np.random.randn(batch_size, VOCAB_SIZE, VOCAB_DIM).astype(np.float32)
        vocab_features[:, PADDING_SYMBOL] = 0.0
        vocab_features[:, START_SYMBOL] = 0.0
        # the last dim is the sum of all other dims
        vocab_features[:, :, -1] = np.sum(vocab_features[:, :, :-1], axis=-1)

        # tgt sequence probabilities, used for off-policy importance sampling correciton
        tgt_probs = np.zeros(batch_size).astype(np.float32)
        # rewards shape: batch_size
        rewards = np.zeros(batch_size).astype(np.float32)
        # truth_idx shape: batch_size
        truth_idx = np.zeros((batch_size, tgt_seq_len)).astype(np.long)
        # src_idx shape: batch_size x seq_len
        src_idx = np.full((batch_size, max_seq_len), PADDING_SYMBOL).astype(np.long)
        # src_src_mask shape: batch_size x seq_len x seq_len
        src_src_mask = np.ones((batch_size, max_seq_len, max_seq_len)).astype(np.int8)
        # tgt_idx_with_start_sym shape: batch_size x (seq_len + 1)
        tgt_idx_with_start_sym = np.full((batch_size, tgt_seq_len + 1), PADDING_SYMBOL).astype(np.long)
        # the first column is starting symbol, used to kick off the decoder
        # the last seq_len columns are real sequence data in shape: batch_size, seq_len
        tgt_idx_with_start_sym[:, 0] = START_SYMBOL

        # src_features shape: batch_size x seq_len x vocab_dim
        # tgt_features shape: batch_size x (seq_len + 1) x vocab_dim
        src_features = np.zeros((batch_size, max_seq_len, vocab_dim)).astype(np.float32)
        tgt_features = np.zeros((batch_size, tgt_seq_len + 1, vocab_dim)).astype(np.float32)

        for i in range(batch_size):
            # random_seq_len = (i % max_seq_len) + 1
            random_seq_len = max_seq_len

            # symbol 0 is used for padding and symbol 1 is used for starting symbol.
            src_idx[i] = np.arange(VOCAB_SIZE)[2:]
            src_idx[i, random_seq_len:] = PADDING_SYMBOL
            src_src_mask[i] = np.tile(src_idx[i] != PADDING_SYMBOL, (max_seq_len, 1))
            src_features[i] = embedding(src_idx[i], vocab_features[i])

            order = 1. if np.sum(user_features[i]) > 0 else -1.
            sort_idx = np.argsort(np.sum(vocab_features[i, 2:2+random_seq_len, :-1], axis=-1) * order) + 2
            truth_idx[i] = sort_idx[:tgt_seq_len]

            tgt_idx_with_start_sym[i, 1:1 + tgt_seq_len] = np.random.permutation(sort_idx)[:tgt_seq_len]
            tgt_features[i] = embedding(tgt_idx_with_start_sym[i], vocab_features[i])
            rewards[i] = reward_function_pairwise(user_features[i], vocab_features[i], tgt_idx_with_start_sym[i, 1:], truth_idx[i])
            tgt_probs[i] = 1. / reduce(lambda a, b: a*b, range(tgt_seq_len, 0, -1))

        # tgt will be further separated into trg (first seq_len columns, including the starting symbol)
        # and trg_y (last seq_len columns, not including the starting symbol) in Batch constructor
        # trg is used to generate target masks and embeddings, trg_y is used as labels

        yield Batch(
            user_features=torch.from_numpy(user_features).to(device),
            src_src_mask=torch.from_numpy(src_src_mask).to(device),
            tgt_idx_with_start_sym=torch.from_numpy(tgt_idx_with_start_sym).to(device),
            truth_idx=torch.from_numpy(truth_idx).to(device),
            src_features=torch.from_numpy(src_features).to(device),
            src_in_idx=torch.from_numpy(src_idx).to(device),
            tgt_features_with_start_sym=torch.from_numpy(tgt_features).to(device),
            rewards=torch.from_numpy(rewards).to(device),
            tgt_probs=torch.from_numpy(tgt_probs).to(device),
        )


def run_epoch(epoch, train_data_iter, eval_data_iter, model, trainer):
    "Standard Training and Logging Function"
    start = time.time()
    train_mses = []

    for i, batch in enumerate(train_data_iter):
        train_mse = trainer.train(batch)
        print(f"{epoch}-th epoch, {i}-th batch train_mse={train_mse}")

    batch = next(eval_data_iter)
    model.eval()
    pred_reward = model(batch).squeeze()
    target_reward = batch.rewards
    print(f"Eval pred_reward={pred_reward[:20]}, target_reward={target_reward[:20]}")
    print(f"Eval MSE={F.mse_loss(pred_reward, target_reward)}")


def main():
    model = Seq2SlateRewardModel(
        state_dim=DIM_USER,
        candidate_dim=VOCAB_DIM,
        num_stacked_layers=NUM_STACKED_LAYERS,
        num_heads=NUM_HEADS,
        dim_model=DIM_MODEL,
        dim_feedforward=DIM_FEEDFORWARD,
        slate_seq_len=TARGET_SEQ_LEN,
    )

    trainer = RewardNetTrainer(
        model, BATCH_SIZE, USE_GPU
    )

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
                device=device,
            ),
            data_gen(
                user_dim=DIM_USER,
                vocab_dim=VOCAB_DIM,
                batch_size=BATCH_SIZE,
                num_batches=NUM_TRAIN_BATCHES,
                max_seq_len=MAX_SEQ_LEN,
                tgt_seq_len=TARGET_SEQ_LEN,
                device=device,
            ),
            model,
            trainer,
        )

# vocab symbol includes padding symbol (0) and sequence starting symbol (1)
PADDING_SYMBOL = 0
START_SYMBOL = 1
DIM_USER = 4
VOCAB_DIM = 5
MAX_SEQ_LEN = 3
TARGET_SEQ_LEN = 3
VOCAB_SIZE = MAX_SEQ_LEN + 2
EPOCH_NUM = 1
DIM_MODEL = 32
DIM_FEEDFORWARD = 512
NUM_STACKED_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 1024
NUM_TRAIN_BATCHES = 1500
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

main()