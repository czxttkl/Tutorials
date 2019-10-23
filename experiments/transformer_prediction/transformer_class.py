import torch.nn as nn
import copy
import torch
import math
import torch.nn.functional as F
import time


# vocab symbol includes padding symbol (0) and sequence starting symbol (1)
PADDING_SYMBOL = 0
START_SYMBOL = 1


def clones(module, N):
    """
    Produce N identical layers.

    :param module: nn.Module class
    :param N: number of copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask, d_k):
    """ Scaled Dot Product Attention """
    # mask shape: batch_size x 1 x seq_len x seq_len

    # scores shape: batch_size x num_heads x seq_len x seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn shape: batch_size x num_heads x seq_len x seq_len
    p_attn = F.softmax(scores, dim=-1)
    # attn shape: batch_size x num_heads x seq_len x d_k
    attn = torch.matmul(p_attn, value)
    return attn, p_attn


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


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, dim_model):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class Encoder(nn.Module):
    "Core encoder is a stack of num_layers layers"

    def __init__(self, layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.dim_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """ Encoder is made up of self-attn and feed forward """

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
    """ Generic num_layers layer decoder with masking."""

    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, tgt_src_mask, tgt_tgt_mask):
        # each layer is one DecoderLayer
        for layer in self.layers:
            x = layer(x, memory, tgt_src_mask, tgt_tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """ Decoder is made of self-attn, src-attn, and feed forward """

    def __init__(self, size, self_attn, src_attn, feed_forward):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 3)

    def forward(self, x, m, tgt_src_mask, tgt_tgt_mask):
        # x is target embedding or the output of previous decoder layer
        # x shape: batch_size, seq_len, dim_model
        # m is the output of the last encoder layer
        # m shape: batch_size, seq_len, dim_model
        # tgt_src_mask shape: batch_size, seq_len, seq_len + 1
        # tgt_tgt_mask shape: batch_size, seq_len, seq_len
        def self_attn_layer_tgt(x):
            return self.self_attn(query=x, key=x, value=x, mask=tgt_tgt_mask)

        def self_attn_layer_src(x):
            return self.self_attn(query=x, key=m, value=m, mask=tgt_src_mask)

        x = self.sublayer[0](x, self_attn_layer_tgt)
        x = self.sublayer[1](x, self_attn_layer_src)
        # return shape: batch_size, seq_len, dim_model
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim_model):
        """ Take in model size and number of heads """
        super(MultiHeadedAttention, self).__init__()
        assert dim_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = dim_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all num_heads heads.
            # mask shape: batch_size, 1, seq_len, seq_len
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from dim_model => num_heads x d_k
        # self.linear[0, 1, 2] is query weight matrix, key weight matrix, and
        # value weight matrix, respectively.
        # l(x) represents the transformed query matrix, key matrix and value matrix
        # l(x) has shape (batch_size, seq_len, dim_model). You can think l(x) as
        # the matrices from a one-head attention; or you can think
        # l(x).view(...).transpose(...) as the matrices of num_heads attentions,
        # each attention has d_k dimension.
        query, key, value = [
            l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: batch_size, num_heads, seq_len, d_k
        x, _ = attention(query, key, value, mask, self.d_k)

        # 3) "Concat" using a view and apply a final linear.
        # each attention's output is d_k dimension. Concat num_heads attention's outputs
        # x shape: batch_size, seq_len, dim_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_feedforward):
        super(PositionwiseFeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_model, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, dim_model),
        )

    def forward(self, x):
        return self.net(x)


class Embedder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Embedder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        # x: raw input features. Shape: batch_size, seq_len, dim_in
        output = self.linear(x) * math.sqrt(self.dim_out)
        # output shape: batch_size, seq_len, dim_out
        return output


class PositionalEncoding(nn.Module):
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


class Seq2SlateRewardModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        num_stacked_layers: int,
        num_heads: int,
        dim_model: int,
        dim_feedforward: int,
        slate_seq_len: int,
    ):
        """
        A reward network that predicts slate reward.

        It uses a transformer-based encoder to encode the items shown in the slate.
        The slate reward is predicted by attending all encoder steps' outputs.

        For convenience, Seq2SlateRewardModel and Seq2SlateTransformerModel share
        the same parameter notations. Therefore, the reward model's encoder is
        actually applied on target sequences (i.e., slates) referred in
        Seq2SlateTransformerModel.

        Note that max_src_seq_len is the
        """
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.num_stacked_layers = num_stacked_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.slate_seq_len = slate_seq_len

        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, dim_model)
        ff = PositionwiseFeedForward(dim_model, dim_feedforward)
        self.encoder = Encoder(
            EncoderLayer(dim_model, c(attn), c(ff)), num_stacked_layers
        )
        self.decoder = Decoder(
            DecoderLayer(dim_model, c(attn), c(attn), c(ff)), num_stacked_layers
        )
        self.candidate_embedder = Embedder(candidate_dim, dim_model // 2)
        self.state_embedder = Embedder(state_dim, dim_model // 2)
        self.positional_encoding = PositionalEncoding(
            dim_model, max_len=2 * self.slate_seq_len
        )
        self.decoder_embedder = nn.Linear(candidate_dim, dim_model)
        self.proj = nn.Linear(dim_model, 1)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _convert_seq2slate_to_reward_model_format(self, input: Batch):
        """
        Reward model applies the encoder on tgt_seq of seq2slate data.
        So we need to re-assemble data: e.g., tgt_seq now becomes src_seq,
        masks should change correspondingly, etc.
        """
        device = input.src_features.device
        batch_size, _, candidate_dim = input.src_features.shape
        _, slate_seq_len, _ = input.tgt_features.shape
        state = input.user_features
        # use
        src_seq = input.src_features[
            torch.arange(batch_size).repeat_interleave(slate_seq_len),
            input.tgt_out_idx.flatten()-2
        ].reshape(batch_size, slate_seq_len, candidate_dim)
        assert torch.all(src_seq[-1] == input.src_features[-1][input.tgt_out_idx[-1]-2])
        src_src_mask = torch.ones(batch_size, slate_seq_len, slate_seq_len).to(device)
        tgt_seq = torch.ones(batch_size, 1, candidate_dim).to(device)
        tgt_tgt_mask = torch.ones(batch_size, 1, 1).to(device)
        return state, src_seq, src_src_mask, tgt_seq, tgt_tgt_mask

    def forward(self, input: Batch):
        """ Encode tgt sequences and predict the slate reward. """
        state, src_seq, src_src_mask, tgt_seq, tgt_tgt_mask = \
            self._convert_seq2slate_to_reward_model_format(input)

        # tgt_src_mask shape: batch_size, 1, src_seq_len
        tgt_src_mask = src_src_mask[:, :1, :]

        memory = self.encode(state, src_seq, src_src_mask)
        # out shape: batch_size, 1, dim_model
        out = self.decode(
            memory=memory,
            state=state,
            tgt_src_mask=tgt_src_mask,
            tgt_seq=tgt_seq,
            tgt_tgt_mask=tgt_tgt_mask,
        ).squeeze(1)

        pred_reward = self.proj(out)
        return pred_reward

    def encode(self, state, src_seq, src_mask):
        # state: batch_size, state_dim
        # src_seq: batch_size, slate_seq_len, dim_candidate
        # src_src_mask shape: batch_size, slate_seq_len, slate_seq_len
        batch_size = src_seq.shape[0]

        # candidate_embed: batch_size, slate_seq_len, dim_model/2
        candidate_embed = self.candidate_embedder(src_seq)
        # state_embed: batch_size, dim_model/2
        state_embed = self.state_embedder(state)
        # transform state_embed into shape: batch_size, slate_seq_len, dim_model/2
        state_embed = state_embed.repeat(1, self.slate_seq_len).reshape(
            batch_size, self.slate_seq_len, -1
        )

        # Input at each encoder step is actually concatenation of state_embed
        # and candidate embed. state_embed is replicated at each encoding step.
        # src_embed shape: batch_size, slate_seq_len, dim_model
        src_embed = self.positional_encoding(
            torch.cat((state_embed, candidate_embed), dim=-1), self.slate_seq_len
        )

        # encoder_output shape: batch_size, slate_seq_len, dim_model
        return self.encoder(src_embed, src_mask)

    def decode(self, memory, state, tgt_src_mask, tgt_seq, tgt_tgt_mask):
        """
        One step decoder. The decoder's output will be used as the input to
        the last layer for predicting slate reward
        """
        # tgt_embed shape: batch_size, 1, dim_model
        tgt_embed = self.decoder_embedder(tgt_seq)
        # shape: batch_size, 1, dim_model
        return self.decoder(tgt_embed, memory, tgt_src_mask, tgt_tgt_mask)


class RewardNetTrainer:
    def __init__(
        self, reward_net: nn.Module, minibatch_size: int, use_gpu: bool = False
    ) -> None:
        self.reward_net = reward_net
        self.use_gpu = use_gpu
        self.minibatch_size = minibatch_size
        self.minibatch = 0
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.opt = torch.optim.Adam(self.reward_net.parameters(), lr=1e-3)

    def train(self, training_input: Batch):
        t1 = time.time()
        target_reward = training_input.rewards.squeeze()

        predicted_reward = self.reward_net(training_input).squeeze()
        mse_loss = self.loss_fn(predicted_reward, target_reward)

        mse_loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        mse_loss = mse_loss.detach().cpu().numpy()

        self.minibatch += 1
        t2 = time.time()
        print("{} batch: mse_loss={}, time={}".format(self.minibatch, mse_loss, t2 - t1))

        return mse_loss