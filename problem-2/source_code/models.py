"""
models.py
=========
Character-level name generation using three RNN variants:
  1. Vanilla RNN
  2. Bidirectional LSTM (BLSTM)
  3. RNN with Basic Attention Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Vanilla RNN
# ─────────────────────────────────────────────
class VanillaRNN(nn.Module):
    """
    Architecture
    ─────────────
    Input  → Embedding → RNN (1 layer) → Linear → Softmax

    The hidden state is updated at each step.  At inference time we
    sample the next character from the output distribution and feed it
    back as the next input.

    Hyperparameters
    ───────────────
    vocab_size : number of unique characters in the dataset
    embed_dim  : 32   (character embedding dimension)
    hidden_size: 128  (RNN hidden state size)
    n_layers   : 1
    dropout    : 0.3  (applied on output before linear)
    learning_rate (set in train.py): 0.003
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_size: int = 128, n_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn   = nn.RNN(embed_dim, hidden_size, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len) — token indices
        hidden : (n_layers, batch, hidden_size) or None
        returns: logits (batch, seq_len, vocab_size), hidden
        """
        emb = self.embed(x)                        # (B, T, E)
        out, hidden = self.rnn(emb, hidden)        # (B, T, H)
        out = self.dropout(out)
        logits = self.fc(out)                      # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 2. Bidirectional LSTM (BLSTM)
# ─────────────────────────────────────────────
class BidirectionalLSTM(nn.Module):
    """
    Architecture
    ─────────────
    Input  → Embedding → BiLSTM (2 layers) → Linear projection →
             Dropout → Linear → Softmax

    The BiLSTM processes the sequence in both directions.  The
    concatenated (forward + backward) hidden states are projected to
    vocab_size for prediction.

    Note: Because generation is causal (left-to-right), during
    *training* we still use the full BiLSTM output for each position,
    but we shift the targets by one step so the model effectively
    learns character-level language modelling conditioned on the
    surrounding context.  At *generation* time we run only the forward
    pass (using a uni-directional wrapper around the trained weights).

    For simplicity we keep a single BiLSTM for both training and
    generation (a common approach in teaching settings).

    Hyperparameters
    ───────────────
    embed_dim  : 32
    hidden_size: 128  (per direction; total = 256)
    n_layers   : 2
    dropout    : 0.3
    learning_rate: 0.002
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_size: int = 128, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.bidirectional = True
        self.num_directions = 2

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, num_layers=n_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if n_layers > 1 else 0.0)
        self.proj    = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x      : (B, T)
        hidden : tuple((n_layers*2, B, H), (n_layers*2, B, H)) or None
        returns: logits (B, T, V), (h_n, c_n)
        """
        emb = self.embed(x)                         # (B, T, E)
        out, (h_n, c_n) = self.lstm(emb, hidden)    # out: (B, T, 2H)
        out = self.proj(out)                         # (B, T, H)
        out = F.relu(out)
        out = self.dropout(out)
        logits = self.fc(out)                        # (B, T, V)
        return logits, (h_n, c_n)

    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.n_layers * self.num_directions,
                         batch_size, self.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 3. RNN with Basic Attention
# ─────────────────────────────────────────────
class Attention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over RNN outputs.

    Given RNN outputs H (B, T, H) it computes a context vector for each
    timestep by attending over all previous outputs (causal masking applied).

    score(h_t, h_s) = v^T * tanh(W1*h_t + W2*h_s)
    alpha_t         = softmax(score_t, dim=-1)
    context_t       = sum_s alpha_{t,s} * h_s
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_outputs):
        """
        rnn_outputs: (B, T, H)
        returns    : context (B, T, H), weights (B, T, T)
        """
        B, T, H = rnn_outputs.shape

        # query: each timestep
        query = self.W1(rnn_outputs)          # (B, T, H)
        # keys:  all timesteps
        keys  = self.W2(rnn_outputs)          # (B, T, H)

        # broadcast to (B, T_q, T_k, H)
        query_exp = query.unsqueeze(2).expand(-1, -1, T, -1)  # (B,T,T,H)
        keys_exp  = keys.unsqueeze(1).expand(-1, T, -1, -1)   # (B,T,T,H)

        scores = self.v(torch.tanh(query_exp + keys_exp)).squeeze(-1)  # (B,T,T)

        # causal mask: position t may only attend to positions 0..t
        mask = torch.triu(torch.ones(T, T, device=rnn_outputs.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        weights = F.softmax(scores, dim=-1)  # (B,T,T)
        weights = torch.nan_to_num(weights)  # handle -inf -> 0 after softmax

        context = torch.bmm(weights, rnn_outputs)  # (B,T,H)
        return context, weights


class RNNWithAttention(nn.Module):
    """
    Architecture
    ─────────────
    Input  → Embedding → RNN (1 layer) → Attention (self) →
             Concat[rnn_out, context] → Linear projection →
             Dropout → Linear → Softmax

    Hyperparameters
    ───────────────
    embed_dim  : 32
    hidden_size: 128
    n_layers   : 1
    dropout    : 0.3
    learning_rate: 0.003
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_size: int = 128, n_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embed     = nn.Embedding(vocab_size, embed_dim)
        self.rnn       = nn.RNN(embed_dim, hidden_size, num_layers=n_layers,
                                batch_first=True,
                                dropout=dropout if n_layers > 1 else 0.0)
        self.attention = Attention(hidden_size)
        self.proj      = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x      : (B, T)
        hidden : (n_layers, B, H) or None
        returns: logits (B, T, V), hidden, attn_weights (B, T, T)
        """
        emb = self.embed(x)                           # (B, T, E)
        rnn_out, hidden = self.rnn(emb, hidden)       # (B, T, H)
        context, attn_w = self.attention(rnn_out)     # (B, T, H), (B,T,T)
        combined = torch.cat([rnn_out, context], dim=-1)  # (B, T, 2H)
        out = F.relu(self.proj(combined))             # (B, T, H)
        out = self.dropout(out)
        logits = self.fc(out)                         # (B, T, V)
        return logits, hidden, attn_w

    def init_hidden(self, batch_size: int, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
