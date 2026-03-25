"""
train.py
========
Trains all three models and saves checkpoints + loss curves.

Usage:
    python train.py
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import NamesDataset, build_vocab, load_names, PAD_TOKEN
from models  import VanillaRNN, BidirectionalLSTM, RNNWithAttention

# ── Reproducibility ───────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────
DATA_PATH    = 'TrainingNames.txt'
CHECKPOINTS  = 'checkpoints'
PLOTS_DIR    = 'plots'
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

EPOCHS       = 30
BATCH_SIZE   = 32
MAX_LEN      = 25
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CFGS = {
    'VanillaRNN': {
        'cls':         VanillaRNN,
        'embed_dim':   32,
        'hidden_size': 128,
        'n_layers':    1,
        'dropout':     0.3,
        'lr':          0.003,
    },
    'BLSTM': {
        'cls':         BidirectionalLSTM,
        'embed_dim':   32,
        'hidden_size': 128,
        'n_layers':    2,
        'dropout':     0.3,
        'lr':          0.002,
    },
    'RNNAttention': {
        'cls':         RNNWithAttention,
        'embed_dim':   32,
        'hidden_size': 128,
        'n_layers':    1,
        'dropout':     0.3,
        'lr':          0.003,
    },
}


def train_epoch(model, loader, optimizer, criterion, device, model_name):
    model.train()
    total_loss = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        if model_name == 'RNNAttention':
            logits, _, _ = model(inp)
        else:
            logits, _ = model(inp)
        # logits: (B, T, V)  tgt: (B, T)
        B, T, V = logits.shape
        loss = criterion(logits.view(B*T, V), tgt.view(B*T))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device, model_name):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            if model_name == 'RNNAttention':
                logits, _, _ = model(inp)
            else:
                logits, _ = model(inp)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), tgt.view(B*T))
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_losses(train_losses, val_losses, model_name, plots_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss', color='steelblue')
    plt.plot(val_losses,   label='Val Loss',   color='tomato', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{model_name} — Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_loss.png'), dpi=150)
    plt.close()


def main():
    # ── Data ──────────────────────────────────────────────
    names = load_names(DATA_PATH)
    print(f"Loaded {len(names)} names.")

    vocab, idx2char = build_vocab(names)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    dataset  = NamesDataset(names, vocab, max_len=MAX_LEN)
    n_val    = max(1, int(0.1 * len(dataset)))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Save vocab
    with open(os.path.join(CHECKPOINTS, 'vocab.json'), 'w') as f:
        json.dump({'vocab': vocab, 'idx2char': {str(k): v for k, v in idx2char.items()}}, f)

    results = {}

    for name, cfg in MODEL_CFGS.items():
        print(f"\n{'='*55}")
        print(f"  Training {name}")
        print(f"{'='*55}")

        cls = cfg['cls']
        model = cls(
            vocab_size  = vocab_size,
            embed_dim   = cfg['embed_dim'],
            hidden_size = cfg['hidden_size'],
            n_layers    = cfg['n_layers'],
            dropout     = cfg['dropout'],
        ).to(DEVICE)

        n_params = model.count_params()
        print(f"  Trainable parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore PAD

        train_losses, val_losses = [], []
        best_val = float('inf')

        for epoch in range(1, EPOCHS + 1):
            tr_loss  = train_epoch(model, train_loader, optimizer, criterion, DEVICE, name)
            val_loss = eval_epoch(model, val_loader, criterion, DEVICE, name)
            scheduler.step()
            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(CHECKPOINTS, f'{name}_best.pt'))

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                      f"train={tr_loss:.4f}  val={val_loss:.4f}")

        plot_losses(train_losses, val_losses, name, PLOTS_DIR)
        print(f"  Best val loss: {best_val:.4f}")

        results[name] = {
            'n_params':     n_params,
            'best_val':     best_val,
            'train_losses': train_losses,
            'val_losses':   val_losses,
            'cfg': {k: str(v) for k, v in cfg.items() if k != 'cls'},
        }

    with open(os.path.join(CHECKPOINTS, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Training complete.  Checkpoints saved to:", CHECKPOINTS)
    return results


if __name__ == '__main__':
    main()
