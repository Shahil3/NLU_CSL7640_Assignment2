"""
generate.py
===========
Loads each trained model and generates names using temperature sampling.

Usage:
    python generate.py
"""

import os
import json
import torch
import torch.nn.functional as F
from dataset import build_vocab, load_names
from models  import VanillaRNN, BidirectionalLSTM, RNNWithAttention

CHECKPOINTS  = 'checkpoints'
GENERATED_DIR = 'generated_names'
os.makedirs(GENERATED_DIR, exist_ok=True)

DATA_PATH    = 'TrainingNames.txt'
N_GENERATE   = 200
MAX_LEN      = 25
TEMPERATURE  = 0.8         # sampling temperature
DEVICE       = torch.device('cpu')

MODEL_CFGS = {
    'VanillaRNN': {
        'cls':         VanillaRNN,
        'embed_dim':   32,
        'hidden_size': 128,
        'n_layers':    1,
        'dropout':     0.0,   # no dropout at inference
    },
    'BLSTM': {
        'cls':         BidirectionalLSTM,
        'embed_dim':   32,
        'hidden_size': 128,
        'n_layers':    2,
        'dropout':     0.0,
    },
    'RNNAttention': {
        'cls':         RNNWithAttention,
        'embed_dim':   32,
        'hidden_size': 128,
        'n_layers':    1,
        'dropout':     0.0,
    },
}


def load_vocab():
    with open(os.path.join(CHECKPOINTS, 'vocab.json')) as f:
        data = json.load(f)
    vocab    = data['vocab']
    idx2char = {int(k): v for k, v in data['idx2char'].items()}
    return vocab, idx2char


def generate_name_vanilla(model, vocab, idx2char, max_len=MAX_LEN, temp=TEMPERATURE):
    model.eval()
    SOS = vocab['<SOS>']
    EOS = vocab['<EOS>']
    with torch.no_grad():
        inp    = torch.tensor([[SOS]], device=DEVICE)
        hidden = model.init_hidden(1, DEVICE)
        name   = ''
        for _ in range(max_len):
            logits, hidden = model(inp, hidden)          # (1,1,V)
            probs  = F.softmax(logits[0, 0] / temp, dim=-1)
            idx    = torch.multinomial(probs, 1).item()
            if idx == EOS:
                break
            ch = idx2char.get(idx, '')
            if ch not in ('<PAD>', '<SOS>', '<EOS>'):
                name += ch
            inp = torch.tensor([[idx]], device=DEVICE)
    return name.capitalize()


def generate_name_blstm(model, vocab, idx2char, max_len=MAX_LEN, temp=TEMPERATURE):
    """
    BLSTM is bidirectional, so we can't truly do step-by-step hidden-state
    generation.  We use a greedy / sampling approach: start with SOS and
    iteratively extend by feeding the growing sequence as a batch of 1.
    """
    model.eval()
    SOS = vocab['<SOS>']
    EOS = vocab['<EOS>']
    with torch.no_grad():
        tokens = [SOS]
        for _ in range(max_len):
            inp    = torch.tensor([tokens], device=DEVICE)           # (1, t)
            logits, _ = model(inp)                                   # (1, t, V)
            last_logit = logits[0, -1]                               # (V,)
            probs  = F.softmax(last_logit / temp, dim=-1)
            idx    = torch.multinomial(probs, 1).item()
            if idx == EOS:
                break
            tokens.append(idx)
        name = ''.join(idx2char.get(i, '') for i in tokens[1:]
                       if idx2char.get(i, '') not in ('<PAD>', '<SOS>', '<EOS>'))
    return name.capitalize()


def generate_name_attention(model, vocab, idx2char, max_len=MAX_LEN, temp=TEMPERATURE):
    model.eval()
    SOS = vocab['<SOS>']
    EOS = vocab['<EOS>']
    with torch.no_grad():
        inp    = torch.tensor([[SOS]], device=DEVICE)
        hidden = model.init_hidden(1, DEVICE)
        name   = ''
        for _ in range(max_len):
            logits, hidden, _ = model(inp, hidden)   # (1,1,V)
            probs  = F.softmax(logits[0, 0] / temp, dim=-1)
            idx    = torch.multinomial(probs, 1).item()
            if idx == EOS:
                break
            ch = idx2char.get(idx, '')
            if ch not in ('<PAD>', '<SOS>', '<EOS>'):
                name += ch
            inp = torch.tensor([[idx]], device=DEVICE)
    return name.capitalize()


GENERATE_FN = {
    'VanillaRNN':   generate_name_vanilla,
    'BLSTM':        generate_name_blstm,
    'RNNAttention': generate_name_attention,
}


def main():
    vocab, idx2char = load_vocab()
    training_names  = set(n.lower() for n in load_names(DATA_PATH))
    all_results     = {}

    for model_name, cfg in MODEL_CFGS.items():
        print(f"\nGenerating names with {model_name} …")
        cls = cfg['cls']
        model = cls(
            vocab_size  = len(vocab),
            embed_dim   = cfg['embed_dim'],
            hidden_size = cfg['hidden_size'],
            n_layers    = cfg['n_layers'],
            dropout     = cfg['dropout'],
        ).to(DEVICE)

        ckpt = os.path.join(CHECKPOINTS, f'{model_name}_best.pt')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        gen_fn = GENERATE_FN[model_name]
        names  = []
        attempts = 0
        while len(names) < N_GENERATE and attempts < N_GENERATE * 10:
            n = gen_fn(model, vocab, idx2char)
            attempts += 1
            if len(n) >= 2:      # discard empty or single-char outputs
                names.append(n)

        # Save generated names
        out_path = os.path.join(GENERATED_DIR, f'{model_name}_generated.txt')
        with open(out_path, 'w') as f:
            for n in names:
                f.write(n + '\n')

        all_results[model_name] = names
        print(f"  Generated {len(names)} names  →  {out_path}")

    return all_results


if __name__ == '__main__':
    main()
