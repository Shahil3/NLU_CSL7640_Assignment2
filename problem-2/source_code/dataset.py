"""
dataset.py
==========
Vocabulary building and PyTorch Dataset for character-level name generation.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


# Special tokens
PAD_TOKEN   = '<PAD>'   # index 0
START_TOKEN = '<SOS>'   # index 1
END_TOKEN   = '<EOS>'   # index 2


def build_vocab(names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build character-level vocabulary from a list of names."""
    chars = sorted(set(''.join(names).lower()))
    vocab = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    idx2char = {v: k for k, v in vocab.items()}
    return vocab, idx2char


def encode_name(name: str, vocab: Dict[str, int]) -> List[int]:
    """Encode a name as a list of character indices (lowercase)."""
    return [vocab[START_TOKEN]] + [vocab[c] for c in name.lower() if c in vocab] + [vocab[END_TOKEN]]


class NamesDataset(Dataset):
    """
    Each sample is (input_seq, target_seq) where:
      input_seq  = [SOS, c1, c2, ..., cn]
      target_seq = [c1,  c2, ..., cn, EOS]
    Both are padded to `max_len`.
    """

    def __init__(self, names: List[str], vocab: Dict[str, int], max_len: int = 30):
        self.vocab   = vocab
        self.max_len = max_len
        self.samples = []
        for name in names:
            encoded = encode_name(name, vocab)
            inp    = encoded[:-1]          # SOS ... last_char
            target = encoded[1:]           # first_char ... EOS
            self.samples.append((inp, target))

    def _pad(self, seq: List[int]) -> List[int]:
        seq = seq[:self.max_len]
        return seq + [0] * (self.max_len - len(seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return (torch.tensor(self._pad(inp), dtype=torch.long),
                torch.tensor(self._pad(tgt), dtype=torch.long))


def load_names(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names
