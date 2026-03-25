"""
evaluate.py
===========
Computes quantitative metrics (Novelty Rate, Diversity) and performs
qualitative analysis for each model's generated names.

Usage:
    python evaluate.py
"""

import os
import json
import re
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH     = 'TrainingNames.txt'
GENERATED_DIR = 'generated_names'
EVAL_DIR      = 'evaluation'
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_NAMES   = ['VanillaRNN', 'BLSTM', 'RNNAttention']
DISPLAY_NAMES = {
    'VanillaRNN':   'Vanilla RNN',
    'BLSTM':        'Bidirectional LSTM',
    'RNNAttention': 'RNN + Attention',
}

# common Indian name patterns for realism check
INDIAN_PATTERNS = [
    r'(av|an|ar|ka|sh|ra|vi|su|pr|am|ma|na|de|di|ni|la|ta|re|sa|bh|ch|dh)',
    r'(a|i|u|e|o)$',  # typical vowel endings
    r'^[A-Z][a-z]+$',  # capitalised word
]


def load_names(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def novelty_rate(generated, training_set):
    """Percentage of generated names NOT in training set (case-insensitive)."""
    train_lower = {n.lower() for n in training_set}
    novel = sum(1 for n in generated if n.lower() not in train_lower)
    return novel / len(generated) * 100 if generated else 0.0


def diversity(generated):
    """Number of unique generated names / total generated."""
    return len(set(g.lower() for g in generated)) / len(generated) * 100 if generated else 0.0


def avg_length(names):
    return np.mean([len(n) for n in names]) if names else 0.0


def valid_names_ratio(names):
    """
    Fraction that look like plausible names:
    2-15 chars, only alpha, first char uppercase.
    """
    valid = sum(1 for n in names if re.match(r'^[A-Z][a-z]{1,14}$', n))
    return valid / len(names) * 100 if names else 0.0


def top_ngrams(names, n=2, k=10):
    from collections import Counter
    all_ng = []
    for name in names:
        w = name.lower()
        all_ng += [w[i:i+n] for i in range(len(w)-n+1)]
    return Counter(all_ng).most_common(k)


def realism_score(names):
    """
    Simple heuristic realism score:
    - Length 4-12 → +1
    - Ends in vowel → +1
    - Contains at least one common Indian bigram → +1
    Max = 3 per name, returned as percentage.
    """
    common_bigrams = {'av','an','ar','ka','sh','ra','vi','su','pr','am',
                      'ma','na','de','di','ni','la','ta','re','sa','bh',
                      'ch','dh','sh','kh','gh','ya','ha','va'}
    scores = []
    for n in names:
        s = 0
        w = n.lower()
        if 4 <= len(w) <= 12:
            s += 1
        if w and w[-1] in 'aeiou':
            s += 1
        bgs = {w[i:i+2] for i in range(len(w)-1)}
        if bgs & common_bigrams:
            s += 1
        scores.append(s / 3)
    return np.mean(scores) * 100 if scores else 0.0


def failure_modes(names):
    """Identify and count common failure modes."""
    modes = {
        'too_short (<3 chars)':    0,
        'too_long  (>15 chars)':   0,
        'non_alpha chars':         0,
        'repeated_chars  (≥3 same)': 0,
        'all_consonants':          0,
    }
    for n in names:
        w = n.lower()
        if len(w) < 3:
            modes['too_short (<3 chars)'] += 1
        if len(w) > 15:
            modes['too_long  (>15 chars)'] += 1
        if not n.isalpha():
            modes['non_alpha chars'] += 1
        if any(w.count(c*3) > 0 for c in set(w)):
            modes['repeated_chars  (≥3 same)'] += 1
        if all(c not in 'aeiou' for c in w):
            modes['all_consonants'] += 1
    return modes


def plot_comparison(metrics, metric_name, ylabel, filename):
    names  = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#4C72B0', '#DD8452', '#55A868']
    plt.figure(figsize=(7, 4))
    bars = plt.bar([DISPLAY_NAMES[n] for n in names], values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.ylabel(ylabel)
    plt.title(metric_name)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, filename), dpi=150)
    plt.close()


def plot_length_distribution(all_names_dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for ax, (mname, names), color in zip(axes, all_names_dict.items(), colors):
        lengths = [len(n) for n in names]
        ax.hist(lengths, bins=range(1, 20), color=color, edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(lengths):.1f}')
        ax.set_title(DISPLAY_NAMES[mname], fontsize=10)
        ax.set_xlabel('Name Length')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Count')
    fig.suptitle('Name Length Distribution by Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'length_distributions.png'), dpi=150)
    plt.close()


def main():
    training_names = load_names(DATA_PATH)
    all_generated  = {}

    for mname in MODEL_NAMES:
        path = os.path.join(GENERATED_DIR, f'{mname}_generated.txt')
        all_generated[mname] = load_names(path)
        print(f"Loaded {len(all_generated[mname])} generated names for {mname}")

    # ── Quantitative metrics ─────────────────────────────
    metrics = {}
    for mname in MODEL_NAMES:
        gen = all_generated[mname]
        metrics[mname] = {
            'novelty_rate':       novelty_rate(gen, training_names),
            'diversity':          diversity(gen),
            'avg_length':         avg_length(gen),
            'valid_names_ratio':  valid_names_ratio(gen),
            'realism_score':      realism_score(gen),
            'total_generated':    len(gen),
            'unique_generated':   len(set(g.lower() for g in gen)),
        }

    print("\n" + "="*65)
    print("  QUANTITATIVE EVALUATION")
    print("="*65)
    header = f"{'Metric':<28}" + "".join(f"{DISPLAY_NAMES[m]:<22}" for m in MODEL_NAMES)
    print(header)
    print("-"*65)
    for key in ['novelty_rate','diversity','avg_length','valid_names_ratio','realism_score']:
        row = f"{key:<28}"
        for mname in MODEL_NAMES:
            val = metrics[mname][key]
            row += f"{val:<22.2f}"
        print(row)

    # Save metrics JSON
    with open(os.path.join(EVAL_DIR, 'quantitative_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ────────────────────────────────────────────
    plot_comparison({m: metrics[m]['novelty_rate']       for m in MODEL_NAMES},
                    'Novelty Rate (%)', 'Novelty Rate (%)',  'novelty_rate.png')
    plot_comparison({m: metrics[m]['diversity']          for m in MODEL_NAMES},
                    'Diversity (%)',    'Diversity (%)',     'diversity.png')
    plot_comparison({m: metrics[m]['realism_score']      for m in MODEL_NAMES},
                    'Realism Score (%)', 'Realism Score (%)', 'realism_score.png')
    plot_length_distribution(all_generated)

    # ── Qualitative analysis ─────────────────────────────
    print("\n" + "="*65)
    print("  QUALITATIVE ANALYSIS — Sample Generated Names (top 20)")
    print("="*65)
    qual_data = {}
    for mname in MODEL_NAMES:
        gen = all_generated[mname]
        print(f"\n  {DISPLAY_NAMES[mname]}")
        print("  " + ", ".join(gen[:20]))
        fm = failure_modes(gen)
        print("  Failure modes:", {k: v for k, v in fm.items() if v > 0})
        bigrams = top_ngrams(gen, n=2, k=5)
        print("  Top bigrams:", bigrams)
        qual_data[mname] = {
            'samples':       gen[:30],
            'failure_modes': fm,
            'top_bigrams':   bigrams,
        }

    with open(os.path.join(EVAL_DIR, 'qualitative_analysis.json'), 'w') as f:
        json.dump(qual_data, f, indent=2)

    print(f"\n✓ Evaluation complete.  Results saved to: {EVAL_DIR}/")
    return metrics, qual_data


if __name__ == '__main__':
    main()
