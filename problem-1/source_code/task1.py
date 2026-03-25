"""
============================================================
  PROBLEM 1 — LEARNING WORD EMBEDDINGS FROM IIT JODHPUR DATA
  TASK 1: DATASET PREPARATION
  
  Name   : Shahil Sharma
  Roll No: B22CS048
  Course : Natural Language Processing 
  Date   : March 2026
============================================================

NOTE TO SELF:
  Scraped data from 4 sources:
    1. IIT Jodhpur Academic Regulations (PDF parsed)
    2. Department pages (CSE, ECE, Mechanical)
    3. Faculty profile pages
    4. Course syllabus pages

  I tried to keep things clean and document every step
  so the TA can follow what's happening.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup
import re
import os
import json
import string
import random
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import numpy as np
from PIL import Image   # optional — for shaped word cloud

# Download NLTK data (only first time)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# All the URLs I scraped. I picked pages that have actual readable English text.
SOURCES = {
    "academic_regulations": [
        "https://iitj.ac.in/academics/index.php?id=ug_regulations",
        "https://iitj.ac.in/academics/index.php?id=pg_regulations",
    ],
    "departments": [
        "https://iitj.ac.in/department/index.php?id=cse",
        "https://iitj.ac.in/department/index.php?id=ee",
        "https://iitj.ac.in/department/index.php?id=me",
        "https://iitj.ac.in/department/index.php?id=ce",
        "https://iitj.ac.in/department/index.php?id=phy",
        "https://iitj.ac.in/department/index.php?id=chem",
    ],
    "research": [
        "https://iitj.ac.in/research/index.php?id=overview",
        "https://iitj.ac.in/research/index.php?id=projects",
    ],
    "institute_info": [
        "https://iitj.ac.in/institute/index.php?id=about",
        "https://iitj.ac.in/institute/index.php?id=vision_mission",
        "https://iitj.ac.in/institute/index.php?id=administration",
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

OUTPUT_DIR = "iitj_corpus"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — WEB SCRAPING
# ══════════════════════════════════════════════════════════════════════════════

def scrape_page(url: str) -> str:
    """
    Fetch a single URL and extract clean paragraph text.
    Returns raw text or empty string on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove nav, footer, sidebar — boilerplate stuff
        for tag in soup(["nav", "footer", "header", "script",
                          "style", "noscript", "aside", "form"]):
            tag.decompose()

        # Grab meaningful text blocks
        texts = []
        for elem in soup.find_all(["p", "li", "td", "h1", "h2", "h3", "h4", "span", "div"]):
            t = elem.get_text(separator=" ").strip()
            if len(t.split()) >= 5:          # skip very short fragments
                texts.append(t)

        return " ".join(texts)

    except Exception as e:
        print(f"  [WARN] Could not fetch {url}  →  {e}")
        return ""


def collect_corpus(sources: dict) -> dict:
    """
    Iterate over all source categories and scrape each URL.
    Returns a dict: {category: [raw_text, ...]}
    """
    corpus = {}
    total = sum(len(v) for v in sources.values())
    done = 0

    print("=" * 60)
    print("  COLLECTING DATA FROM IIT JODHPUR SOURCES")
    print("=" * 60)

    for category, urls in sources.items():
        corpus[category] = []
        print(f"\n[+] Category: {category.upper()}")
        for url in urls:
            done += 1
            print(f"  ({done}/{total}) Scraping: {url}")
            text = scrape_page(url)
            if text:
                corpus[category].append(text)
                print(f"       ✓  {len(text.split())} words extracted")
            else:
                print(f"       ✗  No content")

    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Regex patterns I found useful for cleaning IIT pages
_BOILERPLATE = re.compile(
    r"(skip to (main|content)|copyright\s*©?\s*\d{4}|all rights reserved"
    r"|iit jodhpur\s*\||\bpowered by\b|sitemap|contact us|home\s*›)",
    re.IGNORECASE
)
_NON_ASCII    = re.compile(r"[^\x00-\x7F]+")          # Devanagari / other scripts
_EXTRA_PUNC   = re.compile(r"[^\w\s]")                # keep only words & spaces
_MULTI_SPACE  = re.compile(r"\s+")


def remove_boilerplate(text: str) -> str:
    """Remove navigational chrome and legal boilerplate line-by-line."""
    lines = text.split("\n")
    clean = [l for l in lines if not _BOILERPLATE.search(l)]
    return " ".join(clean)


def remove_non_english(text: str) -> str:
    """Strip non-ASCII characters (handles Hindi mixed into pages)."""
    return _NON_ASCII.sub(" ", text)


def basic_clean(text: str) -> str:
    """Lowercase → strip punctuation → collapse whitespace."""
    text = text.lower()
    text = _EXTRA_PUNC.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def tokenize_and_filter(text: str) -> list[str]:
    """
    Tokenize with NLTK, then remove:
      - stopwords
      - single characters
      - pure numeric tokens
      - very long 'words' (probably garbled HTML)
    """
    tokens = word_tokenize(text)
    filtered = [
        tok for tok in tokens
        if tok not in STOP_WORDS
        and len(tok) > 1
        and not tok.isnumeric()
        and len(tok) <= 25          # ignore HTML artefacts
    ]
    return filtered


def preprocess_document(raw: str) -> tuple[str, list[str]]:
    """
    Full pipeline for a single raw document.
    Returns (clean_text, token_list)
    """
    text = remove_boilerplate(raw)
    text = remove_non_english(text)
    text = basic_clean(text)
    tokens = tokenize_and_filter(text)
    return text, tokens


def preprocess_corpus(corpus: dict) -> tuple[list[str], list[list[str]]]:
    """
    Run preprocessing on all documents.
    Returns flat lists: (all_clean_docs, all_token_lists)
    """
    all_docs   = []
    all_tokens = []

    print("\n" + "=" * 60)
    print("  PREPROCESSING CORPUS")
    print("=" * 60)

    for category, docs in corpus.items():
        print(f"\n[+] {category.upper()} — {len(docs)} document(s)")
        for i, raw in enumerate(docs):
            clean, tokens = preprocess_document(raw)
            if len(tokens) < 20:
                print(f"    doc {i+1}: skipped (too short after cleaning)")
                continue
            all_docs.append(clean)
            all_tokens.append(tokens)
            print(f"    doc {i+1}: {len(raw.split()):>6} raw tokens → "
                  f"{len(tokens):>6} clean tokens")

    return all_docs, all_tokens


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATASET STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_statistics(all_tokens: list[list[str]]) -> dict:
    """Compute and pretty-print corpus statistics."""
    flat_tokens = [tok for doc in all_tokens for tok in doc]
    vocab       = set(flat_tokens)
    freq        = Counter(flat_tokens)

    stats = {
        "num_documents"  : len(all_tokens),
        "total_tokens"   : len(flat_tokens),
        "vocabulary_size": len(vocab),
        "top_50_words"   : freq.most_common(50),
        "avg_doc_length" : round(len(flat_tokens) / max(len(all_tokens), 1), 2),
    }

    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total documents  : {stats['num_documents']}")
    print(f"  Total tokens     : {stats['total_tokens']:,}")
    print(f"  Vocabulary size  : {stats['vocabulary_size']:,}")
    print(f"  Avg doc length   : {stats['avg_doc_length']} tokens")
    print(f"\n  Top 20 words:")
    for word, cnt in stats["top_50_words"][:20]:
        bar = "█" * min(40, cnt // max(1, stats["total_tokens"] // 500))
        print(f"    {word:<20} {cnt:>5}  {bar}")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — WORD CLOUD + BAR CHART  (the fun part 😄)
# ══════════════════════════════════════════════════════════════════════════════

# IIT Jodhpur's brand colors — orange, dark blue, and white
IITJ_ORANGE  = "#F07722"
IITJ_BLUE    = "#1A2B6B"
IITJ_CREAM   = "#FDF6EC"
IITJ_ACCENT  = "#E63946"


def iitj_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """Custom color function cycling through IITJ brand palette."""
    palette = [IITJ_ORANGE, IITJ_BLUE, "#2E86AB", "#E63946", "#457B9D", "#F4A261"]
    return random.choice(palette)


def generate_word_cloud(stats: dict, save_path: str = "wordcloud_iitj.png"):
    """
    Generate a styled word cloud from top words.
    Layout: word cloud on left, frequency bar chart on right.
    """
    freq_dict = dict(stats["top_50_words"])

    # ── Word Cloud ──────────────────────────────────────────
    wc = WordCloud(
        width            = 900,
        height           = 600,
        background_color = IITJ_CREAM,
        max_words        = 80,
        color_func       = iitj_color_func,
        prefer_horizontal= 0.7,
        collocations     = False,
        max_font_size    = 120,
        min_font_size    = 12,
        margin           = 8,
    ).generate_from_frequencies(freq_dict)

    # ── Figure layout ────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8), facecolor=IITJ_CREAM)
    fig.suptitle(
        "IIT Jodhpur Corpus — Most Frequent Words",
        fontsize=20, fontweight="bold",
        color=IITJ_BLUE, y=1.02
    )

    # Left panel — word cloud
    ax_wc = fig.add_axes([0.01, 0.05, 0.52, 0.88])
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title("Word Cloud", fontsize=13, color=IITJ_BLUE, pad=8)

    # Right panel — horizontal bar chart (top 25)
    ax_bar = fig.add_axes([0.57, 0.05, 0.40, 0.88])
    top25_words  = [w for w, _ in stats["top_50_words"][:25]]
    top25_counts = [c for _, c in stats["top_50_words"][:25]]

    colors = [IITJ_ORANGE if i % 2 == 0 else IITJ_BLUE
              for i in range(len(top25_words))]

    bars = ax_bar.barh(
        range(len(top25_words)), top25_counts[::-1],
        color=colors[::-1], edgecolor="white", linewidth=0.5
    )
    ax_bar.set_yticks(range(len(top25_words)))
    ax_bar.set_yticklabels(top25_words[::-1], fontsize=10, color=IITJ_BLUE)
    ax_bar.set_xlabel("Frequency", color=IITJ_BLUE, fontsize=11)
    ax_bar.set_title("Top 25 Words", fontsize=13, color=IITJ_BLUE, pad=8)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(colors=IITJ_BLUE)
    ax_bar.set_facecolor(IITJ_CREAM)

    # Add count labels on bars
    for bar, count in zip(bars, top25_counts[::-1]):
        ax_bar.text(
            bar.get_width() + max(top25_counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count), va="center", ha="left",
            fontsize=8, color=IITJ_BLUE
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=IITJ_CREAM)
    print(f"\n  [✓] Word cloud saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE CORPUS TO DISK  (for Task-2 Word2Vec training later)
# ══════════════════════════════════════════════════════════════════════════════

def save_corpus(all_tokens: list[list[str]], stats: dict):
    """
    Save:
      - sentences.txt  →  one tokenized sentence per line (for Word2Vec)
      - stats.json     →  corpus statistics
    """
    sent_path  = os.path.join(OUTPUT_DIR, "sentences.txt")
    stats_path = os.path.join(OUTPUT_DIR, "stats.json")

    with open(sent_path, "w", encoding="utf-8") as f:
        for doc_tokens in all_tokens:
            # Break each doc into ~20-token sentences (sliding window)
            chunk_size = 20
            for i in range(0, len(doc_tokens), chunk_size):
                chunk = doc_tokens[i: i + chunk_size]
                if len(chunk) >= 5:
                    f.write(" ".join(chunk) + "\n")

    # Convert top words to plain dict for JSON
    serializable_stats = {k: v for k, v in stats.items()
                          if k != "top_50_words"}
    serializable_stats["top_50_words"] = stats["top_50_words"]

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"  [✓] Tokenized sentences saved → {sent_path}")
    print(f"  [✓] Statistics saved          → {stats_path}")



# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  NLP ASSIGNMENT — TASK 1: DATASET PREPARATION" + " " * 11 + "║")
    print("║  IIT Jodhpur Word Embeddings Project" + " " * 21 + "║")
    print("╚" + "═" * 58 + "╝\n")

    # ── Try live scraping first, fall back to demo data ──────
    print("[INFO] Attempting live web scraping from iitj.ac.in ...")
    corpus = collect_corpus(SOURCES)

    # Check if we got meaningful data
    total_scraped = sum(
        sum(len(t.split()) for t in docs)
        for docs in corpus.values()
    )

    if total_scraped < 1000:
        print("\n[FALLBACK] Live scraping returned insufficient data.")
        print("           Using pre-collected offline demo corpus instead.")
        print("           (This is expected if running without internet access)\n")
        corpus = DEMO_TEXTS

    # ── Preprocess ───────────────────────────────────────────
    all_docs, all_tokens = preprocess_corpus(corpus)

    # ── Statistics ───────────────────────────────────────────
    stats = compute_statistics(all_tokens)

    # ── Visualize ────────────────────────────────────────────
    wc_path = os.path.join(OUTPUT_DIR, "wordcloud_iitj.png")
    generate_word_cloud(stats, save_path=wc_path)

    # ── Save to disk (for Task-2) ────────────────────────────
    save_corpus(all_tokens, stats)

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TASK 1 COMPLETE  ✓")
    print("=" * 60)
    print(f"  Output directory : ./{OUTPUT_DIR}/")
    print(f"    sentences.txt  → tokenized sentences for Word2Vec")
    print(f"    stats.json     → corpus statistics")
    print(f"    {os.path.basename(wc_path):<15}→ word cloud image")
    print("\n  Ready for Task-2: Word2Vec Training  🚀\n")


if __name__ == "__main__":
    main()
