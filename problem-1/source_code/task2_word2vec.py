"""
============================================================
  PROBLEM 1 — TASK 2: WORD2VEC MODEL TRAINING
  IIT Jodhpur Corpus — CBOW & Skip-gram with Negative Sampling

  Implemented from scratch using NumPy only.
  No gensim / torch / tensorflow used.
============================================================
"""

import numpy as np
import re, os, json, math, random
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "task2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CORPUS  (IIT Jodhpur dataset from Task-1)
# ─────────────────────────────────────────────────────────────
RAW_CORPUS = [
    """academic regulations indian institute technology jodhpur govern conduct
    undergraduate postgraduate programs students required maintain minimum
    cumulative grade point average continue enrollment senate iit jodhpur supreme
    academic body responsible maintaining standards instruction examination research
    institute student register courses every semester stipulated registration period
    late registration permitted fine specified academic office minimum credit requirement
    bachelor technology program distributed across eight semesters courses classified
    core courses elective courses humanities social sciences courses student must clear
    core courses eligible graduation grading system follows ten point scale letter grades
    student fails core course must repeat course subsequent semester offered academic
    integrity utmost importance plagiarism form treated serious academic misconduct""",

    """postgraduate programs iit jodhpur include master technology doctor philosophy
    master science research admission m tech programs gate score followed institute
    level interview phd admissions held twice year july january research scholars
    expected complete coursework first year subsequently focus research thesis advisory
    committee guides research scholar throughout doctoral program phd scholar must
    publish least one paper reputed journal before submission thesis maximum duration
    completing phd degree six years date joining financial support form scholarship
    provided full time phd m tech students government norms students encouraged
    participate national international conferences present research findings broader
    scientific community""",

    """department computer science engineering iit jodhpur offers undergraduate
    postgraduate doctoral programs department state art computing facilities high
    performance computing clusters dedicated research laboratories research areas
    department span artificial intelligence machine learning natural language processing
    computer vision data science cybersecurity distributed systems theoretical computer
    science faculty members department actively engaged sponsored research projects
    funded government agencies dst drdo meity industry collaborations department hosts
    regular seminars workshops hackathons foster culture innovation students alumni
    department placed leading technology companies research institutions worldwide
    b tech curriculum cse includes courses programming data structures algorithms
    operating systems database management computer networks software engineering""",

    """department electrical engineering offers programs focusing power systems signal
    processing communications vlsi design laboratories department include power electronics
    lab analog digital circuits lab embedded systems lab communications lab department
    collaborates industry partners curriculum development internship opportunities
    research department includes renewable energy systems smart grids biomedical signal
    processing advanced semiconductor devices students participate national competitions
    smart india hackathon texas instruments innovation challenge commendable results
    department offers interdisciplinary courses jointly physics mathematics computer science
    departments provide students broad technical foundation faculty members received funding
    agencies including dst serb ministry new renewable energy""",

    """department mechanical engineering iit jodhpur covers areas including thermal
    engineering manufacturing engineering solid mechanics design engineering computational
    fluid dynamics lab advanced manufacturing lab robotics lab support undergraduate
    postgraduate research activities projects department often address challenges relevant
    desert ecosystem rajasthan including solar thermal systems water harvesting technologies
    dust mitigation strategies solar panels students mechanical engineering undertake eight
    week industrial training program final year gain practical experience department hosts
    annual technical festival students showcase design projects prototypes""",

    """iit jodhpur established several centres excellence foster interdisciplinary research
    centre artificial intelligence data science conducts research machine learning deep
    learning computer vision applications healthcare agriculture smart cities desert
    technology centre focuses technologies relevant arid regions including solar energy
    water conservation sustainable construction materials biotechnology bioinformatics
    research group works genomics protein structure prediction drug discovery computational
    methods institute active collaborations premier national institutions including isro
    barc csir laboratories international universities germany japan france united states
    research funding institute grown significantly total externally funded projects
    exceeding two hundred crore rupees institute encourages faculty file patents several
    patents granted areas solar energy storage materials biomedical devices""",

    """indian institute technology jodhpur established 2008 one eight new iits set
    government india expand higher technical education country institute located historic
    city jodhpur state rajasthan gateway thar desert permanent campus spans eight hundred
    acres features modern academic residential infrastructure institute follows vision
    excellence education research innovation commitment societal impact mission iit jodhpur
    provide world class technical education develop technologies sustainable development
    nurture leaders science engineering management institute senate governing board finance
    committee oversee academic administrative financial matters respectively student body
    iit jodhpur diverse students across india several foreign students enrolled various
    programs cultural technical sports festivals organized student bodies provide vibrant
    campus life placement cell facilitates campus recruitment leading companies visiting
    institute every year internship full time recruitment""",

    """research iit jodhpur spans multiple domains engineering sciences basic sciences
    interdisciplinary areas faculty actively pursuing funded projects national international
    levels new faculty members joining institute bringing fresh perspectives cutting edge
    research methodologies doctoral students contributing significantly knowledge generation
    publication research papers reputed journals conferences indicates healthy research
    culture institute startup ecosystem iit jodhpur growing entrepreneurs alumni students
    founding companies technology various domains institute provides support incubation
    facilities mentoring nascent startups collaborations industries lead sponsored research
    consultancy projects providing real world context academic research""",

    """b tech program computer science engineering includes compulsory internship semester
    students expected work industry leading tech companies research labs iit jodhpur
    placement record consistently strong students recruited companies google microsoft
    amazon adobe samsung qualcomm tcs infosys among many others median package b tech
    graduates seen upward trend past years reflecting quality technical education
    m tech students specializations machine learning systems security networking data
    analytics graduates highly sought industry academia alike phd graduates joining
    faculty positions iisc iit nit research positions tata research microsoft research
    ibm research other leading organizations""",

    """iit jodhpur campus rajasthan showcases desert architecture sustainable design
    buildings designed passive cooling reduce energy consumption laboratories equipped
    latest instruments equipment central library houses extensive collection books
    journals digital resources students hostels provide comfortable living environment
    sports facilities include cricket ground football field tennis courts swimming pool
    gymnasium indoor games room health centre provides medical facilities students
    faculty staff canteen mess provide nutritious food options students campus also
    has atm bank post office several convenience stores meeting daily needs""",
]


# ─────────────────────────────────────────────────────────────
# VOCABULARY BUILDING
# ─────────────────────────────────────────────────────────────

def build_vocab(sentences, min_count=1):
    """Build word→index mapping and frequency table."""
    freq = Counter(w for s in sentences for w in s)
    vocab = {w: i for i, (w, c) in
             enumerate(freq.most_common()) if c >= min_count}
    idx2word = {i: w for w, i in vocab.items()}
    freq_arr  = np.array([freq[idx2word[i]] for i in range(len(vocab))],
                          dtype=np.float32)
    return vocab, idx2word, freq_arr


def tokenize_corpus(raw_corpus):
    sentences = []
    for doc in raw_corpus:
        tokens = re.findall(r"[a-z]+", doc.lower())
        tokens = [t for t in tokens if len(t) > 1]
        if tokens:
            sentences.append(tokens)
    return sentences


# ─────────────────────────────────────────────────────────────
# NEGATIVE SAMPLING TABLE
# ─────────────────────────────────────────────────────────────

def build_neg_table(freq_arr, table_size=1_000_000, power=0.75):
    """Build unigram table for fast negative sampling (Mikolov et al.)."""
    powered = freq_arr ** power
    probs   = powered / powered.sum()
    table   = np.zeros(table_size, dtype=np.int32)
    idx = 0
    p_cumul = 0.0
    for w_idx, p in enumerate(probs):
        p_cumul += p * table_size
        while idx < table_size and idx < p_cumul:
            table[idx] = w_idx
            idx += 1
    return table


def get_negatives(table, target_idx, k):
    """Sample k negative indices, excluding target."""
    negs = []
    while len(negs) < k:
        candidate = table[np.random.randint(0, len(table))]
        if candidate != target_idx:
            negs.append(candidate)
    return negs


# ─────────────────────────────────────────────────────────────
# SIGMOID
# ─────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


# ─────────────────────────────────────────────────────────────
# SKIP-GRAM WITH NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────────

class SkipGramNS:
    """
    Skip-gram model with Negative Sampling.
    Matrices:
      W_in  : (V, d)  — center word embeddings
      W_out : (V, d)  — context word embeddings
    """
    def __init__(self, vocab_size, embed_dim):
        scale = 0.1
        self.W_in  = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale
        self.W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)
        self.V = vocab_size
        self.d = embed_dim

    def train_pair(self, center_idx, context_idx, neg_indices, lr):
        """One SGD update for a single (center, context, negatives) triple."""
        v_c = self.W_in[center_idx]             # (d,)
        u_o = self.W_out[context_idx]           # (d,)

        # Positive sample gradient
        score_pos = sigmoid(np.dot(v_c, u_o))
        g_pos     = (1.0 - score_pos) * lr

        # Negative samples gradient
        neg_vecs   = self.W_out[neg_indices]    # (k, d)
        scores_neg = sigmoid(np.dot(neg_vecs, v_c))  # (k,)
        g_neg      = (-scores_neg * lr)[:, np.newaxis] * neg_vecs  # (k,d)

        # Update W_in (center)
        grad_center  = g_pos * u_o
        grad_center += np.sum(g_neg, axis=0)    # sum neg contributions
        # also: gradient from neg on W_out
        self.W_in[center_idx] += grad_center

        # Update W_out
        self.W_out[context_idx] += g_pos * v_c
        for i, neg_idx in enumerate(neg_indices):
            self.W_out[neg_idx] += (-scores_neg[i] * lr) * v_c

        # Return a rough loss for monitoring
        loss  =  -np.log(score_pos + 1e-7)
        loss += -np.sum(np.log(1.0 - scores_neg + 1e-7))
        return loss

    def get_embeddings(self):
        return self.W_in.copy()


# ─────────────────────────────────────────────────────────────
# CBOW WITH NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────────

class CBOWNS:
    """
    CBOW model with Negative Sampling.
    Context words' embeddings are averaged → predict center word.
    """
    def __init__(self, vocab_size, embed_dim):
        scale = 0.1
        self.W_in  = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale
        self.W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)
        self.V = vocab_size
        self.d = embed_dim

    def train_pair(self, context_indices, center_idx, neg_indices, lr):
        """One SGD update: average context → predict center."""
        ctx_vecs  = self.W_in[context_indices]  # (2w, d)
        h         = ctx_vecs.mean(axis=0)        # (d,)  hidden layer = mean
        u_o       = self.W_out[center_idx]       # (d,)

        # Positive
        score_pos = sigmoid(np.dot(h, u_o))
        g_pos     = (1.0 - score_pos) * lr

        # Negative
        neg_vecs   = self.W_out[neg_indices]
        scores_neg = sigmoid(np.dot(neg_vecs, h))
        g_neg_sum  = np.sum((-scores_neg * lr)[:, np.newaxis] * neg_vecs, axis=0)

        # EH: error signal on hidden layer
        eh = g_pos * u_o + g_neg_sum

        # Propagate to context embeddings
        for ctx_idx in context_indices:
            self.W_in[ctx_idx] += eh / len(context_indices)

        # Update output weights
        self.W_out[center_idx] += g_pos * h
        for i, neg_idx in enumerate(neg_indices):
            self.W_out[neg_idx] += (-scores_neg[i] * lr) * h

        loss  = -np.log(score_pos + 1e-7)
        loss += -np.sum(np.log(1.0 - scores_neg + 1e-7))
        return loss

    def get_embeddings(self):
        return self.W_in.copy()


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def generate_skipgram_pairs(sentences, vocab, window):
    pairs = []
    for sent in sentences:
        idxs = [vocab[w] for w in sent if w in vocab]
        for i, center in enumerate(idxs):
            lo = max(0, i - window)
            hi = min(len(idxs), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, idxs[j]))
    return pairs


def generate_cbow_pairs(sentences, vocab, window):
    pairs = []
    for sent in sentences:
        idxs = [vocab[w] for w in sent if w in vocab]
        for i, center in enumerate(idxs):
            lo = max(0, i - window)
            hi = min(len(idxs), i + window + 1)
            ctx = [idxs[j] for j in range(lo, hi) if j != i]
            if ctx:
                pairs.append((ctx, center))
    return pairs


def train_skipgram(sentences, vocab, freq_arr,
                   embed_dim=50, window=2, neg_k=5,
                   epochs=5, lr=0.025, verbose=True):
    V = len(vocab)
    model = SkipGramNS(V, embed_dim)
    neg_table = build_neg_table(freq_arr)
    pairs = generate_skipgram_pairs(sentences, vocab, window)
    history = []

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        cur_lr = lr * (1 - epoch / (epochs + 1))
        cur_lr = max(cur_lr, lr * 0.0001)

        for center, context in pairs:
            negs = get_negatives(neg_table, context, neg_k)
            loss = model.train_pair(center, context, negs, cur_lr)
            total_loss += loss

        avg_loss = total_loss / max(len(pairs), 1)
        history.append(avg_loss)
        if verbose:
            print(f"    Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  lr={cur_lr:.5f}")

    return model, history


def train_cbow(sentences, vocab, freq_arr,
               embed_dim=50, window=2, neg_k=5,
               epochs=5, lr=0.025, verbose=True):
    V = len(vocab)
    model = CBOWNS(V, embed_dim)
    neg_table = build_neg_table(freq_arr)
    pairs = generate_cbow_pairs(sentences, vocab, window)
    history = []

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        cur_lr = lr * (1 - epoch / (epochs + 1))
        cur_lr = max(cur_lr, lr * 0.0001)

        for ctx, center in pairs:
            negs = get_negatives(neg_table, center, neg_k)
            loss = model.train_pair(ctx, center, negs, cur_lr)
            total_loss += loss

        avg_loss = total_loss / max(len(pairs), 1)
        history.append(avg_loss)
        if verbose:
            print(f"    Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  lr={cur_lr:.5f}")

    return model, history


# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY UTILITIES
# ─────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def most_similar(word, embeddings, vocab, idx2word, topn=5):
    if word not in vocab:
        return []
    idx  = vocab[word]
    vec  = embeddings[idx]
    vec  = vec / (np.linalg.norm(vec) + 1e-8)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    sims  = (embeddings / norms) @ vec
    sims[idx] = -1.0
    top_idx = np.argsort(sims)[::-1][:topn]
    return [(idx2word[i], float(sims[i])) for i in top_idx]


# ─────────────────────────────────────────────────────────────
# EXPERIMENT GRID
# ─────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # (embed_dim, window, neg_k)
    (50,  2, 5),
    (50,  4, 5),
    (50,  2, 10),
    (100, 2, 5),
    (100, 4, 5),
    (100, 4, 10),
]

EVAL_PAIRS = [
    ("research",    "machine"),
    ("engineering", "technology"),
    ("students",    "faculty"),
    ("learning",    "training"),
    ("solar",       "energy"),
    ("computer",    "science"),
    ("jodhpur",     "rajasthan"),
    ("phd",         "research"),
]


def run_all_experiments(sentences, vocab, idx2word, freq_arr):
    results = {"skipgram": [], "cbow": []}

    for embed_dim, window, neg_k in EXPERIMENTS:
        tag = f"d{embed_dim}_w{window}_k{neg_k}"
        print(f"\n{'─'*55}")
        print(f"  CONFIG  dim={embed_dim}  window={window}  neg_k={neg_k}")
        print(f"{'─'*55}")

        # ── Skip-gram ────────────────────────────────────────
        print("  [Skip-gram]")
        sg_model, sg_hist = train_skipgram(
            sentences, vocab, freq_arr,
            embed_dim=embed_dim, window=window, neg_k=neg_k,
            epochs=5, lr=0.025, verbose=True
        )
        sg_emb = sg_model.get_embeddings()

        sg_sims = {f"{a}~{b}": cosine_sim(sg_emb[vocab[a]], sg_emb[vocab[b]])
                   for a, b in EVAL_PAIRS
                   if a in vocab and b in vocab}

        results["skipgram"].append({
            "tag": tag, "embed_dim": embed_dim,
            "window": window, "neg_k": neg_k,
            "final_loss": sg_hist[-1],
            "loss_history": sg_hist,
            "similarities": sg_sims,
            "embeddings": sg_emb,
        })

        # ── CBOW ─────────────────────────────────────────────
        print("  [CBOW]")
        cb_model, cb_hist = train_cbow(
            sentences, vocab, freq_arr,
            embed_dim=embed_dim, window=window, neg_k=neg_k,
            epochs=5, lr=0.025, verbose=True
        )
        cb_emb = cb_model.get_embeddings()

        cb_sims = {f"{a}~{b}": cosine_sim(cb_emb[vocab[a]], cb_emb[vocab[b]])
                   for a, b in EVAL_PAIRS
                   if a in vocab and b in vocab}

        results["cbow"].append({
            "tag": tag, "embed_dim": embed_dim,
            "window": window, "neg_k": neg_k,
            "final_loss": cb_hist[-1],
            "loss_history": cb_hist,
            "similarities": cb_sims,
            "embeddings": cb_emb,
        })

    return results


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — TRAINING LOSS CURVES
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(results):
    IITJ_ORANGE = "#F07722"
    IITJ_BLUE   = "#1A2B6B"
    IITJ_CREAM  = "#FDF6EC"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=IITJ_CREAM)
    fig.suptitle("Figure 1: Training Loss Curves — Skip-gram vs CBOW",
                 fontsize=14, fontweight="bold", color=IITJ_BLUE, y=1.01)

    sg_data = results["skipgram"]
    cb_data = results["cbow"]

    for i, (sg, cb) in enumerate(zip(sg_data, cb_data)):
        ax = axes[i // 3][i % 3]
        ax.set_facecolor(IITJ_CREAM)
        epochs = list(range(1, len(sg["loss_history"]) + 1))
        ax.plot(epochs, sg["loss_history"], color=IITJ_ORANGE,
                marker="o", linewidth=2, markersize=5, label="Skip-gram")
        ax.plot(epochs, cb["loss_history"], color=IITJ_BLUE,
                marker="s", linewidth=2, markersize=5, linestyle="--", label="CBOW")
        ax.set_title(
            f"dim={sg['embed_dim']}  win={sg['window']}  k={sg['neg_k']}",
            fontsize=10, color=IITJ_BLUE
        )
        ax.set_xlabel("Epoch", fontsize=9, color=IITJ_BLUE)
        ax.set_ylabel("Avg Loss", fontsize=9, color=IITJ_BLUE)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=IITJ_BLUE)
        ax.grid(True, alpha=0.3, color="gray")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=IITJ_CREAM)
    plt.close()
    print(f"  ✓ Figure 1 saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — FINAL LOSS COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_final_loss_comparison(results):
    IITJ_ORANGE = "#F07722"
    IITJ_BLUE   = "#1A2B6B"
    IITJ_CREAM  = "#FDF6EC"

    tags   = [r["tag"] for r in results["skipgram"]]
    sg_losses = [r["final_loss"] for r in results["skipgram"]]
    cb_losses = [r["final_loss"] for r in results["cbow"]]

    x = np.arange(len(tags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=IITJ_CREAM)
    ax.set_facecolor(IITJ_CREAM)

    bars1 = ax.bar(x - width/2, sg_losses, width, label="Skip-gram",
                   color=IITJ_ORANGE, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, cb_losses, width, label="CBOW",
                   color=IITJ_BLUE, edgecolor="white", linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7.5, color=IITJ_ORANGE, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7.5, color=IITJ_BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=8.5)
    ax.set_ylabel("Final Avg Loss (Epoch 5)", color=IITJ_BLUE)
    ax.set_title("Figure 2: Final Training Loss — All Configurations",
                 fontsize=13, fontweight="bold", color=IITJ_BLUE)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=IITJ_BLUE)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_final_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=IITJ_CREAM)
    plt.close()
    print(f"  ✓ Figure 2 saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — COSINE SIMILARITY HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_similarity_heatmap(results):
    IITJ_BLUE  = "#1A2B6B"
    IITJ_CREAM = "#FDF6EC"

    # Use best config (lowest final loss) per model
    best_sg = min(results["skipgram"], key=lambda x: x["final_loss"])
    best_cb = min(results["cbow"],     key=lambda x: x["final_loss"])

    pair_labels = list(best_sg["similarities"].keys())
    sg_vals = [best_sg["similarities"][p] for p in pair_labels]
    cb_vals = [best_cb["similarities"][p] for p in pair_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=IITJ_CREAM)

    for ax, vals, title, cfg in zip(
        axes,
        [sg_vals, cb_vals],
        ["Skip-gram", "CBOW"],
        [best_sg["tag"], best_cb["tag"]]
    ):
        ax.set_facecolor(IITJ_CREAM)
        colors = ["#C84B31" if v > 0.3 else "#2E86AB" if v > 0 else "#6C757D"
                  for v in vals]
        bars = ax.barh(pair_labels, vals, color=colors, edgecolor="white")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-0.5, 1.0)
        ax.set_title(f"{title}\n(best config: {cfg})",
                     fontsize=11, color=IITJ_BLUE, fontweight="bold")
        ax.set_xlabel("Cosine Similarity", color=IITJ_BLUE)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8.5, color=IITJ_BLUE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=IITJ_BLUE)

    fig.suptitle("Figure 3: Word-Pair Cosine Similarities — Best Configuration per Model",
                 fontsize=12, fontweight="bold", color=IITJ_BLUE)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_similarities.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=IITJ_CREAM)
    plt.close()
    print(f"  ✓ Figure 3 saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 4 — t-SNE / PCA EMBEDDING VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_embedding_pca(results, vocab, idx2word):
    from sklearn.decomposition import PCA

    IITJ_ORANGE = "#F07722"
    IITJ_BLUE   = "#1A2B6B"
    IITJ_CREAM  = "#FDF6EC"

    best_sg = min(results["skipgram"], key=lambda x: x["final_loss"])
    best_cb = min(results["cbow"],     key=lambda x: x["final_loss"])

    # Interesting domain words to annotate
    SHOW_WORDS = [
        "research", "engineering", "students", "faculty", "learning",
        "solar", "energy", "computer", "science", "jodhpur", "phd",
        "machine", "technology", "laboratory", "campus", "academic",
        "data", "artificial", "intelligence", "training", "programs",
    ]
    show_idx = [vocab[w] for w in SHOW_WORDS if w in vocab]
    show_words = [idx2word[i] for i in show_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=IITJ_CREAM)

    for ax, r, title, color in zip(
        axes,
        [best_sg, best_cb],
        ["Skip-gram", "CBOW"],
        [IITJ_ORANGE, IITJ_BLUE],
    ):
        emb = r["embeddings"]
        pca  = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(emb)

        ax.set_facecolor("#FAFAF6")
        ax.scatter(reduced[:, 0], reduced[:, 1],
                   alpha=0.15, s=8, color=color)

        for idx in show_idx:
            x, y = reduced[idx]
            ax.scatter(x, y, s=60, color=color, zorder=5, edgecolors="white", linewidth=0.8)
            ax.annotate(idx2word[idx], (x, y),
                        fontsize=7.5, color=IITJ_BLUE,
                        xytext=(5, 3), textcoords="offset points",
                        fontweight="bold")

        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var.)", color=IITJ_BLUE)
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var.)", color=IITJ_BLUE)
        ax.set_title(f"{title} Embeddings — PCA\n(config: {r['tag']})",
                     fontsize=11, color=IITJ_BLUE, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=IITJ_BLUE)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Figure 4: 2D PCA Projection of Word Embeddings (Best Config per Model)",
                 fontsize=12, fontweight="bold", color=IITJ_BLUE)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_pca_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=IITJ_CREAM)
    plt.close()
    print(f"  ✓ Figure 4 saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 5 — HYPERPARAMETER EFFECT PLOTS
# ─────────────────────────────────────────────────────────────

def plot_hyperparameter_effects(results):
    IITJ_ORANGE = "#F07722"
    IITJ_BLUE   = "#1A2B6B"
    IITJ_CREAM  = "#FDF6EC"

    def extract(data, key, val):
        return [r for r in data if r[key] == val]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=IITJ_CREAM)
    fig.suptitle("Figure 5: Effect of Hyperparameters on Final Loss",
                 fontsize=13, fontweight="bold", color=IITJ_BLUE)

    params = [
        ("embed_dim", [50, 100], "Embedding Dimension"),
        ("window",    [2, 4],   "Context Window Size"),
        ("neg_k",     [5, 10],  "Negative Samples (k)"),
    ]

    for ax, (param, vals, label) in zip(axes, params):
        ax.set_facecolor(IITJ_CREAM)

        sg_means = [np.mean([r["final_loss"] for r in results["skipgram"] if r[param] == v])
                    for v in vals]
        cb_means = [np.mean([r["final_loss"] for r in results["cbow"] if r[param] == v])
                    for v in vals]

        x = np.arange(len(vals))
        ax.bar(x - 0.2, sg_means, 0.35, label="Skip-gram",
               color=IITJ_ORANGE, edgecolor="white")
        ax.bar(x + 0.2, cb_means, 0.35, label="CBOW",
               color=IITJ_BLUE, edgecolor="white")

        for xi, sg_v, cb_v in zip(x, sg_means, cb_means):
            ax.text(xi - 0.2, sg_v + 0.002, f"{sg_v:.3f}",
                    ha="center", fontsize=8, color=IITJ_ORANGE, fontweight="bold")
            ax.text(xi + 0.2, cb_v + 0.002, f"{cb_v:.3f}",
                    ha="center", fontsize=8, color=IITJ_BLUE, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals])
        ax.set_xlabel(label, color=IITJ_BLUE)
        ax.set_ylabel("Avg Final Loss", color=IITJ_BLUE)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=IITJ_BLUE)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_hyperparam_effects.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=IITJ_CREAM)
    plt.close()
    print(f"  ✓ Figure 5 saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS TABLE AS JSON (for docx report)
# ─────────────────────────────────────────────────────────────

def save_results_table(results):
    table = []
    for sg, cb in zip(results["skipgram"], results["cbow"]):
        table.append({
            "config": sg["tag"],
            "embed_dim": sg["embed_dim"],
            "window": sg["window"],
            "neg_k": sg["neg_k"],
            "sg_final_loss": round(sg["final_loss"], 4),
            "cb_final_loss": round(cb["final_loss"], 4),
            "sg_avg_sim": round(np.mean(list(sg["similarities"].values())), 4),
            "cb_avg_sim": round(np.mean(list(cb["similarities"].values())), 4),
        })

    path = os.path.join(OUTPUT_DIR, "results_table.json")
    # Convert numpy floats to Python floats for JSON serialization
    table_serializable = [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in row.items()} for row in table]
    with open(path, "w") as f:
        json.dump(table_serializable, f, indent=2)
    print(f"  ✓ Results table saved → {path}")

    # Pretty print
    print(f"\n  {'Config':<20} {'SG Loss':>10} {'CB Loss':>10} {'SG Sim':>10} {'CB Sim':>10}")
    print("  " + "─" * 62)
    for row in table:
        print(f"  {row['config']:<20} {row['sg_final_loss']:>10.4f} "
              f"{row['cb_final_loss']:>10.4f} {row['sg_avg_sim']:>10.4f} "
              f"{row['cb_avg_sim']:>10.4f}")

    return path, table


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  TASK 2: WORD2VEC MODEL TRAINING                       ║")
    print("║  Skip-gram + CBOW with Negative Sampling               ║")
    print("║  Implemented from scratch — NumPy only                 ║")
    print("╚" + "═"*56 + "╝\n")

    # Prepare corpus
    sentences = tokenize_corpus(RAW_CORPUS)
    vocab, idx2word, freq_arr = build_vocab(sentences, min_count=1)
    print(f"Corpus: {len(sentences)} documents, "
          f"{sum(len(s) for s in sentences)} tokens, "
          f"vocab={len(vocab)}")

    # Run experiments
    print("\n" + "═"*56)
    print("  RUNNING EXPERIMENTS (6 configs × 2 models = 12 runs)")
    print("═"*56)
    results = run_all_experiments(sentences, vocab, idx2word, freq_arr)

    # Save results
    print("\n" + "═"*56)
    print("  GENERATING FIGURES")
    print("═"*56)
    fig1 = plot_loss_curves(results)
    fig2 = plot_final_loss_comparison(results)
    fig3 = plot_similarity_heatmap(results)
    fig4 = plot_embedding_pca(results, vocab, idx2word)
    fig5 = plot_hyperparameter_effects(results)

    print("\n" + "═"*56)
    print("  RESULTS SUMMARY TABLE")
    print("═"*56)
    json_path, table = save_results_table(results)

    # Save vocab for report
    with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w") as f:
        json.dump({"vocab": list(vocab.keys()), "size": len(vocab)}, f)

    # Save best model nearest neighbors
    best_sg = min(results["skipgram"], key=lambda x: x["final_loss"])
    best_cb = min(results["cbow"],     key=lambda x: x["final_loss"])

    neighbors = {}
    for word in ["research", "engineering", "learning", "solar", "jodhpur"]:
        if word in vocab:
            neighbors[f"sg_{word}"] = most_similar(
                word, best_sg["embeddings"], vocab, idx2word, topn=5)
            neighbors[f"cb_{word}"] = most_similar(
                word, best_cb["embeddings"], vocab, idx2word, topn=5)

    with open(os.path.join(OUTPUT_DIR, "nearest_neighbors.json"), "w") as f:
        json.dump(neighbors, f, indent=2)

    print(f"\n  Top-5 nearest neighbors (Skip-gram, best config):")
    for word in ["research", "engineering", "learning"]:
        if f"sg_{word}" in neighbors:
            sims = ", ".join(f"{w}({s:.2f})" for w,s in neighbors[f"sg_{word}"])
            print(f"    {word:<15} → {sims}")

    print(f"\n  Top-5 nearest neighbors (CBOW, best config):")
    for word in ["research", "engineering", "learning"]:
        if f"cb_{word}" in neighbors:
            sims = ", ".join(f"{w}({s:.2f})" for w,s in neighbors[f"cb_{word}"])
            print(f"    {word:<15} → {sims}")

    print(f"\n{'═'*56}")
    print(f"  ALL OUTPUTS → ./{OUTPUT_DIR}/")
    print(f"{'═'*56}")
    print("  fig1_loss_curves.png")
    print("  fig2_final_loss.png")
    print("  fig3_similarities.png")
    print("  fig4_pca_embeddings.png")
    print("  fig5_hyperparam_effects.png")
    print("  results_table.json")
    print("  nearest_neighbors.json")
    print("\n  ✓ Ready for report writing.\n")

    return results, vocab, idx2word, table, {
        "fig1": fig1, "fig2": fig2, "fig3": fig3,
        "fig4": fig4, "fig5": fig5
    }


if __name__ == "__main__":
    main()