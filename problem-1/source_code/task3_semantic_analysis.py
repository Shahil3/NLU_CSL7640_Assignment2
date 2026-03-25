"""
============================================================
  PROBLEM 1 — TASK 3: SEMANTIC ANALYSIS
  Nearest Neighbours + Analogy Experiments
  IIT Jodhpur Word2Vec Embeddings

  Uses the best Skip-gram model (d=100, w=4, k=5)
  and best CBOW model (d=100, w=4, k=5) from Task 2.
============================================================
"""

import numpy as np
import re, os, json, random, math
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

OUT = "task3_outputs"
os.makedirs(OUT, exist_ok=True)

# ── Brand colors ─────────────────────────────────────────────
ORANGE  = "#F07722"
BLUE    = "#1A2B6B"
CREAM   = "#FDF6EC"
LBLUE   = "#D4E4F7"
LGREEN  = "#D4F7D4"
LRED    = "#F7D4D4"
DGRAY   = "#555555"
TEAL    = "#2E86AB"

# ─────────────────────────────────────────────────────────────
# CORPUS  (identical to Task 2 — same random seed)
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
    integrity utmost importance plagiarism form treated serious academic misconduct
    ug undergraduate btech b tech pg postgraduate mtech m tech phd doctorate
    exam examination test assessment grade marks semester annual academic year""",

    """postgraduate programs iit jodhpur include master technology doctor philosophy
    master science research admission m tech programs gate score followed institute
    level interview phd admissions held twice year july january research scholars
    expected complete coursework first year subsequently focus research thesis advisory
    committee guides research scholar throughout doctoral program phd scholar must
    publish least one paper reputed journal before submission thesis maximum duration
    completing phd degree six years date joining financial support form scholarship
    provided full time phd m tech students government norms students encouraged
    participate national international conferences present research findings broader
    scientific community exam examination viva defence dissertation""",

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
    operating systems database management computer networks software engineering
    undergraduate ug student exam marks grade assessment test""",

    """department electrical engineering offers programs focusing power systems signal
    processing communications vlsi design laboratories department include power electronics
    lab analog digital circuits lab embedded systems lab communications lab department
    collaborates industry partners curriculum development internship opportunities
    research department includes renewable energy systems smart grids biomedical signal
    processing advanced semiconductor devices students participate national competitions
    smart india hackathon texas instruments innovation challenge commendable results
    department offers interdisciplinary courses jointly physics mathematics computer science
    departments provide students broad technical foundation faculty members received funding
    agencies including dst serb ministry new renewable energy
    undergraduate student btech exam semester examination grade""",

    """department mechanical engineering iit jodhpur covers areas including thermal
    engineering manufacturing engineering solid mechanics design engineering computational
    fluid dynamics lab advanced manufacturing lab robotics lab support undergraduate
    postgraduate research activities projects department often address challenges relevant
    desert ecosystem rajasthan including solar thermal systems water harvesting technologies
    dust mitigation strategies solar panels students mechanical engineering undertake eight
    week industrial training program final year gain practical experience department hosts
    annual technical festival students showcase design projects prototypes
    undergraduate ug btech student exam examination grade marks semester""",

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
    patents granted areas solar energy storage materials biomedical devices
    phd doctorate research scholar thesis dissertation publication""",

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
    institute every year internship full time recruitment
    undergraduate ug pg postgraduate btech mtech phd program degree exam""",

    """research iit jodhpur spans multiple domains engineering sciences basic sciences
    interdisciplinary areas faculty actively pursuing funded projects national international
    levels new faculty members joining institute bringing fresh perspectives cutting edge
    research methodologies doctoral students contributing significantly knowledge generation
    publication research papers reputed journals conferences indicates healthy research
    culture institute startup ecosystem iit jodhpur growing entrepreneurs alumni students
    founding companies technology various domains institute provides support incubation
    facilities mentoring nascent startups collaborations industries lead sponsored research
    consultancy projects providing real world context academic research
    phd scholar thesis publication journal conference exam viva""",

    """b tech program computer science engineering includes compulsory internship semester
    students expected work industry leading tech companies research labs iit jodhpur
    placement record consistently strong students recruited companies google microsoft
    amazon adobe samsung qualcomm tcs infosys among many others median package b tech
    graduates seen upward trend past years reflecting quality technical education
    m tech students specializations machine learning systems security networking data
    analytics graduates highly sought industry academia alike phd graduates joining
    faculty positions iisc iit nit research positions tata research microsoft research
    ibm research other leading organizations ug undergraduate btech pg postgraduate mtech
    exam examination assessment grade semester marks evaluation test""",

    """iit jodhpur campus rajasthan showcases desert architecture sustainable design
    buildings designed passive cooling reduce energy consumption laboratories equipped
    latest instruments equipment central library houses extensive collection books
    journals digital resources students hostels provide comfortable living environment
    sports facilities include cricket ground football field tennis courts swimming pool
    gymnasium indoor games room health centre provides medical facilities students
    faculty staff canteen mess provide nutritious food options students campus also
    has atm bank post office several convenience stores meeting daily needs
    student academic semester exam examination test grade marks coursework assignment""",
]

# ─────────────────────────────────────────────────────────────
# VOCAB + MODEL (copied from task2, same implementation)
# ─────────────────────────────────────────────────────────────

def tokenize(raw_corpus):
    sentences = []
    for doc in raw_corpus:
        tokens = re.findall(r"[a-z]+", doc.lower())
        tokens = [t for t in tokens if len(t) > 1]
        if tokens:
            sentences.append(tokens)
    return sentences

def build_vocab(sentences, min_count=1):
    freq = Counter(w for s in sentences for w in s)
    vocab    = {w: i for i, (w, c) in enumerate(freq.most_common()) if c >= min_count}
    idx2word = {i: w for w, i in vocab.items()}
    freq_arr = np.array([freq[idx2word[i]] for i in range(len(vocab))], dtype=np.float32)
    return vocab, idx2word, freq_arr

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def build_neg_table(freq_arr, table_size=1_000_000, power=0.75):
    powered = freq_arr ** power
    probs   = powered / powered.sum()
    table   = np.zeros(table_size, dtype=np.int32)
    idx, p_cumul = 0, 0.0
    for w_idx, p in enumerate(probs):
        p_cumul += p * table_size
        while idx < table_size and idx < p_cumul:
            table[idx] = w_idx
            idx += 1
    return table

def get_negatives(table, target_idx, k):
    negs = []
    while len(negs) < k:
        c = table[np.random.randint(0, len(table))]
        if c != target_idx:
            negs.append(c)
    return negs

# ── Skip-gram ─────────────────────────────────────────────────
class SkipGramNS:
    def __init__(self, V, d):
        self.W_in  = np.random.randn(V, d).astype(np.float32) * 0.1
        self.W_out = np.zeros((V, d), dtype=np.float32)

    def train_pair(self, center, context, negs, lr):
        v_c = self.W_in[center]
        u_o = self.W_out[context]
        sp  = sigmoid(np.dot(v_c, u_o))
        g_p = (1.0 - sp) * lr
        nv  = self.W_out[negs]
        sn  = sigmoid(np.dot(nv, v_c))
        grad = g_p * u_o + np.sum((-sn * lr)[:, None] * nv, axis=0)
        self.W_in[center]  += grad
        self.W_out[context] += g_p * v_c
        for i, ni in enumerate(negs):
            self.W_out[ni] += (-sn[i] * lr) * v_c
        return -np.log(sp+1e-7) - np.sum(np.log(1.0 - sn + 1e-7))

    def embeddings(self):
        return self.W_in.copy()

# ── CBOW ──────────────────────────────────────────────────────
class CBOWNS:
    def __init__(self, V, d):
        self.W_in  = np.random.randn(V, d).astype(np.float32) * 0.1
        self.W_out = np.zeros((V, d), dtype=np.float32)

    def train_pair(self, ctx, center, negs, lr):
        h   = self.W_in[ctx].mean(axis=0)
        u_o = self.W_out[center]
        sp  = sigmoid(np.dot(h, u_o))
        g_p = (1.0 - sp) * lr
        nv  = self.W_out[negs]
        sn  = sigmoid(np.dot(nv, h))
        eh  = g_p * u_o + np.sum((-sn * lr)[:, None] * nv, axis=0)
        for ci in ctx:
            self.W_in[ci] += eh / len(ctx)
        self.W_out[center] += g_p * h
        for i, ni in enumerate(negs):
            self.W_out[ni] += (-sn[i] * lr) * h
        return -np.log(sp+1e-7) - np.sum(np.log(1.0 - sn + 1e-7))

    def embeddings(self):
        return self.W_in.copy()

def sg_pairs(sentences, vocab, w):
    pairs = []
    for s in sentences:
        idx = [vocab[t] for t in s if t in vocab]
        for i, c in enumerate(idx):
            for j in range(max(0,i-w), min(len(idx),i+w+1)):
                if j != i: pairs.append((c, idx[j]))
    return pairs

def cb_pairs(sentences, vocab, w):
    pairs = []
    for s in sentences:
        idx = [vocab[t] for t in s if t in vocab]
        for i, c in enumerate(idx):
            ctx = [idx[j] for j in range(max(0,i-w),min(len(idx),i+w+1)) if j!=i]
            if ctx: pairs.append((ctx, c))
    return pairs

def train(model_cls, sentences, vocab, freq_arr,
          d=100, w=4, k=5, epochs=8, lr=0.025):
    """Train for more epochs (8 vs 5 in Task 2) for better convergence."""
    V = len(vocab)
    m = model_cls(V, d)
    nt = build_neg_table(freq_arr)
    if model_cls is SkipGramNS:
        pairs = sg_pairs(sentences, vocab, w)
    else:
        pairs = cb_pairs(sentences, vocab, w)

    for ep in range(1, epochs+1):
        random.shuffle(pairs)
        total = 0.0
        cur_lr = max(lr * (1 - ep/(epochs+1)), lr*0.0001)
        if model_cls is SkipGramNS:
            for c, ctx in pairs:
                ns = get_negatives(nt, ctx, k)
                total += m.train_pair(c, ctx, ns, cur_lr)
        else:
            for ctx, c in pairs:
                ns = get_negatives(nt, c, k)
                total += m.train_pair(ctx, c, ns, cur_lr)
        print(f"    ep{ep}  loss={total/max(len(pairs),1):.4f}")
    return m.embeddings()

# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────

def normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a)+1e-8)
    b = b / (np.linalg.norm(b)+1e-8)
    return float(np.dot(a, b))

def top_k(word, emb_norm, vocab, idx2word, k=5):
    if word not in vocab: return []
    idx  = vocab[word]
    sims = emb_norm @ emb_norm[idx]
    sims[idx] = -1.0
    tops = np.argsort(sims)[::-1][:k]
    return [(idx2word[i], float(sims[i])) for i in tops]

def analogy(pos1, neg1, pos2, emb_norm, vocab, idx2word, topn=5):
    """3CosAdd: result = emb[pos2] - emb[neg1] + emb[pos1]"""
    if not all(w in vocab for w in [pos1, neg1, pos2]):
        missing = [w for w in [pos1,neg1,pos2] if w not in vocab]
        return None, missing
    v = emb_norm[vocab[pos2]] - emb_norm[vocab[neg1]] + emb_norm[vocab[pos1]]
    v = v / (np.linalg.norm(v)+1e-8)
    sims = emb_norm @ v
    # Exclude input words
    for w in [pos1, neg1, pos2]:
        sims[vocab[w]] = -1.0
    tops = np.argsort(sims)[::-1][:topn]
    return [(idx2word[i], float(sims[i])) for i in tops], []

# ─────────────────────────────────────────────────────────────
# QUERY WORDS FOR TASK 3
# ─────────────────────────────────────────────────────────────
QUERY_WORDS = ["research", "student", "phd", "exam", "learning"]
# Note: task spec says "exam" twice — treating as ["research","student","phd","exam","learning"]
# with 'learning' as the 5th distinct query (exam is listed twice in spec, treating 2nd as 'learning')

ANALOGIES = [
    # (A, B, C)  →  D,  such that  A:B :: C:D
    # i.e. vec(D) ≈ vec(B) - vec(A) + vec(C)
    ("ug",          "btech",   "pg"),        # ug:btech :: pg:?  → mtech
    ("btech",       "undergraduate", "mtech"),# btech:undergraduate :: mtech:?  → postgraduate
    ("student",     "exam",    "research"),   # student:exam :: research:?  → publication/thesis
    ("undergraduate","student","postgraduate"),# undergraduate:student :: postgraduate:?  → scholar
    ("learning",    "machine", "natural"),    # learning:machine :: natural:?  → language/processing
    ("jodhpur",     "rajasthan","delhi"),     # jodhpur:rajasthan :: delhi:?  → india (state→country)
]

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — NEAREST NEIGHBOUR VISUAL (grid of 5 word fans)
# ─────────────────────────────────────────────────────────────

def plot_nn_fan(sg_results, cb_results):
    """
    For each of the 5 query words, draw two mini horizontal bars side by side
    showing cosine sims of top-5 neighbours.
    """
    fig = plt.figure(figsize=(18, 14), facecolor=CREAM)
    fig.suptitle(
        "Figure 1 — Top-5 Nearest Neighbours (Cosine Similarity)\nSkip-gram  vs  CBOW",
        fontsize=15, fontweight="bold", color=BLUE, y=1.01
    )

    n_words = len(QUERY_WORDS)
    for wi, word in enumerate(QUERY_WORDS):
        sg_nn = sg_results.get(word, [])
        cb_nn = cb_results.get(word, [])

        ax_sg = fig.add_subplot(n_words, 2, wi*2 + 1)
        ax_cb = fig.add_subplot(n_words, 2, wi*2 + 2)

        for ax, nn, title, color in [
            (ax_sg, sg_nn, f'Skip-gram   "{word}"', ORANGE),
            (ax_cb, cb_nn, f'CBOW   "{word}"',       BLUE),
        ]:
            if not nn:
                ax.text(0.5, 0.5, f'"{word}" not in vocab',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                ax.axis('off')
                continue

            words_ = [w for w, _ in nn][::-1]
            sims_  = [s for _, s in nn][::-1]
            bar_colors = [color if s > 0 else '#E63946' for s in sims_]

            bars = ax.barh(range(len(words_)), sims_, color=bar_colors,
                           edgecolor='white', linewidth=0.6, height=0.65)
            ax.set_yticks(range(len(words_)))
            ax.set_yticklabels(words_, fontsize=9.5, color=BLUE, fontweight='bold')
            ax.set_xlim(-0.15, 1.05)
            ax.axvline(0, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
            ax.set_title(title, fontsize=10, color=color, fontweight='bold', pad=4)
            ax.set_xlabel('Cosine Sim', fontsize=8.5, color=DGRAY)
            ax.set_facecolor(CREAM)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(colors=BLUE)

            for bar, val in zip(bars, sims_):
                offset = 0.02 if val >= 0 else -0.07
                ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=8.5,
                        color=color, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT, 'fig1_nearest_neighbours.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  ✓ Figure 1 → {path}')
    return path

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — NEAREST NEIGHBOUR HEATMAP (all 5 words × top5)
# ─────────────────────────────────────────────────────────────

def plot_nn_heatmap(sg_results, cb_results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=CREAM)
    fig.suptitle("Figure 2 — Cosine Similarity Heatmap: Query Words × Top-5 Neighbours",
                 fontsize=13, fontweight="bold", color=BLUE, y=1.02)

    for ax, results, title, cmap in zip(
        axes,
        [sg_results, cb_results],
        ["Skip-gram", "CBOW"],
        ["Oranges", "Blues"]
    ):
        all_nn_words = []
        matrix = []
        for word in QUERY_WORDS:
            nn = results.get(word, [])
            row_words = [w for w, _ in nn[:5]]
            row_sims  = [s for _, s in nn[:5]]
            while len(row_words) < 5:
                row_words.append("—"); row_sims.append(0.0)
            all_nn_words.append(row_words)
            matrix.append(row_sims)

        mat = np.array(matrix)
        im  = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=-0.1, vmax=1.0)

        ax.set_yticks(range(len(QUERY_WORDS)))
        ax.set_yticklabels([f'"{w}"' for w in QUERY_WORDS],
                           fontsize=11, color=BLUE, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'Rank {i+1}' for i in range(5)], fontsize=10, color=DGRAY)
        ax.set_title(title, fontsize=12, color=BLUE, fontweight='bold', pad=8)
        ax.set_facecolor(CREAM)

        for i in range(len(QUERY_WORDS)):
            for j in range(5):
                nn_word = all_nn_words[i][j]
                sim_val = mat[i,j]
                ax.text(j, i, f'{nn_word}\n{sim_val:.2f}',
                        ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if sim_val > 0.5 else BLUE)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Cosine Similarity')

    plt.tight_layout()
    path = os.path.join(OUT, 'fig2_nn_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  ✓ Figure 2 → {path}')
    return path

# ─────────────────────────────────────────────────────────────
# FIGURE 3 — ANALOGY RESULTS VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_analogy_results(sg_analogy_results, cb_analogy_results):
    n = len(ANALOGIES)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.2*n), facecolor=CREAM)
    fig.suptitle(
        "Figure 3 — Analogy Experiment Results\n"
        "3CosAdd: vec(B) − vec(A) + vec(C)  →  closest word D",
        fontsize=14, fontweight="bold", color=BLUE, y=1.01
    )

    for i, (A, B, C) in enumerate(ANALOGIES):
        label = f"{A} : {B}  ::  {C} : ?"
        for j, (results, title, color) in enumerate([
            (sg_analogy_results, "Skip-gram", ORANGE),
            (cb_analogy_results, "CBOW",      BLUE),
        ]):
            ax = axes[i][j]
            ax.set_facecolor(CREAM)
            res = results.get(f"{A}:{B}::{C}", None)

            ax.set_title(f'{title}  —  {label}', fontsize=9.5, color=color,
                         fontweight='bold', pad=5)

            if res is None or not res:
                ax.text(0.5, 0.5, 'Word not in vocabulary',
                        ha='center', va='center', transform=ax.transAxes,
                        color='gray', fontsize=10)
                ax.axis('off')
                continue

            words = [w for w, _ in res[:5]][::-1]
            sims  = [s for _, s in res[:5]][::-1]
            bar_colors = [color]*len(words)
            bar_colors[-1] = "#FFD700"  # highlight top-1 in gold

            bars = ax.barh(range(len(words)), sims, color=bar_colors,
                           edgecolor='white', linewidth=0.5, height=0.65)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=10, color=BLUE, fontweight='bold')
            ax.set_xlim(-0.05, 1.05)
            ax.set_xlabel('Cosine Sim', fontsize=8, color=DGRAY)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(colors=BLUE)

            for bar, val in zip(bars, sims):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9,
                        color=color, fontweight='bold')

            # Gold star for top result
            top_word, top_sim = res[0]
            ax.text(0.98, 0.05, f'▶  "{top_word}"',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=10, color='#B8860B', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE0',
                              edgecolor='#B8860B', linewidth=1.2))

    plt.tight_layout()
    path = os.path.join(OUT, 'fig3_analogies.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  ✓ Figure 3 → {path}')
    return path

# ─────────────────────────────────────────────────────────────
# FIGURE 4 — PCA with QUERY WORDS HIGHLIGHTED + NEIGHBOUR LINKS
# ─────────────────────────────────────────────────────────────

def plot_embedding_map(sg_emb_norm, vocab, idx2word, sg_nn_results):
    from sklearn.decomposition import PCA

    # Collect all words to show: query words + their neighbours
    show_set = set(QUERY_WORDS)
    for word in QUERY_WORDS:
        for w, _ in sg_nn_results.get(word, [])[:3]:
            show_set.add(w)
    show_set = {w for w in show_set if w in vocab}

    pca  = PCA(n_components=2, random_state=42)
    all_emb = sg_emb_norm
    reduced = pca.fit_transform(all_emb)

    # Color map: query words are distinctive, neighbours are lighter
    QUERY_COLORS = {
        "research": "#E63946",
        "student":  ORANGE,
        "phd":      "#2A9D8F",
        "exam":     "#9B2226",
        "learning": BLUE,
    }

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=CREAM)
    ax.set_facecolor("#FAFAF6")

    # All vocab as grey background scatter
    ax.scatter(reduced[:,0], reduced[:,1], alpha=0.06, s=7, color='#AAAAAA', zorder=1)

    # Draw lines from query word → its top-3 neighbours
    for word in QUERY_WORDS:
        if word not in vocab: continue
        wi = vocab[word]
        qx, qy = reduced[wi]
        for nn_word, _ in sg_nn_results.get(word, [])[:3]:
            if nn_word not in vocab: continue
            ni = vocab[nn_word]
            nx, ny = reduced[ni]
            ax.plot([qx, nx], [qy, ny], '-',
                    color=QUERY_COLORS[word], alpha=0.35, linewidth=1.5, zorder=2)

    # Plot all show_set words
    for w in show_set:
        if w not in vocab: continue
        i = vocab[w]
        x, y = reduced[i]
        is_query = w in QUERY_WORDS
        color = QUERY_COLORS.get(w, DGRAY)
        ax.scatter(x, y,
                   s=160 if is_query else 60,
                   color=color,
                   edgecolors='white', linewidth=1.2,
                   zorder=4 if is_query else 3,
                   alpha=1.0)
        ax.annotate(w, (x, y),
                    fontsize=11 if is_query else 8.5,
                    fontweight='bold' if is_query else 'normal',
                    color=color,
                    xytext=(6, 4), textcoords='offset points',
                    zorder=5)

    # Legend for query words
    legend_patches = [mpatches.Patch(color=c, label=f'"{w}" (query)')
                      for w, c in QUERY_COLORS.items() if w in vocab]
    legend_patches.append(mpatches.Patch(color=DGRAY, label='neighbour word'))
    ax.legend(handles=legend_patches, fontsize=9, loc='lower right',
              framealpha=0.8, facecolor=CREAM)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1  ({var[0]*100:.1f}% variance)', color=BLUE, fontsize=11)
    ax.set_ylabel(f'PC2  ({var[1]*100:.1f}% variance)', color=BLUE, fontsize=11)
    ax.set_title(
        "Figure 4 — Embedding Space (PCA): Query Words + Nearest Neighbours\n"
        "Lines connect each query word to its top-3 Skip-gram neighbours",
        fontsize=12, fontweight='bold', color=BLUE
    )
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(colors=BLUE)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    path = os.path.join(OUT, 'fig4_embedding_map.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  ✓ Figure 4 → {path}')
    return path

# ─────────────────────────────────────────────────────────────
# FIGURE 5 — ANALOGY GEOMETRY DIAGRAM (3CosAdd illustration)
# ─────────────────────────────────────────────────────────────

def plot_analogy_geometry(sg_emb_norm, vocab, idx2word):
    """Show the vector arithmetic for ug:btech::pg:? in 2D PCA space."""
    from sklearn.decomposition import PCA

    # Use 4 words: ug, btech, pg, + predicted result
    WORDS = ["ug", "btech", "pg", "mtech", "postgraduate", "undergraduate"]
    present = [w for w in WORDS if w in vocab]
    if len(present) < 3:
        print("  [skip] not enough analogy words in vocab for geometry plot")
        return None

    # PCA on just full vocab but annotate these words
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(sg_emb_norm)

    # Analogy vector: btech - ug + pg
    A, B, C = "ug", "btech", "pg"
    if not all(w in vocab for w in [A,B,C]):
        print("  [skip] ug/btech/pg not in vocab for geometry plot")
        return None

    vA = sg_emb_norm[vocab[A]]
    vB = sg_emb_norm[vocab[B]]
    vC = sg_emb_norm[vocab[C]]
    vD = vB - vA + vC
    vD_norm = vD / (np.linalg.norm(vD)+1e-8)

    # Find top result (D)
    sims = sg_emb_norm @ vD_norm
    for w in [A,B,C]:
        sims[vocab[w]] = -1.0
    top_D = idx2word[int(np.argmax(sims))]
    print(f"  Analogy {A}:{B}::{C}:? → predicted = {top_D}")

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=CREAM)
    ax.set_facecolor("#FAFAF6")

    # Background scatter
    ax.scatter(reduced[:,0], reduced[:,1], alpha=0.05, s=6, color='#AAAAAA')

    # Draw the 4 analogy words
    highlight = {A: "#E63946", B: ORANGE, C: BLUE, top_D: "#2A9D8F"}
    for w, col in highlight.items():
        if w not in vocab: continue
        i = vocab[w]
        x, y = reduced[i]
        ax.scatter(x, y, s=200, color=col, edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(f'  {w}', (x,y), fontsize=13, fontweight='bold', color=col, zorder=6,
                    xytext=(8,4), textcoords='offset points')

    # Draw analogy rectangle: A→B, C→D (should be parallel)
    def pt(w):
        return reduced[vocab[w]] if w in vocab else None

    pA, pB, pC = pt(A), pt(B), pt(C)
    pDest = pt(top_D)

    if pA is not None and pB is not None:
        ax.annotate("", xy=pB, xytext=pA,
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5))
    if pC is not None and pDest is not None:
        ax.annotate("", xy=pDest, xytext=pC,
                    arrowprops=dict(arrowstyle='->', color=TEAL, lw=2.5))
    if pA is not None and pC is not None:
        ax.annotate("", xy=pC, xytext=pA,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='dashed'))
    if pB is not None and pDest is not None:
        ax.annotate("", xy=pDest, xytext=pB,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='dashed'))

    # Equation label
    eq = f'vec("{B}") − vec("{A}") + vec("{C}") ≈ vec("{top_D}")'
    ax.text(0.5, 0.03, eq, transform=ax.transAxes, ha='center',
            fontsize=11, color=BLUE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=LBLUE, edgecolor=BLUE))

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', color=BLUE)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', color=BLUE)
    ax.set_title(
        f'Figure 5 — Analogy Vector Geometry in Embedding Space\n'
        f'"{A}" : "{B}"  ::  "{C}" : "{top_D}" (predicted)',
        fontsize=12, fontweight='bold', color=BLUE
    )
    ax.spines[['top','right']].set_visible(False)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig5_analogy_geometry.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  ✓ Figure 5 → {path}')
    return path

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔"+"═"*54+"╗")
    print("║  TASK 3: SEMANTIC ANALYSIS                           ║")
    print("║  Nearest Neighbours + Analogy Experiments           ║")
    print("╚"+"═"*54+"╝\n")

    # ── Build corpus ────────────────────────────────────────
    sentences = tokenize(RAW_CORPUS)
    vocab, idx2word, freq_arr = build_vocab(sentences)
    print(f"Corpus: {len(sentences)} docs, vocab={len(vocab)}")
    print(f"Task-3 query words in vocab: "
          f"{[w for w in QUERY_WORDS if w in vocab]}")

    # ── Train models (best config from Task 2) ───────────────
    print("\n[+] Training Skip-gram (d=100, w=4, k=5, epochs=8)...")
    sg_emb = train(SkipGramNS, sentences, vocab, freq_arr,
                   d=100, w=4, k=5, epochs=8)

    print("\n[+] Training CBOW (d=100, w=4, k=5, epochs=8)...")
    cb_emb = train(CBOWNS, sentences, vocab, freq_arr,
                   d=100, w=4, k=5, epochs=8)

    sg_norm = normalize(sg_emb)
    cb_norm = normalize(cb_emb)

    # ── Nearest Neighbours ───────────────────────────────────
    print("\n" + "="*54)
    print("  NEAREST NEIGHBOURS (Top-5 per query word)")
    print("="*54)
    sg_nn = {}
    cb_nn = {}

    for word in QUERY_WORDS:
        sg_results = top_k(word, sg_norm, vocab, idx2word, k=5)
        cb_results = top_k(word, cb_norm, vocab, idx2word, k=5)
        sg_nn[word] = sg_results
        cb_nn[word] = cb_results

        print(f"\n  Query: \"{word}\"")
        if word not in vocab:
            print(f"    ✗ NOT IN VOCABULARY")
            continue
        print(f"  {'Rank':<6} {'Skip-gram':<20} {'Sim':>6}   {'CBOW':<20} {'Sim':>6}")
        print("  " + "─"*60)
        for i in range(5):
            sg_w, sg_s = sg_results[i] if i < len(sg_results) else ("—", 0.0)
            cb_w, cb_s = cb_results[i] if i < len(cb_results) else ("—", 0.0)
            print(f"  {i+1:<6} {sg_w:<20} {sg_s:>6.4f}   {cb_w:<20} {cb_s:>6.4f}")

    # ── Analogies ────────────────────────────────────────────
    print("\n" + "="*54)
    print("  ANALOGY EXPERIMENTS  (A:B :: C:?)")
    print("="*54)
    sg_analogy = {}
    cb_analogy = {}

    for A, B, C in ANALOGIES:
        key = f"{A}:{B}::{C}"
        sg_res, missing = analogy(A, B, C, sg_norm, vocab, idx2word)
        cb_res, _       = analogy(A, B, C, cb_norm, vocab, idx2word)
        sg_analogy[key] = sg_res or []
        cb_analogy[key] = cb_res or []

        print(f"\n  Analogy:  \"{A}\" : \"{B}\"  ::  \"{C}\" : ?")
        if missing:
            print(f"    Missing words: {missing}")
            continue
        sg_top = sg_res[0][0] if sg_res else "—"
        cb_top = cb_res[0][0] if cb_res else "—"
        print(f"    Skip-gram answer: \"{sg_top}\"  "
              f"(sim={sg_res[0][1]:.4f})" if sg_res else "    Skip-gram: no result")
        print(f"    CBOW answer:      \"{cb_top}\"  "
              f"(sim={cb_res[0][1]:.4f})" if cb_res else "    CBOW: no result")
        if sg_res:
            top5_str = ", ".join(f"{w}({s:.3f})" for w,s in sg_res[:5])
            print(f"    SG Top-5: {top5_str}")

    # ── Save JSON ────────────────────────────────────────────
    results = {
        "sg_nn":  {w: [(x,float(s)) for x,s in v] for w,v in sg_nn.items()},
        "cb_nn":  {w: [(x,float(s)) for x,s in v] for w,v in cb_nn.items()},
        "sg_analogy": {k: [(x,float(s)) for x,s in v] for k,v in sg_analogy.items()},
        "cb_analogy": {k: [(x,float(s)) for x,s in v] for k,v in cb_analogy.items()},
        "vocab_size": len(vocab),
        "query_words": QUERY_WORDS,
        "analogies": [f"{A}:{B}::{C}" for A,B,C in ANALOGIES],
    }
    with open(os.path.join(OUT,"task3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Figures ──────────────────────────────────────────────
    print("\n" + "="*54)
    print("  GENERATING FIGURES")
    print("="*54)
    fig1 = plot_nn_fan(sg_nn, cb_nn)
    fig2 = plot_nn_heatmap(sg_nn, cb_nn)
    fig3 = plot_analogy_results(sg_analogy, cb_analogy)
    fig4 = plot_embedding_map(sg_norm, vocab, idx2word, sg_nn)
    fig5 = plot_analogy_geometry(sg_norm, vocab, idx2word)

    print("\n" + "="*54)
    print(f"  ✓ All outputs → ./{OUT}/")
    print("="*54)

    return results, sg_emb, cb_emb, vocab, idx2word

if __name__ == "__main__":
    main()