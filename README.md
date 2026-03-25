# NLU CSL7640 — Assignment 2

**Course:** Natural Language Understanding (CSL7640)
**Student:** Shahil Sharma | Roll No: B22CS048
**Institute:** IIT Jodhpur

---

## Overview

This repository contains solutions to two problems from Assignment 2 of CSL7640. Both problems explore core NLU concepts — learning meaningful word representations from raw text, and generating text sequences at the character level using recurrent architectures.

```
.
├── problem-1/               # Word Embeddings from IIT Jodhpur Corpus
│   ├── iitj_corpus/         # Scraped & preprocessed corpus data
│   │   ├── sentences.txt        # Tokenized sentences (input to Word2Vec)
│   │   ├── stats.json           # Corpus statistics
│   │   └── wordcloud_iitj.png   # Word frequency visualisation
│   └── source_code/
│       ├── task1.py             # Dataset preparation & web scraping
│       ├── task2_word2vec.py    # Word2Vec (Skip-gram + CBOW) from scratch
│       ├── task3_semantic_analysis.py  # Nearest neighbours & analogy experiments
│       ├── task-4.py            # PCA / t-SNE embedding visualisation
│       ├── task2_outputs/       # Plots, vocab, results from Task 2
│       └── task3_outputs/       # Plots and results from Task 3
│
└── problem-2/               # Character-level Name Generation with RNN Variants
    ├── TrainingNames.txt        # 1,000 Indian names dataset
    ├── VanillaRNN_generated.txt
    ├── BLSTM_generated.txt
    ├── RNNAttention_generated.txt
    ├── RNN_Name_Generation_Report.pdf
    └── source_code/
        ├── dataset.py           # Vocabulary & PyTorch Dataset
        ├── models.py            # All three model architectures
        ├── train.py             # Training loop for all models
        ├── generate.py          # Name generation via temperature sampling
        ├── evaluate.py          # Novelty rate, diversity, realism metrics
        └── make_report.py       # PDF report generator
```

---

## Problem 1 — Learning Word Embeddings from IIT Jodhpur Data

### Objective

Build a domain-specific word embedding model trained on text scraped from the IIT Jodhpur website, implement Word2Vec from scratch (no gensim/PyTorch), and perform semantic analysis on the resulting embeddings.

---

### Task 1 — Dataset Preparation (`task1.py`)

Scrapes text from multiple sections of the IIT Jodhpur website (academic regulations, department pages, research overview, institute information) and builds a clean corpus for Word2Vec training.

**Data sources:**
- Academic Regulations (UG & PG)
- Department pages: CSE, EE, ME, CE, Physics, Chemistry
- Research overview & projects
- Institute about/vision/administration pages

**Preprocessing pipeline:**
1. HTML boilerplate removal (nav, footer, sidebar)
2. Non-ASCII / Devanagari character stripping
3. Lowercasing and punctuation removal
4. NLTK tokenization with stopword filtering
5. Short token and numeric token removal

**Corpus statistics (from `iitj_corpus/stats.json`):**

| Property | Value |
|---|---|
| Documents | 7 |
| Total tokens | 707 |
| Vocabulary size | 423 |
| Avg. doc length | 101 tokens |
| Top words | research, department, institute, jodhpur, students, engineering |

**Outputs:** `iitj_corpus/sentences.txt`, `iitj_corpus/stats.json`, `iitj_corpus/wordcloud_iitj.png`

> **Note:** The script includes a fallback offline demo corpus in case `iitj.ac.in` is unreachable.

---

### Task 2 — Word2Vec Training (`task2_word2vec.py`)

Implements both **Skip-gram** and **CBOW** with **Negative Sampling** from scratch using NumPy only. 6 hyperparameter configurations are trained and compared.

**Architecture:**
- Two weight matrices: `W_in` (input/embedding) and `W_out` (output/context)
- Negative sampling with unigram frequency table (power = 0.75, Mikolov et al.)
- Linear learning rate decay per epoch

**Hyperparameter grid (6 configurations × 2 models = 12 runs):**

| Config | Embed Dim | Window | Neg Samples |
|---|---|---|---|
| 1 | 50 | 2 | 5 |
| 2 | 50 | 4 | 5 |
| 3 | 50 | 2 | 10 |
| 4 | 100 | 2 | 5 |
| 5 | 100 | 4 | 5 ← **best** |
| 6 | 100 | 4 | 10 |

**Training settings:** 5 epochs, learning rate = 0.025 (linear decay)

**Evaluation word pairs for cosine similarity:**
`research~machine`, `engineering~technology`, `students~faculty`, `learning~training`, `solar~energy`, `computer~science`, `jodhpur~rajasthan`, `phd~research`

**Outputs in `task2_outputs/`:**
- `fig1_loss_curves.png` — Loss curves for all 12 runs
- `fig2_final_loss.png` — Final loss comparison bar chart
- `fig3_similarities.png` — Cosine similarity heatmap across word pairs
- `fig4_pca_embeddings.png` — PCA projection of word embeddings
- `fig5_hyperparam_effects.png` — Effect of dim/window/neg_k on final loss
- `nearest_neighbors.json` — Top-5 nearest neighbours for key words
- `results_table.json` — All experiment metrics
- `vocab.json` — Vocabulary mapping
- `Task2_Word2Vec_Report.docx` — Full written report

---

### Task 3 — Semantic Analysis (`task3_semantic_analysis.py`)

Uses the best models from Task 2 (Skip-gram: d=100, w=4, k=5 and CBOW: d=100, w=4, k=5) for deeper semantic analysis. Trains for 8 epochs for better convergence.

**Experiments:**

1. **Nearest Neighbours** — Top-5 most similar words for query terms (research, engineering, students, learning, solar, computer, jodhpur, phd) using cosine similarity on L2-normalised embeddings.

2. **Analogy Experiments** — Vector arithmetic: `king - man + woman = queen` style queries applied to the IIT Jodhpur domain (e.g., `research - lab + students`, `solar - energy + water`).

3. **Embedding Space Visualisation** — t-SNE / PCA 2D projection of embedding space coloured by semantic cluster.

**Outputs in `task3_outputs/`:**
- `fig1_nearest_neighbours.png` — Fan-style nearest neighbour plot
- `fig2_nn_heatmap.png` — Similarity heatmap for neighbours
- `fig3_analogies.png` — Analogy result visualisation
- `fig4_embedding_map.png` — 2D embedding space map
- `fig5_analogy_geometry.png` — Vector geometry illustration
- `task3_results.json` — Full results

---

### Task 4 — Embedding Visualisation (`task-4.py`)

Standalone script for PCA and t-SNE visualisation of word clusters. Pre-defined clusters:
- **Departments:** cse, electrical, mechanical, biosciences, physics, mathematics
- **Academic Programs:** btech, mtech, phd, curriculum, semester, thesis
- **Campus Life:** hostel, library, sports, mess, laboratory, students

Supports both `method='pca'` and `method='tsne'` projection.

---

### Running Problem 1

**Install dependencies:**
```bash
pip install numpy matplotlib requests beautifulsoup4 nltk wordcloud pillow scikit-learn
```

**Run in order:**
```bash
cd problem-1/source_code

# Task 1 — build corpus
python task1.py

# Task 2 — train Word2Vec
python task2_word2vec.py

# Task 3 — semantic analysis
python task3_semantic_analysis.py

# Task 4 — embedding visualisation (requires trained model in memory; see script comments)
python task-4.py
```

---

## Problem 2 — Character-Level Name Generation Using RNN Variants

### Objective

Design and compare three recurrent sequence models for character-level Indian name generation. Train on 1,000 Indian names and evaluate using novelty, diversity, and realism metrics.

---

### Task 0 — Dataset (`TrainingNames.txt`)

1,000 Indian names (male and female) spanning Sanskrit-origin Hindu names, South Indian names (Tamil, Telugu, Kannada, Malayalam), and pan-Indian names. Stored one name per line.

**Statistics:**
| Property | Value |
|---|---|
| Total names | 1,000 |
| Vocabulary size | 29 (26 alpha + PAD/SOS/EOS) |
| Avg. name length | 7.4 characters |
| Min / Max length | 3 / 20 characters |

---

### Task 1 — Model Architectures (`models.py`)

Three models implemented from scratch in PyTorch:

#### 1. Vanilla RNN
```
Embedding(29→32) → RNN(32→128, 1 layer) → Dropout(0.3) → Linear(128→29)
```
- Trainable parameters: **25,405**
- Learning rate: 0.003

#### 2. Bidirectional LSTM (BLSTM)
```
Embedding(29→32) → BiLSTM(32→256, 2 layers) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→29)
```
- Trainable parameters: **598,717**
- Learning rate: 0.002

#### 3. RNN with Causal Attention
```
Embedding(29→32) → RNN(32→128) → CausalAttention(128→128) →
Concat(256) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→29)
```
Attention uses additive (Bahdanau-style) scoring with a causal mask to prevent attending to future positions.
- Trainable parameters: **91,197**
- Learning rate: 0.003

**All models:** batch size = 32, epochs = 30, optimizer = Adam, LR scheduler = StepLR(step=30, γ=0.5), loss = CrossEntropy (PAD ignored).

---

### Task 2 — Quantitative Evaluation

200 names generated per model (temperature = 0.8):

| Metric | Vanilla RNN | BLSTM | RNN + Attention |
|---|---|---|---|
| Novelty Rate (%) | 87.0 | **100.0** | 87.0 |
| Diversity (%) | **97.5** | 99.5 | 77.5 |
| Valid Names Ratio (%) | **100.0** | 16.5 | 99.5 |
| Realism Score (%) | 76.5 | 19.2 | **80.2** |
| Avg. Name Length | 6.84 | 22.04 | 5.75 |
| Best Val Loss | 1.8982 | 0.0014 | 1.9447 |

> BLSTM's 100% novelty and near-zero val loss are misleading — it memorises training data bidirectionally but cannot generate coherently in the autoregressive (left-to-right) mode required at inference time.

---

### Task 3 — Qualitative Analysis

**Vanilla RNN** — Good phonological authenticity. Sample: *Shebdam, Manandran, Abhikan, Chenkara, Rajandra, Ananda, Prashant*. Occasional over-long morpheme concatenations (e.g. *Anantraumadhan*).

**BLSTM** — Produces garbled sequences: *Qewwooomoiducq, Codygoducqeegfivdooqezezm*. 167/200 names exceed 15 characters. Root cause: train-generate gap from bidirectional processing.

**RNN + Attention** — Highest realism. Sample: *Anitabha, Sarstan, Vishila, Dhitan, Aayan, Anushit, Chilkarthan*. Slight mode-collapse tendency around the *Anit-* prefix family due to the attention mechanism reinforcing productive patterns.

**Winner: RNN + Attention** — best realism (80.2%), near-perfect validity (99.5%).

---

### Running Problem 2

**Install dependencies:**
```bash
pip install torch numpy matplotlib reportlab
```

**Run in order:**
```bash
cd problem-2/source_code

# Train all three models (saves checkpoints/)
python train.py

# Generate 200 names per model (saves generated_names/)
python generate.py

# Compute metrics and plots (saves evaluation/)
python evaluate.py

# Generate PDF report
python make_report.py
```

**File outputs:**

| File | Description |
|---|---|
| `checkpoints/*.pt` | Saved model weights |
| `checkpoints/vocab.json` | Character vocabulary |
| `plots/*_loss.png` | Training/validation loss curves |
| `generated_names/*.txt` | 200 generated names per model |
| `evaluation/quantitative_metrics.json` | Novelty, diversity, realism scores |
| `evaluation/qualitative_analysis.json` | Sample names, failure modes, bigrams |
| `evaluation/*.png` | Comparison bar charts and length distributions |
| `RNN_Name_Generation_Report.pdf` | Full project report |

---

## Requirements

**Problem 1:**
```
numpy
matplotlib
requests
beautifulsoup4
nltk
wordcloud
pillow
scikit-learn
```

**Problem 2:**
```
torch>=1.9
numpy
matplotlib
reportlab
```

Install all at once:
```bash
pip install numpy matplotlib requests beautifulsoup4 nltk wordcloud pillow scikit-learn torch reportlab
```

---

## Key Findings

**Problem 1:** Skip-gram and CBOW with d=100, window=4, k=5 negative samples give the best embeddings on the IIT Jodhpur domain corpus. Skip-gram generally captures rarer word relationships better; CBOW is faster and works well for frequent words.

**Problem 2:** For autoregressive character generation, unidirectional architectures (Vanilla RNN, RNN+Attention) outperform BiLSTM. Attention improves phonological realism at the cost of some diversity. BiLSTM should only be used for discriminative (not generative) NLU tasks.

---

## References

1. Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR.
2. Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS.
3. Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
4. Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
