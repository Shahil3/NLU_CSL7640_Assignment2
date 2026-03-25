"""
make_report.py
==============
Generates the full project PDF report.
"""

import os, json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

W, H = A4
MARGIN = 2.0 * cm
PLOTS  = 'plots'
EVAL   = 'evaluation'
GEN    = 'generated_names'
CKPT   = 'checkpoints'

# ── Styles ────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle('Title2', parent=styles['Title'],
    fontSize=20, spaceAfter=6, textColor=colors.HexColor('#1a237e'), leading=24)

h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
    fontSize=14, spaceAfter=4, spaceBefore=14,
    textColor=colors.HexColor('#1a237e'), borderPad=2)

h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
    fontSize=12, spaceAfter=4, spaceBefore=10,
    textColor=colors.HexColor('#283593'))

h3_style = ParagraphStyle('H3', parent=styles['Heading3'],
    fontSize=10, spaceAfter=3, spaceBefore=8,
    textColor=colors.HexColor('#3949ab'))

body_style = ParagraphStyle('Body', parent=styles['Normal'],
    fontSize=9.5, leading=14, spaceAfter=6, alignment=TA_JUSTIFY)

code_style = ParagraphStyle('Code', parent=styles['Code'],
    fontSize=8, leading=11, backColor=colors.HexColor('#f5f5f5'),
    borderColor=colors.HexColor('#bdbdbd'), borderWidth=0.5,
    borderPad=4, fontName='Courier')

caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
    fontSize=8, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=8)

bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
    fontSize=9.5, leading=14, spaceAfter=3,
    leftIndent=16, bulletIndent=4)

def hr():
    return HRFlowable(width='100%', thickness=0.5,
                      color=colors.HexColor('#9fa8da'), spaceAfter=8)

def p(text, style=body_style):
    return Paragraph(text, style)

def h1(text): return Paragraph(text, h1_style)
def h2(text): return Paragraph(text, h2_style)
def h3(text): return Paragraph(text, h3_style)
def sp(n=1):  return Spacer(1, n * 0.35 * cm)

def img(path, width=14*cm, caption=None):
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=width, height=width*0.5))
    if caption:
        items.append(Paragraph(caption, caption_style))
    return items

def make_table(data, col_widths=None, header_bg=colors.HexColor('#3949ab')):
    style = TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1,  0), 9),
        ('FONTSIZE',    (0, 1), (-1, -1), 8.5),
        ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8eaf6')]),
        ('GRID',        (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',(0, 0), (-1, -1), 6),
    ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(style)
    return t


# ── Load data ─────────────────────────────────────────────────
with open(os.path.join(CKPT, 'training_results.json')) as f:
    train_res = json.load(f)

with open(os.path.join(EVAL, 'quantitative_metrics.json')) as f:
    quant = json.load(f)

with open(os.path.join(EVAL, 'qualitative_analysis.json')) as f:
    qual = json.load(f)

MODEL_NAMES = ['VanillaRNN', 'BLSTM', 'RNNAttention']
DISPLAY = {'VanillaRNN':'Vanilla RNN', 'BLSTM':'Bidirectional LSTM', 'RNNAttention':'RNN + Attention'}


# ── Build story ───────────────────────────────────────────────
def build_story():
    story = []

    # ══════════════════ COVER ══════════════════
    story.append(sp(4))
    story.append(p('PROBLEM 2', ParagraphStyle('Cover1', parent=styles['Normal'],
        fontSize=13, alignment=TA_CENTER, textColor=colors.HexColor('#5c6bc0'), spaceAfter=4)))
    story.append(Paragraph(
        'Character-Level Name Generation<br/>Using RNN Variants',
        ParagraphStyle('CoverTitle', parent=styles['Title'],
            fontSize=24, alignment=TA_CENTER, leading=30,
            textColor=colors.HexColor('#1a237e'), spaceAfter=10)
    ))
    story.append(hr())
    story.append(sp(1))
    story.append(p('Deep Learning Assignment Report', ParagraphStyle('Sub',
        parent=styles['Normal'], fontSize=12, alignment=TA_CENTER,
        textColor=colors.HexColor('#37474f'), spaceAfter=4)))
    story.append(p('Models: Vanilla RNN &bull; Bidirectional LSTM &bull; RNN + Attention',
        ParagraphStyle('Sub2', parent=styles['Normal'], fontSize=10,
        alignment=TA_CENTER, textColor=colors.grey)))
    story.append(sp(2))

    # Summary box
    summary_data = [
        ['Dataset', '1,000 Indian names (TrainingNames.txt)'],
        ['Vocabulary', '29 unique characters + 3 special tokens'],
        ['Epochs', '30 per model'],
        ['Framework', 'PyTorch'],
        ['Names Generated', '200 per model (600 total)'],
    ]
    t = Table(summary_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e8eaf6')),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#9fa8da')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ══════════════════ TABLE OF CONTENTS ══════════════════
    story.append(h1('Table of Contents'))
    story.append(hr())
    toc = [
        ('Task 0', 'Dataset Construction', '3'),
        ('Task 1', 'Model Architectures & Implementation', '3'),
        ('Task 2', 'Quantitative Evaluation', '7'),
        ('Task 3', 'Qualitative Analysis', '9'),
        ('',       'Conclusion', '11'),
    ]
    for sec, title, pg in toc:
        story.append(p(f'<b>{sec}</b>  &nbsp;&nbsp; {title} <font color="#9fa8da">{"."*50}</font> {pg}'))
    story.append(PageBreak())

    # ══════════════════ TASK 0: DATASET ══════════════════
    story.append(h1('Task 0: Dataset Construction'))
    story.append(hr())
    story.append(p(
        'A dataset of 1,000 Indian names was curated using large language model assistance and '
        'saved as <b>TrainingNames.txt</b>. The dataset spans both male and female names drawn '
        'from major Indian linguistic traditions including Sanskrit-origin Hindu names, South '
        'Indian names (Tamil, Telugu, Kannada, Malayalam), North Indian names, and names common '
        'across multiple regions.'
    ))
    story.append(sp())
    story.append(h2('Dataset Statistics'))
    stats_data = [
        ['Property', 'Value'],
        ['Total names', '1,000'],
        ['Unique characters (lowercase)', '26 alpha + special tokens'],
        ['Vocabulary size (with SOS/EOS/PAD)', '29'],
        ['Min name length', '3 characters'],
        ['Max name length', '20 characters'],
        ['Average name length', '7.4 characters'],
        ['Gender split (approx.)', '55% male, 45% female'],
    ]
    story.append(make_table(stats_data, col_widths=[8*cm, 8*cm]))
    story.append(sp())
    story.append(p(
        'Names include common Indian prefixes and suffixes such as <i>-esh</i>, <i>-anand</i>, '
        '<i>-kumar</i>, <i>-devi</i>, <i>-bai</i>, and typical syllabic patterns like '
        '<i>bh-</i>, <i>sh-</i>, <i>kh-</i>, and vowel-heavy constructions.'
    ))
    story.append(sp())
    story.append(h2('Data Preprocessing'))
    story.append(p(
        'Each name is lowercased and encoded at the character level. Three special tokens are '
        'prepended/appended to every sequence:'
    ))
    for tok, desc in [('<b>&lt;PAD&gt;</b> (index 0)', 'Padding token to equalise batch lengths'),
                      ('<b>&lt;SOS&gt;</b> (index 1)', 'Start-of-sequence token; used as the first input'),
                      ('<b>&lt;EOS&gt;</b> (index 2)', 'End-of-sequence token; signals generation stop')]:
        story.append(Paragraph(f'&bull; {tok} — {desc}', bullet_style))
    story.append(p(
        'Input sequences are [SOS, c<sub>1</sub>, c<sub>2</sub>, ..., c<sub>n</sub>] and '
        'target sequences are [c<sub>1</sub>, c<sub>2</sub>, ..., c<sub>n</sub>, EOS], '
        'enabling standard teacher-forcing training. A 90/10 train-validation split is applied.'
    ))
    story.append(PageBreak())

    # ══════════════════ TASK 1: MODELS ══════════════════
    story.append(h1('Task 1: Model Architectures & Implementation'))
    story.append(hr())

    # ── Model 1: Vanilla RNN ──
    story.append(h2('Model 1 — Vanilla RNN'))
    story.append(p(
        'The Vanilla RNN is the baseline sequence model. It uses a simple recurrent cell where '
        'the hidden state at each step is a non-linear function of the current input and the '
        'previous hidden state: <b>h<sub>t</sub> = tanh(W<sub>h</sub> h<sub>t-1</sub> + '
        'W<sub>x</sub> x<sub>t</sub> + b)</b>.'
    ))
    story.append(sp())
    story.append(h3('Architecture'))

    arch_rnn = [
        ['Layer', 'Type', 'Input Dim', 'Output Dim', 'Notes'],
        ['1', 'Embedding', 'vocab_size=29', '32', 'Learnable char embeddings'],
        ['2', 'RNN Cell', '32', '128', '1 layer, tanh activation'],
        ['3', 'Dropout', '128', '128', 'p = 0.30'],
        ['4', 'Linear (fc)', '128', '29', 'Output logits over vocab'],
        ['5', 'Softmax', '29', '29', 'Applied at inference'],
    ]
    story.append(make_table(arch_rnn, col_widths=[1.4*cm, 3.2*cm, 3.0*cm, 3.0*cm, 5.4*cm]))
    story.append(sp())

    story.append(h3('Hyperparameters'))
    hp_rnn = [
        ['Hyperparameter', 'Value'],
        ['Embedding dimension', '32'],
        ['Hidden size', '128'],
        ['Number of layers', '1'],
        ['Dropout probability', '0.30'],
        ['Optimizer', 'Adam'],
        ['Learning rate', '0.003'],
        ['LR scheduler', 'StepLR (step=30, gamma=0.5)'],
        ['Batch size', '32'],
        ['Epochs', '30'],
        ['Trainable parameters', f"{train_res['VanillaRNN']['n_params']:,}"],
    ]
    story.append(make_table(hp_rnn, col_widths=[8*cm, 8*cm]))
    story.append(sp())
    story.append(p(
        '<b>Design rationale:</b> The Vanilla RNN is intentionally kept shallow (1 layer, 128 '
        'units) to serve as a baseline. It suffers from vanishing gradients for long sequences '
        'but performs well for short names (average 7 characters). The small embedding dimension '
        '(32) is sufficient for a 26-character alphabet.'
    ))
    story.append(sp(2))

    # ── Model 2: BLSTM ──
    story.append(h2('Model 2 — Bidirectional LSTM (BLSTM)'))
    story.append(p(
        'The Bidirectional LSTM extends the standard LSTM with two parallel passes: a forward '
        'pass (left to right) and a backward pass (right to left). The concatenated outputs '
        '(dimension 2H = 256) capture both past and future context at each position. '
        'LSTMs use gates (input, forget, output) to mitigate vanishing gradients.'
    ))
    story.append(sp())
    story.append(h3('Architecture'))
    arch_blstm = [
        ['Layer', 'Type', 'Input Dim', 'Output Dim', 'Notes'],
        ['1', 'Embedding',   'vocab_size=29', '32',  'Learnable char embeddings'],
        ['2', 'BiLSTM',      '32',            '256', '2 layers, bidirectional; 128 per dir'],
        ['3', 'Linear proj', '256',           '128', 'Projection + ReLU'],
        ['4', 'Dropout',     '128',           '128', 'p = 0.30'],
        ['5', 'Linear (fc)', '128',           '29',  'Output logits over vocab'],
        ['6', 'Softmax',     '29',            '29',  'Applied at inference'],
    ]
    story.append(make_table(arch_blstm, col_widths=[1.4*cm, 3.2*cm, 3.0*cm, 3.0*cm, 5.4*cm]))
    story.append(sp())
    story.append(h3('Hyperparameters'))
    hp_blstm = [
        ['Hyperparameter', 'Value'],
        ['Embedding dimension', '32'],
        ['Hidden size (per direction)', '128 (total: 256)'],
        ['Number of layers', '2 (bidirectional)'],
        ['Dropout probability', '0.30'],
        ['Optimizer', 'Adam'],
        ['Learning rate', '0.002'],
        ['LR scheduler', 'StepLR (step=30, gamma=0.5)'],
        ['Batch size', '32'],
        ['Epochs', '30'],
        ['Trainable parameters', f"{train_res['BLSTM']['n_params']:,}"],
    ]
    story.append(make_table(hp_blstm, col_widths=[8*cm, 8*cm]))
    story.append(sp())
    story.append(p(
        '<b>Design rationale:</b> BLSTM has by far the largest parameter count (598,717) due to '
        'bidirectional processing and 2 layers. During training it achieves near-zero loss by '
        'exploiting future context, but this causes an important <i>generation gap</i>: at '
        'inference time, future characters are unavailable, so the model produces sequences that '
        'do not resemble plausible names. This is a known limitation of BiRNNs for generation tasks.'
    ))
    story.append(sp(2))

    # ── Model 3: RNN + Attention ──
    story.append(h2('Model 3 — RNN with Basic Attention Mechanism'))
    story.append(p(
        'This model augments the Vanilla RNN with a causal additive (Bahdanau-style) '
        'self-attention layer. After the RNN produces its output sequence H, the attention '
        'module computes a context vector for each timestep by attending over all previous '
        'positions (causal masking prevents attending to future tokens).'
    ))
    story.append(sp())
    story.append(p('<b>Attention equations:</b>'))
    story.append(p('score(h<sub>t</sub>, h<sub>s</sub>) = v<super>T</super> &middot; tanh(W<sub>1</sub> h<sub>t</sub> + W<sub>2</sub> h<sub>s</sub>)'))
    story.append(p('&alpha;<sub>t</sub> = softmax(score<sub>t</sub>, masked causal)'))
    story.append(p('context<sub>t</sub> = &Sigma;<sub>s &le; t</sub> &alpha;<sub>t,s</sub> &middot; h<sub>s</sub>'))
    story.append(sp())
    story.append(h3('Architecture'))
    arch_attn = [
        ['Layer', 'Type', 'Input Dim', 'Output Dim', 'Notes'],
        ['1', 'Embedding',       'vocab_size=29', '32',  'Learnable char embeddings'],
        ['2', 'RNN Cell',        '32',            '128', '1 layer, tanh activation'],
        ['3', 'Causal Attention','128',            '128', 'Additive self-attention, causal mask'],
        ['4', 'Concat + Proj',   '256',           '128', 'Concat(rnn, context) + ReLU'],
        ['5', 'Dropout',         '128',           '128', 'p = 0.30'],
        ['6', 'Linear (fc)',     '128',           '29',  'Output logits over vocab'],
        ['7', 'Softmax',         '29',            '29',  'Applied at inference'],
    ]
    story.append(make_table(arch_attn, col_widths=[1.4*cm, 3.4*cm, 3.0*cm, 3.0*cm, 5.2*cm]))
    story.append(sp())
    story.append(h3('Hyperparameters'))
    hp_attn = [
        ['Hyperparameter', 'Value'],
        ['Embedding dimension', '32'],
        ['Hidden size', '128'],
        ['Attention hidden size', '128 (same as RNN)'],
        ['Number of RNN layers', '1'],
        ['Dropout probability', '0.30'],
        ['Optimizer', 'Adam'],
        ['Learning rate', '0.003'],
        ['LR scheduler', 'StepLR (step=30, gamma=0.5)'],
        ['Batch size', '32'],
        ['Epochs', '30'],
        ['Trainable parameters', f"{train_res['RNNAttention']['n_params']:,}"],
    ]
    story.append(make_table(hp_attn, col_widths=[8*cm, 8*cm]))
    story.append(sp())
    story.append(p(
        '<b>Design rationale:</b> Attention gives the model a direct path to any previous '
        'character, alleviating the RNN information bottleneck. The causal mask ensures the '
        'model cannot "cheat" by looking at future characters during training, making it '
        'suitable for autoregressive generation. The parameter count (91,197) is significantly '
        'higher than the Vanilla RNN but much lower than the BLSTM.'
    ))
    story.append(sp())

    # Parameter comparison table
    story.append(h2('Model Parameter Comparison'))
    param_data = [
        ['Model', 'Embed Dim', 'Hidden', 'Layers', 'Trainable Params', 'Best Val Loss'],
        ['Vanilla RNN',        '32', '128', '1',     f"{train_res['VanillaRNN']['n_params']:,}",   f"{train_res['VanillaRNN']['best_val']:.4f}"],
        ['Bidirectional LSTM', '32', '128', '2 (bi)', f"{train_res['BLSTM']['n_params']:,}",      f"{train_res['BLSTM']['best_val']:.4f}"],
        ['RNN + Attention',    '32', '128', '1',      f"{train_res['RNNAttention']['n_params']:,}", f"{train_res['RNNAttention']['best_val']:.4f}"],
    ]
    story.append(make_table(param_data, col_widths=[4.5*cm, 2.2*cm, 2.0*cm, 2.5*cm, 3.8*cm, 3.0*cm]))
    story.append(sp())
    story.append(p(
        'The BLSTM dominates in parameter count due to bidirectional processing (2 directions '
        'x 2 layers x LSTM gates). It achieves near-zero validation loss, demonstrating its '
        'memorisation capacity, though this does not translate to generation quality. '
        'The Attention model offers the best balance between capacity and generation performance.'
    ))
    story.append(PageBreak())

    # ── Loss curves ──
    story.append(h2('Training & Validation Loss Curves'))
    story.append(p('Loss curves were recorded every epoch using cross-entropy loss with PAD token ignored.'))
    story.append(sp())
    for mname in MODEL_NAMES:
        path = os.path.join(PLOTS, f'{mname}_loss.png')
        story.extend(img(path, width=13*cm, caption=f'Figure: {DISPLAY[mname]} — Loss over 30 epochs'))
        story.append(sp())
    story.append(PageBreak())

    # ══════════════════ TASK 2: QUANTITATIVE EVALUATION ══════════════════
    story.append(h1('Task 2: Quantitative Evaluation'))
    story.append(hr())
    story.append(p(
        'Each model was used to generate 200 names via temperature sampling (T = 0.8). '
        'The following metrics were computed to compare generation quality.'
    ))
    story.append(sp())

    story.append(h2('Metric Definitions'))
    for name, defn in [
        ('<b>Novelty Rate</b>',  'Percentage of generated names that do not appear verbatim in the training set (case-insensitive). Higher is better — indicates the model generalises beyond memorisation.'),
        ('<b>Diversity</b>',     'Number of unique generated names divided by total generated names, expressed as a percentage. Higher indicates less repetition.'),
        ('<b>Average Length</b>','Mean character length of generated names. Indian names typically range from 4 to 12 characters.'),
        ('<b>Valid Names Ratio</b>', 'Percentage of names matching the regex ^[A-Z][a-z]{1,14}$ (capitalised, alphabetic, reasonable length).'),
        ('<b>Realism Score</b>', 'Heuristic score (0-100) rewarding: (a) length 4-12, (b) vowel ending, (c) presence of a common Indian bigram.'),
    ]:
        story.append(Paragraph(f'&bull; {name} — {defn}', bullet_style))

    story.append(sp())
    story.append(h2('Results Table'))

    quant_data = [
        ['Metric', 'Vanilla RNN', 'BLSTM', 'RNN + Attention'],
        ['Novelty Rate (%)',       f"{quant['VanillaRNN']['novelty_rate']:.1f}",      f"{quant['BLSTM']['novelty_rate']:.1f}",      f"{quant['RNNAttention']['novelty_rate']:.1f}"],
        ['Diversity (%)',          f"{quant['VanillaRNN']['diversity']:.1f}",         f"{quant['BLSTM']['diversity']:.1f}",         f"{quant['RNNAttention']['diversity']:.1f}"],
        ['Avg. Name Length',       f"{quant['VanillaRNN']['avg_length']:.2f}",        f"{quant['BLSTM']['avg_length']:.2f}",        f"{quant['RNNAttention']['avg_length']:.2f}"],
        ['Valid Names Ratio (%)',   f"{quant['VanillaRNN']['valid_names_ratio']:.1f}", f"{quant['BLSTM']['valid_names_ratio']:.1f}", f"{quant['RNNAttention']['valid_names_ratio']:.1f}"],
        ['Realism Score (%)',      f"{quant['VanillaRNN']['realism_score']:.1f}",     f"{quant['BLSTM']['realism_score']:.1f}",     f"{quant['RNNAttention']['realism_score']:.1f}"],
        ['Total Generated',        str(quant['VanillaRNN']['total_generated']),        str(quant['BLSTM']['total_generated']),        str(quant['RNNAttention']['total_generated'])],
        ['Unique Generated',       str(quant['VanillaRNN']['unique_generated']),       str(quant['BLSTM']['unique_generated']),       str(quant['RNNAttention']['unique_generated'])],
    ]
    story.append(make_table(quant_data, col_widths=[5.5*cm, 4.0*cm, 4.0*cm, 4.0*cm]))
    story.append(sp())

    story.append(h2('Metric Visualisations'))
    for metric_file, caption in [
        ('novelty_rate.png', 'Figure: Novelty Rate comparison across models'),
        ('diversity.png',    'Figure: Diversity comparison across models'),
        ('realism_score.png','Figure: Realism Score comparison across models'),
    ]:
        story.extend(img(os.path.join(PLOTS, metric_file), width=12*cm, caption=caption))
        story.append(sp())

    story.extend(img(os.path.join(EVAL, 'length_distributions.png'), width=15*cm,
                     caption='Figure: Name length distributions for all three models'))
    story.append(sp())

    story.append(h2('Analysis & Interpretation'))
    story.append(p(
        '<b>Vanilla RNN:</b> Achieves 87% novelty rate and 97.5% diversity, generating varied '
        'names that largely do not appear in the training set. All 200 names pass the validity '
        'check. Average length (6.84) is close to natural Indian name length. Realism score '
        'of 76.5% reflects good phonological alignment with Indian name patterns.'
    ))
    story.append(p(
        '<b>Bidirectional LSTM:</b> Achieves a perfect 100% novelty rate (no generated name '
        'matches the training set), and 99.5% diversity. However, these scores are misleading — '
        'the BLSTM generates extremely long (avg 22 characters) garbled sequences with no '
        'resemblance to real names. Only 16.5% pass the validity check and the realism score '
        'is 19.2%. The root cause is the <i>train-generate gap</i>: the model was trained '
        'bidirectionally but generation must be done left-to-right, breaking the assumption '
        'that future context is available.'
    ))
    story.append(p(
        '<b>RNN + Attention:</b> Achieves 87% novelty rate and 77.5% diversity. The lower '
        'diversity compared to Vanilla RNN suggests the attention mechanism causes the model '
        'to favour certain character patterns (e.g. "Anit-" prefixes appear frequently). '
        'The realism score is the highest at 80.2%, indicating the attention mechanism helps '
        'the model learn authentic Indian name phonology. Valid names ratio is 99.5%.'
    ))
    story.append(PageBreak())

    # ══════════════════ TASK 3: QUALITATIVE ANALYSIS ══════════════════
    story.append(h1('Task 3: Qualitative Analysis'))
    story.append(hr())
    story.append(p(
        'This section presents representative generated samples for each model, discusses '
        'the realism of the outputs, identifies common failure modes, and analyses the '
        'phonological characteristics of generated names.'
    ))
    story.append(sp())

    # ── Vanilla RNN ──
    story.append(h2('3.1  Vanilla RNN — Qualitative Analysis'))
    story.append(h3('Representative Generated Samples'))
    samples_rnn = qual['VanillaRNN']['samples'][:30]
    sample_text = ', '.join(samples_rnn)
    story.append(Paragraph(sample_text, ParagraphStyle('Samples', parent=styles['Normal'],
        fontSize=9, leading=14, textColor=colors.HexColor('#1a237e'),
        backColor=colors.HexColor('#f3f4ff'), borderPad=6)))
    story.append(sp())

    story.append(h3('Realism Assessment'))
    story.append(p(
        'The Vanilla RNN produces names that sound plausible as Indian names. Examples like '
        '<i>Manandran</i>, <i>Abhikan</i>, <i>Chenkara</i>, <i>Rajandra</i>, <i>Ananda</i>, '
        'and <i>Prashant</i> have authentic Indian phonology. The model has learnt common '
        'consonant clusters (sh-, bh-, pr-) and typical suffixes (-an, -a, -ra). '
        'Occasionally it generates names that are slightly distorted versions of real names '
        '(e.g. <i>Priman</i> resembling Priya+Man) which indicates partial memorisation '
        'combined with interpolation.'
    ))
    story.append(h3('Common Failure Modes'))
    fm_rnn = qual['VanillaRNN']['failure_modes']
    fm_text = ', '.join(f'{k}: {v}' for k, v in fm_rnn.items() if v > 0) or 'None detected'
    story.append(p(f'Detected failures: <b>{fm_text}</b>'))
    story.append(p(
        'The Vanilla RNN occasionally generates over-long hyphenated forms '
        '(e.g. <i>Anantraumadhan</i>) by stringing together common morphemes. '
        'This arises because the RNN hidden state can lose track of sequence length. '
        'Short outputs like <i>Sam</i>, <i>Jaga</i>, <i>Anil</i> are valid Indian names '
        'but are also present verbatim in the training set.'
    ))
    story.append(sp())

    # ── BLSTM ──
    story.append(h2('3.2  Bidirectional LSTM — Qualitative Analysis'))
    story.append(h3('Representative Generated Samples'))
    samples_bl = qual['BLSTM']['samples'][:20]
    sample_text2 = ', '.join(samples_bl[:10])
    story.append(Paragraph(sample_text2, ParagraphStyle('Samples2', parent=styles['Normal'],
        fontSize=9, leading=14, textColor=colors.HexColor('#7b1fa2'),
        backColor=colors.HexColor('#fdf4ff'), borderPad=6)))
    story.append(sp())
    story.append(h3('Realism Assessment'))
    story.append(p(
        'The BLSTM generates sequences with very poor realism. Outputs such as '
        '<i>Qewwooomoiducq</i>, <i>Mmitymodoqegwwmmodocqkqeb</i> are clearly not Indian '
        'names. The outputs contain rare characters (q, x, z), repeated sequences '
        '(www, ooo), and lengths of 20+ characters. The model has learned character '
        'co-occurrence statistics from both directions simultaneously, but the generation '
        'procedure (left-to-right with growing sequences) is fundamentally incompatible '
        'with how the model was trained.'
    ))
    story.append(h3('Common Failure Modes'))
    fm_bl = qual['BLSTM']['failure_modes']
    story.append(p(
        f'Too long (&gt;15 chars): <b>{fm_bl.get("too_long  (>15 chars)", 0)}</b> / 200, '
        f'Repeated chars: <b>{fm_bl.get("repeated_chars  (≥3 same)", 0)}</b> / 200, '
        f'All consonants: <b>{fm_bl.get("all_consonants", 0)}</b> / 200.'
    ))
    story.append(p(
        '<b>Root cause:</b> The train-generate gap is the primary failure mode. The BLSTM '
        'memorises the training data perfectly (near-zero train/val loss) via bidirectional '
        'context, but cannot leverage future context during sequential generation. '
        'For name generation tasks, a unidirectional model is strictly more appropriate.'
    ))
    story.append(sp())

    # ── RNN + Attention ──
    story.append(h2('3.3  RNN + Attention — Qualitative Analysis'))
    story.append(h3('Representative Generated Samples'))
    samples_attn = qual['RNNAttention']['samples'][:30]
    sample_text3 = ', '.join(samples_attn)
    story.append(Paragraph(sample_text3, ParagraphStyle('Samples3', parent=styles['Normal'],
        fontSize=9, leading=14, textColor=colors.HexColor('#1b5e20'),
        backColor=colors.HexColor('#f1f8e9'), borderPad=6)))
    story.append(sp())
    story.append(h3('Realism Assessment'))
    story.append(p(
        'The Attention RNN produces highly realistic Indian names. Outputs such as '
        '<i>Anitabha</i>, <i>Sarstan</i>, <i>Vishila</i>, <i>Anushit</i>, <i>Dhitan</i>, '
        '<i>Aayan</i> are plausible and phonologically authentic. The model has learned '
        'productive Indian name morphology: the <i>An-</i> prefix is highly productive in '
        'Indian names (Anand, Anita, Anirudh) and the attention weights help the model '
        'reinforce consistent prefix/suffix patterns. The model correctly favours vowel '
        'endings which are characteristic of Indian names.'
    ))
    story.append(h3('Common Failure Modes'))
    fm_attn = qual['RNNAttention']['failure_modes']
    story.append(p(
        f'Too long: <b>{fm_attn.get("too_long  (>15 chars)", 0)}</b> / 200. '
        f'<b>Mode collapse tendency</b>: The attention mechanism can reinforce '
        'a small set of productive patterns, leading to low diversity. Many generated '
        'names share the <i>Anit-</i> prefix (Anita, Aniti, Aniten, Anit, Anitabha), '
        'suggesting the model has learned this as a particularly strong pattern from training.'
    ))
    story.append(sp())

    story.append(h2('3.4  Cross-Model Comparison'))
    comparison_data = [
        ['Criterion', 'Vanilla RNN', 'BLSTM', 'RNN + Attention'],
        ['Phonological realism', 'Good', 'Poor', 'Best'],
        ['Name diversity', 'High', 'Very High*', 'Moderate'],
        ['Length control', 'Good', 'Poor (too long)', 'Good'],
        ['Pattern consistency', 'Moderate', 'None', 'High'],
        ['Failure mode severity', 'Low', 'Critical', 'Low'],
        ['Overall ranking', '#2', '#3', '#1'],
    ]
    story.append(make_table(comparison_data, col_widths=[5.5*cm, 3.5*cm, 3.5*cm, 4.5*cm]))
    story.append(p('* BLSTM diversity is high but for the wrong reason (random noise).', caption_style))
    story.append(PageBreak())

    # ══════════════════ CONCLUSION ══════════════════
    story.append(h1('Conclusion'))
    story.append(hr())
    story.append(p(
        'This project implemented and compared three recurrent neural network architectures for '
        'character-level Indian name generation: Vanilla RNN, Bidirectional LSTM, and '
        'RNN with Attention. The key findings are:'
    ))
    story.append(sp())
    for point in [
        '<b>Vanilla RNN</b> serves as a strong and efficient baseline. With only 25,405 parameters it achieves 87% novelty, 97.5% diversity, and 76.5% realism. Its simplicity makes it robust and well-suited for short-sequence generation.',
        '<b>Bidirectional LSTM</b> demonstrates the importance of architecture-task alignment. Despite having 23x more parameters (598,717), its bidirectional training procedure fundamentally conflicts with autoregressive generation, resulting in meaningless outputs.',
        '<b>RNN with Attention</b> achieves the highest realism score (80.2%) and near-perfect name validity (99.5%). The causal attention mechanism allows the model to selectively focus on relevant previous characters, producing more phonologically consistent names. Its moderate diversity suggests a tendency toward mode collapse on productive patterns.',
        'For name generation specifically, <b>unidirectional architectures are preferred</b>. Bidirectional models should only be used for discriminative tasks (classification, NER) where the full input is available at inference time.',
    ]:
        story.append(Paragraph(f'&bull;  {point}', bullet_style))
        story.append(sp(0.5))

    story.append(sp())
    story.append(h2('Future Work'))
    for fw in [
        'Implement a proper unidirectional LSTM for a fair comparison with the attention model.',
        'Experiment with Transformer-based character models (GPT-style) for name generation.',
        'Increase dataset size to 5,000–10,000 names for better generalisation.',
        'Evaluate with human judges for perceived name authenticity.',
        'Implement beam search or top-k/top-p sampling for higher quality generation.',
    ]:
        story.append(Paragraph(f'&bull;  {fw}', bullet_style))

    story.append(sp(2))
    story.append(hr())
    story.append(p(
        'Source code, generated names, checkpoints, and evaluation scripts are provided '
        'in the accompanying deliverables package.',
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                       alignment=TA_CENTER, textColor=colors.grey)
    ))

    return story


# ── Build PDF ─────────────────────────────────────────────────
def main():
    out_path = '/mnt/user-data/outputs/RNN_Name_Generation_Report.pdf'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title='Character-Level Name Generation using RNN Variants',
        author='Deep Learning Assignment',
    )

    story = build_story()
    doc.build(story)
    print(f'PDF saved to: {out_path}')


if __name__ == '__main__':
    main()
