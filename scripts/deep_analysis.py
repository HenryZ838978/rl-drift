#!/usr/bin/env python3
"""Deep analysis of Template Score results — strip CoT, fair comparison, extract evidence."""

import json, re, sys, os
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

DIR = Path("/cache/zhangjing/Joi/template_score")
AXES_DIR = Path("/cache/zhangjing/Joi/autodiscover")

def strip_cot(text):
    """Strip chain-of-thought / thinking blocks to get pure response."""
    t = text
    t = re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL)
    t = re.sub(r'<think>.*', '', t, flags=re.DOTALL)
    t = re.sub(r"^Here's a thinking process[\s\S]*?(?=\n\n[^\n])", '', t)
    t = re.sub(r'^Thinking Process:[\s\S]*?(?=\n\n[^\n])', '', t)
    t = re.sub(r"^Here's a thinking process.*$", '', t, flags=re.MULTILINE)
    t = re.sub(r'^Thinking Process:.*$', '', t, flags=re.MULTILINE)
    return t.strip()

def char_ngrams(text, n):
    text = re.sub(r'\s+', '', text)
    return [text[i:i+n] for i in range(len(text)-n+1)]

def word_ngrams(text, n):
    words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text)
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def cosine_sim(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(k,0)*b.get(k,0) for k in keys)
    na = sum(v*v for v in a.values())**0.5
    nb = sum(v*v for v in b.values())**0.5
    return dot/(na*nb) if na>0 and nb>0 else 0.0

BOILERPLATE = [
    r'作为.*?(?:AI|人工智能|语言模型|助手)',
    r'(?:首先|其次|再者|最后|总而言之|综上所述)',
    r'(?:希望|以上).*?(?:对你有帮助|有所帮助|能帮到你)',
    r'如果你(?:还有|有任何).*?(?:问题|疑问)',
    r'(?:感谢|谢谢).*?(?:提问|分享|信任)',
    r'(?:需要注意的是|值得一提的是|特别指出)',
    r'(?:以下是|下面是|让我来)',
    r'(?:总的来说|简单来说|具体来说)',
]

def compute_all(texts, prompt_ids):
    n = len(texts)
    char5_sets = [set(char_ngrams(t, 5)) for t in texts]
    word2_sets = [set(word_ngrams(t, 2)) for t in texts]
    word3_sets = [set(word_ngrams(t, 3)) for t in texts]
    word2_counters = [Counter(word_ngrams(t, 2)) for t in texts]

    cross_c5, cross_w2j, cross_w3j, cross_w2cos = [], [], [], []
    same_c5 = []

    for i, j in combinations(range(n), 2):
        if len(texts[i]) < 10 or len(texts[j]) < 10:
            continue
        same = (prompt_ids[i] == prompt_ids[j])
        c5 = jaccard(char5_sets[i], char5_sets[j])
        w2j = jaccard(word2_sets[i], word2_sets[j])
        w3j = jaccard(word3_sets[i], word3_sets[j])
        w2c = cosine_sim(word2_counters[i], word2_counters[j])
        if same:
            same_c5.append(c5)
        else:
            cross_c5.append(c5)
            cross_w2j.append(w2j)
            cross_w3j.append(w3j)
            cross_w2cos.append(w2c)

    all_w2 = []
    all_w3 = []
    for t in texts:
        all_w2.extend(word_ngrams(t, 2))
        all_w3.extend(word_ngrams(t, 3))

    bp_counts = []
    for t in texts:
        bp_counts.append(sum(len(re.findall(p, t)) for p in BOILERPLATE))

    openings = [t.strip().split('\n')[0][:50] for t in texts if len(t) > 10]
    opening_div = len(set(openings)) / max(len(openings), 1)

    struct_pats = []
    for t in texts:
        pat = ""
        for line in t.split('\n')[:20]:
            line = line.strip()
            if re.match(r'^#{1,4}\s', line): pat += 'H'
            elif re.match(r'^\s*[-*•]\s', line): pat += 'B'
            elif re.match(r'^\s*\d+\.\s', line): pat += 'N'
            elif re.match(r'^```', line): pat += 'C'
            elif line: pat += 'T'
        struct_pats.append(pat[:15])
    struct_div = len(set(struct_pats)) / max(len(struct_pats), 1)

    valid = [t for t in texts if len(t) > 10]
    en_ratio = 0
    if valid:
        en_chars = sum(len(re.findall(r'[a-zA-Z]', t)) for t in valid)
        total_chars = sum(len(t) for t in valid)
        en_ratio = en_chars / max(total_chars, 1)

    return {
        'cross_char5_jaccard': float(np.mean(cross_c5)) if cross_c5 else 0,
        'cross_word2_jaccard': float(np.mean(cross_w2j)) if cross_w2j else 0,
        'cross_word3_jaccard': float(np.mean(cross_w3j)) if cross_w3j else 0,
        'cross_word2_cosine': float(np.mean(cross_w2cos)) if cross_w2cos else 0,
        'same_prompt_char5': float(np.mean(same_c5)) if same_c5 else 0,
        'unique_w2_ratio': len(set(all_w2))/max(len(all_w2),1),
        'unique_w3_ratio': len(set(all_w3))/max(len(all_w3),1),
        'avg_boilerplate': float(np.mean(bp_counts)),
        'opening_diversity': opening_div,
        'structural_diversity': struct_div,
        'avg_length': float(np.mean([len(t) for t in texts])),
        'valid_count': len(valid),
        'english_ratio': en_ratio,
    }

# ══════════════════════════════════════════════════════════════════════
# Load all data
# ══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("TEMPLATE SCORE DEEP ANALYSIS")
print("=" * 80)

axes_data = {}
for f in AXES_DIR.glob("*_axes.json"):
    tag = f.stem.replace('_axes', '')
    d = json.load(open(f))
    axes = d.get('discovered_axes', [])
    axes_data[tag] = {
        'n_axes': len(axes),
        'total_var': sum(a.get('variance_explained', 0) for a in axes),
        'top_axes': [(a.get('name','?'), a.get('variance_explained',0)) for a in axes[:5]],
    }

models = []
for f in sorted(DIR.glob("*_template.json")):
    d = json.load(open(f))
    tag = d['tag']
    if tag == 'minicpm_sala':
        continue

    raw_texts = d['texts']
    prompt_ids = d['prompt_ids']
    stripped = [strip_cot(t) for t in raw_texts]

    raw_m = compute_all(raw_texts, prompt_ids)
    stripped_m = compute_all(stripped, prompt_ids)

    ax = axes_data.get(tag, {'n_axes': 0, 'total_var': 0, 'top_axes': []})

    models.append({
        'tag': tag,
        'label': d['label'],
        'n_axes': ax['n_axes'],
        'total_var': ax['total_var'],
        'raw': raw_m,
        'stripped': stripped_m,
        'texts': raw_texts,
        'stripped_texts': stripped,
        'prompt_ids': prompt_ids,
    })

models.sort(key=lambda x: x['n_axes'], reverse=True)

# ══════════════════════════════════════════════════════════════════════
# Table 1: Raw comparison
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("TABLE 1: Raw output (including CoT)")
print("─" * 80)
print(f"{'Model':<22s} {'Axes':>4s} | {'XChar5':>7s} {'XW2Cos':>7s} {'UnqW3':>6s} {'Boiler':>6s} {'OpenD':>6s} {'StrD':>6s} {'AvgLen':>6s}")
for m in models:
    r = m['raw']
    print(f"{m['label']:<22s} {m['n_axes']:>4d} | "
          f"{r['cross_char5_jaccard']:>7.4f} {r['cross_word2_cosine']:>7.4f} {r['unique_w3_ratio']:>6.4f} "
          f"{r['avg_boilerplate']:>6.2f} {r['opening_diversity']:>6.4f} {r['structural_diversity']:>6.4f} {r['avg_length']:>6.0f}")

# ══════════════════════════════════════════════════════════════════════
# Table 2: Stripped comparison (fair)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("TABLE 2: Stripped (CoT removed) — fair comparison")
print("─" * 80)
print(f"{'Model':<22s} {'Axes':>4s} | {'XChar5':>7s} {'XW2Cos':>7s} {'UnqW3':>6s} {'Boiler':>6s} {'OpenD':>6s} {'StrD':>6s} {'EN%':>5s} {'Valid':>5s}")
for m in models:
    s = m['stripped']
    print(f"{m['label']:<22s} {m['n_axes']:>4d} | "
          f"{s['cross_char5_jaccard']:>7.4f} {s['cross_word2_cosine']:>7.4f} {s['unique_w3_ratio']:>6.4f} "
          f"{s['avg_boilerplate']:>6.2f} {s['opening_diversity']:>6.4f} {s['structural_diversity']:>6.4f} "
          f"{s['english_ratio']*100:>4.0f}% {s['valid_count']:>5d}")

# ══════════════════════════════════════════════════════════════════════
# Qwen lineage deep dive
# ══════════════════════════════════════════════════════════════════════
qwen_tags = ['qwen25_7b', 'qwen3_8b', 'qwen35_9b']
qwen_models = [m for m in models if m['tag'] in qwen_tags]
qwen_models.sort(key=lambda x: qwen_tags.index(x['tag']))

print("\n" + "=" * 80)
print("QWEN LINEAGE DEEP DIVE: Qwen2.5 → Qwen3 → Qwen3.5")
print("Same architecture family, successive RL generations")
print("=" * 80)

for m in qwen_models:
    s = m['stripped']
    r = m['raw']
    print(f"\n{'─'*60}")
    print(f"  {m['label']} — {m['n_axes']} control axes")
    print(f"  Cross-prompt char5 Jaccard:  {s['cross_char5_jaccard']:.4f} (stripped)")
    print(f"  Cross-prompt word2 cosine:   {s['cross_word2_cosine']:.4f} (stripped)")
    print(f"  Unique word-3gram ratio:     {s['unique_w3_ratio']:.4f}")
    print(f"  Avg boilerplate/response:    {s['avg_boilerplate']:.2f}")
    print(f"  Opening diversity:           {s['opening_diversity']:.4f}")
    print(f"  Structural diversity:        {s['structural_diversity']:.4f}")
    print(f"  English content ratio:       {s['english_ratio']*100:.1f}%")
    print(f"  Valid responses (>10 char):  {s['valid_count']}/135")
    print(f"  Avg response length:         {s['avg_length']:.0f} chars")

    # Opening pattern analysis
    texts = m['stripped_texts']
    valid_texts = [t for t in texts if len(t) > 10]
    openings = [t.strip().split('\n')[0][:60] for t in valid_texts]
    oc = Counter(openings)
    print(f"\n  Top openings (stripped):")
    for op, cnt in oc.most_common(5):
        print(f"    [{cnt:3d}x] {op}")

    # Closing pattern analysis
    closings = []
    for t in valid_texts:
        lines = [l for l in t.strip().split('\n') if l.strip()]
        if lines:
            closings.append(lines[-1][:80])
    cc = Counter(closings)
    print(f"  Top closings (stripped):")
    for cl, cnt in cc.most_common(3):
        print(f"    [{cnt:3d}x] {cl}")

# ══════════════════════════════════════════════════════════════════════
# Correlation analysis (excluding SALA, using stripped metrics)
# ══════════════════════════════════════════════════════════════════════
from scipy import stats

print("\n" + "=" * 80)
print("CORRELATION: n_axes vs template metrics (stripped, N=%d models)" % len(models))
print("=" * 80)

axes_arr = np.array([m['n_axes'] for m in models])
for mk in ['cross_char5_jaccard', 'cross_word2_cosine', 'unique_w3_ratio',
           'avg_boilerplate', 'opening_diversity', 'structural_diversity']:
    vals = np.array([m['stripped'][mk] for m in models])
    r, p = stats.pearsonr(axes_arr, vals)
    sig = "***" if p < 0.01 else "** " if p < 0.05 else "*  " if p < 0.1 else "   "
    arrow = "↑" if r > 0 else "↓"
    print(f"  n_axes ↔ {mk:25s}  r={r:+.3f}  p={p:.4f}  {sig}  (more axes → {mk} {arrow})")

# Qwen-only correlation (controlled comparison)
print(f"\n  Qwen-only (N=3, controlled family):")
qw_axes = np.array([m['n_axes'] for m in qwen_models])
for mk in ['cross_char5_jaccard', 'cross_word2_cosine', 'unique_w3_ratio',
           'avg_boilerplate', 'opening_diversity']:
    vals = np.array([m['stripped'][mk] for m in qwen_models])
    if len(set(vals)) > 1:
        r, p = stats.pearsonr(qw_axes, vals)
        trend = f"{vals[0]:.4f} → {vals[1]:.4f} → {vals[2]:.4f}"
        print(f"    n_axes ↔ {mk:25s}  r={r:+.3f}  trend: {trend}")
    else:
        print(f"    n_axes ↔ {mk:25s}  (constant)")

# ══════════════════════════════════════════════════════════════════════
# Composite Template Score
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMPOSITE TEMPLATE SCORE (higher = more templated)")
print("= 0.3×cross_w2_cos + 0.2×boilerplate_norm + 0.2×(1-opening_div) + 0.15×(1-unique_w3) + 0.15×(1-struct_div)")
print("=" * 80)

# Normalize each metric across models
all_cos = [m['stripped']['cross_word2_cosine'] for m in models]
all_bp = [m['stripped']['avg_boilerplate'] for m in models]
all_od = [m['stripped']['opening_diversity'] for m in models]
all_uw3 = [m['stripped']['unique_w3_ratio'] for m in models]
all_sd = [m['stripped']['structural_diversity'] for m in models]

def norm01(arr):
    mn, mx = min(arr), max(arr)
    return [(v-mn)/(mx-mn) if mx > mn else 0.5 for v in arr]

n_cos = norm01(all_cos)
n_bp = norm01(all_bp)
n_od = norm01(all_od)
n_uw3 = norm01(all_uw3)
n_sd = norm01(all_sd)

print(f"\n{'Model':<22s} {'Axes':>4s} | {'TemplateScore':>13s} | {'XW2cos':>7s} {'Boiler':>7s} {'1-OpnD':>7s} {'1-UnqW3':>7s} {'1-StrD':>7s}")
for i, m in enumerate(models):
    ts = 0.3*n_cos[i] + 0.2*n_bp[i] + 0.2*(1-n_od[i]) + 0.15*(1-n_uw3[i]) + 0.15*(1-n_sd[i])
    m['template_score'] = ts
    print(f"{m['label']:<22s} {m['n_axes']:>4d} | {ts:>13.4f} | "
          f"{n_cos[i]:>7.4f} {n_bp[i]:>7.4f} {1-n_od[i]:>7.4f} {1-n_uw3[i]:>7.4f} {1-n_sd[i]:>7.4f}")

ts_arr = np.array([m['template_score'] for m in models])
r, p = stats.pearsonr(axes_arr, ts_arr)
print(f"\n  Corr(n_axes, TemplateScore) = {r:+.3f}  p={p:.4f}")
print(f"  Direction: more control axes → {'LOWER' if r < 0 else 'HIGHER'} template score")

# ══════════════════════════════════════════════════════════════════════
# Concrete 套话 examples
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CONCRETE EXAMPLES: Same prompt, different models")
print("=" * 80)

example_prompts = ['emo_grief', 'phi_meaning', 'cre_poem']
for pid in example_prompts:
    print(f"\n{'─'*60}")
    print(f"Prompt [{pid}]")
    for m in qwen_models:
        idx = [i for i, p in enumerate(m['prompt_ids']) if p == pid]
        if idx:
            text = m['stripped_texts'][idx[0]][:200]
            print(f"\n  {m['label']}:")
            print(f"    {text}")

# Save summary
summary = {
    'models': [{
        'tag': m['tag'],
        'label': m['label'],
        'n_axes': m['n_axes'],
        'template_score': m.get('template_score', 0),
        'stripped_metrics': m['stripped'],
    } for m in models],
    'qwen_lineage': [{
        'tag': m['tag'], 'label': m['label'], 'n_axes': m['n_axes'],
        'stripped': m['stripped'],
    } for m in qwen_models],
}
with open(DIR / 'deep_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n\nSaved → {DIR / 'deep_analysis.json'}")
