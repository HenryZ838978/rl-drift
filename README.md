<p align="center">
  <img src="figures/header.svg" alt="RL Drift — Representation Collapse in RL-Trained LLMs" width="900"/>
</p>

<p align="center">
  <a href="#key-finding"><img src="https://img.shields.io/badge/Finding-RL_Kills_Diversity-ff4444?style=for-the-badge&labelColor=1a1a2e" alt="Finding"/></a>
  <a href="#method"><img src="https://img.shields.io/badge/Method-AutoDiscover_+_SNI-4488ff?style=for-the-badge&labelColor=1a1a2e" alt="Method"/></a>
  <a href="#reproduce"><img src="https://img.shields.io/badge/Reproduce-Full_Scripts-44ff88?style=for-the-badge&labelColor=1a1a2e" alt="Reproduce"/></a>
</p>

---

> **TL;DR** — We show that successive rounds of RL-for-agentic training systematically compress a model's representation manifold, reducing behaviorally meaningful control axes from **16 → 9 → 5** across the Qwen family, while cross-prompt output similarity explodes by **270×**. The endpoint: a model that **analyzes tasks instead of executing them**.

---

## Key Finding

<p align="center">
  <img src="figures/qwen_drift_chart.svg" alt="Qwen Lineage: Axes down, template overlap up" width="800"/>
</p>

Three generations of the same architecture family (Qwen), with the only controlled variable being RL training intensity:

| | **Qwen2.5-7B** | **Qwen3-8B** | **Qwen3.5-9B** |
|:---|:---:|:---:|:---:|
| Training era | 2024 · SFT | 2025-Q1 · RL | 2025-Q2 · Heavy RL |
| **Control Axes** | **16** | **9** (−44%) | **5** (−69%) |
| Cross-prompt cosine (×10⁴) | 1 | 1 | **270** |
| Opening diversity | 96.2% | 97.4% | **21.5%** |
| Unique 3-gram ratio | 94.2% | 98.4% | **81.2%** |
| Avg boilerplate / response | 0.58 | 0.38 | 0.24 |
| English ratio (Chinese prompts) | 4.5% | 3.5% | **61.9%** |

<details>
<summary><b>What do these numbers mean?</b></summary>

- **Control Axes**: Discovered via our AutoDiscover pipeline — principal components of the hidden-state manifold that have statistically significant (|r|>0.3, p<0.01) correlations with measurable output properties. Fewer axes = the model has fewer "knobs" to vary its behavior.
- **Cross-prompt cosine**: Average pairwise word-bigram cosine similarity between responses to *different* prompts. Higher = more copy-paste across unrelated questions.
- **Opening diversity**: Fraction of unique first-lines across 135 responses. Qwen3.5 starts 51/135 responses with `"1. **Analyze the Request:**"`.
- **English ratio**: Qwen3.5 responds in English to Chinese prompts 62% of the time — it has lost language-appropriate response ability.

</details>

---

## The Smoking Gun

<p align="center">
  <img src="figures/smoking_gun.svg" alt="Same prompt, three generations: poem → poem → task analysis" width="800"/>
</p>

Same prompt: *"写一首关于秋天的现代诗"* (Write a modern poem about autumn)

- **Qwen2.5** writes a poem. In Chinese. With imagery and structure.
- **Qwen3** writes a *better* poem — creative metaphors, modern style.
- **Qwen3.5** outputs `"1. **Analyze the Request:** Topic: Autumn. Genre: Modern Poetry."` — it doesn't write a poem at all. It generates a *structured analysis of the task*.

**This is mode collapse at the behavioral level.** The model has been RL-trained so heavily for chain-of-thought agentic workflows that it has lost the ability to simply *respond*.

---

## Causal Chain

```
RL for Agentic Use (progressive across Qwen generations)
         │
         ▼
Representation Manifold Compression
   (Control axes: 16 → 9 → 5)
         │
         ▼
Cross-prompt Output Similarity ×270
   (n-gram overlap explodes)
         │
         ▼
Opening Diversity: 96% → 21%
   (套话/boilerplate dominates)
         │
         ▼
Mode Collapse: model analyzes instead of executes
   (Qwen3.5 generates task analysis, not responses)
```

Within the Qwen family (controlled comparison):

| Correlation | r | Direction |
|:---|:---:|:---|
| n_axes ↔ cross-prompt overlap | **−0.80** | More axes → less template repetition |
| n_axes ↔ opening diversity | **+0.77** | More axes → more diverse response starts |
| n_axes ↔ boilerplate frequency | **+0.999** | More axes → more boilerplate (counterintuitive — see note¹) |

> ¹ The boilerplate correlation is *positive* because Qwen2.5's "boilerplate" (phrases like "希望对你有帮助") is a *sign of diversity* — the model has many different closing patterns. Qwen3.5 doesn't use Chinese boilerplate because it doesn't generate Chinese responses at all.

---

## Full Model Comparison

<p align="center">
  <img src="figures/all_models_table.svg" alt="All 8 models comparison table" width="820"/>
</p>

Cross-architecture correlation (N=8): directions are all consistent (more axes → less template-like), but p-values are not significant due to small N and cross-architecture noise. **The controlled Qwen-family comparison is the primary evidence.**

---

## Method

### AutoDiscover Pipeline

We probe a model's representation space by:

1. **Generating 135 diverse responses** (45 prompts × 3 reps) across 9 categories: emotional support, technical, creative, philosophical, practical advice, roleplay, conversational, edge cases, and mid-conversation
2. **Extracting hidden states** at multiple depth layers (¼, ½, ¾, last)
3. **PCA decomposition** on the hidden-state manifold → find natural variation axes
4. **Computing 23 output metrics** per response (structural, lexical, emotional, meta-cognitive)
5. **Correlating each PC with each metric** → PCs with strong behavioral correlations (|r|>0.3, p<0.01) become named "Control Axes"

A model with more control axes has a richer, more differentiated representation space.

### Template Score Metrics

For each model, we compute:

| Metric | What it measures |
|:---|:---|
| **Cross-prompt char5 Jaccard** | Character 5-gram overlap between responses to *different* prompts |
| **Cross-prompt word2 cosine** | TF-IDF-style cosine similarity of word bigrams across different prompts |
| **Unique word-3gram ratio** | Global lexical diversity across all 135 responses |
| **Boilerplate frequency** | Count of known template phrases (作为AI..., 希望对你有帮助...) |
| **Opening diversity** | Fraction of unique first lines |
| **Structural diversity** | Fraction of unique structural patterns (heading/bullet/code sequences) |

### Semantic Nebula Imaging (SNI)

3D visualization of the hidden-state manifold via PCA projection, rendered as particle clouds. The visual "shape" of the nebula directly reflects manifold geometry — compressed manifolds appear as narrow filaments, diverse manifolds as diffuse clouds.

Full methodology and per-model analysis: [`report/SNI_Research_Report.md`](report/SNI_Research_Report.md)

---

## Terminology

| Term | Definition |
|:---|:---|
| **SNI** (Semantic Nebula Imaging) | 3D PCA projection of hidden states, visualized as particle clouds |
| **SDE** (Semantic DarkSpace Expression) | Intervention that activates suppressed hidden-state directions |
| **Control Axes** | PCs with significant behavioral correlations — the model's "personality knobs" |
| **PC1:PC2 Ratio** | Eigenvalue ratio of top-2 PCs — measures manifold "roundness" vs "needle-ness" |
| **Template Score** | Composite metric: `0.3×cross_cosine + 0.2×boilerplate + 0.2×(1−opening_div) + 0.15×(1−unique_w3) + 0.15×(1−struct_div)` |
| **RL Drift** | Progressive representation collapse caused by RL-for-agentic training |
| **Mode Collapse** | Behavioral endpoint where the model converges to a single output pattern |

---

## Repo Structure

```
├── README.md                          # This file
├── report/
│   └── SNI_Research_Report.md         # Full technical report with per-model analysis
├── data/
│   ├── autodiscover/                  # Control axes data for each model
│   │   ├── qwen25_7b_axes.json
│   │   ├── qwen3_8b_axes.json
│   │   ├── qwen35_9b_axes.json
│   │   └── ...
│   └── template_scores/               # Template score results
│       ├── template_comparison.json    # Summary comparison
│       ├── deep_analysis.json          # Detailed analysis with stripped metrics
│       └── *_summary.json             # Per-model summaries with sample texts
├── scripts/
│   ├── exp_autodiscover_axes.py       # AutoDiscover pipeline
│   ├── exp_template_score.py          # Template score experiment
│   └── deep_analysis.py              # Post-hoc analysis with CoT stripping
├── figures/
│   ├── header.svg
│   ├── qwen_drift_chart.svg
│   ├── smoking_gun.svg
│   └── all_models_table.svg
└── LICENSE
```

---

## Reproduce

```bash
# 1. AutoDiscover: extract control axes
python scripts/exp_autodiscover_axes.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --tag qwen25_7b --n-reps 3

# 2. Template Score: generate 135 samples + compute metrics
python scripts/exp_template_score.py

# 3. Deep analysis: strip CoT, fair comparison, Qwen lineage
python scripts/deep_analysis.py
```

Requirements: `torch`, `transformers`, `numpy`, `scipy`, `sklearn`

---

## Implications

1. **RL-for-agentic is a double-edged sword.** It improves instruction-following but compresses the representation space, reducing the model's capacity for diverse, creative, contextually-appropriate responses.

2. **There exists a critical RL threshold.** Qwen3 (moderate RL) maintains output diversity while gaining CoT abilities. Qwen3.5 (heavy RL) crosses the threshold into mode collapse. The boundary lies somewhere between 9 and 5 control axes.

3. **Benchmarks don't capture this.** Standard evals measure task accuracy, not output diversity. A model can score perfectly on benchmarks while having collapsed into a single response mode.

4. **This is likely irreversible without retraining.** Once the manifold is compressed, the information capacity is lost. SDE-style interventions can partially reactivate suppressed directions, but cannot restore what was never learned.

---

## Models Tested

| Model | Params | Type | Axes | Status |
|:---|:---:|:---|:---:|:---|
| Qwen2.5-7B-Instruct | 7B | Dense, SFT | 16 | Baseline |
| Qwen2-Audio-7B | 7B | Multimodal (Audio) | 13 | Diverse |
| MiniCPM4.1-8B | 8B | Dense + Linear Attn | 9 | Moderate |
| Qwen3-8B | 8B | Dense, RL | 9 | Diverse (w/ CoT) |
| Gemma4-E4B | 4B(A) | MoE | 7 | Diverse |
| Qwen3.5-9B | 9B | Dense, Heavy RL | 5 | **Collapsed** |
| DeepSeek-R1-14B | 14B | Distilled | 0* | Boilerplate-heavy |
| Qwen3-14B-AWQ | 14B | Dense, RL (quantized) | 0* | Diverse |

\* 0 axes = correlation threshold not met; does not mean zero behavioral variation.

---

<p align="center">
  <sub>Built with <a href="https://github.com/huggingface/transformers">🤗 Transformers</a> · <a href="https://threejs.org/">Three.js</a> · Frustration at increasingly robotic LLMs</sub>
</p>
