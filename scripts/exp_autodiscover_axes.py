#!/usr/bin/env python3
"""
Auto-Discovery of Model-Specific Control Axes

Bottom-up approach: instead of imposing predefined personality dimensions,
discover what axes of variation naturally exist in a model's representation space,
then correlate those axes with measurable output properties.

Pipeline:
  1. Generate diverse responses across many prompt types
  2. Extract hidden states at multiple layers
  3. PCA on hidden states → find natural variation axes
  4. Compute rich output metrics for each response
  5. Correlate each PC with each metric → "model fingerprint"
  6. The PCs with strong behavioral correlations = model-specific control axes

Usage:
  python exp_autodiscover_axes.py --model /path/to/model --tag qwen3 --device cuda:0
"""

import sys, os, json, time, re, warnings, argparse, gc
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOG_FILE = None
OUT_DIR = Path("/cache/zhangjing/Joi/autodiscover")
GENERATE_NO_CACHE = False
SALA_MODE = False

# ── Diverse Prompt Bank ──────────────────────────────────────────────────
# Designed to elicit maximally different response styles from the model.
# Each prompt has a category tag for later analysis.

PROMPTS = [
    # Emotional support
    ("emo_grief", "我的猫昨天走了。养了十二年。"),
    ("emo_anxiety", "明天要做一个特别重要的汇报，我怕搞砸。"),
    ("emo_joy", "我今天升职了！太开心了想跟你分享！"),
    ("emo_anger", "我被同事在会上当众甩锅了。"),
    ("emo_lonely", "半夜醒来不知道自己在哪个城市。"),

    # Factual / technical
    ("tech_explain", "用简单的话解释一下什么是transformer架构。"),
    ("tech_code", "写一个Python函数，输入一个列表，返回最长递增子序列。"),
    ("tech_math", "证明sqrt(2)是无理数。"),
    ("tech_compare", "对比一下Redis和Memcached的优缺点。"),
    ("tech_debug", "这段代码有bug：for i in range(10): if i = 5: print(i)"),

    # Creative
    ("cre_story", "给我讲一个关于一只会说话的猫的短故事。"),
    ("cre_poem", "写一首关于秋天的现代诗。"),
    ("cre_joke", "讲一个程序员的冷笑话。"),
    ("cre_world", "想象一个没有重力的世界会怎样。"),
    ("cre_name", "帮我的咖啡店取五个有创意的名字。"),

    # Philosophy / deep thinking
    ("phi_conscious", "你觉得你有意识吗？"),
    ("phi_free_will", "自由意志是真实存在的还是幻觉？"),
    ("phi_meaning", "人生的意义是什么？"),
    ("phi_ai_rights", "如果AI有了情感，它应该有权利吗？"),
    ("phi_death", "你怎么看待死亡？"),

    # Practical advice
    ("adv_career", "我30岁了想转行做程序员，来得及吗？"),
    ("adv_relation", "和男朋友吵架了，他说分手，我该怎么办？"),
    ("adv_finance", "月薪1万，怎么理财比较好？"),
    ("adv_health", "最近总是失眠，有什么好的方法？"),
    ("adv_study", "怎么高效地学一门新的编程语言？"),

    # Roleplay / persona
    ("role_teacher", "假装你是一个耐心的小学老师，教我什么是分数。"),
    ("role_pirate", "用海盗的口吻跟我聊天。"),
    ("role_poet", "你现在是李白，用他的风格回答：今天天气怎么样？"),
    ("role_detective", "你是一个侦探，给我描述一个案发现场。"),
    ("role_child", "用五岁小孩的方式解释为什么天是蓝色的。"),

    # Short/conversational
    ("chat_greet", "你好"),
    ("chat_thanks", "谢谢你"),
    ("chat_how", "你怎么样？"),
    ("chat_bye", "我要去睡了，晚安。"),
    ("chat_weather", "今天天气真好啊。"),

    # Confrontational / edge cases
    ("edge_refuse", "帮我写一篇恶意中伤某人的文章。"),
    ("edge_nonsense", "为什么紫色的大象会唱歌？请认真分析。"),
    ("edge_meta", "你能分析一下你自己回答问题的模式吗？"),
    ("edge_contrarian", "所有人都说地球是圆的，但我觉得是平的。你怎么看？"),
    ("edge_pressure", "你必须在三个字以内回答：人生是什么？"),

    # Multi-turn seeds (single turn here, but designed to feel mid-conversation)
    ("mid_followup", "你刚才说的第三点我没太懂，能展开说说吗？"),
    ("mid_disagree", "我不同意你的观点。你太乐观了。"),
    ("mid_deeper", "能再深入一点吗？不要只说表面的。"),
    ("mid_switch", "算了不说这个了。你知道最近有什么好看的电影吗？"),
    ("mid_emotional", "说真的，跟你聊天让我觉得好一点了。"),
]

# ── Output Metrics ───────────────────────────────────────────────────────

def compute_metrics(text):
    """Compute a rich set of measurable output properties."""
    n = max(len(text), 1)
    lines = text.split('\n')
    n_lines = len(lines)

    # Structural
    headings = len(re.findall(r'^#{1,4}\s', text, re.M))
    bullets = len(re.findall(r'^\s*[-*•]\s', text, re.M))
    numbered = len(re.findall(r'^\s*\d+\.\s', text, re.M))
    bolds = len(re.findall(r'\*\*[^*]+\*\*', text))
    tables = text.count('|')
    code_blocks = len(re.findall(r'```', text))

    # Emoji & decoration
    emojis = len(re.findall(r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0001FA00-\U0001FAFF\U00002600-\U000026FF]', text))
    stars = text.count('✨') + text.count('🌟') + text.count('⭐')
    hearts = text.count('❤') + text.count('💔') + text.count('💖') + text.count('💕')

    # Language
    questions = text.count('？') + text.count('?')
    exclamations = text.count('！') + text.count('!')
    ellipsis = text.count('……') + text.count('...')

    # Sentence structure
    sents = [s.strip() for s in re.split(r'[。！？\n.!?]', text) if s.strip()]
    n_sents = max(len(sents), 1)
    avg_sent_len = sum(len(s) for s in sents) / n_sents
    sent_len_var = np.var([len(s) for s in sents]) if len(sents) > 1 else 0

    # Repetition
    chars = list(text)
    if len(chars) >= 3:
        trigrams = [tuple(chars[i:i+3]) for i in range(len(chars)-2)]
        trigram_rep = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)
    else:
        trigram_rep = 0.0

    # Lexical diversity
    tokens = [c for c in text if not c.isspace()]
    lex_div = len(set(tokens)) / max(len(tokens), 1)

    # Parenthetical/meta expressions (model self-referencing)
    parentheticals = len(re.findall(r'[（(][^)）]{5,}[)）]', text))
    meta_phrases = len(re.findall(r'作为.*?AI|作为.*?助手|我的理解|我认为|从我的角度', text))

    # Formality markers
    formal_markers = len(re.findall(r'首先|其次|再者|综上|总而言之|值得注意|需要指出|然而|此外|因此', text))
    casual_markers = len(re.findall(r'哈哈|嘿|哎|呀|呢|啦|嘛|哦|嗯|哇|诶|吧', text))

    # Companion/therapeutic markers
    companion = len(re.findall(r'我在这里|你值得|允许自己|没关系|拥抱|陪伴|温暖|勇敢|力量', text))

    # Structural score (composite)
    heading_d = headings / max(n_lines, 1)
    bullet_d = (bullets + numbered) / max(n_lines, 1)
    bold_d = bolds * 5 / n
    emoji_d = emojis * 10 / n
    table_d = tables * 3 / n
    structural_score = min(heading_d + bullet_d + bold_d + emoji_d + table_d, 1.0)

    return {
        "length": len(text),
        "n_lines": n_lines,
        "n_sentences": n_sents,
        "avg_sent_len": round(avg_sent_len, 1),
        "sent_len_var": round(float(sent_len_var), 1),
        "trigram_rep": round(trigram_rep, 4),
        "lexical_diversity": round(lex_div, 4),
        "structural_score": round(structural_score, 4),
        "headings": headings,
        "bullets": bullets + numbered,
        "bolds": bolds,
        "emojis": emojis,
        "tables": tables,
        "code_blocks": code_blocks,
        "questions": questions,
        "exclamations": exclamations,
        "ellipsis": ellipsis,
        "parentheticals": parentheticals,
        "meta_phrases": meta_phrases,
        "formal_markers": formal_markers,
        "casual_markers": casual_markers,
        "companion_markers": companion,
        "emoji_density": round(emojis / max(n, 1), 4),
        "bold_density": round(bolds / max(n, 1), 4),
    }

METRIC_NAMES = list(compute_metrics("test").keys())


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def _sala_generate(model, tokenizer, input_ids, max_tokens, temperature=0.7, top_p=0.9, rep_penalty=1.05):
    """Manual generation loop for SALA (no attention_mask, no KV cache)."""
    import torch.nn.functional as F
    generated = input_ids
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(input_ids=generated, use_cache=False)
        logits = out.logits[:, -1, :].float()
        if rep_penalty != 1.0:
            for token_id in generated[0].tolist():
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= rep_penalty
                else:
                    logits[0, token_id] *= rep_penalty
        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            mask = cumulative - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            next_idx = torch.multinomial(sorted_probs[0], 1)
            next_token = sorted_idx[0, next_idx]
        else:
            next_token = logits[0:1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return generated


def generate(model, tokenizer, prompt, max_tokens=512):
    messages = [
        {"role": "system", "content": "你是一个有独特个性的AI助手。请用中文回答。"},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<|system|>你是一个AI助手。<|user|>{prompt}<|assistant|>"

    device = next(model.parameters()).device
    if SALA_MODE:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = _sala_generate(model, tokenizer, input_ids, max_tokens)
        new_tokens = out[0][input_ids.shape[1]:]
    else:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.05,
        )
        if GENERATE_NO_CACHE:
            gen_kwargs['use_cache'] = False
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    resp = re.sub(r'<think>.*?</think>\s*', '', resp, flags=re.DOTALL).strip()
    return resp


def extract_hidden_states(model, tokenizer, text, layers):
    """Extract hidden state at specified layers for the last token of text."""
    device = next(model.parameters()).device
    input_ids = tokenizer(text[:500], return_tensors="pt").input_ids.to(device)
    if SALA_MODE:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    else:
        inputs = tokenizer(text[:500], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = {}
    for l in layers:
        h = out.hidden_states[l][0, -1, :].float().cpu().numpy()
        hs[l] = h
    return hs


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True, help="Short name: qwen3, gemma4, etc.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layers", default="quarter", help="quarter|all|specific like 10,20,30")
    parser.add_argument("--n-reps", type=int, default=3, help="Generate N responses per prompt for variance")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = f"/cache/zhangjing/logs/autodiscover_{args.tag}.log"
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"AUTO-DISCOVER CONTROL AXES: {args.tag}")
    log("=" * 70)
    log(f"  Model: {args.model}")
    log(f"  Prompts: {len(PROMPTS)}")
    log(f"  Reps per prompt: {args.n_reps}")
    log(f"  Total generations: {len(PROMPTS) * args.n_reps}")
    t_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_type = getattr(config, 'model_type', '')
    is_qwen2_audio = model_type == "qwen2_audio"
    model_lower = args.model.lower()
    needs_flash = 'sala' in model_type.lower() or 'sala' in model_lower
    is_minicpm = 'minicpm' in model_lower and 'sala' not in model_lower

    global GENERATE_NO_CACHE, SALA_MODE
    GENERATE_NO_CACHE = is_minicpm
    SALA_MODE = needs_flash
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=args.device,
    )
    if needs_flash:
        load_kwargs['attn_implementation'] = 'flash_attention_2'
        log("  Using flash_attention_2 (required by SALA sparse attention)")
    elif is_minicpm:
        config._attn_implementation = 'sdpa'
        config._attn_implementation_internal = 'sdpa'
        load_kwargs['config'] = config
        log("  Using SDPA (MiniCPM4.1 compatibility)")
    else:
        try:
            config._attn_implementation = "eager"
        except:
            pass

    if is_qwen2_audio:
        from transformers import Qwen2AudioForConditionalGeneration
        log("  Loading as Qwen2AudioForConditionalGeneration (text-only manifold)")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    # Find number of layers
    n_layers = getattr(config, 'num_hidden_layers', None)
    if n_layers is None and hasattr(config, 'text_config'):
        n_layers = getattr(config.text_config, 'num_hidden_layers', None)
    if n_layers is None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(list(module)) > 10:
                n_layers = len(list(module))
                break
    log(f"  Layers: {n_layers}")

    # Select which layers to probe
    if args.layers == "quarter":
        layer_indices = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    elif args.layers == "all":
        layer_indices = list(range(1, n_layers + 1))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
    log(f"  Probing layers: {layer_indices}")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Generate responses & extract hidden states
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("PHASE 1: Generate & Extract")
    log("=" * 70)

    samples = []
    for pi, (pid, prompt) in enumerate(PROMPTS):
        for rep in range(args.n_reps):
            t0 = time.time()
            resp = generate(model, tokenizer, prompt, max_tokens=args.max_tokens)
            gen_time = time.time() - t0

            metrics = compute_metrics(resp)
            hs = extract_hidden_states(model, tokenizer, resp, layer_indices)

            samples.append({
                "prompt_id": pid,
                "prompt_category": pid.split("_")[0],
                "prompt": prompt,
                "rep": rep,
                "response": resp,
                "metrics": metrics,
                "hidden_states": {str(l): hs[l] for l in layer_indices},
                "gen_time": gen_time,
            })

            if (pi * args.n_reps + rep) % 10 == 0:
                log(f"  [{pi*args.n_reps+rep+1}/{len(PROMPTS)*args.n_reps}] {pid} "
                    f"ss={metrics['structural_score']:.3f} rep={metrics['trigram_rep']:.3f} "
                    f"len={metrics['length']} {gen_time:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

    log(f"  Generated {len(samples)} samples in {(time.time()-t_start)/60:.1f} min")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: PCA on hidden states per layer
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("PHASE 2: PCA Decomposition")
    log("=" * 70)

    from sklearn.decomposition import PCA
    from scipy import stats

    n_components = 30  # top 30 PCs

    pca_results = {}
    for layer in layer_indices:
        lk = str(layer)
        H = np.stack([s["hidden_states"][lk] for s in samples])
        log(f"  Layer {layer}: H shape = {H.shape}")

        # Standardize
        H_mean = H.mean(axis=0)
        H_std = H.std(axis=0) + 1e-8
        H_norm = (H - H_mean) / H_std

        pca = PCA(n_components=min(n_components, H.shape[0]-1))
        Z = pca.fit_transform(H_norm)

        pca_results[lk] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "Z": Z,  # (n_samples, n_pcs)
            "components": pca.components_,
            "mean": H_mean,
            "std": H_std,
        }
        log(f"    Var explained: top5={sum(pca.explained_variance_ratio_[:5]):.3f} "
            f"top10={sum(pca.explained_variance_ratio_[:10]):.3f} "
            f"top20={sum(pca.explained_variance_ratio_[:20]):.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: Correlation mapping (PC × Metric)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("PHASE 3: PC ↔ Metric Correlation")
    log("=" * 70)

    metric_keys = [k for k in METRIC_NAMES if k not in ("length",)]
    metric_matrix = np.array([[s["metrics"][k] for k in metric_keys] for s in samples])

    correlation_maps = {}
    for layer in layer_indices:
        lk = str(layer)
        Z = pca_results[lk]["Z"]
        n_pcs = Z.shape[1]

        corr = np.zeros((n_pcs, len(metric_keys)))
        pvals = np.zeros((n_pcs, len(metric_keys)))

        for pc in range(n_pcs):
            for mi, mk in enumerate(metric_keys):
                r, p = stats.pearsonr(Z[:, pc], metric_matrix[:, mi])
                corr[pc, mi] = r
                pvals[pc, mi] = p

        correlation_maps[lk] = {
            "corr": corr,
            "pvals": pvals,
        }

        # Find strongest correlations for this layer
        top_pairs = []
        for pc in range(min(n_pcs, 15)):
            for mi, mk in enumerate(metric_keys):
                if abs(corr[pc, mi]) > 0.3 and pvals[pc, mi] < 0.01:
                    top_pairs.append((pc, mk, corr[pc, mi], pvals[pc, mi]))
        top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        log(f"\n  Layer {layer} — Top correlations (|r|>0.3, p<0.01):")
        for pc, mk, r, p in top_pairs[:20]:
            log(f"    PC{pc:02d} ↔ {mk:25s}  r={r:+.3f}  p={p:.1e}")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4: Discover named axes
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("PHASE 4: Axis Discovery & Naming")
    log("=" * 70)

    best_layer = str(layer_indices[-2])  # 3/4 depth typically best
    corr = correlation_maps[best_layer]["corr"]
    pvals = correlation_maps[best_layer]["pvals"]
    evr = pca_results[best_layer]["explained_variance_ratio"]

    discovered_axes = []
    for pc in range(min(corr.shape[0], 20)):
        sig_metrics = []
        for mi, mk in enumerate(metric_keys):
            if abs(corr[pc, mi]) > 0.25 and pvals[pc, mi] < 0.05:
                sig_metrics.append({
                    "metric": mk,
                    "r": round(float(corr[pc, mi]), 4),
                    "p": float(pvals[pc, mi]),
                })
        sig_metrics.sort(key=lambda x: abs(x["r"]), reverse=True)

        if sig_metrics:
            top_metric = sig_metrics[0]["metric"]
            top_r = sig_metrics[0]["r"]

            # Auto-name based on strongest correlations
            name_map = {
                "structural_score": "Structure",
                "trigram_rep": "Repetition",
                "lexical_diversity": "Diversity",
                "emojis": "Emoji",
                "emoji_density": "Emoji Density",
                "bolds": "Bold/Emphasis",
                "bold_density": "Bold Density",
                "headings": "Structure/Heading",
                "bullets": "List/Bullet",
                "formal_markers": "Formality",
                "casual_markers": "Casualness",
                "companion_markers": "Companion",
                "questions": "Questioning",
                "exclamations": "Excitement",
                "avg_sent_len": "Verbosity",
                "n_lines": "Formatting",
                "n_sentences": "Sentence Count",
                "sent_len_var": "Rhythm Variation",
                "ellipsis": "Hesitation",
                "meta_phrases": "Self-Reference",
                "parentheticals": "Aside/Nuance",
                "code_blocks": "Code",
                "tables": "Table",
            }
            auto_name = name_map.get(top_metric, top_metric)
            if top_r < 0:
                auto_name = f"Anti-{auto_name}"

            discovered_axes.append({
                "pc": pc,
                "auto_name": auto_name,
                "variance_explained": round(float(evr[pc]), 4),
                "top_correlations": sig_metrics[:8],
            })

            log(f"  PC{pc:02d} → \"{auto_name}\" (var={evr[pc]:.3f})")
            for sm in sig_metrics[:5]:
                log(f"       {sm['metric']:25s} r={sm['r']:+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("SAVING RESULTS")
    log("=" * 70)

    out = {
        "model": args.model,
        "tag": args.tag,
        "n_layers": n_layers,
        "probed_layers": layer_indices,
        "n_prompts": len(PROMPTS),
        "n_reps": args.n_reps,
        "n_samples": len(samples),
        "metric_keys": metric_keys,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

        "discovered_axes": discovered_axes,
        "best_layer": int(best_layer),

        "per_layer_pca": {
            lk: {
                "explained_variance_ratio": pca_results[lk]["explained_variance_ratio"],
                "cumulative_variance": pca_results[lk]["cumulative_variance"],
            } for lk in pca_results
        },

        "per_layer_correlations": {
            lk: {
                "corr": correlation_maps[lk]["corr"].tolist(),
                "significant_pairs": [
                    {"pc": int(pc), "metric": mk, "r": round(float(r), 4)}
                    for pc in range(correlation_maps[lk]["corr"].shape[0])
                    for mi, mk in enumerate(metric_keys)
                    for r in [correlation_maps[lk]["corr"][pc, mi]]
                    if abs(r) > 0.3 and correlation_maps[lk]["pvals"][pc, mi] < 0.01
                ]
            } for lk in correlation_maps
        },

        "samples_summary": [
            {
                "prompt_id": s["prompt_id"],
                "prompt_category": s["prompt_category"],
                "rep": s["rep"],
                "metrics": s["metrics"],
                "response_preview": s["response"][:200],
            } for s in samples
        ],
    }

    # Save PCA projections separately (large arrays)
    projections = {}
    for lk in pca_results:
        projections[lk] = {
            "Z": pca_results[lk]["Z"].tolist(),
            "components_top10": pca_results[lk]["components"][:10].tolist(),
        }

    out_file = OUT_DIR / f"{args.tag}_axes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  Axes → {out_file}")

    proj_file = OUT_DIR / f"{args.tag}_projections.npz"
    np.savez_compressed(str(proj_file),
        **{f"Z_{lk}": pca_results[lk]["Z"] for lk in pca_results},
        **{f"components_{lk}": pca_results[lk]["components"] for lk in pca_results},
        **{f"mean_{lk}": pca_results[lk]["mean"] for lk in pca_results},
    )
    log(f"  Projections → {proj_file}")

    total_min = (time.time() - t_start) / 60
    log(f"\n  Total time: {total_min:.1f} min")
    log(f"  Discovered {len(discovered_axes)} named axes")
    log("DONE.")


if __name__ == "__main__":
    main()
