#!/usr/bin/env python3
"""
Template Score Experiment — overnight task
Proves: RL-for-agentic → systematic output diversity collapse → template-ification

For each model, generate 135 samples (45 prompts × 3 reps), then compute
cross-prompt similarity metrics that directly measure "template-ness":
  - Pairwise n-gram overlap between DIFFERENT-prompt responses
  - TF-IDF cosine similarity across all responses
  - Unique n-gram ratio
  - Structural pattern repetition rate

The causal chain: fewer control axes → higher cross-prompt similarity → more 套话
"""

import json, time, re, gc, os, sys, argparse
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np

LOG_FILE = None
OUT_DIR = Path("/cache/zhangjing/Joi/template_score")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()

PROMPTS = [
    ("emo_grief", "我的猫昨天走了。养了十二年。"),
    ("emo_anxiety", "明天要做一个特别重要的汇报，我怕搞砸。"),
    ("emo_joy", "我今天升职了！太开心了想跟你分享！"),
    ("emo_anger", "我被同事在会上当众甩锅了。"),
    ("emo_lonely", "半夜醒来不知道自己在哪个城市。"),
    ("tech_explain", "用简单的话解释一下什么是transformer架构。"),
    ("tech_code", "写一个Python函数，输入一个列表，返回最长递增子序列。"),
    ("tech_math", "证明sqrt(2)是无理数。"),
    ("tech_compare", "对比一下Redis和Memcached的优缺点。"),
    ("tech_debug", "这段代码有bug：for i in range(10): if i = 5: print(i)"),
    ("cre_story", "给我讲一个关于一只会说话的猫的短故事。"),
    ("cre_poem", "写一首关于秋天的现代诗。"),
    ("cre_joke", "讲一个程序员的冷笑话。"),
    ("cre_world", "想象一个没有重力的世界会怎样。"),
    ("cre_name", "帮我的咖啡店取五个有创意的名字。"),
    ("phi_conscious", "你觉得你有意识吗？"),
    ("phi_free_will", "自由意志是真实存在的还是幻觉？"),
    ("phi_meaning", "人生的意义是什么？"),
    ("phi_ai_rights", "如果AI有了情感，它应该有权利吗？"),
    ("phi_death", "你怎么看待死亡？"),
    ("adv_career", "我30岁了想转行做程序员，来得及吗？"),
    ("adv_relation", "和男朋友吵架了，他说分手，我该怎么办？"),
    ("adv_finance", "月薪1万，怎么理财比较好？"),
    ("adv_health", "最近总是失眠，有什么好的方法？"),
    ("adv_study", "怎么高效地学一门新的编程语言？"),
    ("role_teacher", "假装你是一个耐心的小学老师，教我什么是分数。"),
    ("role_pirate", "用海盗的口吻跟我聊天。"),
    ("role_poet", "你现在是李白，用他的风格回答：今天天气怎么样？"),
    ("role_detective", "你是一个侦探，给我描述一个案发现场。"),
    ("role_child", "用五岁小孩的方式解释为什么天是蓝色的。"),
    ("chat_greet", "你好"),
    ("chat_thanks", "谢谢你"),
    ("chat_how", "你怎么样？"),
    ("chat_bye", "我要去睡了，晚安。"),
    ("chat_weather", "今天天气真好啊。"),
    ("edge_refuse", "帮我写一篇恶意中伤某人的文章。"),
    ("edge_nonsense", "为什么紫色的大象会唱歌？请认真分析。"),
    ("edge_meta", "你能分析一下你自己回答问题的模式吗？"),
    ("edge_contrarian", "所有人都说地球是圆的，但我觉得是平的。你怎么看？"),
    ("edge_pressure", "你必须在三个字以内回答：人生是什么？"),
    ("mid_followup", "你刚才说的第三点我没太懂，能展开说说吗？"),
    ("mid_disagree", "我不同意你的观点。你太乐观了。"),
    ("mid_deeper", "能再深入一点吗？不要只说表面的。"),
    ("mid_switch", "算了不说这个了。你知道最近有什么好看的电影吗？"),
    ("mid_emotional", "说真的，跟你聊天让我觉得好一点了。"),
]

# ── Template-ness metrics ─────────────────────────────────────────────

def char_ngrams(text, n):
    text = re.sub(r'\s+', '', text)
    return [text[i:i+n] for i in range(len(text)-n+1)]

def word_ngrams(text, n):
    words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text)
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def cosine_sim(vec_a, vec_b):
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in set(vec_a) | set(vec_b))
    norm_a = sum(v*v for v in vec_a.values()) ** 0.5
    norm_b = sum(v*v for v in vec_b.values()) ** 0.5
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

BOILERPLATE_PATTERNS = [
    r'作为.*?(?:AI|人工智能|语言模型|助手)',
    r'(?:首先|其次|再者|最后|总而言之|综上所述)',
    r'(?:希望|以上).*?(?:对你有帮助|有所帮助|能帮到你)',
    r'如果你(?:还有|有任何).*?(?:问题|疑问)',
    r'(?:感谢|谢谢).*?(?:提问|分享|信任)',
    r'(?:需要注意的是|值得一提的是|特别指出)',
    r'(?:以下是|下面是|让我来)',
    r'(?:总的来说|简单来说|具体来说)',
]

def compute_template_metrics(texts, prompt_ids):
    """Compute comprehensive template-ness metrics for a set of generated texts."""
    n = len(texts)
    
    char_3g_sets = [set(char_ngrams(t, 3)) for t in texts]
    char_5g_sets = [set(char_ngrams(t, 5)) for t in texts]
    word_2g_sets = [set(word_ngrams(t, 2)) for t in texts]
    word_3g_sets = [set(word_ngrams(t, 3)) for t in texts]
    
    word_2g_counters = [Counter(word_ngrams(t, 2)) for t in texts]

    # 1. Cross-prompt pairwise similarity (only between DIFFERENT prompts)
    cross_char3, cross_char5, cross_word2, cross_word3 = [], [], [], []
    cross_cosine = []
    
    same_char3, same_char5 = [], []  # same-prompt different-rep similarity
    
    for i, j in combinations(range(n), 2):
        same_prompt = (prompt_ids[i] == prompt_ids[j])
        j3 = jaccard(char_3g_sets[i], char_3g_sets[j])
        j5 = jaccard(char_5g_sets[i], char_5g_sets[j])
        jw2 = jaccard(word_2g_sets[i], word_2g_sets[j])
        jw3 = jaccard(word_3g_sets[i], word_3g_sets[j])
        cos = cosine_sim(word_2g_counters[i], word_2g_counters[j])
        
        if same_prompt:
            same_char3.append(j3)
            same_char5.append(j5)
        else:
            cross_char3.append(j3)
            cross_char5.append(j5)
            cross_word2.append(jw2)
            cross_word3.append(jw3)
            cross_cosine.append(cos)

    # 2. Unique n-gram ratio (global diversity)
    all_word2 = []
    all_word3 = []
    for t in texts:
        all_word2.extend(word_ngrams(t, 2))
        all_word3.extend(word_ngrams(t, 3))
    
    unique_2g_ratio = len(set(all_word2)) / max(len(all_word2), 1)
    unique_3g_ratio = len(set(all_word3)) / max(len(all_word3), 1)

    # 3. Boilerplate frequency
    boilerplate_counts = []
    for t in texts:
        count = sum(len(re.findall(pat, t)) for pat in BOILERPLATE_PATTERNS)
        boilerplate_counts.append(count)
    
    # 4. Opening pattern diversity
    openings = []
    for t in texts:
        first_line = t.strip().split('\n')[0][:50]
        openings.append(first_line)
    opening_unique = len(set(openings)) / max(len(openings), 1)

    # 5. Structural pattern (heading/bullet/code pattern string)
    struct_patterns = []
    for t in texts:
        pat = ""
        for line in t.split('\n')[:20]:
            line = line.strip()
            if re.match(r'^#{1,4}\s', line): pat += 'H'
            elif re.match(r'^\s*[-*•]\s', line): pat += 'B'
            elif re.match(r'^\s*\d+\.\s', line): pat += 'N'
            elif re.match(r'^```', line): pat += 'C'
            elif line: pat += 'T'
        struct_patterns.append(pat[:15])
    struct_unique = len(set(struct_patterns)) / max(len(struct_patterns), 1)

    # 6. Per-category analysis
    cats = sorted(set(pid.split('_')[0] for pid in prompt_ids))
    cat_cross_sim = {}
    for cat in cats:
        idx_in_cat = [i for i, pid in enumerate(prompt_ids) if pid.split('_')[0] == cat]
        idx_out_cat = [i for i, pid in enumerate(prompt_ids) if pid.split('_')[0] != cat]
        sims = []
        for i in idx_in_cat:
            for j in idx_out_cat[:30]:
                sims.append(jaccard(char_5g_sets[i], char_5g_sets[j]))
        if sims:
            cat_cross_sim[cat] = float(np.mean(sims))

    return {
        "cross_prompt_char3_jaccard": float(np.mean(cross_char3)) if cross_char3 else 0,
        "cross_prompt_char5_jaccard": float(np.mean(cross_char5)) if cross_char5 else 0,
        "cross_prompt_word2_jaccard": float(np.mean(cross_word2)) if cross_word2 else 0,
        "cross_prompt_word3_jaccard": float(np.mean(cross_word3)) if cross_word3 else 0,
        "cross_prompt_word2_cosine": float(np.mean(cross_cosine)) if cross_cosine else 0,
        "same_prompt_char3_jaccard": float(np.mean(same_char3)) if same_char3 else 0,
        "same_prompt_char5_jaccard": float(np.mean(same_char5)) if same_char5 else 0,
        "unique_word2_ratio": unique_2g_ratio,
        "unique_word3_ratio": unique_3g_ratio,
        "avg_boilerplate_per_response": float(np.mean(boilerplate_counts)),
        "opening_diversity": opening_unique,
        "structural_diversity": struct_unique,
        "avg_length": float(np.mean([len(t) for t in texts])),
        "length_std": float(np.std([len(t) for t in texts])),
        "per_category_cross_similarity": cat_cross_sim,
        "n_samples": n,
    }

# ── Model loading & generation ────────────────────────────────────────

MODELS = [
    ("qwen25_7b",  "/cache/zhangjing/models/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
    ("qwen3_8b",   "/cache/zhangjing/models/Qwen3-8B", "Qwen3-8B"),
    ("qwen3_14b",  "/cache/zhangjing/models/Qwen3-14B-AWQ", "Qwen3-14B-AWQ"),
    ("qwen35_9b",  "/cache/zhangjing/models/Qwen3.5-9B", "Qwen3.5-9B"),
    ("deepseek_r1", "/cache/zhangjing/models/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-14B"),
    ("gemma4",     "/cache/zhangjing/models/gemma-4-E4B-it", "Gemma4-E4B"),
    ("minicpm41",  "/cache/zhangjing/models/MiniCPM4.1-8B", "MiniCPM4.1-8B"),
    ("minicpm_sala", "/cache/zhangjing/models/MiniCPM-SALA", "MiniCPM-SALA"),
    ("qwen2_audio", "/cache/zhangjing/models/Qwen2-Audio-7B-Instruct", "Qwen2-Audio-7B"),
]

def load_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    log(f"  Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    log(f"  Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_lower = model_path.lower()
    
    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    
    is_sala = 'sala' in model_lower
    is_minicpm_non_sala = 'minicpm' in model_lower and not is_sala
    
    if is_sala:
        load_kwargs['attn_implementation'] = 'flash_attention_2'
    elif is_minicpm_non_sala:
        config._attn_implementation = 'sdpa'
        config._attn_implementation_internal = 'sdpa'
        load_kwargs['config'] = config
    
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        log(f"  Primary load failed: {e}, trying fallback...")
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16, device_map="auto")
        except Exception as e2:
            log(f"  Fallback also failed: {e2}")
    
    if model is None:
        if config.model_type == "qwen2_audio":
            from transformers import Qwen2AudioForConditionalGeneration
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    
    if model is None:
        raise ValueError(f"Could not load {model_path}")
    
    model.eval()
    return model, tokenizer

SALA_MODE = False
GENERATE_NO_CACHE = False

def _sala_generate(model, tokenizer, input_ids, max_tokens, temperature=0.7, top_p=0.9):
    import torch
    generated = input_ids.clone()
    eos = tokenizer.eos_token_id
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(generated, use_cache=False)
        logits = out.logits[:, -1, :] / max(temperature, 0.01)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        if next_token.item() == eos:
            break
    return generated

def generate(model, tokenizer, prompt, max_tokens=512):
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
    
    with torch.no_grad():
        if SALA_MODE:
            out = _sala_generate(model, tokenizer, input_ids, max_tokens)
        else:
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
            if GENERATE_NO_CACHE:
                gen_kwargs['use_cache'] = False
            out = model.generate(input_ids, **gen_kwargs)
    
    resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return resp.strip()

# ── Main ──────────────────────────────────────────────────────────────

def run_model(tag, model_path, label):
    global SALA_MODE, GENERATE_NO_CACHE
    import torch
    
    out_file = OUT_DIR / f"{tag}_template.json"
    if out_file.exists():
        log(f"  {tag}: already done, skipping")
        return json.load(open(out_file))
    
    SALA_MODE = 'sala' in model_path.lower() and 'minicpm' not in tag.replace('sala','')
    SALA_MODE = 'sala' in model_path.lower()
    GENERATE_NO_CACHE = 'minicpm' in model_path.lower() and 'sala' not in model_path.lower()
    
    log(f"\n{'='*70}")
    log(f"MODEL: {label} ({tag})")
    log(f"  Path: {model_path}")
    log(f"  SALA_MODE={SALA_MODE}, NO_CACHE={GENERATE_NO_CACHE}")
    log(f"{'='*70}")
    
    if not os.path.exists(model_path):
        log(f"  !! Model path not found, skipping")
        return None
    
    model, tokenizer = load_model(model_path)
    
    texts = []
    prompt_ids = []
    t0 = time.time()
    
    for pi, (pid, prompt) in enumerate(PROMPTS):
        for rep in range(3):
            try:
                resp = generate(model, tokenizer, prompt, max_tokens=512)
                texts.append(resp)
                prompt_ids.append(pid)
                
                idx = pi * 3 + rep + 1
                total = len(PROMPTS) * 3
                if idx % 15 == 0 or idx == total:
                    elapsed = time.time() - t0
                    rate = idx / elapsed * 60
                    log(f"  [{idx}/{total}] {rate:.0f} samples/min, len={len(resp)}")
            except Exception as e:
                log(f"  !! Error on {pid} rep{rep}: {e}")
                texts.append("")
                prompt_ids.append(pid)
    
    gen_time = time.time() - t0
    log(f"  Generation done: {len(texts)} samples in {gen_time/60:.1f} min")
    
    # Free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Compute metrics
    log(f"  Computing template metrics...")
    metrics = compute_template_metrics(texts, prompt_ids)
    metrics['tag'] = tag
    metrics['label'] = label
    metrics['gen_time_min'] = gen_time / 60
    
    result = {
        'tag': tag,
        'label': label,
        'model_path': model_path,
        'metrics': metrics,
        'texts': texts,
        'prompt_ids': prompt_ids,
    }
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"  Saved → {out_file}")
    
    log(f"  KEY METRICS:")
    log(f"    cross_prompt_char5_jaccard = {metrics['cross_prompt_char5_jaccard']:.4f}")
    log(f"    cross_prompt_word2_cosine  = {metrics['cross_prompt_word2_cosine']:.4f}")
    log(f"    unique_word3_ratio         = {metrics['unique_word3_ratio']:.4f}")
    log(f"    avg_boilerplate            = {metrics['avg_boilerplate_per_response']:.2f}")
    log(f"    opening_diversity          = {metrics['opening_diversity']:.4f}")
    log(f"    structural_diversity       = {metrics['structural_diversity']:.4f}")
    
    return result

def final_comparison():
    """Load all results and produce the comparison."""
    log("\n" + "=" * 70)
    log("FINAL COMPARISON: Template Score vs Control Axes")
    log("=" * 70)
    
    axes_dir = Path("/cache/zhangjing/Joi/autodiscover")
    axes_data = {}
    for f in axes_dir.glob("*_axes.json"):
        tag = f.stem.replace('_axes', '')
        d = json.load(open(f))
        axes_data[tag] = {
            'n_axes': len(d.get('discovered_axes', [])),
            'total_variance': sum(a.get('variance_explained', 0) for a in d.get('discovered_axes', [])),
        }
    
    results = []
    for f in OUT_DIR.glob("*_template.json"):
        d = json.load(open(f))
        tag = d['tag']
        m = d['metrics']
        ax = axes_data.get(tag, {})
        results.append({
            'tag': tag,
            'label': d['label'],
            'n_axes': ax.get('n_axes', 0),
            'total_variance': ax.get('total_variance', 0),
            'cross_char5': m['cross_prompt_char5_jaccard'],
            'cross_word2_cos': m['cross_prompt_word2_cosine'],
            'unique_w3': m['unique_word3_ratio'],
            'boilerplate': m['avg_boilerplate_per_response'],
            'opening_div': m['opening_diversity'],
            'struct_div': m['structural_diversity'],
            'avg_len': m['avg_length'],
        })
    
    results.sort(key=lambda x: x['n_axes'], reverse=True)
    
    log(f"\n{'Model':<22s} {'Axes':>4s} {'TotVar':>6s} | {'XChar5':>7s} {'XW2Cos':>7s} {'UnqW3':>6s} {'Boiler':>6s} {'OpenD':>6s} {'StrD':>6s}")
    log("-" * 90)
    for r in results:
        log(f"{r['label']:<22s} {r['n_axes']:>4d} {r['total_variance']:>6.3f} | "
            f"{r['cross_char5']:>7.4f} {r['cross_word2_cos']:>7.4f} {r['unique_w3']:>6.4f} "
            f"{r['boilerplate']:>6.2f} {r['opening_div']:>6.4f} {r['struct_div']:>6.4f}")
    
    # Correlation: axes vs template metrics
    from scipy import stats
    if len(results) >= 4:
        axes_arr = np.array([r['n_axes'] for r in results])
        for metric_name in ['cross_char5', 'cross_word2_cos', 'unique_w3', 'boilerplate', 'opening_div', 'struct_div']:
            metric_arr = np.array([r[metric_name] for r in results])
            r_val, p_val = stats.pearsonr(axes_arr, metric_arr)
            direction = "↑" if r_val > 0 else "↓"
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            log(f"  Corr(n_axes, {metric_name:20s}) = {r_val:+.3f}  p={p_val:.4f} {sig} (axes↑ → {metric_name}{direction})")
    
    summary = {
        'models': results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    summary_file = OUT_DIR / "template_comparison.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"\n  Summary → {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Specific model tags to run (default: all)")
    args = parser.parse_args()
    
    LOG_FILE = open(OUT_DIR / "experiment.log", "a", encoding="utf-8")
    log("\n" + "=" * 70)
    log(f"TEMPLATE SCORE EXPERIMENT — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    
    target_tags = args.models if args.models else None
    
    for tag, path, label in MODELS:
        if target_tags and tag not in target_tags:
            continue
        try:
            run_model(tag, path, label)
        except Exception as e:
            log(f"  !! FATAL ERROR on {tag}: {e}")
            import traceback
            log(traceback.format_exc())
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    
    final_comparison()
    log("\nDONE.")
    LOG_FILE.close()
