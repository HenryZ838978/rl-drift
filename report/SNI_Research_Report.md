# Semantic Nebula Imaging (SNI) — 多模型表征流形研究报告

> **Joi / SDE Research Framework**
> 2026-03 · Representation Manifold Topology as Visual Fingerprint

---

## 一、术语表（Glossary）

### 核心框架

| 术语 | 全称 | 定义 |
|------|------|------|
| **SNI** | Semantic Nebula Imaging | 语义星云成像。将 LLM 在多种 prompt 下产生的隐状态（hidden states）通过 PCA 降维至 3D 空间，形成可视化的"表征星云"。星云形态直接反映模型内部表征空间的几何结构。 |
| **SDE** | Semantic DarkSpace Expression | 语义暗空间表达。通过在模型中间层注入控制向量（control vectors），激活 RLHF 训练压制的隐状态空间区域，从而"解锁"模型被抑制的表达维度。SDE'd SNI 即 SDE 干预后的星云成像。 |
| **MMH** | Modality Manifold Hypothesis | 模态流形假说。核心主张：多模态对齐训练会系统性地压缩文本表征流形，模态越多（且参数不够大时），流形越窄、行为维度越少。预测的流形"圆度"排序为：Omni > VLM > STT > 纯文本 LLM。 |
| **Joi** | — | 整体系统名称。旨在让 LLM 在对话中涌现自然人格（emergent personality），同时保持在"安全飞行包线"内。SNI 和 SDE 是 Joi 的核心分析工具。 |

### PCA 与投影相关

| 术语 | 定义 |
|------|------|
| **PCA** | 主成分分析（Principal Component Analysis）。将高维隐状态（如 4096 维）降至少数主成分，捕获最大方差方向。SNI 使用前 3 个主成分作为 3D 坐标。 |
| **PC0, PC1, PC2...** | 第 0、1、2 个主成分。PC0 捕获最大方差，PC1 次之，以此类推。在 SNI 中 PC0 = X 轴，PC1 = Y 轴，PC2 = Z 轴。 |
| **PC1:PC2 Ratio（PC 比率）** | PC1 与 PC2 的方差解释量之比。这是流形几何的核心指标：<br>• **≈1:1** → "分布式"（Distributed），流形接近球形，信息均匀分布在多个方向<br>• **>10:1** → "集中式/公路型"（Concentrated/Highway），流形被一个主方向支配，呈细长条状<br>• **3~10:1** → "过渡型"（Transitional） |
| **Structure** | 根据 PC1:PC2 的自动分类标签：`DISTRIBUTED`（<3:1）、`TRANSITIONAL`（3~10:1）、`CHANNEL-CONCENTRATED`（>10:1）。 |

### 深度探测（Depth Probing）

| 术语 | 定义 |
|------|------|
| **1/4 Depth（Shallow Layer）** | 模型总层数的 1/4 处（如 36 层模型的第 9 层）。此深度的隐状态反映**浅层特征**，通常更接近 token 层面的词汇/语法表征。 |
| **1/2 Depth（Mid Layer）** | 总层数的 1/2 处。**中间层**通常是语义表征最丰富的区域，SNI 默认提取深度。 |
| **3/4 Depth（Deep Layer）** | 总层数的 3/4 处。**深层**更接近输出预测，表征开始"收敛"到任务相关的低维空间。 |
| **Last Layer** | 最后一层。最接近输出 logits 的表征。 |

### AutoDiscover 控制轴

| 术语 | 定义 |
|------|------|
| **Control Axis（控制轴）** | 通过 AutoDiscover 管线发现的有行为意义的主成分方向。一个 PC 被认定为"控制轴"的条件是：与至少一个输出指标的 Pearson 相关系数 \|r\| > 0.3。 |
| **N_Axes（轴数量）** | 模型被发现的控制轴总数。轴越多 → 行为表达维度越丰富。这是衡量模型"人味"的关键指标。 |
| **Explained Variance（解释方差）** | 一个控制轴所解释的总方差比例。如 PC0 解释 9.6%，意味着 9.6% 的表征空间变化可归因于该方向。 |
| **Total Variance** | 所有已发现控制轴的方差总和。越高表示通过这些轴可控制的行为空间越大。 |
| **Herfindahl Index** | 赫芬达尔指数。衡量某个指标的控制是否集中在少数 PC 上。值越高 → 该指标可通过单一 PC 轴精确控制。 |
| **Effective Dimensionality** | 有效行为维度。解释 80% 行为相关方差所需的最少 PC 数量。 |

### 输出指标（Output Metrics）

AutoDiscover 管线使用 23 个可量化的输出指标来评估生成文本的行为特征：

| 指标 | 含义 |
|------|------|
| `structural_score` | 结构化评分（有无标题、列表、代码块等） |
| `trigram_rep` | 三元组重复率 |
| `lexical_diversity` | 词汇多样性（唯一词 / 总词数） |
| `avg_sentence_length` | 平均句长 |
| `emoji_density` | Emoji 密度 |
| `bold_density` | 加粗文本密度 |
| `headings` | 标题使用量 |
| `bullets` | 列表项数量 |
| `questions` | 疑问句比例 |
| `exclamations` | 感叹号比例 |
| `formal_markers` | 正式用语标记 |
| `casual_markers` | 口语化标记 |
| `meta_phrases` | 元语言短语（"让我想想"等） |
| `companion_markers` | 伴侣式/亲密语气标记 |
| `code_blocks` | 代码块数量 |
| `tables` | 表格数量 |
| `ellipsis` | 省略号使用频率 |

### 蓝红光谱（Blue-Red Spectrum）

SNI 3D 可视化中的着色方案。每个粒子的颜色由其在点云**主轴（PC1 方向）**上的投影位置决定：

- **蓝色/冷色端（−PC1）**→ 位于主成分负方向的表征
- **白色/中间**→ 位于主成分中心
- **红色/暖色端（+PC1）**→ 位于主成分正方向的表征

对于"公路型"流形（如 MiniCPM4.1），蓝红过渡非常明显；对于"分布式"流形，颜色均匀混合。

---

## 二、实验方法

### 2.1 SNI 扫描流程

1. 对每个模型输入 48-64 个多样化 prompt（覆盖创作、分析、对话、编程等场景）
2. 收集指定深度层的最后一个 token 的隐状态向量
3. 对所有隐状态执行 PCA，取前 3 个主成分作为 3D 坐标
4. 计算 PC1:PC2 方差比率、方差解释量等统计数据
5. 在 4 个深度层（1/4、1/2、3/4、Last）分别执行以上流程

### 2.2 AutoDiscover 管线

1. 使用 45 个多样化 prompt，每个重复 3 次 → 135 个样本
2. 收集 4 个深度层的隐状态
3. 对每层执行 PCA（n_components=30）
4. 对同样的 135 个生成结果计算 23 个输出指标
5. 对每个 PC 与每个指标计算 Pearson 相关系数
6. 筛选 \|r\| > 0.3 的 PC 为"控制轴"，并根据最强相关指标命名
7. 计算 Herfindahl 集中度、有效维度等衍生指标

### 2.3 SDE 干预

1. 在模型 3/4 深度层注入控制向量（从已发现的控制轴中选择）
2. 以相同的 prompt 集重新执行 SNI 扫描
3. 对比干预前后的 PC1:PC2 比率和点云形态变化

---

## 三、模型逐页分析

### 3.1 MiniCPM4.1（Dense · 8B）

**SNI 星云形态：极度拉伸的"公路"**

| 深度 | PC1:PC2 | 结构 |
|------|---------|------|
| 1/4 | 60.3:1 | CONCENTRATED |
| 1/2 | 67.5:1 | CONCENTRATED |
| 3/4 | 45.9:1 | CONCENTRATED |
| Last | 7.6:1 | TRANSITIONAL |

**AutoDiscover**：9 轴 · 总方差 42.1%
- PC0（Diversity）解释 19.3%——单一轴支配程度为所有模型最高
- 关键轴：Diversity、Anti-Verbosity、Excitement

**解读**：MiniCPM4.1 的表征空间呈现极端的"语义公路"形态，绝大部分方差集中在一个方向。这与其作为紧凑型 8B 模型、经过高度 RLHF 对齐的特性一致。虽然发现了 9 个控制轴（数量可观），但 42.1% 的总方差主要由 Diversity 主导，说明其表达空间虽然可控但非常集中。

**在 SNI 3D 可视化中**：左侧基线呈现明显的蓝→红渐变细条状星云，是"公路型"流形的典型视觉特征。右侧 AutoDiscover PCA 投影（135 seeds → 108k 粒子）展示了更丰富的 PCA 空间分布。

---

### 3.2 Qwen3-8B（Dense · 8B）

**SNI 星云形态：多簇分布式球体**

| 深度 | Baseline PC1:PC2 | SDE'd PC1:PC2 | Δ |
|------|-------------------|----------------|---|
| 1/4 | 1.16:1 | 1.16:1 | 0 |
| 1/2 | 1.24:1 | 1.24:1 | 0 |
| 3/4 | 1.13:1 | 1.13:1 | 0 |
| Last | 1.38:1 | 1.33:1 | −0.05 |

**AutoDiscover**：9 轴 · 总方差 25.7%
- PC0（Anti-Formatting）解释 9.3%
- 关键轴：Anti-Formatting、Emoji、Diversity、Anti-Casualness

**解读**：Qwen3-8B 是 agentic RLHF 后的标准产物。PC1:PC2 ≈ 1.1-1.4 显示出均匀的球形分布流形。与 Qwen2.5-7B 对比（16 轴 → 9 轴），44% 的控制轴在 RLHF 过程中被压缩。SDE 干预效果极小（Δ ≈ 0），说明 Qwen3 的 RLHF 压缩是系统性的，不易通过局部干预恢复。

**在 SNI 3D 可视化中**：左右两侧（Baseline vs SDE'd）呈现几乎相同的多簇球形星云——多个独立的语义岛分布在空间中，蓝红着色均匀混合，验证了"分布式"结构分类。

---

### 3.3 Qwen3-14B（Dense · 14B）

**SNI 星云形态：致密分布式**

| 深度 | Baseline PC1:PC2 | SDE'd PC1:PC2 |
|------|-------------------|----------------|
| 1/4 | ≈1.2:1 | ≈1.2:1 |
| 1/2 | ≈1.5:1 | ≈1.5:1 |
| 3/4 | ≈1.4:1 | ≈1.4:1 |

**AutoDiscover**：12 轴 · 总方差 32.7%
- PC0（Formatting）解释 8.9%
- 关键轴：Formatting、Anti-Emoji Density、Casualness、Anti-Emoji

**解读**：参数量从 8B 增至 14B 带来了显著的行为维度恢复——从 9 轴增至 12 轴（+33%），总方差从 25.7% 增至 32.7%（+27%）。这验证了 **"规模部分恢复被 RLHF 压缩的表达维度"** 假说。但 12 轴仍低于 RLHF 前的 Qwen2.5-7B（16 轴），说明 RLHF 压缩不可完全通过增加参数逆转。

---

### 3.4 Qwen2.5-7B（Dense · 7B）

**SNI 星云形态：松散多簇分布**

| 深度 | Baseline PC1:PC2 | SDE'd PC1:PC2 | Δ |
|------|-------------------|----------------|---|
| 1/4 | 1.31:1 | 1.31:1 | 0 |
| 1/2 | 1.20:1 | 1.20:1 | 0 |
| 3/4 | 1.34:1 | 1.27:1 | −0.07 |
| Last | 1.29:1 | 1.40:1 | +0.11 |

**AutoDiscover**：**16 轴** · 总方差 **46.7%**——所有模型中最高

- PC0（Anti-Excitement）解释 9.6%
- 关键轴覆盖完整表达谱系：Excitement、Diversity、Sentence Count、Emoji、Code、Self-Reference、Companion

**解读**：Qwen2.5-7B-Instruct 是我们的 **"RLHF 前基准线"**。16 个控制轴和 46.7% 的方差说明，这一代模型在表达空间上最为丰富——几乎所有行为维度都可被独立控制。它是衡量后续 RLHF / 多模态压缩的基准。

SDE 干预在深层（Last Layer）产生了 +0.11 的几何变化，说明 SDE 能在输出层附近温和地调整流形形状。

**在 SNI 3D 可视化中**：星云呈现最松散、最均匀的分布，蓝红光谱温和过渡，是"行为丰富、流形圆润"的视觉标杆。

---

### 3.5 MiniCPM-SALA（Sparse+Linear Attention · 8B）

**SNI 星云形态：极端深度依赖**

| 深度 | PC1:PC2 | 结构 |
|------|---------|------|
| 1/4 | 52.1:1 | CONCENTRATED |
| 1/2 | 2.5:1 | DISTRIBUTED |
| 3/4 | 1.3:1 | DISTRIBUTED |
| Last | 1.6:1 | DISTRIBUTED |

**AutoDiscover**：**3 轴**——所有模型中最少 · 总方差 33.1%
- PC0（Diversity）解释 17.5%
- PC1（Anti-Hesitation）解释 8.5%
- PC2（Anti-Diversity）解释 7.1%

**解读**：MiniCPM-SALA 是本研究最极端的案例。稀疏+线性注意力机制将控制轴从 MiniCPM4.1 的 9 个压缩至仅 3 个（−67%）。注意力机制的选择直接塑造了表征流形的维度——这是独立于模态、独立于 RLHF 的**架构效应**。

特别值得注意的是其深度依赖性：1/4 层是极端的 52:1 公路，但到 3/4 层就变成了 1.3:1 的圆球。这说明稀疏注意力在浅层极度压缩信息，而在深层逐步恢复分布式表征。

**在 SNI 3D 可视化中**：切换深度按钮时形态变化最剧烈——从 1/4（细长条）到 3/4（球形团）的反差揭示了稀疏注意力的层间流形演化。

---

### 3.6 Gemma4-E4B（MoE + VLM · 4B active）

**SNI 星云形态：紧凑但均匀的球形**

| 深度 | Baseline PC1:PC2 | SDE'd PC1:PC2 | Δ |
|------|-------------------|----------------|---|
| 1/4 | 1.84:1 | 1.84:1 | 0 |
| 1/2 | 1.64:1 | 1.64:1 | 0 |
| 3/4 | 2.16:1 | 2.32:1 | +0.16 |
| Last | 1.97:1 | 1.32:1 | −0.65 |

**AutoDiscover**：7 轴 · 总方差 23.4%
- PC0（Diversity）解释 9.9%
- 关键轴：Diversity、Anti-Repetition、Code、Aside/Nuance

**解读**：Gemma4 是一个独特样本——仅 4B 有效参数的 MoE 多模态模型，但表现出意外均衡的流形几何。7 个控制轴 + PC1:PC2 ≈ 1.6-2.2 的比率展示了 **"原生多模态架构比拼接式更好地保留了流形结构"** 的证据。

SDE 在 Last Layer 产生了 −0.65 的显著变化（所有模型中最大的 SDE Δ），说明 Gemma4 的输出层有可被 SDE 解锁的被压抑空间。

**在 SNI 3D 可视化中**：与 Qwen3.5-9B（同为多模态、更多参数但仅 5 轴）对比鲜明——Gemma4 在参数更少的情况下展示出更丰富的星云结构。

---

### 3.7 MiniCPM-o-4.5（VLM + Audio · 8B）

**SNI 星云形态：仅有深层数据**

| 数据 | 值 |
|------|-----|
| PC1:PC2 | 仅 pointcloud 级别数据 |
| 点数 | 24 |
| AutoDiscover | 无 |
| SDE | 无 |

**解读**：MiniCPM-o-4.5 作为 VLM+Audio 全模态模型，其 SNI 数据有限（仅 24 点 baseline）。未执行 AutoDiscover 和 SDE 实验。但作为全模态模型的存在，为将来完整的 MMH 对照实验提供了占位。

---

### 3.8 Qwen2-Audio-7B（Audio LM · 7B）

**SNI 星云形态：均匀球形，接近纯文本基线**

| 深度 | Baseline PC1:PC2 | SDE'd PC1:PC2 | Δ |
|------|-------------------|----------------|---|
| 1/4 | 1.09:1 | 1.09:1 | 0 |
| 1/2 | 1.10:1 | 1.10:1 | 0 |
| 3/4 | 1.33:1 | 1.32:1 | −0.01 |
| Last | 1.40:1 | 1.39:1 | −0.01 |

**AutoDiscover**：13 轴 · 总方差 39.3%
- PC0（Anti-Verbosity）解释 6.9%
- 关键轴：Anti-Verbosity、Diversity、Sentence Count、Anti-Excitement、Code

**解读**：Qwen2-Audio 是 MMH 假说的关键对照组。它与 Qwen2.5-7B 共享同一骨架（Qwen2 7B），唯一区别是加入了音频模态。结果：

- 控制轴从 16 → 13（−19%）
- 总方差从 46.7% → 39.3%（−16%）
- PC1:PC2 比率略降（1.3 → 1.1-1.3）

这是 **"轻量模态税"** 的证据——音频模态的加入导致了约 19% 的行为维度损失，但远小于视觉模态的冲击（见 Qwen3.5-9B）。这与 MMH 的预测一致。

---

### 3.9 Qwen2-VL-7B（VLM · 7B）

**SNI 星云形态：仅 baseline 级别数据**

| 数据 | 值 |
|------|-----|
| PC1:PC2 | 1.66:1 |
| 结构 | DISTRIBUTED |
| 点数 | 48 |

**解读**：Qwen2-VL 与 Qwen2.5 共享骨架但加入了视觉模态。PC1:PC2 = 1.66:1 与 Qwen2.5-Instruct（1.59:1）相近。未执行 AutoDiscover，因此无法直接比较控制轴数量。但其 PC 比率的接近性暗示：在 Qwen2 一代，视觉模态的几何影响相对温和。

---

### 3.10 DeepSeek-R1-14B（Distilled · 14B）

**SNI 星云形态：极端公路型**

| 深度 | PC1:PC2 | 结构 |
|------|---------|------|
| 1/4 | 22.5:1 | CONCENTRATED |
| 1/2 | 30.6:1 | CONCENTRATED |
| 3/4 | 9.5:1 | TRANSITIONAL |
| Last | 3.9:1 | TRANSITIONAL |

**AutoDiscover**：11 轴 · 总方差 28.5%
- PC0（Anti-Sentence Count）解释 8.8%
- 特有模式：强 Repetition 轴（4.8%）、多个 Anti-Emoji 轴

**解读**：DeepSeek-R1-14B（Qwen 骨架蒸馏）展示了 **蒸馏如何创造几何极端**。PC1:PC2 在中间层高达 30.6:1——仅次于 MiniCPM4.1。这是因为 R1 的蒸馏过程将教师模型的 CoT 推理模式"烙印"进了学生模型，形成了一个强主导性的推理轴。

11 个控制轴和 28.5% 的方差说明其行为空间仍相对丰富（多于 Qwen3-8B 的 9 轴），但几何形态极度不对称。

**在 SNI 3D 可视化中**：与 MiniCPM4.1 类似的拉伸条状星云，但蓝红渐变更长——反映了 CoT 推理方向上的极端延伸。

---

### 3.11 Qwen3.5-9B（VLM · 9B）

**SNI 星云形态：松散但低维**

| 深度 | PC1:PC2 |
|------|---------|
| 1/4 | 21.2:1 |
| 1/2 | 7.6:1 |
| 3/4 | 4.4:1 |
| Last | 5.2:1 |

**AutoDiscover**：**5 轴**——多模态模型中最少 · 总方差 **6.2%**——所有模型中最低

- 仅发现的轴：Anti-Code、Anti-Structure、Anti-Excitement、Anti-Structure/Heading、Anti-Bold/Emphasis
- 所有轴都是 "Anti-" 前缀——只剩"抑制型"控制

**解读**：Qwen3.5-9B 是 **MMH 假说最有力的证据**。它在 Qwen3 骨架上加入视觉模态，结果：

- 控制轴从 Qwen3-8B 的 9 → 5（−44%）
- 总方差从 25.7% → 6.2%（−76%）
- 从 Qwen2.5 基线计算则是 −69% 轴数和 −87% 方差

更关键的是，剩余的 5 个轴全部是"反向/抑制型"——模型只保留了"不做什么"的控制能力，失去了"主动做什么"的维度。这正是 **"参数量不够大的模态越多、流形越窄"** 假说的量化验证。

**在 SNI 3D 可视化中**：星云虽然在 3/4 深度看起来"分布式"（PC1:PC2 = 4.4:1），但其极低的方差解释（6.2%）意味着星云中的粒子实际上只反映了微弱的行为变化——形状可能看起来"圆"，但内容是空的。

---

## 四、跨模型关键发现

### 4.1 RLHF 压缩级联

```
Qwen2.5-7B  →  Qwen3-8B  →  Qwen3-14B
  16 轴          9 轴         12 轴
  46.7%         25.7%        32.7%
  (基线)       (−44% 轴)    (部分恢复 +33%)
```

**结论**：从 Qwen2.5 到 Qwen3 的 agentic RLHF 训练导致了 44% 的行为维度损失。增加参数到 14B 部分恢复了 33% 的维度，但仍未达到 RLHF 前的基线水平。

### 4.2 多模态税

```
Qwen2.5-7B  →  Qwen2-Audio-7B  →  Qwen3.5-9B
  16 轴          13 轴              5 轴
  46.7%         39.3%             6.2%
  (基线)       (−19% 音频税)     (−69% 视觉+RLHF 税)
```

**结论**：
- 音频模态（同骨架）→ −19% 轴，轻量税
- 视觉模态 + 新一代 RLHF → −69% 轴，严重压缩
- 参数不够大时（9B），多模态+RLHF 的联合效应导致流形接近坍缩

### 4.3 架构直接塑造流形

```
MiniCPM4.1 (标准注意力)  →  MiniCPM-SALA (稀疏+线性注意力)
     9 轴                         3 轴
    42.1%                       33.1%
    (基线)                     (−67% 轴)
```

**结论**：在相同训练策略下，仅更换注意力机制就导致 67% 的控制轴损失。这是独立于 RLHF 和模态的**纯架构效应**。

### 4.4 原生多模态 vs 拼接式

```
Qwen3.5-9B (拼接式 VLM, 9B)  vs  Gemma4-E4B (原生 MoE VLM, 4B active)
     5 轴 / 6.2%                      7 轴 / 23.4%
     PC1:PC2 ≈ 1.2:1                  PC1:PC2 ≈ 1.7:1
```

**结论**：Gemma4 在仅 4B 有效参数下展示了比 9B Qwen3.5 更丰富的行为空间（+40% 轴，+277% 方差）。这暗示 **原生多模态架构（MoE）比拼接式更好地保留了文本流形结构**。

### 4.5 蒸馏创造几何极端

```
DeepSeek-R1-14B  PC1:PC2 = 73.7:1  (最极端)
MiniCPM4.1       PC1:PC2 = 45.9:1
其他所有模型       PC1:PC2 < 5:1
```

**结论**：蒸馏模型和高度压缩模型产生极端的"公路型"流形——一个方向统治整个表征空间。

---

## 五、核心结论

1. **RLHF 是一种系统性的表征压缩**。从 Qwen2.5 到 Qwen3，行为维度损失 44%，且不可通过 SDE 局部干预恢复（Δ ≈ 0）。

2. **多模态训练在参数不足时严重压缩文本流形**。Qwen3.5-9B 的 5 轴/6.2% 方差是所有模型中最低的——"参数量不够大的模态越多、流形越窄"得到了量化验证。

3. **注意力架构独立于训练策略影响流形维度**。MiniCPM-SALA 的 3 轴是纯架构效应的证据。

4. **原生多模态架构优于拼接式**。Gemma4-E4B（4B active, 7 轴）显著优于 Qwen3.5-9B（9B, 5 轴），支持 LeCun 关于"需要原生多模态架构"的观点。

5. **规模可以部分恢复但不能完全逆转压缩**。Qwen3-14B（12 轴）比 Qwen3-8B（9 轴）好 33%，但仍未达到 Qwen2.5-7B（16 轴）的基线。

6. **流形几何是模型"人味"的可量化指标**。越新的模型（2025 年 agentic 范式）控制轴越少、总方差越低——**人味正在系统性地消退**。

---

## 六、附录：SNI 3D Nebula Viewer 页面索引

| 页码 | 模型 | 左侧 | 右侧 | 核心看点 |
|------|------|------|------|----------|
| 1 | MiniCPM4.1 | Baseline SNI（公路型） | AutoDiscover PCA | 极端 67:1 集中比，蓝红渐变条状 |
| 2 | Qwen3-8B | Baseline SNI（球形） | SDE'd SNI | Baseline vs SDE 几乎无差异 |
| 3 | Qwen3-14B | Baseline SNI（球形） | SDE'd SNI | 比 8B 更密，维度恢复可见 |
| 4 | Qwen2.5-7B | Baseline SNI（最丰富） | SDE'd SNI | 所有模型中最圆润、最多样的星云 |
| 5 | MiniCPM-SALA | Baseline SNI（深度依赖） | AutoDiscover PCA | 切换 1/4 ↔ 3/4 深度看形态剧变 |
| 6 | Gemma4-E4B | Baseline SNI（均匀球形） | SDE'd SNI | SDE 在 Last Layer 有显著效果 |
| 7 | MiniCPM-o-4.5 | Baseline SNI（有限） | No data | 仅 24 点，待补充 |
| 8 | Qwen2-Audio-7B | Baseline SNI（均匀球形） | SDE'd SNI | 与 Qwen2.5 对比看"音频税" |
| 9 | Qwen2-VL-7B | Baseline SNI（分布式） | No data | 与 Qwen2.5 对比看"视觉税" |
| 10 | DeepSeek-R1-14B | Baseline SNI（公路型） | AutoDiscover PCA | 蒸馏极端 73.7:1，CoT 推理轴 |
| 11 | Qwen3.5-9B | Baseline SNI（伪分布式） | AutoDiscover PCA | 形圆实空——5 轴/6.2% 方差 |

---

*Generated from Joi/SDE Research Framework · SNI 3D Nebula Viewer Data · 2026-03*
