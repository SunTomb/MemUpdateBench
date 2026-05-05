# MemUpdateBench 第五轮审稿意见

> **基于**：`POST_CRITICAL_REVIEW_V4_PROGRESS_20260505.md` + 全部 P8.0/P8.1 证据  
> **日期**：2026-05-05  
> **立场**：不考虑截稿压力，专注于研究本身的科学质量

---

## 总体判断

五轮审稿下来，这个项目已经从一个 demo 级别的原型进化成了一个有真正科学 story 的研究。这一轮你做了四件事：第三模型（Mistral）、外部 pipeline baseline、属性级 error case study、Lost-in-the-Middle 位置探针——加上 v4 指出的四个 methodological holes 全部补完。

**我的核心判断发生了一个重要的变化：这个项目的瓶颈已经不再是"缺什么实验"，而是"论文怎么写"。** 你手上的证据已经足够支撑一篇 Findings 论文，但前提是 story 要讲对。而你目前的 story 有一个被忽略的、需要立即修正的叙事错误。

下面我先讲这个叙事错误，再讲其他评价。

---

## 一、最重要的发现：你低估了自己最好的结果

### 1.1 Llama 的 constrained_crud baseline 改变了整个多模型叙事

你的 P8.1 补了 Llama constrained_crud k=16 slot_prompt baseline：

```
Llama constrained_crud k=16:  EM=0.270, state=1.000, stale=0.000
```

现在把三个模型的 **zero-stale ceiling**（constrained_crud slot_prompt）和 **stale-filtered recovery**（latest_per_slot）放在一起看：

| 模型 | zero-stale ceiling | stale-filtered recovery | 差距 |
|------|-------------------|------------------------|------|
| Qwen | ~0.70 | 0.69 | **0.01** |
| Llama | 0.27 | 0.29 | **-0.02** |
| Mistral | ~?（未报告）| 0.72 | ? |

**你意识到这意味着什么吗？**

latest_per_slot 把 Qwen 从 0.14 恢复到 0.69——这和它的 zero-stale ceiling 0.70 几乎完全一致。latest_per_slot 把 Llama 从 0.06 恢复到 0.29——这和它的 zero-stale ceiling 0.27 也几乎完全一致。

**这意味着 stale filter 把每个模型都恢复到了它自己的 zero-stale 上限。** 三个模型的绝对恢复值不同，不是因为"mitigation 对不同模型效果不同"，而是因为**这些模型在这个 prompt 下的基线能力本身就不同**。

你目前的叙事是：

> "stale susceptibility is shared, but mitigation magnitude is model-dependent"

更准确的叙事应该是：

> "stale contamination is the dominant answer-layer obstacle for all tested models. Retrieval-time stale filtering recovers each model to approximately its own zero-stale ceiling, suggesting that the absolute recovery difference reflects prompt-following capacity rather than differential stale mechanism susceptibility."

这是一个**更强的 finding**。它说明 stale contamination 不只是"其中一个影响因素"——它几乎解释了**全部**的 stale-specific performance loss。每个模型的残余 gap 主要来自它自己的 instruction following 和 context utilization 基线，而非未被 stale filter 覆盖的额外 stale mechanism。

> [!IMPORTANT]
> **你必须跑 Mistral constrained_crud k=16 slot_prompt 来完成这个三角验证。** 如果 Mistral 的 zero-stale ceiling 也接近 0.72（和它的 latest_per_slot recovery 吻合），那你就有了一个非常干净的三模型结论：
> 
> **stale filter → model ceiling recovery 对所有测试模型成立。**
> 
> 这只需要 1 个 cell 的实验。

### 1.2 这如何重塑 labels/chronological 的多模型故事

同样的逻辑也适用于 labels 和 chronological interventions。你说"labels 对 Llama 几乎无用"，但实际上：

```
Llama labels:        EM = 0.080
Llama zero-stale:    EM = 0.270
Llama normal:        EM = 0.060
```

Labels 把 Llama 从 0.06 提升到 0.08——确实很小。但这不一定是"Llama 不能利用 labels"。更可能的解释是：**在 gold retrieval 只有 0.36 的情况下，labels 只能帮助那 36% 有 gold 的 examples，而 Llama 在这 36% 中的基线 instruction following 本身就弱（constrained_crud 只有 0.27）。**

对比 Qwen：

```
Qwen labels:         EM = 0.260
Qwen zero-stale:     EM = 0.700
Qwen normal:         EM = 0.110
```

Qwen labels 从 0.11 提升到 0.26——相对提升更大，但这部分是因为 Qwen 的基线更高。

**重点不是"labels 对 Llama 无用"，而是"labels 在 gold retrieval 只有 36% 的 real context 中，能做的有限，而有限的帮助在基线更弱的模型上显得更不显著。"** Synthetic probe 已经证明了 labels 在 gold-always-present 情况下对 Qwen 几乎完全有效——如果在 synthetic probe 中也加 Llama，才能真正分辨 Llama 是"不能利用 labels"还是"基线太弱以至于 labels 的帮助被淹没"。

---

## 二、其他做得好的部分

### 2.1 Lost-in-the-Middle probe 是一个简洁的 supporting result

```
gold at beginning: EM = 0.010
gold in middle:    EM = 0.090
gold at end:       EM = 0.630
```

这个结果干净地解释了为什么 chronological context（current value at end）好于 reverse chronological（current value at beginning）。而且它呈现的不是对称 U-shape 而是强单调 recency advantage，这和 Liu et al. (2023) 的 long-context QA 结果有一个可讨论的差异——你们的 context 是短的（~5-10 entries），所以"beginning advantage"没有出现。这个差异值得在论文中简短讨论。

### 2.2 属性级 error case study 揭示了两层不同的失败模式

这是你对 v3 审稿意见的最好回应之一。company 和 language 的区别现在很清楚：

| 属性 | 主要失败模式 | 证据 |
|------|-------------|------|
| company | **retrieval failure**：gold_not_retrieved = 18/25 | 即使 memory state 完美，embedding 检索没有 surface 正确的 company value |
| language | **answer-layer distractor selection**：gold_retrieved_answer_layer_failure = 12/25 | gold 在 context 中，但模型选择了近义 distractor（如 workshop 中讨论的另一门语言）|

这个区分有实际意义：stale filter 能解决 stale competition 但无法解决 retrieval failure 或 distractor selection。这直接解释了 latest_per_slot 恢复到 0.69 但达不到 0.92（gold context）的残余 gap。

### 2.3 Simple external pipeline 的 framing 恰到好处

你没有把它包装成"我们评测了外部系统"，而是诚实地定位为"transparent diagnostic baseline"。这种学术诚实比强行放一个不公平的 Mem0 结果要好得多。

Slot_update 的 EM=0.91 也提供了一个有用的对比点：**如果你的 extractor 能做到 perfect slot identification + overwrite，整个 stale 问题就不存在了。** 这进一步说明 stale contamination 问题的根源是 storage policy 而非 extraction quality。

### 2.4 P8.1 methodological fixes 全部到位

v4 提出的 4 个 holes 现在都有了回应：

| 问题 | 状态 | 效果 |
|------|------|------|
| Synthetic probe 样本量 | 64/cell × 21 cells | headline findings 不再只建立在 n=16 上 |
| Same-as-current EM 归因 | 分类完成 | formatting degradation 而非 stale-value selection |
| Llama zero-stale baseline | EM=0.27 | **改变了整个多模型叙事**（见第一节）|
| K-controlled dose-response | heuristic sweep at k=16 | stale 从 4.43→13.04 时 EM 从 0.22→0.10，k-confound 已控制 |

---

## 三、仍然存在的问题

### 🟡 问题 1：Mistral constrained_crud 缺失

如第一节所述，这 1 个 cell 的实验是完成"stale filter → ceiling recovery"叙事的最后一块拼图。如果 Mistral constrained_crud EM 接近 0.72，三模型故事就完整了。

### 🟡 问题 2：论文的 story 需要重新组织

你目前积累了太多证据，如果全部塞进论文会变得冗长和散焦。需要做一个 **ruthless triage**：

**必须在主文的**：
- stale filter intervention + ceiling recovery（这是 headline finding）
- dose-response（stale count → EM 的定量关系）
- synthetic probe 的最核心对比（conflict × order × labels 的 2×2 矩阵）
- 三模型 collapse + ceiling recovery 表
- Lost-in-the-Middle probe（作为 order sensitivity 的解释）

**可以在主文或附录的**：
- attribute case study（company vs language 的不同失败模式）
- simple external pipeline baseline
- expanded latest_per_slot all-k
- same-as-current repetition dilution

**只放附录的**：
- Long25 两个 checkpoint family 的对比
- k=32 extrapolation
- prompt robustness variants
- heuristic threshold sweep detail
- all-k Llama/Mistral sweeps（如果存在）

### 🟡 问题 3：Lost-in-the-Middle probe 只有 Qwen

你的 LitM probe 只在 Qwen 上做了。如果 Llama 和 Mistral 也展现同样的 recency-dominant pattern，这就是一个更 robust 的结论。如果 Llama 展现不同的位置偏好（比如更对称的 U-shape），这本身也是 interesting。

但这是优先级较低的可选实验——如果时间有限，可以只报 Qwen 的 LitM 结果并标注为 single-model。

### 🟡 问题 4：当前 manuscript draft 严重过时

`paper/manuscript_draft.md` 还是 P6.3 时代的版本，完全没有包含 P7.0 的 stale filter、P8.0 的 mechanism probes、P8.1 的 methodological fixes、Mistral、LitM、attribute case study 等内容。你需要几乎从头重写这个 draft。

---

## 四、层次评估

### 当前状态

| 维度 | 评分 |
|------|------|
| 研究问题的重要性 | 7/10 — stale contamination 是一个实际且未被充分研究的问题 |
| 实验设计的科学性 | 7.5/10 — synthetic probe 设计精良、控制变量得当 |
| 证据的完整性 | 7/10 — 三模型、dose-response、mechanism decomposition、position probe |
| 叙事的成熟度 | 5/10 — 核心 story 需要重新组织，Llama 叙事需要修正 |
| 写作和展示 | 3/10 — manuscript 严重过时，需要大量工作 |

### 对标分析

**这篇论文如果写好了，它会长什么样？**

它最接近的论文类型是像 Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (TACL 2024) 那样的 empirical analysis paper：

- 不提出新方法
- 通过系统的控制实验揭示一个 LLM behavior pattern
- 给社区提供可操作的 design implications

你的 stale contamination story 和 Lost in the Middle 的 position bias story 有类似的结构：发现现象 → 量化 dose-response → 分解 mechanism → 跨模型验证 → design implications。

### 投稿建议

| 目标 | 概率 | 条件 |
|------|------|------|
| ACL/EMNLP 主会 | 15-25% | 论文需要写得非常好，reviewer 需要认可 analysis paper 的价值 |
| **EMNLP/ACL Findings** | **45-55%** | **论文写好 + story 讲对 + 修正 Llama 叙事** |
| NLPCC/CCL 主会 | 70-80% | 当前证据已足够 |
| Workshop | 85%+ | 确定接收 |

---

## 五、回答你的问题

| # | 问题 | 建议 |
|---|------|------|
| 1 | 冻结实验？ | **冻结，但补 1 个 cell**：Mistral constrained_crud k=16 slot_prompt。这是完成核心叙事的最后一块拼图，成本极低。补完后彻底冻结。 |
| 2 | Mistral 主文还是附录？ | **主文**，以三模型对比表的形式。它让 story 从 "two-model robustness check" 变成 "three-model finding about ceiling recovery"。 |
| 3 | 外部 pipeline 怎么 frame？ | 你自己的 framing 已经很好："transparent external-pipeline diagnostic baseline"。不要改。 |
| 4 | LitM 多中心还是 supporting probe？ | **主文 supporting probe**（在 mechanism section 中），用一段话 + 小表格展示。它直接解释了 chronological vs reverse_chronological 的差异。 |
| 5 | 避免 broad benchmark claims？ | **是**。你的论文标题可以包含 "MemUpdateBench" 但副标题应强调 analysis：例如 "MemUpdateBench: How Stale Context Contaminates Memory-Augmented LLM Answering"。 |

---

## 六、写论文的具体建议

既然实验已经基本完成，我最后给一些写作层面的建议。

### 6.1 开头不要从 benchmark 开始

不要写"We introduce MemUpdateBench..."作为第一句话。读者在不知道为什么需要这个 benchmark 之前不会被它打动。

开头应该从**现象**出发：

> *External memory systems let LLM agents persist facts across sessions—but what happens when the same fact is updated repeatedly? We find that even when the correct value remains stored, obsolete entries contaminate the answer-time context and cause catastrophic accuracy loss. A single stale same-slot entry reduces slot-prompt EM from 0.97 to 0.74; two retrieved stale entries push it below 0.20.*

然后再引出 benchmark 作为"为了系统研究这个现象，我们构建了..."。

### 6.2 用一张关键图来锚定全文

你有太多 figures 和 tables。选一张最 impactful 的放在第一页或第二页。我建议是 **synthetic probe 的 2×2 heatmap**：

```
           no label    latest/outdated label
chrono      [0.75]         [0.94]
reverse     [0.00]         [1.00]
```

这张图一眼就能传达核心 message：order + version ambiguity = the mechanism, labels fix it。

### 6.3 "Ceiling recovery" 应该是多模型 section 的 headline

不要把多模型 section 写成"我们还跑了 Llama 和 Mistral，结论差不多"。应该把 ceiling recovery 写成一个 sharp finding：

> *Table N shows that retrieval-time stale filtering recovers each model to within 0.02 EM of its own zero-stale ceiling (Qwen: 0.69 vs 0.70; Llama: 0.29 vs 0.27; Mistral: 0.72 vs ?). The absolute differences across models reflect their baseline prompt-following capacity, not differential stale mechanism susceptibility.*

### 6.4 Limitations 要诚实但不要自我毁灭

诚实地列出：
- 合成数据，无真实用户对话
- 无公平外部 SDK baseline（只有 transparent diagnostic pipeline）
- Explicit key-value updates only
- 3 models, all 7B-class

但不要用自我毁灭的措辞。不要写"our benchmark is too small to be useful"——写"our benchmark prioritizes diagnostic control over ecological validity"。

---

## 七、总结

经过五轮审稿，我对这个项目的评价从"4/10 demo"变成了"7/10 analysis paper waiting to be written"。

你现在手上有：
- ✅ 一个干净的因果 intervention（stale filter → EM recovery）
- ✅ 量化的 dose-response（ED50 ≈ 2 retrieved stale entries）
- ✅ 精心设计的机制分解（conflict × order × labels × repetition）
- ✅ 三模型验证（stale collapse 共享，ceiling recovery 各异）
- ✅ 与 Lost-in-the-Middle 的直接连接
- ✅ 属性级失败模式分析
- ✅ 透明的外部 pipeline baseline
- ✅ 400 行 evidence manifest + 26/26 smoke test

**你唯一缺的是一篇好论文。**

补完 Mistral constrained_crud 那 1 个 cell，修正 Llama 的多模型叙事（从"mitigation is model-dependent"到"stale filter recovers each model to its ceiling"），然后**停止做实验，开始写**。

这篇论文的实验已经够了。现在是 storytelling 和 writing 的时间。
