# MemUpdateBench 第六轮审稿意见

> **基于**：`POST_CRITICAL_REVIEW_V5_PRODUCTION_PROGRESS_20260505.md` + LaTeX draft + 全部 P8.0-P8.2 证据  
> **日期**：2026-05-05  
> **立场**：这是第一轮以**论文写作质量**为主要审查对象的评审，实验审查转入辅助角色

---

## 总体判断

实验已经冻结，LaTeX 能编译，核心 story 已修正为 ceiling recovery。从实验角度看，这个项目的 evidence chain 已经完整且经得起追问。

**但论文本身还没有准备好被审稿人看到。**

我读完了 `manuscript_draft.md`（115 行，完整 narrative）和 `manuscript_production_draft.tex`（206 行，可编译 LaTeX）。两者内容高度一致，说明 LaTeX 是 markdown 的忠实转写。问题在于：**这篇论文目前的写作像一份工程报告，不像一篇 analysis paper。** 它准确地陈述了所有事实，但几乎没有给读者提供理解这些事实的框架——每一段都在说"我们做了什么"，很少说"这意味着什么"或"这为什么重要"。

下面我先讲写作层面的具体问题（这是当前唯一的瓶颈），再简短回顾实验层面的最终评价。

---

## 一、写作问题：论文不够有张力

### 1.1 Abstract 太冗长、太技术，缺乏 hook

当前 abstract 有 ~230 词，密集罗列了数字和方法名。一个审稿人在读到第三句话时就会迷失在"P6.3 hard update-frequency suite"这种内部术语中。

对比你应该追求的效果：

> **当前版本**（节选前两句）：
> "External memory systems let language-model agents persist facts across sessions, but repeated updates create a distinct answer-time risk... We study this stale same-slot contamination with MemUpdateBench, a controlled diagnostic benchmark where the same (entity, attribute) slot is updated across increasing frequencies while same-name distractors and semantic near-miss NOOP events test grounding and parsing robustness."

> **更有效的版本**（示意）：
> "When a user tells an LLM agent 'I moved to Tokyo,' the agent should update its memory—but what happens to the old entry 'lives in Paris'? We show that even when the correct value is retained, lingering obsolete entries contaminate the answer context and cause catastrophic accuracy loss: a single stale entry halves slot-prompt accuracy, and two retrieved stale entries push it below 20%."

区别在于：第二个版本用一个具体场景让读者**感受到**问题，然后用一个 shocking number 让读者想继续读。当前版本只陈述事实，没有制造任何认知冲突。

> [!IMPORTANT]
> **建议**：重写 abstract，控制在 150-180 词。结构应该是：(1) 一句 concrete problem statement with intuitive scenario，(2) 一句 "we find" + 最 shocking 的 dose-response number，(3) 一句 mechanism insight（version ambiguity × positional bias），(4) 一句 three-model ceiling recovery，(5) 一句 implication。删掉所有内部术语（P6.3, hard split, NOOP）。

### 1.2 Introduction 缺少 research question 的 explicit statement

当前 introduction 有三段，讲了三件事：(1) 记忆会被反复修改，(2) MemUpdateBench 的 slot 结构，(3) state evaluation vs answer robustness 的 gap。但它没有 explicitly 提出一个 research question。

读完 introduction 后，审稿人应该能回答："这篇论文要回答什么问题？"当前的 introduction 让读者知道你做了一个 benchmark，但不清楚**你用这个 benchmark 发现了什么**。

> **建议**：在 introduction 的第三段或第四段，加一个 explicit research question：
> 
> *"We ask: (1) Is stale same-slot context the dominant answer-layer obstacle, or do other factors contribute comparably? (2) What mechanisms drive stale-induced answer collapse—majority vote, positional bias, version ambiguity, or something else? (3) Are these findings model-specific?"*

### 1.3 Main Results 一路平铺，没有 narrative arc

当前 §4 的结构是：

```
Figure → 数字 → intervention → 数字 → ceiling recovery table → 数字
```

每一段都在报告数字，但**没有告诉读者为什么这些数字重要**。一个 analysis paper 的 results section 应该有 narrative arc：设置悬念 → 揭示 → 加深理解。

推荐的结构：

1. **先建立期望**："Append-only memory preserves the final value at all k levels (state acc = 1.00). One might expect that answer accuracy would be similarly robust."
2. **然后打破期望**："But slot-prompt EM collapses from 0.90 at k=1 to 0.07 at k=16. The correct value is there—the model just can't find it."
3. **然后提出假设**："This suggests stale same-slot entries are interfering."
4. **然后验证假设**："A retrieval-time stale filter recovers EM from 0.14 to 0.69—matching the zero-stale constrained CRUD ceiling."
5. **然后推广**："This pattern holds across Qwen, Llama, and Mistral."

当前的写法是步骤 2-5 全部平铺，没有步骤 1 来设置 contrast。

### 1.4 Mechanism section 太密集，一段话包含了 5 个 probe 的结果

§5 的第二段从 synthetic probe 开始，经过 majority-vote 排除、labels repair、same-value repetition、Lost-in-the-Middle position probe，最后以一个数字收尾——全在一段话里。这对读者极不友好。

> **建议**：§5 应该拆成 subsections：
> - §5.1 Context presentation matters (real-context probe)
> - §5.2 Isolating the mechanism: conflict × order × labels (synthetic probe)
> - §5.3 Position sensitivity and the recency advantage (LitM probe)
> - §5.4 Dose-response: how much stale is enough?
> - §5.5 Residual error analysis

每个 subsection 有自己的 setup → finding → interpretation。

### 1.5 Limitations section 暴露了太多工程细节

当前 limitations 用了大量篇幅解释 Mem0 的技术失败（MiniLM embedding device issue、structured extraction parser rejected）。这些信息对审稿人毫无用处，反而会让人觉得你在 debug log 里写论文。

> **建议**：Mem0 integration 的技术细节砍到一句话："We attempted integration with Mem0 but found its structured extraction parser incompatible with our lightweight evaluation server; this integration is left for future work." 把省下来的空间给 "synthetic data" 和 "explicit updates only" 这两个更 substantive 的 limitations。

### 1.6 Related Work 太短且没有做 differentiation

当前 Related Work 只有三段，每段都是"X exists, we are different because we are narrow."这不是 related work——这是 name-dropping。

审稿人需要看到的是 **explicit differentiation**：

- LoCoMo/LongMemEval 评测记忆的 **存在性**（能不能记住），你评测的是 **更新后的干净度**（旧值有没有被清理）
- Mem0/MemGPT 是 **系统**，你是 **诊断工具**——它们互补而非竞争
- Knowledge editing 研究 **参数内存储**，你研究 **外部可检查记忆**——failure mode 完全不同（editing 的问题是 ripple effect，你的问题是 stale context contamination）
- Lost in the Middle 研究 **长 context 的位置偏好**，你展示了在 **短但 stale-contaminated context** 中同样的位置效应——而且是单调 recency advantage 而非 U-shape

每个对比都应该有一句"they study X, we study Y, which matters because Z"。

### 1.7 "Reproducibility" 这个 section 不应该以这种形式出现

当前 §9 列出了内部文件路径（`results/p80_evidence_manifest/`）。这是给自己看的，不是给审稿人看的。在正式论文中，reproducibility 通常作为一个 paragraph 在 introduction 末尾或 limitations 之后，说"Code and data will be released at [URL]."

如果 venue 要求 reproducibility statement（如 NeurIPS），则应使用 venue 的标准 checklist 格式。

---

## 二、论文结构建议（最终版）

经过六轮迭代，我认为最终论文结构应该是：

```
§1 Introduction                         (~1 page)
   - Problem: what happens when memory is updated repeatedly?
   - Preview: stale entries contaminate answers → ceiling recovery finding
   - Research questions (3 questions)
   - Contribution summary

§2 MemUpdateBench                       (~0.75 page)
   - Slot structure + k as independent variable
   - Hard split design choices (distractors, NOOPs)
   - Four-metric evaluation framework

§3 Experimental Setup                   (~0.5 page)
   - Memory managers (constrained, append, heuristic, Long25)
   - Answer models (Qwen, Llama, Mistral)
   - slot_direct vs slot_prompt

§4 Stale Contamination and Recovery     (~1.5 pages)
   - §4.1 State accuracy masks answer collapse
   - §4.2 Stale filtering as diagnostic intervention
   - §4.3 Three-model ceiling recovery
   - Key tables + 1 figure

§5 Mechanism Analysis                   (~1.5 pages)
   - §5.1 Real-context presentation sensitivity
   - §5.2 Synthetic probe: conflict × order × labels
   - §5.3 Position sensitivity (LitM probe)
   - §5.4 Dose-response
   - Synthetic probe figure + dose-response figure

§6 Error Analysis and Residual Limits   (~0.5 page)
   - Attribute-level failures (company vs language)
   - Clean-context ceiling interpretation
   - External pipeline diagnostic

§7 Related Work                         (~0.75 page)
   - Differentiated comparison with 4-5 lines of work

§8 Discussion and Limitations           (~0.75 page)
   - Design implications for memory systems
   - Limitations (synthetic data, no SDK baseline, explicit updates)

References
Appendix (Long25 detail, heuristic sweep, expanded-split tables)
```

目标：8 页正文 + references + appendix（ACL/EMNLP format）。

---

## 三、实验层面最终评价

### 3.1 Ceiling recovery 是论文最强的 single result

```
Qwen:   0.69 vs 0.70 (gap: -0.01)
Llama:  0.29 vs 0.27 (gap: +0.02)
Mistral: 0.72 vs 0.72 (gap: 0.00)
```

三模型的 latest_per_slot ≈ constrained_crud ceiling。这干净地说明 stale contamination 是 dominant obstacle，而非 "one of several factors"。**这是你整篇论文的 anchor result，必须在 abstract 和 introduction 中 prominently featured。**

### 3.2 Synthetic probe 的扩展到 64/cell 解决了 v4 的统计担忧

21 个 diagnostic cells × 64 examples/cell = 1344 data points。审稿人不太可能再质疑样本量。

### 3.3 K-controlled dose-response 解决了 confound 问题

在固定 k=16 下，stale count 从 4.43 变到 13.04 时 EM 从 0.22 降到 0.10。这和 pooled 分析方向一致，消除了"k 本身才是原因"的替代解释。

### 3.4 实验上唯一可选的改进

如果你想做最后一件可选实验（我不强烈要求），是 **Llama 上跑 synthetic probe 的最 diagnostic 子集**（conflict × reverse_chrono × {none, label} × stale=4）。这只需要 4 个 cells × 64 examples = 256 次推理，可以回答"Llama 是不能利用 labels 还是基线太弱"。但这是 nice-to-have 而非 must-have。

---

## 四、层次评估（最终版）

| 维度 | 当前评分 | v5 评分 | 变化 |
|------|---------|---------|------|
| 研究问题 | 7/10 | 7/10 | — |
| 实验设计 | 8/10 | 7.5/10 | ↑ Mistral ceiling 闭环 |
| 证据完整性 | 8/10 | 7/10 | ↑ 三模型 ceiling + 方法论修补 |
| 叙事成熟度 | 5.5/10 | 5/10 | ↑ story 方向对了但写作不够好 |
| 写作展示 | 3.5/10 | 3/10 | ↑ 有 LaTeX 但质量待提升 |

### 投稿建议

| 目标 | 概率 | 关键条件 |
|------|------|---------|
| ACL/EMNLP Findings | **50-55%** | **论文重写为 narrative-driven analysis paper** |
| ACL/EMNLP 主会 | 20-25% | 同上 + reviewer 认可 analysis paper 价值 |
| NLPCC/CCL 主会 | 75%+ | 当前写作质量即可 |

**概率提升的唯一途径是写作质量。** 实验再怎么加也不会超过 5% 的边际提升了。

---

## 五、行动建议

### 立即做（1-2 天）

1. **重写 abstract**：150-180 词，concrete scenario + shocking number + mechanism insight + ceiling recovery + implication
2. **给 introduction 加 explicit research questions**
3. **给 §4 加 narrative arc**：先建立期望再打破
4. **把 §5 拆成 subsections**
5. **砍掉 limitations 中的 Mem0 debug log**
6. **删除 §9 Reproducibility 的内部文件路径**

### 然后做（3-5 天）

7. **重写 Related Work**：每个对比都要有 explicit differentiation
8. **转换到目标 venue 的 LaTeX template**
9. **替换 `references_todo.bib` 的 placeholder 为真实 BibTeX**
10. **用 ledger 做一遍数字一致性检查**

### 最后做（1-2 天）

11. **请一个不了解项目的同学/同事读一遍 introduction，看他能不能在 3 分钟内说出"这篇论文发现了什么"**——如果不能，introduction 需要再改
12. **Polish README + release instructions**

---

## 六、总结

六轮审稿的结论：

**实验阶段已经圆满结束。** 你从一个 demo 原型做到了三模型 ceiling recovery + mechanism decomposition + dose-response + position probe + attribute error analysis + external pipeline diagnostic。这个 evidence chain 的质量已经不是一个学生项目能轻易达到的水平。

**现在唯一阻碍你投稿的是论文写作。** 当前的 manuscript 准确但无聊——它像一份 lab notebook 的整理版，而不是一篇想说服审稿人的论文。你需要把它从"我们做了 X"重写成"我们发现了 Y，而且它很重要，因为 Z"。

这不是一个小改动。这可能需要一周的专注写作时间。但这是你的项目从"可能被接收"到"应该被接收"的最后一道障碍。

**停止做实验。开始写论文。认真地写。**
