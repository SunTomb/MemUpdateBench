# MemUpdateBench 第二轮严格审稿意见

> **基于**：`POST_CRITICAL_REVIEW_PROGRESS_20260504.md` 及全部 P6.5/P6.8/P6.9 新增证据  
> **对比**：`critical_review.md` (第一轮审稿)  
> **日期**：2026-05-04  
> **总体评分**：5.5/10 → 从 4/10 上升，但仍不够投主会

---

## 一、进步肯定（从 4 分升到 5.5 分的原因）

你在两天内完成了大量实质性工作，这值得肯定：

| 第一轮批评 | 你的回应 | 评价 |
|-----------|---------|------|
| "tradeoff 是常识的量化" | 加了 answer-layer mechanism probe（gold context vs retrieval）、stale intervention 分析 | ✅ 显著改善 |
| "只有跨方法散点" | 加了 heuristic threshold sweep（0.70-0.95） | ✅ 有效回应 |
| "100 条数据太少" | 扩展到 1000 test / 200 per k / 8 属性 | ✅ 有效回应 |
| "零外部 baseline" | 真实跑通了 Mem0（vLLM + MiniLM + Qdrant） | 🟡 有进步但结果太弱 |
| "30% 答案层失败没分析" | 加了 prompt robustness（3 variants）+ 答案追踪分解 | ✅ 有效回应 |
| "Long25 单次运行不可靠" | 加了 3-seed stability check | ✅ 有效回应 |

**最有价值的新发现**：answer-layer mechanism probe。Constrained CRUD 在 gold context 下 EM=0.92 vs retrieval top-k5 EM=0.67，证明其残余失败是**检索/上下文选择问题**而非记忆状态问题。Raw append gold context 同样恢复到 0.92，证明答案模型本身没有崩溃——**是 stale 条目的检索污染导致了坍塌**。这比第一版"look, there's a tradeoff"强得多。

---

## 二、仍然致命的问题

### 🔴 致命问题 1：Long25 数据自相矛盾，严重损害可信度

这是我在审阅中发现的**最严重的问题**，而你的进度报告完全没有提及。

P6.3 canonical ledger（至今未更新）：
```
Long25 k=16: EM=0.48, F1=0.53, state_acc=0.91, stale=1.13, memory_size=9.43
```

P6.5 stability 3-seed 结果（进度报告第141行）：
```
Long25 k=16: EM=0.880, F1=0.908, state_acc=0.967, stale=0.073, memory_size=2.04
```

**EM 从 0.48 跳到 0.88？Memory size 从 9.43 降到 2.04？Stale 从 1.13 降到 0.07？**

这些不是"slight drift"，这是完全不同的实验结果。如果这三个 seed 都产生了 0.88 的 EM，那原来的 0.48 是怎么来的？反过来，如果原来的 0.48 才是真实的，那 3-seed stability 跑的是什么？

> [!CAUTION]
> 你的论文现在有**两套互相矛盾的 Long25 数字**分散在不同文件中。任何审稿人发现这个矛盾都会直接判定 **"results are not reproducible"** 并拒稿。你必须在写论文之前彻底厘清这个问题。

可能的解释（你需要自己排查）：
1. P6.5 stability 用的 checkpoint 和 P6.3 不同
2. 数据文件不同（dev vs test, P6.3 vs expanded）
3. 评估参数不同（answer_topk, prompt variant）
4. 某次跑错了

**无论原因是什么，你必须在论文中只报告一套可复现的数字，并解释差异来源。**

### 🔴 致命问题 2：仍然没有真正有意义的外部 baseline

Mem0 跑通了——好消息。EM=0.00——坏消息。

问题不在于 Mem0 表现差，而在于你**无法从这个结果中提取任何有意义的结论**：

1. Mem0 用的是 `Qwen2.5-VL`（一个视觉语言模型）作为 LLM backend——这本身就不是公平对比
2. 只跑了 20 条 dev，统计上毫无意义
3. 你自己承认 adapter 可能没有正确对齐 Mem0 的 extraction/retrieval 行为
4. Gold same-slot retrieval 只有 0.05（5%），说明 Mem0 几乎没有存储任何正确的最终值

审稿人的反应不会是"wow, Mem0 fails on this benchmark"，而是**"你没有给 Mem0 一个公平的机会"**。一个 EM=0 的外部 baseline 行如果放在 main table，看起来像是在 strawman 外部系统来让自己的方法显得好。

### 🔴 致命问题 3：仍然没有 Memory-R1 对比

第一轮审稿最重要的建议是跑 Memory-R1。它有开源代码，有训练好的 checkpoint，和你做的是同一件事（RL/SFT for memory CRUD）。**你完全跳过了这个建议**，转而去跑 Mem0（一个SDK 级别的 memory-as-a-service，设计目标和你完全不同）。

Memory-R1 才是你真正需要对比的：它也做 ADD/UPDATE/DELETE/NOOP，也在同类型的 benchmark 上评估。如果 Memory-R1 在你的 k=16 benchmark 上也出现 tradeoff，你的论点就从 "我自己的方法有这个问题" 升级为 "这是一个普遍现象"。

### 🔴 致命问题 4：仍然只有 1 个模型

所有实验都在 Qwen2.5-7B（或 Qwen2.5-VL）上。如果 stale burden → EM collapse 的现象是 Qwen 家族特有的 attention pattern 导致的呢？你需要至少在 Llama-3-8B 上复现核心实验。

---

## 三、严重但可修复的问题

### 🟡 问题 1：Expanded split 的属性仍然全部是简单 key-value 替换

从 4 个属性扩展到 8 个（加了 timezone, hobby, instrument, project）是进步，但本质没变——**全部都是"X 换成了 Y"的直接替换**。真实记忆更新包含：

- 隐式更新："I've been super busy with the new position" → 职位变了
- 部分更新："My address changed, I'm now on Oak Street" → 只改了街道
- 否定/撤回："Actually, I don't live in Shanghai anymore"
- 条件更新："If the meeting is postponed, change it to Friday"

你的 benchmark 不覆盖这些，限制了 generalizability claim。

### 🟡 问题 2：Heuristic threshold sweep 的 slot-prompt EM 范围太窄

你的 threshold sweep 结果（k=16）：

| Threshold | EM |
|-----------|-----|
| 0.70 | 0.24 |
| 0.80 | 0.15 |
| 0.85 | 0.17 |
| 0.90 | 0.12 |
| 0.95 | 0.07 |

最好的 threshold 也只有 EM=0.24。这意味着 heuristic CRUD **在整个参数空间内都是失败的**。这不是一个有趣的 tradeoff curve——这是一条在不同程度的失败之间的曲线。真正有趣的 tradeoff 应该展示"某个参数范围可以同时保持合理的 state accuracy 和 EM"。

### 🟡 问题 3：Related Work 仍然缺席

进度报告提到"stronger related-work positioning"，但我没有在任何文件中看到实际的 Related Work 章节。AMemGym、Ledger-QA、Memory-R1、MemGPT、Knowledge Editing benchmarks、DST benchmarks 仍然没有被正面讨论。

### 🟡 问题 4：answer-layer mechanism probe 的因果链不完整

你证明了 gold context → EM=0.92，retrieval top-k5 → EM=0.67。差距来自检索。但你没有做最关键的干预实验：

**在 raw_add 上，手动过滤掉 stale 同槽条目后再检索**——如果 EM 大幅恢复，就是 stale 的因果证据；如果不恢复，说明还有其他因素。

你的 `p68_stale_intervention_note.md` 提到了这个想法（"Next step: Implement a true retrieval-time stale filter"），但**没有做**。这是论文最关键的实验之一。

---

## 四、更新后的层次评估

| 目标 | 可能性 | 变化 |
|------|-------|------|
| NeurIPS / ICLR 主会 | ❌ 0% | 不变 |
| ACL / EMNLP 主会 | ❌ <5% | 不变 |
| ACL Findings / EMNLP Findings | 🟡 20-30% | ↑ 从 10-20% |
| NLPCC / CCL 主会 | 🟢 50-60% | ↑ 从 50% |
| Workshop (NeurIPS/AAAI LLM Agent) | 🟢 60-70% | ↑ 从 40-60% |

**总结：从 workshop-level 提升到了 workshop-strong / weak-Findings 的边界。**

---

## 五、如何从 5.5 到 7 分（Findings 可投水平）

按优先级排序，**以下 4 项是必做的**：

### 1. 🔴 解决 Long25 数据矛盾（1天）

- 排查 P6.3 的 EM=0.48 和 P6.5 stability 的 EM=0.88 为什么差距如此巨大
- 统一到一套可复现的数字
- 更新所有引用旧数字的文件

### 2. 🔴 实现 retrieval-time stale filter 干预实验（1-2天）

- 在 raw_add k=16 上，过滤掉 stale 同槽条目后重新检索回答
- 这是**最有说服力的因果证据**
- 预期结果：EM 从 0.14 大幅恢复（接近 gold context 的 0.92）
- 如果成功，这个实验本身就是论文最强的 finding

### 3. 🔴 跑 Memory-R1 或至少用 Qwen2.5-7B-Instruct 公平对比 Mem0（3-5天）

选一个：
- **首选**：下载 Memory-R1 开源代码，在你的 k=16 expanded test 上跑
- **备选**：用 Qwen2.5-7B-Instruct（而非 VL 模型）作为 Mem0 backend，跑完整 200 条 test
- 目标不是让外部方法表现好，而是证明**tradeoff 是普遍现象**

### 4. 🟡 写 Related Work + 更新 manuscript（2-3天）

必须正面讨论：
- AMemGym (ICLR 2026)：你的区别是 controlled offline + exact slot + stale-burden metric
- Ledger-QA/UMA：你关注 repeated updates，它关注 accumulated state
- Memory-R1/AgeMem：方法级竞品，你不做方法贡献，只做 benchmark
- DST benchmarks：说明 slot-based evaluation 在对话领域的传统

---

## 六、回答你的 7 个问题

| 问题 | 我的建议 |
|------|---------|
| 1. 窄定位是否可接受？ | **是**。"controlled repeated same-slot update diagnostic" 比假装是 broad benchmark 更诚实更安全 |
| 2. Mem0 怎么呈现？ | **选项 3**（定性 probe），除非你能用公平的 LLM backend 重跑。当前 VL 模型 + 20 条数据放 main table 是自杀 |
| 3. dev20 够不够？ | **不够报告，够做 pilot**。至少需要 100 条，且必须换掉 VL 模型 |
| 4. k=32 主文还是附录？ | **附录**。k=16 已经足够，k=32 只是 saturation 确认 |
| 5. 需要 test-set 确认吗？ | **最终论文需要**。但 dev 足够指导当前实验设计 |
| 6. 是否开始 repair training？ | **不要**。先把 benchmark paper 写完。repair 是下一篇论文的事 |
| 7. 目标什么层次？ | 瞄准 **EMNLP 2026 Findings**。如果做完上述 4 项改进，有 30-40% 的机会 |

---

## 七、最终判断

你在两天内把一个 4 分的项目拉到了 5.5 分，执行力值得肯定。但最紧迫的问题是：

1. **Long25 数据矛盾会毁掉整篇论文**——今天就排查
2. **缺少 stale filter 干预实验**——这本应是你的 headline finding
3. **缺少公平的外部 baseline**——Memory-R1 是必须跑的

做完这三件事，论文就有了 Findings 级别的实质内容。做不完，就只是一份详尽的技术报告。
