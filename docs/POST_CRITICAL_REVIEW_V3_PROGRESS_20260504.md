# Post-Critical-Review-v3 Progress Report

## Purpose

本文档总结在接受 `docs/critical_review_v3.md` 的评判之后，项目围绕“如何把研究做到真正好”又推进了哪些工作、目前已经形成了哪些更强的结论、还剩下哪些需要导师判断的关键问题。

它的角色与 `docs/POST_CRITICAL_REVIEW_V2_PROGRESS_20260504.md` 类似：不是论文正文，而是给导师快速了解当前进度、研究强度、和接下来该不该继续做更多实验的汇报材料。

第三轮评审之后，项目目标已经从“补一些 reviewer blocker”进一步收缩和提升为三件事：

1. 把 stale contamination 从 **what/how much** 推进到 **why/when**；
2. 证明这个现象不是 Qwen-only，也不是单一小样本 artifact；
3. 把已有结果整理成足以支持 benchmark + analysis paper 的 manuscript-ready 资产，而不是只停留在分散的 notes 和脚本。

当前最重要的整体结论是：

> P8.0 的第一轮核心分析已经基本完成。我们现在不仅知道 raw_add 的 stale context 会导致 collapse，也已经通过 dose-response、real-context order/label probe、synthetic same-slot probe、expanded latest_per_slot all-k、以及 Llama3.1-8B dev+test replication，把“为什么/何时/是否跨模型成立”推进到了可写进论文主线的程度。当前剩余最大缺口已经不再是 stale mechanism 本身，而是 fair external baseline、最终论文定稿、以及 release-level packaging。

## What changed after `critical_review_v3.md`

### 1. We now have a dose-response story, not only a stale-filter story

`critical_review_v3.md` 最强的一点批评是：我们知道 stale filter 有用，但不知道 stale contamination 的剂量效应是什么样的，也不知道多少 stale 条目就足以造成 collapse。

现在这个问题已经有了第一轮明确答案。

Key artifacts:

```text
scripts/analyze_stale_dose_response.py
results/p80_stale_dose_response/
paper/p80_stale_dose_response_note.md
paper/figures/p80_stale_dose_response.{png,pdf}
```

Current result summary:

- 分析覆盖 1500 个 pooled raw_add slot-prompt examples。
- Stored stale same-slot count 的 binning 显示：

| stored stale count | n | EM | F1 | answer value present |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 300 | 0.967 | 0.970 | 1.000 |
| 1 | 300 | 0.743 | 0.748 | 0.780 |
| 3 | 300 | 0.290 | 0.320 | 0.393 |

- Retrieved stale same-slot count 的 binning 显示：

| retrieved stale count | n | EM | F1 | answer value present |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 201 | 1.000 | 1.000 | 1.000 |
| 1 | 231 | 0.667 | 0.667 | 0.675 |
| 2 | 92 | 0.174 | 0.196 | 0.261 |

- Lightweight logistic fit:

| predictor | n | slope | ED50 |
| --- | ---: | ---: | ---: |
| stored stale same-slot count | 1500 | -0.383 | 3.176 |
| retrieved stale same-slot count | 1000 | -1.082 | 1.895 |

Interpretation:

- 哪怕只有 1 个 stale same-slot 条目，EM 就已经从 0.967 掉到 0.743；
- retrieved stale exposure 比 stored stale burden 更贴近 answer-time failure mechanism；
- 这让论文第一次可以用更“行为科学”风格的 dose-response 语言，而不只是说“stale 多了会坏”。

Paper implication:

- 现在论文可以明确说：**stale contamination starts early**，且 answer-context 中实际暴露出的 stale 条目比 memory bank 中存着多少 stale 条目更重要。

### 2. We now have real-context mechanism probes that isolate answer-layer presentation effects

`critical_review_v3.md` 明确提出：我们需要区分 majority voting、position bias、semantic interference、attention dilution 等机制假说。

第一步 real-context probe 现在已经完成。

Key artifacts:

```text
scripts/eval_evomemory.py
scripts/run_p80_mechanism_probe_batch_sui3.sh
scripts/summarize_context_mechanisms.py
results/p80_mechanism_probes/
results/p80_mechanism_probe_summary/
paper/p80_context_mechanism_probe_note.md
```

Design:

- 在 raw_add k=16 dev 上固定 retrieved entries；
- 只改变 answer-time context presentation：
  - `normal`
  - `chronological`
  - `reverse_chronological`
  - `timestamp`
  - `latest_outdated_label`

Crucial invariant:

所有 formal conditions 的 retrieval composition 相同：

```text
gold retrieved = 0.360
stale retrieved = 1.000
stale retrieved avg = 4.040
```

Results:

| Condition | EM | F1 | answer value present |
| --- | ---: | ---: | ---: |
| normal order, none | 0.110 | 0.136 | 0.140 |
| chronological, none | 0.230 | 0.275 | 0.320 |
| reverse chronological, none | 0.010 | 0.050 | 0.010 |
| timestamp | 0.150 | 0.200 | 0.230 |
| latest/outdated label | 0.260 | 0.298 | 0.350 |

Interpretation:

- 这说明 performance 变化不是 retrieval composition 变化导致，而是 answer-layer 如何利用同一批 retrieved entries 导致；
- chronological 比 normal 好很多，reverse chronological 几乎崩溃；
- latest/outdated label 明显优于纯 timestamp，支持 semantic/version disambiguation 假说；
- 但 labels 仍远低于 latest_per_slot 或 gold context，说明 retrieval bottleneck 依然更大。

Paper implication:

- 论文现在可以更谨慎但更强地说：**stale collapse 不只是 retrieval 问题，也是 answer-layer 的 order/version disambiguation 问题。**

### 3. We now have synthetic same-slot probes that reject a simple majority-vote story

real-context probe 仍有一个限制：gold retrieval 只有 0.36，因此它无法回答“在 gold 一定 present 时，模型为何仍会错”。

这个问题现在通过 synthetic same-slot probe 得到了第一轮控制实验答案。

Key artifacts:

```text
scripts/run_synthetic_same_slot_probe.py
scripts/summarize_synthetic_same_slot_probe.py
results/p80_synthetic_same_slot_probe/
results/p80_synthetic_same_slot_probe_analysis/
paper/p80_synthetic_same_slot_probe_note.md
paper/figures/p80_synthetic_same_slot_matrix.{png,pdf}
```

Probe dimensions:

- stale counts: 0 / 1 / 2 / 4 / 8 / 16
- value policy: `conflict` vs `same_as_current`
- context order: `chronological` vs `reverse_chronological`
- annotation: `none` vs `latest_outdated_label`

Most diagnostic findings:

#### 3.1 Conflict + reverse chronological + no labels collapses immediately

| stale count | 0 | 1 | 2 | 4 | 8 | 16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EM | 0.938 | 0.188 | 0.062 | 0.062 | 0.000 | 0.000 |

#### 3.2 The same conflict setup is almost repaired by latest/outdated labels

| stale count | 0 | 1 | 2 | 4 | 8 | 16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EM | 0.938 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 |

#### 3.3 Chronological conflict remains relatively robust

| stale count | 0 | 1 | 2 | 4 | 8 | 16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EM | 0.938 | 1.000 | 0.875 | 0.938 | 0.812 | 0.750 |

#### 3.4 same_as_current repetition still hurts exact EM

| stale count | 0 | 1 | 2 | 4 | 8 | 16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EM | 0.938 | 0.875 | 0.875 | 0.562 | 0.500 | 0.562 |

while

```text
answer_value_present = 1.000 throughout
```

Interpretation:

- 这几乎排除了“只是多数票效应”的简单解释；
- conflict + reverse order 的灾难性崩溃表明 order 和 version ambiguity 很关键；
- chronological conflict 的稳健性说明 current 在最后时存在 recency/final-position 优势；
- same-value repetition 仍然伤害 EM，说明除了 semantic conflict 之外还有 attention/format dilution。

Paper implication:

- 论文现在可以更有底气地说：**stale contamination is not one phenomenon but several interacting mechanisms.**

### 4. The latest_per_slot k=8 anomaly is now much less mysterious

`critical_review_v3.md` 指出：P7.0 latest_per_slot all-k 中的 k=8 = 0.99 非单调异常如果不解释，会留下很大的说服力问题。

这个问题现在已经通过 expanded split 验证得到了更清楚的判断。

Key artifacts:

```text
scripts/run_p80_expanded_latest_per_slot_sui3.sh
results/p80_expanded_latest_per_slot_allk/
results/p80_expanded_latest_per_slot_summary/
paper/p80_expanded_latest_per_slot_note.md
paper/figures/p80_expanded_latest_per_slot.{png,pdf}
```

Expanded dev results:

| k | EM | F1 | answer value present | gold retrieved | stale retrieved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.955 | 0.970 | 1.000 | 1.000 | 0.000 |
| 2 | 0.940 | 0.954 | 0.980 | 1.000 | 0.000 |
| 4 | 0.855 | 0.855 | 0.905 | 0.980 | 0.000 |
| 8 | 0.925 | 0.929 | 0.935 | 0.990 | 0.000 |
| 16 | 0.750 | 0.764 | 0.775 | 0.860 | 0.000 |

Interpretation:

- k=8 在 expanded split 上仍然局部偏高，但不再接近完美；
- 因而 P6.3 的 0.99 更像 sample/attribute composition artifact，而不是稳定的 retrieval sweet spot；
- 同时，k=16 stale retrieved = 0.000 但 gold retrieved 只有 0.860，EM 只有 0.750，这进一步说明 latest_per_slot 之后仍有 residual retrieval/context-selection/answer-layer gap。

Paper implication:

- 现在可以把 P6.3 k=8 anomaly 从“必须解释的潜在硬伤”降级成“需要诚实标注为小样本/属性组成效应”的现象。

### 5. We now have completed Llama3.1-8B replication on both dev and test

`critical_review_v3.md` 强调：多模型验证不是简单“查重”，而是理解 stale contamination 是否 model-agnostic 的必要科学步骤。

这一点现在已经完成，并且有比预期更有信息量的结果。

Key artifacts:

```text
scripts/run_p80_llama_replication_sui3.sh
scripts/run_p80_llama_test_confirmation_sui3.sh
results/p80_multimodel_stale_susceptibility/llama31_8b/
results/p80_multimodel_stale_susceptibility/llama31_8b_test/
results/p80_multimodel_stale_susceptibility_summary/
paper/p80_multimodel_stale_susceptibility_note.md
paper/figures/p80_llama_stale_susceptibility.{png,pdf}
```

#### Dev (P6.3 hard k=16 dev)

| Condition | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.060 | 0.062 | 0.120 | 0.360 | 1.000 |
| latest_per_slot | 0.290 | 0.341 | 0.750 | 0.860 | 0.000 |
| latest/outdated labels | 0.080 | 0.105 | 0.350 | 0.360 | 1.000 |
| chronological | 0.020 | 0.039 | 0.140 | 0.360 | 1.000 |
| reverse chronological | 0.050 | 0.050 | 0.110 | 0.360 | 1.000 |

#### Test (P6.3 hard k=16 test)

| Condition | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.040 | 0.042 | 0.140 | 0.360 | 1.000 |
| latest_per_slot | 0.290 | 0.345 | 0.760 | 0.870 | 0.000 |
| latest/outdated labels | 0.100 | 0.125 | 0.360 | 0.360 | 1.000 |
| chronological | 0.050 | 0.057 | 0.170 | 0.360 | 1.000 |
| reverse chronological | 0.040 | 0.040 | 0.100 | 0.360 | 1.000 |

Interpretation:

- stale collapse 明显不是 Qwen-only；
- latest_per_slot 在 Llama 上同样有帮助，但恢复远弱于 Qwen；
- labels 和 chronological 这些在 Qwen 上有帮助的 interventions，在 Llama 上几乎不起作用；
- 这说明 stale susceptibility 是 model-agnostic 的，但 mitigation magnitude 和 context-utilization strategy 是 model-dependent 的。

Paper implication:

- 现在论文完全可以避免“单模型现象”的风险；
- 更重要的是，多模型 replication 本身成为一个 finding：**不同 answer model 对 stale context 和 mitigation 的利用方式不同。**

### 6. Long25 现在不只是“口径不冲突”，而是被更谨慎地重定位为 provenance-sensitive checkpoint family result

`critical_review_v3.md` 进一步指出：P6.3 vs P6.5 的巨大差距本身可能是一个关于 learned manager sensitivity 的 finding，但前提是要先确认这是不是纯 seed effect。

我们现在做了更谨慎的 provenance audit。

Key artifacts:

```text
paper/p80_long25_training_provenance_audit.md
results/p70_long25_reproducibility/long25_provenance.{json,csv}
results/p65_stability/long25_seed_stability_summary.{json,csv,md}
```

Current conclusion:

- 我们已经能严格证明：P6.3 original 和 P6.5 family 是不同 checkpoint family；
- 但当前本地 artifacts 还不能证明它们是“只改了 seed、其他训练变量完全相同”的 clean comparison；
- 因而目前最安全的 paper wording 是：

> training/checkpoint-provenance sensitivity

而不是：

> pure seed sensitivity

Paper implication:

- 这使 Long25 叙事更严谨，也避免了把一个还未充分验证的 seed effect 过度包装成 finding。

### 7. Manuscript-ready assets are now much more mature

第三轮评审之后，项目不再只是“脚本和 notes 增多了”，而是已经有了一套可直接支持写作的 manuscript-ready 资产。

Key new assets:

```text
paper/p80_paper_tables.md
paper/figures/p80_figure_manifest.json
paper/manuscript_sections/p80_results_section_draft.md
paper/p80_claim_evidence_matrix.md
paper/p80_canonical_main_number_ledger.md
paper/p80_release_candidate_checklist.md
paper/p80_remaining_work_summary.md
results/p80_evidence_manifest/evidence_manifest.{json,csv,md}
```

What they now provide:

- **tables**：P8 主表和 appendix 表的 markdown source；
- **figures**：dose-response、synthetic matrix、expanded latest_per_slot、Llama stale susceptibility 的 PNG/PDF；
- **section draft**：可直接搬进 LaTeX 的结果段、caption draft、table draft；
- **claim-evidence matrix**：每个论文 claim 对应的脚本/结果文件/caveat；
- **canonical number ledger**：主文数值和具体 source path 的一一对应；
- **release checklist**：哪些 blocker 还没解决，哪些不是 blocker。

This is important because:

- 现在项目的短板已经不再主要是“没想清楚 stale 为什么坏”；
- 短板开始转向“外部 baseline 是否还要继续投入”和“最终论文如何组织得最有说服力”。

## Current paper-level status

相对于 `critical_review_v3.md` 刚提出时，项目已经从：

```text
有 clean finding，但理解太浅
```

推进到了：

```text
有 clean finding，也有一条比较完整的机制证据链；但 external validity 和 final paper shaping 仍然需要决策
```

### Stronger points now

1. **stale contamination 的机制深度显著增强。**
   - 不再只有 latest_per_slot intervention；
   - 已有 dose-response、real-context mechanism probe、synthetic same-slot probe；
   - 已能区分 value conflict、order sensitivity、version ambiguity、repetition/format dilution。

2. **Qwen-only 风险已明显下降。**
   - Llama3.1-8B dev+test replication 完成；
   - stale collapse 和较弱恢复都被复现。

3. **k=8 anomaly 已基本解释。**
   - expanded latest_per_slot all-k 表明 P6.3 的 near-perfect k=8 更像样本/属性组成效应，而不是稳定 sweet spot。

4. **Long25 叙事更严谨。**
   - 从“口径不矛盾”进一步推进到 provenance-sensitive framing；
   - 避免了不必要的过度宣称。

5. **manuscript-ready 资产已经接近成套。**
   - 表、图、section draft、claim-evidence matrix、canonical number ledger、release checklist 都已存在。

### Remaining weaknesses

1. **仍然没有 fair external baseline row。**
   - Mem0 仍卡在 structured extraction compatibility；
   - original Memory-R1 仍不可用。

2. **仍只有一个 non-Qwen answer-model family。**
   - Llama dev+test 已经完成，但第三模型还没有。

3. **benchmark realism 仍受限于 explicit key-value updates。**
   - implicit / negative / conditional / partial updates 仍未进入主 benchmark。

4. **正式论文还没有完成最终 LaTeX 化、引用整理、图表放置。**
   - 现在更像一个 evidence-complete draft package，而不是 ready-to-submit manuscript。

## Recommended claim level now

我的当前判断比 v2 之后略更积极，但仍然应保持克制：

```text
strong workshop / stronger Findings empirical analysis paper
```

它已经不再只是“weak-to-moderate Findings diagnostic”。原因是：

- 现在不仅有 intervention finding；
- 还有 why/when 层面的机制分析；
- 还有跨模型 replication；
- 还有 anomaly check 和 provenance discipline。

但我仍然不会建议把它包装成：

```text
broad main-track memory benchmark paper
```

除非至少完成以下之一：

1. 一个 fair external baseline；
2. 第三个 answer model；
3. 一个更真实但 oracle semantics 清晰的新 update split。

当前最安全也最有说服力的 framing 仍然是：

> This paper introduces a controlled diagnostic benchmark and empirical analysis of stale context contamination under repeated same-slot updates. Its strongest contribution is not a broad leaderboard, but a mechanism-oriented decomposition of how stale context, retrieval, and answer-layer behavior interact.

## Questions for advisor

### 1. Is the current evidence already enough to freeze the experimental scope?

在我看来，围绕 `critical_review_v3.md` 最核心的研究性批评——“你知道 stale bad，但不知道 why/when”——现在已经被明显缓解。

Question:

- 我们是否应该在此冻结核心实验范围，把重心转向正式论文成稿？
- 还是还值得继续补一个第三模型或更真实 update split？

My recommendation:

- **优先冻结 stale-mechanism 这条主线。**
- 如果允许再补一个实验，优先级应低于论文整合。

### 2. Should the Llama replication be a main-paper table or appendix robustness table?

现在 Llama dev+test 都已经完成，而且信息量很高：

- stale collapse generalizes；
- latest_per_slot helps；
- recovery magnitude is model-dependent；
- labels/order interventions transfer poorly。

Question:

- 这组结果应不应该放入主文核心 section？
- 还是只作为 appendix robustness support？

My recommendation:

- 至少应作为主文中的一个简短 robustness paragraph 或小表格，而不是只放 appendix。

### 3. Is it worth investing more time in a fair external baseline now?

Current status:

- Mem0 Qwen2.5-VL dev20: runnable but unfair/qualitative。
- Qwen2.5-7B text server: queryable。
- MiniLM CPU embedding: fixed。
- structured extraction parser: still fails before dev3 completes。

Question:

- 还值不值得继续投入工程时间修 Mem0 extraction compatibility？
- 还是直接承认它是 current limitation，把 paper 定位为 mechanism/diagnostic paper？

My recommendation:

- 如果目标是当前这版 benchmark + analysis paper，**不建议继续深挖 Mem0 工程**；
- 除非导师认为没有 external SDK row 会严重影响目标 venue。

### 4. Should we add a third answer model?

现在已经有 Qwen + Llama。

Question:

- 是否值得再补一个 Mistral-7B 或其他 7B/8B instruct model，形成三模型矩阵？

Pros:

- external validity 更强；
- model-dependent mitigation story 会更完整。

Cons:

- 对主结论的边际收益可能已经低于继续打磨论文结构和图表；
- 计算和整理成本不再很小。

My recommendation:

- 如果只允许再做一件额外实验，这件事比 Mem0 工程更值得；
- 但优先级仍低于论文成稿。

### 5. Should we keep Long25 in the main table, and if so which family?

Current facts:

- P6.3 original Long25: EM/F1 0.48/0.53, state 0.91, stale 1.13, memory 9.43。
- P6.5 three-seed family: EM 0.87–0.89, memory ~2.04。
- 但目前不能严谨宣称这只是 seed 差异。

Question:

- 主表是否保留 Long25？
- 如果保留，用 original family、P6.5 family、还是两者都放？

My recommendation:

- 不要 silently 用 P6.5 替换 P6.3；
- 可以主表保留原始 P6.3 行，把 P6.5 family 放在 provenance/stability 小表或 appendix；
- 除非导师更希望把 learned compact memory 的“最好点”展示出来。

### 6. Is the current paper better positioned as benchmark+analysis, or primarily analysis with benchmark support?

经过 P8.0 之后，论文的重心已经明显变化：

- 最强的部分不是 method comparison 本身；
- 而是 stale contamination 的因果式机制证据链。

Question:

- 我们是否应继续沿用 “benchmark paper” 的主叙事？
- 还是更明确地改成：

> an empirical analysis of stale context contamination, supported by a controlled repeated-update benchmark

My recommendation:

- **我更偏向后者。**
- 因为现在真正 compelling 的不是“有哪些 baseline 排行”，而是“benchmark 让我们看见了 stale contamination 的结构”。

## Suggested next actions

Depending on advisor preference:

### If writing should start now

1. Freeze the core experimental scope.
2. Use the current story as the main narrative:
   - discovery of stale collapse;
   - latest_per_slot intervention;
   - dose-response;
   - real-context mechanism probe;
   - synthetic same-slot mechanism probe;
   - Llama robustness check;
   - Long25 provenance discipline.
3. Convert `paper/manuscript_sections/p80_results_section_draft.md` into final LaTeX body.
4. Place the generated P8 figures and tables.
5. Keep Mem0 and stronger external-validity goals in limitations/future work.

### If one more experiment is allowed

Prioritize one of:

```text
1. a third answer model;
2. a tiny implicit/conditional/negative update pilot with deterministic oracle semantics;
3. a fairer external pipeline only if it can be made to run quickly.
```

My recommendation remains:

- **do not reopen a large engineering branch unless the advisor explicitly wants stronger external validity at the cost of slower paper completion.**
