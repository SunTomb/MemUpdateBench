# Post-Critical-Review-v4 Progress Report

## Purpose

本文档总结在接受 `docs/critical_review_v4.md` 的评判之后，项目围绕“做到真正有说服力”又推进了哪些工作、哪些 v4 指出的 methodological holes 已经补上、哪些仍然没有做、以及现在需要导师判断的关键问题。

它的角色与 `docs/POST_CRITICAL_REVIEW_V3_PROGRESS_20260504.md` 类似：不是论文正文，而是给导师快速了解当前状态、证据强度、和下一步是否应进入最终论文成稿阶段的汇报材料。

第四轮评审之后，项目目标从 P8.0 的“形成 mechanistic story”进一步变成：

1. 修补 v4 指出的关键可信度短板；
2. 证明 stale contamination 不只是 Qwen/Llama pair artifact；
3. 给论文补一个透明 external-pipeline diagnostic row；
4. 把 attribute sensitivity 和 Lost-in-the-Middle 关系变成可写入正文的分析，而不是停留在猜测。

当前最重要的整体结论是：

> v4 之后，导师提出的“做到真正有说服力”的追加要求已经完成：第三模型 Mistral、simple extract-then-store external pipeline、company/language gold-retrieved error case study、Lost-in-the-Middle gold-position probe，以及 v4 后续指出的 4 个“1 周 methodological holes”都已经有结果并整合进 notes / ledger / table pack / workflow。项目现在的短板已经从“缺关键补强实验”转向“最终论文如何取舍主文空间、完成 LaTeX/citation/consistency/release polishing”。

## What changed after `critical_review_v4.md`

### 1. We now have a third answer model, and it is more informative than a checkmark

`critical_review_v4.md` 建议如果要做到真正有说服力，应在 Qwen + Llama 之外补第三个模型，例如 Mistral-7B 或 Phi-3。

当前已经完成 Mistral-7B-Instruct-v0.1 的 P6.3 hard k=16 dev matrix。

Key artifacts:

```text
scripts/run_p80_third_model_replication_sui3.sh
results/p80_multimodel_stale_susceptibility/mistral7b/
results/p80_multimodel_stale_susceptibility_summary/mistral7b_context_summary.{json,csv,md}
paper/p80_third_model_mistral_note.md
paper/p80_multimodel_stale_susceptibility_note.md
```

Available model resources checked on Sui-3:

```text
/NAS/HuggingFaceModels/Mistral-7B-Instruct-v0.1
/NAS/HuggingFaceModels/Phi-3-mini-4k-instruct
```

Mistral results:

| Condition | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.080 | 0.177 | 0.140 | 0.360 | 1.000 |
| latest_per_slot | 0.720 | 0.735 | 0.750 | 0.860 | 0.000 |
| latest/outdated labels | 0.300 | 0.332 | 0.350 | 0.360 | 1.000 |
| chronological | 0.150 | 0.182 | 0.190 | 0.360 | 1.000 |
| reverse chronological | 0.040 | 0.117 | 0.060 | 0.360 | 1.000 |

Comparison with earlier Qwen/Llama dev results:

| Condition | Qwen EM | Llama EM | Mistral EM |
| --- | ---: | ---: | ---: |
| normal top-k5 | 0.110 | 0.060 | 0.080 |
| latest_per_slot | 0.690 | 0.290 | 0.720 |
| latest/outdated labels | 0.260 | 0.080 | 0.300 |
| chronological | 0.230 | 0.020 | 0.150 |
| reverse chronological | 0.010 | 0.050 | 0.040 |

Interpretation:

- stale collapse 现在已经跨 Qwen / Llama / Mistral 三个 answer model family 成立；
- Mistral 的 latest_per_slot recovery 很强，接近 Qwen，而不是 Llama；
- 因此 Llama 的弱恢复不能简单代表所有非 Qwen 模型；
- 更准确的 finding 是：**stale susceptibility is shared, but mitigation magnitude and context-utilization strategy are model-dependent**。

Paper implication:

- 现在论文可以避免 “two-model anecdote” 风险；
- Mistral 让 cross-model section 更像一个 finding，而不是 robustness check；
- 但当前 Mistral 只有 dev，没有 test-split confirmation。是否需要 Mistral test 取决于导师对主文强度和时间成本的判断。

### 2. We now have a simple external extract-then-store pipeline baseline

`critical_review_v4.md` 明确建议：不要继续深挖 Mem0，如果需要 external baseline，可以自己实现一个简单 extract-then-store pipeline。

这项已经完成。

Key artifacts:

```text
scripts/eval_simple_external_pipeline.py
scripts/summarize_simple_external_pipeline.py
results/p80_simple_external_pipeline/
results/p80_simple_external_pipeline_summary/simple_external_pipeline_summary.{json,csv,md}
paper/p80_simple_external_pipeline_note.md
```

Design:

- 使用项目 parser 从 event 中抽取 inspectable `(entity, attribute, value)` records；
- 比较两种 storage policy：
  - `append`: parsed records 全部追加，作为 raw append analogue；
  - `slot_update`: same `(entity, attribute)` overwrite，作为最简单 structured external pipeline；
- 使用同一套 state / stale / compactness / answer metrics；
- memory records 可保存和检查。

Results:

| Baseline | answer | EM | F1 | state | stale same-slot | memory size | answer value present |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| append parsed-only | slot_direct | 1.000 | 1.000 | 1.000 | 14.250 | 31.000 | 1.000 |
| append parsed-only | slot_prompt | 0.140 | 0.177 | 1.000 | 14.250 | 31.000 | 0.170 |
| slot_update parsed-only | slot_direct | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 1.000 |
| slot_update parsed-only | slot_prompt | 0.910 | 0.926 | 1.000 | 0.000 | 2.000 | 1.000 |

Interpretation:

- 同一个 transparent extractor 下，append storage 复现 stale collapse；
- slot overwrite storage 得到 compact、zero-stale、高 answer robustness 的 clean external-pipeline row；
- 这不是 Mem0 替代品，也不是外部 SDK leaderboard row；
- 但它满足导师要求：证明 benchmark 可以评价一个非 learned、可 inspect 的 extract-then-store memory pipeline。

Paper implication:

- 外部 baseline 缺口现在可以更温和地表述：没有 fair external SDK row，但有 transparent external-pipeline diagnostic row；
- Mem0 structured extraction failure 可以留在 limitations / appendix，而不再是主线 blocker；
- 论文如果定位为 empirical analysis paper，这一行足以支持 external diagnostic utility 的最低要求。

### 3. Attribute sensitivity is now more precise: company and language fail for different reasons

`critical_review_v4.md` 要求解释 attribute sensitivity，特别是 company/language 在 gold retrieved 条件下为什么仍然答错。

现有 P8 note 已经有初版 attribute analysis，但 v4 后进一步加了 gold-retrieved-wrong-answer case extraction 和 error-type taxonomy。

Key artifacts:

```text
scripts/analyze_attribute_failures.py
results/p80_attribute_error_analysis/company_language_error_type_summary.csv
results/p80_attribute_error_analysis/gold_retrieved_wrong_cases.{csv,md}
paper/p80_attribute_error_case_study.md
```

k=16 company/language error-type summary:

| Method | Attribute | Error type | N |
| --- | --- | --- | ---: |
| Constrained CRUD | company | correct | 7 |
| Constrained CRUD | company | gold_not_retrieved | 18 |
| Constrained CRUD | language | correct | 3 |
| Constrained CRUD | language | gold_not_retrieved | 10 |
| Constrained CRUD | language | gold_retrieved_answer_layer_failure | 12 |
| Raw append | company | correct | 3 |
| Raw append | company | gold_not_retrieved | 18 |
| Raw append | company | gold_retrieved_stale_competition | 4 |
| Raw append | language | correct | 4 |
| Raw append | language | gold_not_retrieved | 21 |

Interpretation:

- `company` k=16 的主要失败不是 gold retrieved 后仍答错，而是 clean-state retrieval/context-selection failure：Constrained CRUD 下 18/25 是 `gold_not_retrieved`；
- `language` 更符合导师提出的问题：Constrained CRUD 下 12/25 是 `gold_retrieved_answer_layer_failure`；
- language 的典型 gold-retrieved wrong cases 是：gold update 已经在 context 中，但 retrieved context 同时有强 near-miss distractors，例如 “discussed TypeScript in a workshop”，模型复制 distractor value；
- 因此 company/language 不应混为一谈：它们揭示的是两个不同 failure layer。

Paper implication:

- 论文可以更细地说：attribute sensitivity is not one thing；
- `company` 暴露的是 retrieval/context-selection under clean memory；
- `language` 暴露的是 gold-retrieved answer-layer distractor selection；
- 这有助于解释为什么 latest_per_slot 不能完全达到 gold-context 上界。

### 4. We now have a direct Lost-in-the-Middle-style gold-position probe

`critical_review_v4.md` 建议显式比较 “Lost in the Middle”：把 gold 放在 context 不同位置。

这项已经完成。

Key artifacts:

```text
scripts/run_lost_in_middle_probe.py
scripts/summarize_lost_in_middle_probe.py
results/p80_lost_in_middle_probe/qwen25_k16_dev/lost_in_middle_results.json
results/p80_lost_in_middle_probe_summary/lost_in_middle_summary.{json,csv,md}
paper/p80_lost_in_middle_probe_note.md
```

Design:

- fixed synthetic context set；
- same gold entry + same distractors；
- only move gold entry to:
  - beginning；
  - middle；
  - end；
- Qwen2.5-7B-Instruct；
- P6.3 hard k=16 dev；
- 100 examples；
- 8 distractors。

Results:

| Gold position | N | EM | F1 | answer value present |
| --- | ---: | ---: | ---: | ---: |
| beginning | 100 | 0.010 | 0.073 | 0.040 |
| middle | 100 | 0.090 | 0.183 | 0.190 |
| end | 100 | 0.630 | 0.654 | 0.720 |

Interpretation:

- 这是一个非常强的位置效应；
- 固定 context set 只移动 gold entry 时，gold-at-end 明显好于 middle/beginning；
- 这不是标准 long-context QA 中常见的“开头和结尾都好，中间差”的对称 U-shape，而更像 MemUpdateBench slot answering 下的 **strong final-position / recency advantage**；
- 它直接解释了为什么 chronological context 更好、reverse chronological 更差：current value 出现在后面时模型更容易用它。

Paper implication:

- 论文现在可以明确连接 stale contamination 和 Lost-in-the-Middle / position bias literature；
- 但措辞应谨慎：我们的结果是 Lost-in-the-Middle-style gold-position sensitivity，不应声称复现了所有 long-context setting 中的完整 U-shape。

### 5. The P8/P8.1 evidence package has been refreshed and now includes the v4 additions

Key refreshed artifacts:

```text
results/p80_evidence_manifest/evidence_manifest.{json,csv,md}
paper/p80_paper_tables.md
paper/figures/p80_figure_manifest.json
paper/p80_canonical_main_number_ledger.md
paper/p80_release_candidate_checklist.md
paper/p80_remaining_work_summary.md
WORKFLOW.md
```

Current validation after P8.1:

```text
PYTHONPATH=. python -m py_compile ...
PYTHONPATH=. python scripts/package_p80_paper_tables.py
PYTHONPATH=. python scripts/package_p80_figures.py
PYTHONPATH=. python scripts/build_evidence_manifest.py --result_root results --output_dir results/p80_evidence_manifest
PYTHONPATH=. python scripts/smoke_test.py
SMOKE TEST: 26/26 passed
```

Current manifest count:

```text
400 result rows
```

The P8/P8.1 table pack now includes:

- dose-response;
- P8.1 fixed-k heuristic threshold dose-response;
- real-context mechanism probe;
- Llama stale susceptibility;
- P8.1 Llama constrained zero-stale control;
- Mistral third-model stale susceptibility;
- P8.1 expanded synthetic same-slot diagnostic subset;
- simple extract-then-store external pipeline;
- Lost-in-the-Middle gold-position probe;
- expanded latest_per_slot all-k.

## Current paper-level status after v4 additions

Compared with the status at the time of `docs/critical_review_v4.md`, the project has moved from:

```text
structured mechanistic story, but still missing several credibility upgrades
```

to:

```text
structured mechanistic story plus cross-model, external-pipeline, attribute, position-sensitivity, and P8.1 methodological-rigor support
```

### Stronger points now

1. **Cross-model evidence is stronger.**
   - Qwen + Llama + Mistral now all show stale collapse;
   - Mistral shows strong recovery like Qwen;
   - Llama remains a useful counterexample for weaker mitigation transfer.

2. **External diagnostic utility is no longer purely aspirational.**
   - Mem0 remains unfair/blocked;
   - but simple extract-then-store pipeline is complete and inspectable;
   - append vs slot_update cleanly reproduces the benchmark's core stale-vs-overwrite distinction.

3. **Attribute sensitivity is more interpretable.**
   - company and language no longer have to be lumped together;
   - company is mostly gold-not-retrieved;
   - language has genuine gold-retrieved answer-layer failures.

4. **Position bias is now directly linked to Lost-in-the-Middle-style behavior.**
   - gold-at-end 0.630 vs middle 0.090 vs beginning 0.010 is a clear result;
   - this strengthens the “version ambiguity × positional bias” mechanism story.

5. **The P8.1 methodological caveats are now closed rather than pending.**
   - expanded synthetic selected cells now have 64 examples per cell;
   - same_as_current exact-match failures have a dedicated classification;
   - Llama has a zero-stale constrained CRUD control;
   - dose-response now has a fixed-k heuristic threshold sweep.

6. **Release/provenance infrastructure is stronger.**
   - table pack, ledger, checklist, notes, and evidence manifest all include v4 and P8.1 additions.

### Remaining weaknesses

The main remaining weaknesses are not the same as before.

#### 1. v4's “1-week methodological holes” are now addressed

`critical_review_v4.md` also recommended four lower-cost methodological fixes before freezing:

1. expand the most diagnostic synthetic probe cells to 64+ examples per cell;
2. inspect same_as_current EM failures where answer_value_present=1;
3. run Llama constrained_crud k=16 baseline to control instruction-following weakness;
4. build a k-controlled dose-response using heuristic threshold sweep at fixed k.

Current status:

| v4 1-week fix | Status | Key result |
| --- | --- | --- |
| expanded synthetic diagnostic subset | done | 21 selected cells, 64 examples per cell; reverse-chronological conflict collapse and label repair replicate |
| same_as_current formatting/error inspection | done | 76 EM-fail / answer-value-present cases classified; failures are treated as non-exact answer-surface degradation rather than stale-value selection |
| Llama constrained_crud zero-stale baseline | done | slot_prompt EM/F1 0.270/0.321 despite state 1.000 and stale 0.000; slot_direct 1.000 |
| k-controlled dose-response from heuristic sweep | done | at fixed k=16, stale rises 4.43 -> 13.04 while slot_prompt EM falls 0.220 -> 0.100 and slot_direct stays 1.000 |

These results close the main methodological-rigor caveats from v4. They do not change the headline story, but they make it safer: the synthetic mechanism result is no longer just a 16-example pilot, Llama's weak recovery has an explicit zero-stale control, and the stale dose-response is supported both by pooled raw_add analysis and by a within-k threshold sweep.

#### 2. Mistral has dev but not test confirmation

- Llama has dev + test;
- Mistral currently has dev only;
- given the strength of the Mistral result, test confirmation would be nice but not obviously necessary unless Mistral becomes a headline table.

#### 3. Simple external pipeline is transparent but not a real SDK

- This satisfies the v4 suggestion to avoid Mem0 and implement extract-then-store;
- but it should not be overclaimed as an external ecosystem baseline;
- paper wording should call it a transparent external-pipeline diagnostic baseline.

#### 4. Final manuscript is still not submission-ready

Still remaining:

- final LaTeX conversion;
- citation cleanup;
- figure/table placement;
- final numerical consistency pass;
- README/release packaging decision.

## Recommended claim level now

The current evidence is stronger than at v4 review time.

My updated recommendation is:

```text
strong ACL/EMNLP Findings empirical analysis paper
```

The paper still should not be framed as:

```text
broad main-track comprehensive memory benchmark
```

because:

- external SDK baseline is still missing;
- data remains controlled and synthetic;
- implicit/negative/conditional updates remain future work;
- repair training is not included.

But it is now much more defensible as:

> An empirical analysis of stale context contamination in memory-augmented LLMs, supported by a controlled repeated-update benchmark and mechanism probes.

The strongest paper-level contribution is now:

> stale-context contamination is not merely a storage artifact; it is an answer-layer failure mode shaped by retrieved stale exposure, version ambiguity, context position, model-specific context use, and attribute-specific distractor semantics.

## Questions for advisor

### 1. Should experiments now be frozen?

Current situation:

- the v4 “2-3 week truly convincing” additions are complete;
- the v4 “1-week methodological holes” are also complete;
- validation passes with manifest count 400 and smoke test 26/26.

Question:

- Should we freeze the experimental scope now and move fully to paper production?
- Or does the advisor want any remaining optional robustness check, such as Mistral test confirmation?

My recommendation:

- Freeze experiments now.
- Do not open new benchmark splits, new external SDK engineering, repair training, or additional prompt sweeps.
- If the advisor wants exactly one optional extra experiment, Mistral test confirmation is the cleanest candidate, but it is not necessary unless Mistral becomes a headline main-text table.

### 2. Should Mistral be main text or appendix?

Mistral result is strong:

- normal top-k5 EM 0.080;
- latest_per_slot EM 0.720;
- labels EM 0.300.

Question:

- Should Mistral be in the main multi-model table with Qwen/Llama?
- Or should it be appendix robustness with one main-text sentence?

My recommendation:

- Put a compact three-model table in the main text if space allows.
- It makes the model-dependent mitigation story much stronger.

### 3. How should the simple external pipeline be framed?

Question:

- Should it be presented as an external baseline row?
- Or as an external-pipeline diagnostic sanity check?

My recommendation:

- Use the latter wording.
- It is transparent and useful, but because it uses the project parser, it should not be overclaimed as a fair external SDK comparison.

Suggested wording:

> To test whether the benchmark can diagnose a non-learned external memory pipeline, we implement a transparent extract-then-store baseline with inspectable memory records. It is not intended as an external SDK leaderboard row.

### 4. How central should Lost-in-the-Middle be?

Question:

- Should the Lost-in-the-Middle probe be a main mechanism finding?
- Or a discussion/appendix connection to prior long-context work?

My recommendation:

- Include it in the main mechanism section, but as a supporting probe rather than a new main axis.
- It directly explains why chronological/final-position contexts help.

### 5. Should we still avoid broad benchmark claims?

My recommendation remains yes.

Even after v4 additions, the best framing is not “comprehensive benchmark.” It is:

> controlled benchmark-supported empirical analysis of stale context contamination.

## Suggested next actions

### Recommended next actions after P8.1

1. Freeze experimental scope at the current P8/P8.1 package.
2. Convert `paper/manuscript_draft.md` and `paper/manuscript_sections/p80_results_section_draft.md` into final LaTeX.
3. Use `paper/p80_canonical_main_number_ledger.md` for final number checking.
4. Decide main vs appendix placement for:
   - Mistral;
   - simple external pipeline;
   - Lost-in-the-Middle;
   - attribute case study;
   - P8.1 synthetic / heuristic / Llama-control rigor rows.
5. Keep Mem0 and real external SDK comparisons in limitations/future work.
6. Polish README/release instructions and decide result-artifact packaging.

### What not to reopen unless explicitly requested

- broad external baseline engineering;
- new benchmark splits;
- repair training;
- implicit / negative / conditional update expansion;
- additional prompt sweeps;
- all-k Llama/Mistral matrix sweeps.

## Bottom line

After v4, the project has completed the main credibility upgrades needed for a convincing empirical analysis paper:

- third model;
- simple external pipeline;
- attribute-sensitive case study;
- Lost-in-the-Middle position probe;
- P8.1 methodological-rigor pass;
- refreshed evidence/ledger/table infrastructure.

The remaining decision is no longer “what experiment is still missing,” but:

> how should we compress and present the completed P8/P8.1 evidence package in the final paper, while keeping the claim narrow and defensible?
