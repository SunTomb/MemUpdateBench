# Post-Critical-Review-v2 Progress Report

## Purpose

本文档总结在完成 `docs/CRITICAL_REVIEW_V2_RESPONSE_20260504.md` 之后，项目又继续推进的工作、目前形成的结论，以及仍需要老师判断的问题。它的作用类似 `docs/POST_CRITICAL_REVIEW_PROGRESS_20260504.md`：不是论文正文，而是给老师快速判断下一步投入方向的汇报材料。

第二轮严格评审之后，项目的核心目标已经从“继续堆叠结果”转为三件事：

1. 消除会直接伤害可信度的复现/口径问题；
2. 补上 stale burden 的更强机制证据；
3. 诚实判断外部 baseline 是否足以进入主表。

当前最重要的整体结论是：

> Long25 口径矛盾已经解决；raw_add 的 stale/context 机制证据已经显著加强；P8.0 已经补上 dose-response、上下文机制探针、synthetic same-slot 探针、Llama3.1-8B 多模型复现，以及 expanded latest_per_slot 异常检查；Related Work 和 manuscript 叙事已经补强。公平外部 baseline 仍未完成，Mem0 text-backend 路径目前卡在 structured extraction 兼容性，而不是模型权重或 CUDA embedding。

## What changed after `CRITICAL_REVIEW_V2_RESPONSE_20260504.md`

### 1. Mem0 text-only backend was pushed from “being configured” to a clear failure verdict

在 `CRITICAL_REVIEW_V2_RESPONSE_20260504.md` 初版中，外部 baseline 的状态仍是：Qwen2.5-7B-Instruct text server 正在配置，之后希望跑更公平的 Mem0 dev20/dev100。

后续实际推进后，状态已经更明确。

Key artifacts:

```text
scripts/serve_openai_compatible_transformers.py
configs/mem0_qwen25_text_qdrant_minilm384_cpu.json
paper/p70_external_baseline_text_backend_probe.md
paper/p70_external_baseline_fairness_note.md
```

Cluster setup:

```text
node: Sui-3-Wu
project path: /NAS/yesh/MemUpdateBench
model: /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct
endpoint: http://127.0.0.1:8013/v1
embedder: MiniLM forced to CPU
vector store: local Qdrant, 384-dim collection
```

What succeeded:

- A minimal project-local OpenAI-compatible server was added without modifying unrelated project environments.
- The server successfully answered:
  - `GET /v1/models`;
  - simple `POST /v1/chat/completions` checks.
- The earlier CUDA failure in MiniLM embedding was fixed by forcing the Mem0 HuggingFace embedder to CPU through:

```json
"model_kwargs": {
  "device": "cpu"
}
```

What failed:

- Mem0 with Qwen2.5-7B-Instruct text backend did not complete k=16 dev20.
- Reducing the LLM generation cap from 512 to 128 tokens did not solve the issue.
- A smaller k=16 dev3 smoke also failed before the first completed example.
- Logs repeatedly showed Mem0 structured-extraction parse errors, for example:

```text
Error parsing extraction response: Expecting ',' delimiter
Error parsing extraction response: Extra data
```

Interpretation:

- The remaining blocker is no longer missing Qwen text weights.
- It is also no longer CUDA initialization for MiniLM.
- The blocker is compatibility between Mem0's structured extraction parser and the lightweight transformers OpenAI-compatible server output.

Paper implication:

- The completed Mem0 Qwen2.5-VL dev20 result remains only a qualitative/appendix feasibility probe.
- The Qwen2.5-7B-Instruct text-backend attempt should be reported as an attempted fairness improvement that failed at adapter/extraction compatibility.
- Mem0 should not appear as a fair main-table external baseline unless we first fix extraction prompts/server JSON behavior and rerun dev20 successfully.

### 2. The external-baseline notes were updated to avoid stale or over-optimistic wording

After the text-backend failure verdict, several documents were updated so they no longer imply that the fair Mem0 run is merely waiting for a server to be configured.

Updated files:

```text
docs/CRITICAL_REVIEW_V2_RESPONSE_20260504.md
docs/POST_CRITICAL_REVIEW_PROGRESS_20260504.md
paper/p70_external_baseline_text_backend_probe.md
paper/p70_external_baseline_fairness_note.md
WORKFLOW.md
```

Key wording change:

- Old status: text-only endpoint is being configured; next step is dev20 -> dev100 -> test200.
- New status: text-only endpoint is queryable, CPU embedding works, but Mem0 structured extraction fails before dev3 completes.

This makes the current external-baseline status more defensible. We can now tell reviewers/advisors exactly where the fair Mem0 path failed.

### 3. The stale-filter intervention is now finalized as an answer-time retrieval rewrite, not a new method

This was already mostly completed before the final cleanup, but the wording has now been made consistent across docs.

Key artifacts:

```text
scripts/eval_evomemory.py
scripts/smoke_test.py
scripts/summarize_stale_filter_intervention.py
results/p70_stale_filter_intervention/
results/p70_stale_filter_intervention_summary/stale_filter_summary.{json,md}
results/p70_stale_filter_intervention_summary/stale_filter_allk_filtered.md
paper/p70_stale_filter_intervention_note.md
paper/p70_stale_filter_allk_note.md
paper/p70_stale_filter_extension_note.md
```

Implementation:

```text
--retrieval_policy latest_per_slot
```

Important caveat now consistently stated:

- It is not a pure filter over the original top-k context.
- It retrieves from the full store, deduplicates by `(entity, attribute)`, and keeps the latest entry per slot.
- Therefore it should be described as a slot-aware answer-time retrieval rewrite or intervention, not as a deployable proposed method.

Main k=16 dev result:

| Condition | EM | F1 | State acc. | Stale same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_add normal top-k5 | 0.140 | 0.173 | 1.000 | 14.25 | 52.00 |
| raw_add latest_per_slot | 0.690 | 0.703 | 1.000 | 14.25 | 52.00 |

All-k latest-per-slot dev sweep:

| k | EM | F1 | Answer value present | State acc. | Stale same-slot | Memory size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 2.00 |
| 2 | 0.910 | 0.930 | 0.940 | 1.000 | 1.00 | 5.00 |
| 4 | 0.850 | 0.857 | 0.870 | 1.000 | 3.00 | 12.00 |
| 8 | 0.990 | 0.997 | 1.000 | 1.000 | 6.75 | 25.00 |
| 16 | 0.690 | 0.703 | 0.710 | 1.000 | 14.25 | 52.00 |

Interpretation:

- raw_add 的 memory writes 没有改变，stale burden 和 memory size 仍然很高；
- 只改变 answer-time context selection 就显著恢复 EM/F1；
- 这支持“raw append 的 slot-prompt collapse 很大程度来自 stale/non-final slot exposure 和 latest-slot recall failure”；
- 但因为它扩大了 retrieval scope，不应声称这是纯 top-k stale filter 的因果隔离。

### 4. Long25 provenance decision remains resolved and should be preserved in the paper

No new Long25 experiment was added after the response document, but the decision was propagated into workflow/manuscript-facing notes.

Current decision:

- P6.3 original Long25 row and P6.5 stability row are separate checkpoint families.
- They use the same k=16 P6.3 hard test split but different checkpoints.
- They should not be mixed as if they are the same run.

Canonical evidence:

```text
results/p70_long25_reproducibility/long25_provenance.json
results/p70_long25_reproducibility/long25_provenance.csv
paper/p70_long25_reproducibility_note.md
```

Paper implication:

- Either choose one canonical family for the main table;
- or show both explicitly as separate checkpoint families with checkpoint provenance.

My recommendation remains to avoid using the stronger P6.5 stability numbers as a silent replacement for the original P6.3 Long25 row.

### 5. P8.0 mechanism analysis and multi-model replication substantially strengthen the paper

After `critical_review_v3.md`, the project shifted from short-term blocker cleanup to a fuller benchmark + analysis-paper plan. The first P8.0 execution batch is now complete.

Key artifacts:

```text
scripts/build_evidence_manifest.py
scripts/analyze_stale_dose_response.py
scripts/analyze_attribute_failures.py
scripts/run_p80_mechanism_probe_batch_sui3.sh
scripts/run_synthetic_same_slot_probe.py
scripts/summarize_context_mechanisms.py
scripts/summarize_synthetic_same_slot_probe.py
scripts/run_p80_llama_replication_sui3.sh
scripts/run_p80_expanded_latest_per_slot_sui3.sh
scripts/package_p80_paper_tables.py
paper/p80_stale_dose_response_note.md
paper/p80_context_mechanism_probe_note.md
paper/p80_synthetic_same_slot_probe_note.md
paper/p80_multimodel_stale_susceptibility_note.md
paper/p80_expanded_latest_per_slot_note.md
paper/p80_long25_training_provenance_audit.md
paper/p80_paper_tables.md
paper/manuscript_sections/p80_results_section_draft.md
```

Main new findings:

1. **Dose-response:** pooled raw_add slot-prompt analysis covers 1500 examples. EM drops from 0.967 at stored stale count 0 to 0.743 at stale count 1 and 0.290 at stale count 3. Retrieved-stale ED50 is lower than stored-stale ED50, suggesting answer-context exposure is the closer failure mechanism.
2. **Real-context mechanism probe:** with retrieved entries held fixed at gold retrieval 0.360 and stale retrieved 1.000, context presentation changes EM from 0.110 normal to 0.230 chronological, 0.010 reverse chronological, and 0.260 with latest/outdated labels. This shows answer-layer order and version-disambiguation effects independent of retrieval composition.
3. **Synthetic same-slot probe:** when gold is always present, conflict + reverse chronological + no labels collapses immediately, while latest/outdated labels nearly repair it. Chronological conflict remains much more robust, and same-value repetition still hurts exact EM. This rejects a simple majority-vote-only explanation.
4. **Llama3.1-8B replication:** Llama raw_add normal k=16 collapses on both dev and test (dev EM/F1 0.060/0.062; test 0.040/0.042) with stale retrieved rate 1.000. latest_per_slot removes stale retrieval and improves to 0.290/0.341 on dev and 0.290/0.345 on test, so stale susceptibility is not Qwen-only. However, recovery is weaker than Qwen's 0.690 EM on dev, so mitigation magnitude is model-dependent.
5. **Expanded latest_per_slot all-k:** expanded dev results are 0.955/0.940/0.855/0.925/0.750 EM for k=1/2/4/8/16. The earlier P6.3 k=8 near-perfect 0.990 attenuates, so it should be treated as partly sample/attribute composition rather than a robust retrieval sweet spot.
6. **Long25 provenance audit:** current artifacts support checkpoint/training-provenance sensitivity, not a clean seed-only claim. P6.3 original and P6.5 three-seed family should remain separately labeled.

Paper implication:

- The paper can now be framed more strongly as an empirical analysis of stale context contamination, not only a method-comparison benchmark.
- The strongest contribution is now the mechanism evidence chain: normal -> latest_per_slot -> dose-response -> real-context order/label probes -> synthetic same-slot controls -> Llama replication.

### 6. Related work and manuscript positioning are now better aligned with the narrower claim

The related-work draft and manuscript were updated before the final cleanup, and P8.0 results have now been integrated into the manuscript narrative.

Relevant files:

```text
paper/manuscript_sections/related_work_positioning_draft.md
paper/manuscript_draft.md
```

Current positioning:

> MemUpdateBench is a controlled diagnostic benchmark for repeated explicit updates to the same `(entity, attribute)` memory slot. It is not a broad long-term memory benchmark or a general memory-agent leaderboard.

The related work now discusses:

- LoCoMo / LongMemEval;
- AMemGym;
- Ledger-QA / UMA;
- Memory-R1 / AgeMem;
- Mem0 / MemGPT / Letta;
- dynamic knowledge and memory editing;
- dialogue state tracking.

Remaining manuscript work:

- citation polishing;
- venue-specific tightening;
- deciding how prominently to show Mem0 failed-fairness attempts.

## Current paper-level status

The project is stronger than after the first critical review and materially stronger than the state that triggered `critical_review_v2.md`, but it still should not be overclaimed.

### Stronger points now

1. **Long25 reproducibility risk is resolved.**
   - The conflicting numbers are explained by different checkpoint families.
   - Provenance artifacts exist.

2. **The stale-burden mechanism is much stronger.**
   - Gold-context diagnostics showed large recoverability.
   - latest-per-slot answer-time rewrite shows large recovery without changing writes.
   - dose-response, real-context order/label probes, and synthetic same-slot controls now explain why stale context hurts.
   - Llama3.1-8B replication shows the collapse is not Qwen-only.

3. **The narrow benchmark claim is now better defended.**
   - Expanded split and k=32 support the mechanism story.
   - Expanded latest_per_slot all-k checks the k=8 anomaly and avoids overclaiming it.
   - Limitations around explicit key-value updates are now accepted rather than hidden.

4. **External baseline status is more honest.**
   - Mem0 is not simply “untried.”
   - Qwen2.5-VL Mem0 is runnable but unfair/qualitative.
   - Qwen2.5-7B text Mem0 was attempted and failed for a concrete adapter reason.

5. **Related work is no longer absent.**
   - The current draft covers the main neighboring areas.

### Remaining weaknesses

1. **No fair external baseline row yet.**
   - Mem0 text-backend extraction fails.
   - Original Memory-R1 is unavailable.

2. **Only one non-Qwen answer-model family so far.**
   - Llama3.1-8B-Instruct replication is complete on both dev and test and is already useful.
   - A third answer model would further strengthen external validity, but is no longer as urgent as before.

3. **Benchmark realism remains limited.**
   - The current expanded split is still explicit key-value substitution.
   - It does not cover implicit, negative, conditional, or partial updates.

4. **The heuristic threshold sweep should not be oversold.**
   - It is useful as a parameterized failure-region analysis.
   - It is not a strong Pareto frontier or method-family success curve.

## Recommended claim level now

My recommendation is still:

```text
strong workshop / weak-to-moderate Findings empirical diagnostic
```

I would not recommend claiming this as a broad main-track benchmark paper unless at least one of the following is completed:

1. a fair external baseline row, preferably Mem0 with stable text extraction or original Memory-R1;
2. a third answer-model family beyond the completed Llama3.1-8B-Instruct dev+test replication;
3. a substantially more realistic update split with implicit/conditional/negative/partial updates and reliable oracle semantics.

For the current evidence package, the safest framing is:

> This paper introduces a controlled diagnostic benchmark showing that repeated same-slot updates separate final-state reliability, stale same-slot burden, memory compactness, and answer-layer robustness. The strongest contribution is the mechanism evidence around stale context exposure, not a broad external-memory leaderboard.

## Questions for advisor

### 1. Should we freeze the current paper scope and start final writing?

The core reviewer-risk issues are now mostly handled except fair external baseline and multi-model replication. Should we now freeze the claim as a controlled diagnostic paper and spend time on manuscript integration, or continue running additional experiments?

My recommendation: freeze the scope unless the target venue requires a stronger external baseline.

### 2. How should the failed Mem0 text-backend attempt be presented?

Current facts:

- Qwen2.5-VL Mem0 dev20 completes but is unfair and very weak: EM/F1 0.00/0.05 after value extraction.
- Qwen2.5-7B-Instruct text endpoint is queryable.
- CPU MiniLM embedding works.
- Mem0 text-backend extraction fails before dev3 completes.

Possible presentation choices:

1. appendix-only qualitative feasibility note;
2. short limitation paragraph without table;
3. no manuscript mention, only internal report;
4. invest more time in Mem0-specific extraction prompt/server JSON repair.

My recommendation: option 1 or 2. Do not put Mem0 in the main table.

### 3. Is it worth fixing Mem0 extraction compatibility now?

Fixing Mem0 could mean:

- patching the lightweight server to enforce stricter JSON-only output;
- modifying Mem0 extraction prompts or parser behavior;
- using a more standard vLLM/OpenAI serving stack instead of the minimal transformers server;
- trying a different text model.

This could produce a fairer external row, but it may become engineering-heavy and distract from the current benchmark contribution.

Question: should this be a priority before writing, or deferred as future external-baseline work?

My recommendation: defer unless the paper target is raised above workshop/weak Findings.

### 4. How should we present the completed Llama3.1-8B-Instruct replication?

The one-model-family concern is now materially reduced by completed Llama dev+test replication.

Completed Llama result:

| Condition | EM | F1 | stale retrieved |
| --- | ---: | ---: | ---: |
| raw_add normal top-k5 | 0.060 | 0.062 | 1.000 |
| raw_add latest_per_slot | 0.290 | 0.341 | 0.000 |

Interpretation:

- stale-context collapse is not Qwen-only;
- latest_per_slot helps after removing stale retrieved entries;
- recovery is much weaker than Qwen, and labels/order interventions transfer poorly.

Question: should this be a main-paper table, an appendix table, or a short robustness paragraph?

My recommendation: include a concise main-paper robustness table or paragraph, because it directly answers the single-answer-model concern without requiring another large baseline.

### 5. Which Long25 family should be canonical in the main table?

Options:

1. Use original P6.3 Long25 checkpoint row because it belongs to the original canonical result table.
2. Use P6.5 three-seed family as the stronger learned compact-memory point, but label it as a reseeded checkpoint family.
3. Show both in appendix/provenance table and keep main table focused on deterministic baselines plus mechanism interventions.

My recommendation: do not silently replace P6.3 with P6.5. If using P6.5, explicitly label it as a separate reseeded family.

### 6. Should k=32 stay appendix-only?

k=32 confirms saturation and supports extrapolation, but k=16 already shows the main tradeoff clearly.

My recommendation: appendix only.

### 7. Should repair training remain deferred?

Diagnostics suggest possible repair targets:

- Long25 operation selection / NOOP discrimination;
- answer-layer context robustness;
- Mem0 extraction/update prompt behavior.

However, repair training may shift the paper from benchmark/diagnostic to method improvement.

My recommendation: defer repair training until the benchmark paper scope is settled.

## Suggested next actions

Depending on advisor preference:

### If writing should start now

1. Lock the claim as controlled repeated same-slot update diagnostics.
2. Build the main result table around:
   - constrained_slot_crud;
   - raw_add;
   - heuristic threshold sweep;
   - Long25 with explicit checkpoint provenance;
   - latest_per_slot as mechanism intervention, not method.
3. Put Mem0 and k=32 in appendix/limitations.
4. Tighten related work and limitations.
5. Make sure every main-table number maps to a specific `results/**/evomemory_results.json`.

### If one more experiment is allowed

Now that Llama3.1-8B-Instruct dev+test replication is complete, prioritize one of:

```text
1. a fairer external diagnostic baseline with stable structured extraction;
2. a third answer model;
3. a small implicit/conditional/negative update pilot with deterministic oracle semantics.
```

This addresses the one-model-family criticism more directly than further Mem0 engineering.

### If external baseline is mandatory

Do not scale Mem0 yet. First fix the extraction path:

1. enforce JSON-only output from the local server or use a standard vLLM/OpenAI stack;
2. rerun k=16 dev3;
3. rerun k=16 dev20;
4. only then consider dev100/test200.

### If benchmark realism must be expanded

Design a new opt-in split for implicit/negative/conditional/partial updates only after gold oracle semantics are fully specified. Do not mix these into the current explicit-update split without a deterministic oracle sanity pass.

## Validation status

Recent local validation after the final cleanup:

```text
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/analyze_action_pathology.py scripts/analyze_answer_layer_mechanism.py scripts/analyze_stale_intervention.py scripts/eval_mem0_baseline.py scripts/merge_evomemory_shards.py scripts/summarize_update_frequency.py scripts/summarize_prompt_robustness.py scripts/summarize_heuristic_threshold.py scripts/summarize_stale_filter_intervention.py scripts/serve_openai_compatible_transformers.py scripts/smoke_test.py
python scripts/smoke_test.py
python -m json.tool configs/mem0_qwen25_text_qdrant_minilm384_cpu.json
```

Smoke result:

```text
SMOKE TEST: 26/26 passed
```

## Bottom line

The project now has a much cleaner evidence package than before `critical_review_v2.md`:

- reproducibility issue: resolved;
- stale mechanism issue: strongly addressed;
- related-work issue: addressed at draft level;
- external baseline issue: honestly probed but still unresolved;
- one-model issue: still unresolved.

The next strategic decision is whether to stop experimentation and write a careful diagnostic paper, or invest in one additional strengthening direction. My recommendation is to prioritize manuscript integration, with Llama answer-model replication as the most valuable optional experiment if time remains.
