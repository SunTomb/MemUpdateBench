# Post-Critical-Review Progress Report

## Purpose

This document summarizes what has changed after reading `docs/critical_review.md`, which judged the previous MemUpdateBench package as likely too close to an internal controlled demo for a strong ACL/EMNLP/NeurIPS/ICLR-style benchmark paper. The goal is to help the advisor review whether the project has responded to the main criticisms in the right direction, and to identify the next decisions that need guidance.

## What the critical review changed

The review made clear that the main problem was not figure polish or prose packaging. The core risks were:

1. no real external memory baseline;
2. limited data scale and attribute diversity;
3. insufficient mechanism analysis behind stale burden and answer-layer failures;
4. a tradeoff curve that was too cross-method and not enough same-method-family;
5. weak related-work positioning against long-term memory benchmarks, external memory systems, dialogue state tracking, and knowledge editing.

The project direction was therefore changed from paper-packaging-first to evidence-first. The current positioning is narrower:

> MemUpdateBench is a controlled diagnostic benchmark for repeated updates to the same `(entity, attribute)` memory slot. It is not a broad long-term memory benchmark or a general memory-agent leaderboard.

The main thesis is now that repeated same-slot updates separate four quantities that are often conflated:

- final-state reliability;
- stale same-slot burden;
- memory compactness;
- slot-conditioned answer robustness.

## Completed response to the review

### 1. Prompt robustness and answer traces

The project added prompt-variant and answer-trace instrumentation around `slot_prompt` evaluation.

Key artifacts:

```text
results/p65_prompt_robustness/
results/p65_prompt_robustness_summary/
results/p65_diagnostics/k16_prompt_diagnostics.json
paper/p65_prompt_robustness_note.md
paper/p65_diagnostic_findings.md
scripts/summarize_prompt_robustness.py
scripts/analyze_action_pathology.py
```

Main finding:

- Raw append remains near collapse under prompt variants at k=16.
- Constrained CRUD remains much stronger but still has a clean-state answer-layer gap.
- Long25 remains between Raw append and Constrained CRUD.
- Long25 invalid action rate is near zero, so future repair should target operation selection or NOOP discrimination rather than output-format cleanup.

### 2. Answer-layer mechanism analysis

The project added controlled answer-layer diagnostics comparing normal retrieved context with oracle gold-only context.

Key artifacts:

```text
results/p68_answer_layer_diagnostics/
paper/p68_answer_layer_mechanism_note.md
scripts/analyze_answer_layer_mechanism.py
```

Full k=16 dev findings:

| Method | Context | EM | State acc. | Gold retrieved | Stale retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Constrained CRUD | retrieval top-k5 | 0.67 | 1.00 | 0.86 | 0.00 |
| Constrained CRUD | gold context | 0.92 | 1.00 | 1.00 | 0.00 |
| Raw append | retrieval top-k5 | 0.14 | 1.00 | 0.36 | 1.00 |
| Raw append | gold context | 0.92 | 1.00 | 1.00 | 0.00 |

Interpretation:

- Constrained CRUD's residual gap is largely retrieval/context selection and answer-layer behavior, not stale-state retention.
- Raw append's collapse is dominated by stale same-slot contamination and low effective gold retrieval under normal context.

### 3. Stale-burden intervention analysis

The project added intervention-style analysis for Raw append.

Key artifacts:

```text
results/p68_stale_intervention/
paper/p68_stale_intervention_note.md
scripts/analyze_stale_intervention.py
```

Main finding:

- Raw append top-k5 to gold-only context improves EM by +0.78.
- Increasing retrieval from top-k5 to top-k10 increases gold retrieval but decreases EM, because stale entries remain present.

This supports a stale-competition mechanism rather than a simple “retrieve more memories” fix.

### 4. Same-method-family tradeoff curve

The project added a heuristic CRUD threshold sweep.

Key artifacts:

```text
results/p68_heuristic_threshold_sweep/
results/p68_heuristic_threshold_summary/
paper/p68_heuristic_tradeoff_note.md
scripts/summarize_heuristic_threshold.py
```

At k=16:

| Threshold | State acc. | Slot-prompt EM | Stale same-slot | Memory size |
| ---: | ---: | ---: | ---: | ---: |
| 0.70 | 1.00 | 0.24 | 4.43 | 11.57 |
| 0.80 | 1.00 | 0.15 | 6.49 | 20.47 |
| 0.85 | 1.00 | 0.17 | 7.43 | 26.67 |
| 0.90 | 1.00 | 0.12 | 9.29 | 32.72 |
| 0.95 | 1.00 | 0.07 | 13.04 | 42.20 |

Interpretation:

- The tradeoff is now shown within one method family, not only across unrelated methods.
- State accuracy remains perfect while stale burden and answer robustness vary strongly.

### 5. Long25 stability

The previous Long25 result was checked across three independently trained seeds using sharded Tang-3 evaluation.

Key artifacts:

```text
results/p65_stability/long25_seed_stability_summary.{json,csv,md}
paper/p65_long25_stability_note.md
scripts/merge_evomemory_shards.py
scripts/run_p65_long25_sharded_tang3.sh
```

k=16 test slot-prompt across seeds 11/22/33:

| Metric | Mean | Std | Range |
| --- | ---: | ---: | ---: |
| EM | 0.880 | 0.008 | 0.870-0.890 |
| F1 | 0.908 | 0.004 | 0.903-0.913 |
| State accuracy | 0.967 | 0.021 | 0.940-0.990 |
| Stale same-slot | 0.073 | 0.021 | 0.050-0.100 |
| Final memory size | 2.040 | 0.016 | 2.020-2.060 |

Interpretation:

- Long25 is stable enough to report as a learned compact-memory point.
- It is compact and low-stale, but not a perfect upper bound.

### 6. Data scale and diversity expansion

The project added a separate opt-in expanded split. It does not overwrite P6.3.

Key artifacts:

```text
data/evomemory_update_frequency_expanded_p68_{train,dev,test}.json
data/evomemory_update_frequency_expanded_k{1,2,4,8,16}_p68_{dev,test}.json
results/p68_expanded_oracle/
results/p68_expanded_baselines/
results/p68_expanded_baselines_summary/
paper/p68_expanded_split_note.md
paper/p68_expanded_baseline_note.md
```

Scale:

| Split | Examples | Per k |
| --- | ---: | ---: |
| train | 2500 | 500 |
| dev | 1000 | 200 |
| test | 1000 | 200 |

Attributes:

```text
location, company, preference, language, timezone, hobby, instrument, project
```

Deterministic oracle sanity:

- k=1/2/4/8/16 dev all reach EM/F1/state accuracy 1.00.

Expanded deterministic baselines confirm the state/stale/memory invariant:

- Constrained CRUD keeps stale same-slot at 0.00.
- Raw append grows to 14.12 stale same-slot entries and memory size 51.00 at k=16.

### 7. Expanded split model-backed slot-prompt evaluation

After deterministic sanity, the expanded split was evaluated under model-backed `slot_prompt` on Tang-2.

Key artifacts:

```text
results/p69_expanded_slot_prompt/
results/p69_expanded_slot_prompt_summary/
results/p69_expanded_slot_prompt_allk/
results/p69_expanded_slot_prompt_allk_summary/
paper/p69_expanded_slot_prompt_note.md
scripts/run_p69_expanded_slot_prompt_tang2.sh
scripts/run_p69_expanded_slot_prompt_allk_tang2.sh
```

Expanded dev all-k slot-prompt results:

| Method | k | EM | F1 | State acc. | Stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | 1 | 1.000 | 1.000 | 1.000 | 0.00 |
| Constrained CRUD | 2 | 0.965 | 0.965 | 1.000 | 0.00 |
| Constrained CRUD | 4 | 0.875 | 0.875 | 1.000 | 0.00 |
| Constrained CRUD | 8 | 0.925 | 0.926 | 1.000 | 0.00 |
| Constrained CRUD | 16 | 0.675 | 0.688 | 1.000 | 0.00 |
| Raw append | 1 | 1.000 | 1.000 | 1.000 | 0.00 |
| Raw append | 2 | 0.725 | 0.725 | 1.000 | 1.00 |
| Raw append | 4 | 0.315 | 0.344 | 1.000 | 3.00 |
| Raw append | 8 | 0.095 | 0.096 | 1.000 | 6.88 |
| Raw append | 16 | 0.140 | 0.163 | 1.000 | 14.12 |

Interpretation:

- The expanded split reproduces the P6.3 mechanism story at larger scale and with more attributes.
- Raw append collapses as stale same-slot entries appear.
- Constrained CRUD remains clean-state but still has a high-k answer-layer gap.

### 8. k=32 extrapolation

The project added k=32 only as an opt-in extrapolation point, not as a replacement main axis.

Key artifacts:

```text
data/evomemory_update_frequency_expanded_k32_p69k32_{dev,test}.json
results/p69_k32_oracle/
results/p69_k32_slot_direct/
results/p69_k32_slot_direct_summary/
results/p69_k32_slot_prompt/
results/p69_k32_slot_prompt_summary/
paper/p69_k32_extrapolation_note.md
scripts/run_p69_k32_slot_prompt_tang2.sh
```

k=32 dev results:

| Method | Answer mode | EM | F1 | State acc. | Stale same-slot | Memory size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | slot-direct | 1.000 | 1.000 | 1.000 | 0.00 | 42.00 |
| Raw append | slot-direct | 1.000 | 1.000 | 1.000 | 28.50 | 103.00 |
| Constrained CRUD | slot-prompt | 0.655 | 0.655 | 1.000 | 0.00 | 42.00 |
| Raw append | slot-prompt | 0.155 | 0.172 | 1.000 | 28.50 | 103.00 |

Interpretation:

- k=32 confirms saturation and is useful as appendix evidence.
- k=16 remains the better main endpoint because it already shows the tradeoff clearly.

### 9. Real Mem0 external baseline probe

The external baseline status changed significantly. Mem0 is no longer only blocked at install/import time.

Key artifacts:

```text
scripts/eval_mem0_baseline.py
configs/mem0_vllm_qdrant_minilm384.json
results/p69_external_baselines/mem0_vllm_qdrant384_k16_dev20/
results/p69_external_baselines/mem0_vllm_qdrant384_k16_dev20_v2/
paper/p69_external_baseline_result_note.md
```

Cluster resources found:

- Tang-2 has a live OpenAI-compatible vLLM endpoint:
  - `http://127.0.0.1:8011/v1`
  - served model: `Qwen2.5-VL`
- Local MiniLM embedding model exists under `/NAS/yesh/hf_cache/hub/...`.
- Qdrant client is available through the isolated Mem0 vendor path.

Mem0 now runs end-to-end with:

```text
llm.provider = vllm
embedder.provider = huggingface
vector_store.provider = qdrant
```

Adapter improvements:

- supports Mem0 v2 `filters` API;
- supports `--reuse_memory_instance` to avoid local Qdrant lock conflicts;
- stores `search_payload` and `memory_texts`;
- extracts values from returned memory text instead of using full returned sentences as predictions.

Observed dev20 results:

| Variant | EM | F1 | Inspectable memory | Stale same-slot | Same-slot | Gold same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First returned text | 0.00 | 0.00 | 1.00 | 1.15 | 1.20 | 0.05 | 20.00 |
| Value extraction | 0.00 | 0.05 | 1.00 | 1.00 | 1.05 | 0.05 | 20.00 |

Interpretation:

- Mem0 is now a real runnable external baseline probe.
- The current off-the-shelf configuration performs very poorly on exact repeated same-slot final-value tracking.
- The failure is not merely formatting: after value extraction, returned values are still mostly stale or wrong.
- This is useful negative external-validity evidence, but not yet a polished external leaderboard row.

## Current paper-level progress

The project has improved substantially relative to the critical review:

1. It now has deeper mechanism evidence, not just descriptive tables.
2. It has same-method-family tradeoff evidence.
3. It has larger and more diverse data.
4. It has expanded model-backed slot-prompt results.
5. It has k=32 extrapolation as appendix evidence.
6. It has a real Mem0 external-system probe, although the current result is negative and preliminary.
7. It has stronger related-work positioning.

The remaining weakness is not that the project has no response to the review. The remaining weakness is that the external baseline result is still a small dev20 probe and is currently very poor. The paper must decide how to present that result without overclaiming.

## Questions for advisor

### 1. Is the current positioning acceptable?

Should the paper continue to position MemUpdateBench narrowly as:

> a controlled repeated same-slot update diagnostic benchmark

rather than a broad long-term memory benchmark?

This seems more defensible after the review, but it lowers the ambition of the benchmark claim.

### 2. How should we present Mem0?

We now have a real Mem0 run using local vLLM + MiniLM + Qdrant. The result is poor:

```text
k=16 dev20, EM=0.00, F1=0.05 after value extraction
```

Should this be included as:

1. a main-table external baseline row;
2. an appendix feasibility/negative-result row;
3. only a qualitative external-baseline probe;
4. or should we invest in a Mem0-specific prompt/adapter before reporting it?

My current recommendation is option 2 or 3, unless the advisor wants the paper to emphasize external systems more strongly.

### 3. Is dev20 enough for the Mem0 negative result?

Because dev20 is already 0 EM and inspection shows stale/wrong values, running dev100 may mostly confirm the same problem. But reviewers may distrust a 20-example external row.

Should we:

- run dev100 despite expected low value;
- keep dev20 as a feasibility/diagnostic result only;
- or first improve the Mem0 extraction/update prompt before scaling?

### 4. Should k=32 appear in the main paper or appendix?

k=32 confirms saturation:

- Raw append stale same-slot: 28.50;
- Raw append memory size: 103.00;
- Raw append slot-prompt EM/F1: 0.155/0.172.

But k=16 already shows the tradeoff clearly. My recommendation is appendix only.

### 5. Is expanded split all-k dev enough, or do we need test-set confirmation?

Expanded all-k dev reproduces the mechanism strongly. Should we run the same all-k slot-prompt sweep on the expanded test split, or is dev sufficient for now because the main paper still uses P6.3 hard as the canonical benchmark?

### 6. Should we start repair training?

Current diagnostics suggest:

- Raw append fails due to stale contamination.
- Constrained CRUD fails due to answer-layer/context behavior.
- Long25 is stable but not perfect.
- Mem0 off-the-shelf returns stale/wrong values.

Repair training could target operation selection or NOOP discrimination, but it may distract from the benchmark paper. Should repair remain deferred, or should a small targeted repair experiment be added?

### 7. What venue/claim level should we target?

After these additions, the project seems stronger than the version criticized in `critical_review.md`, but still may not be a broad main-track benchmark paper because external baselines remain limited.

Possible claim levels:

1. workshop-strong controlled diagnostic;
2. Findings-level empirical diagnostic paper;
3. main-track benchmark paper after stronger external baseline integration.

My current recommendation is to aim for a strong Findings-style empirical diagnostic paper unless the advisor wants more external-system integration before writing.

## Suggested next actions after advisor feedback

Depending on the answers above:

1. If Mem0 should be reported: run dev100 and write an appendix table.
2. If Mem0 should be improved: modify extraction/update prompts and re-run dev20 before scaling.
3. If expanded test confirmation is required: run expanded all-k test slot-prompt on Tang/Sui.
4. If repair is desired: design a small targeted repair dataset from action-pathology findings.
5. If paper writing should start: update the manuscript around the new P6.8/P6.9 evidence and move k=32/Mem0 details to appendix.
