# Response to Critical Review v2

## Purpose

This document maps each major criticism in `docs/critical_review_v2.md` to the concrete follow-up evidence now available, the remaining gaps, and the paper-level decision that should follow. It is written for advisor review before the next manuscript pass.

## Executive status

The second review identified four fatal issues and four serious-but-fixable issues. After the P7.0 follow-up, the status is:

| Review issue | Status | Short answer |
| --- | --- | --- |
| Long25 numbers conflict | Resolved | The two result families use different checkpoints; they must be reported separately with provenance. |
| No meaningful external baseline | Partially addressed, still main gap | Mem0 can run; Qwen2.5-VL result is qualitative only. A project-local Qwen2.5-7B-Instruct text server is queryable, but Mem0 extraction parsing fails before dev3 completes. |
| No original Memory-R1 | Not resolved | Only project-local approximation exists; original code/checkpoint still unavailable unless supplied or separately downloaded. |
| Only one model family | Partially addressed with Llama dev+test replication | Llama3.1-8B k=16 reruns are complete on both dev and test: raw_add normal collapses to EM/F1 0.060/0.062 on dev and 0.040/0.042 on test, while latest_per_slot improves to 0.290/0.341 on dev and 0.290/0.345 on test. |
| Expanded split still simple key-value | Accepted limitation | Should be framed as controlled explicit-update diagnostic, not broad realistic memory benchmark. |
| Heuristic threshold sweep weak | Reframed | Sweep shows parameterized failure region; not a strong method-family success curve. |
| Related work missing | Addressed in draft | Related-work draft now discusses AMemGym, Ledger-QA/UMA, Memory-R1/AgeMem, DST, editing, Mem0/Letta. |
| Missing stale filter intervention | Addressed strongly | P7.0 latest-per-slot answer-time retrieval rewrite completed at all k, with careful wording that it is not a pure top-k filter. |

## 1. Long25 data inconsistency

### Review criticism

The review correctly pointed out that the progress report mixed two incompatible-looking Long25 k=16 result lines:

```text
P6.3 original: EM/F1 0.48/0.53, state 0.91, stale 1.13, memory 9.43
P6.5 stability: EM/F1 0.880/0.908, state 0.967, stale 0.073, memory 2.04
```

### Follow-up evidence

Artifacts:

```text
results/p70_long25_reproducibility/long25_provenance.json
results/p70_long25_reproducibility/long25_provenance.csv
paper/p70_long25_reproducibility_note.md
```

Finding:

- Both families use `data/evomemory_update_frequency_hard_k16_p63_test.json`.
- They do **not** use the same checkpoint.
- P6.3 original uses:

```text
outputs/constrained_sft_curriculum_long25/best
```

- P6.5 seed stability uses:

```text
outputs/p65_long25_seed{11,22,33}/best
```

### Decision

Do not present the P6.5 stability numbers as a reproduction of the exact P6.3 Long25 checkpoint. Report them as a separate reseeded checkpoint family or choose one canonical family explicitly.

Recommended paper wording:

> The original P6.3 Long25 checkpoint produced EM/F1 0.48/0.53 at k=16. A later three-seed retraining family on the same k=16 test split produced EM 0.87-0.89 with much smaller memory. We report these as separate checkpoint families with explicit provenance.

## 2. External baseline fairness

### Review criticism

The review correctly argued that the current Mem0 result should not be used as a main-table external baseline because it uses Qwen2.5-VL, only dev20, and may not give Mem0 a fair text-only setup.

### Follow-up evidence

Artifacts:

```text
paper/p70_external_baseline_fairness_note.md
paper/p70_external_baseline_text_backend_probe.md
configs/mem0_qwen25_text_qdrant_minilm384.json
scripts/serve_openai_compatible_transformers.py
```

Current status:

- Existing completed Mem0 probe:

```text
Qwen2.5-VL + MiniLM + Qdrant, k=16 dev20
EM/F1 after value extraction: 0.00 / 0.05
```

- It is runnable and inspectable, but not fair enough for a main table.
- NAS contains text-only model weights:

```text
/NAS/HuggingFaceModels/Qwen2.5-7B-Instruct
/NAS/HuggingFaceModels/Llama3.1-8B-Instruct
```

- The standard `gmsra` environment lacks `vllm`, `fastapi`, and `uvicorn`.
- To avoid modifying unrelated project environments, a minimal project-local OpenAI-compatible transformers server was added and launched under `/NAS/yesh/MemUpdateBench`.
- That server passed `/v1/models` and simple chat checks with Qwen2.5-7B-Instruct, and Mem0 was rerun with CPU MiniLM embeddings.
- However, k=16 dev20 and dev3 both stalled before the first completed example because Mem0's extraction parser repeatedly rejected the model's JSON-like responses.

### Decision

The paper should keep Mem0 as qualitative/appendix feasibility only. The text-only server itself is now confirmed queryable, but the current Mem0 adapter path is not stable enough for a fair run because structured extraction fails before even dev3 completes.

The next fair-run sequence requires first fixing Mem0 extraction compatibility, then running:

```text
k=16 dev20 smoke -> k=16 dev100 -> k=16 test200 -> optional all-k test
```

## 3. Memory-R1 comparison

### Review criticism

The review argues that Memory-R1 is the most relevant external method because it is closer to learned CRUD memory management.

### Follow-up evidence

Local repository search finds only:

```text
baselines/memory_r1_agent.py
```

This is a project-local approximation using this repo's `MemoryStore` and `MemoryManager`; it is not the original Memory-R1 code/checkpoint.

### Decision

Do not report project-local `memory_r1_agent.py` as original Memory-R1.

Paper wording should be:

> Original Memory-R1 evaluation is an important future external baseline. The current repository contains only a project-local approximation, which we do not report as the original method.

If advisor requires this row, the next step is to provide or fetch official Memory-R1 code/checkpoints and allocate a separate reproduction sprint.

## 4. One-model concern

### Review criticism

The review asks whether stale burden collapse is Qwen-specific and suggests reproducing core experiments on Llama-3-8B.

### Current status

NAS contains Llama-family weights, including:

```text
/NAS/HuggingFaceModels/Llama3.1-8B-Instruct
```

No Llama rerun has been completed yet.

### Decision

This remains a valid limitation. It should be handled after the fair Mem0/text-backend path, because both need a stable text-only serving/evaluation setup.

Recommended wording:

> Current model-backed answering experiments use the Qwen-family answer model. Multi-model answer-layer replication, especially Llama-3.1-8B-Instruct, is a priority for future strengthening but is not yet complete.

## 5. Expanded split realism

### Review criticism

The expanded split still consists of explicit key-value substitutions rather than implicit, partial, negative, or conditional updates.

### Response

This is correct. It should be accepted as a limitation, not argued away.

Recommended paper decision:

- Keep narrow claim: controlled repeated explicit same-slot update diagnostic.
- Do not claim broad coverage of realistic memory-update language.
- Present implicit/partial/negative/conditional updates as future benchmark extensions.

## 6. Heuristic threshold sweep weakness

### Review criticism

The heuristic sweep does not show a good method-family tradeoff because EM remains low across thresholds.

### Response

This is mostly correct. The sweep is still useful because it shows parameterized behavior within one method family, but it should not be oversold as a strong success curve.

Recommended paper decision:

- Use it as evidence that threshold tuning changes stale burden and compactness while not solving answer robustness.
- Do not call it a strong Pareto frontier.

## 7. Related work

### Review criticism

Related work was still not manuscript-ready.

### Follow-up evidence

Artifacts:

```text
paper/manuscript_sections/related_work_positioning_draft.md
paper/manuscript_draft.md
```

The draft now discusses:

- long-term memory benchmarks such as LoCoMo and LongMemEval;
- AMemGym and agent-memory benchmarks;
- Ledger-QA / UMA;
- Memory-R1 / AgeMem;
- Mem0 / MemGPT / Letta;
- dynamic knowledge and memory editing;
- dialogue state tracking.

### Decision

This issue is addressed at draft level, but the final manuscript still needs citation polishing and venue-specific tightening.

## 8. Missing stale-filter intervention

### Review criticism

The review identified this as the most important missing mechanism experiment.

### Follow-up evidence

Artifacts:

```text
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

Important caveat:

- This is not a pure filter over the normal top-k context.
- It retrieves from the full store, then deduplicates by `(entity, attribute)` and keeps only the latest entry per slot.
- Therefore it should be called a slot-aware answer-time retrieval rewrite.

Main result:

| Condition | k | EM | F1 | State acc. | Stale same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 16 | 0.140 | 0.173 | 1.000 | 14.25 | 52.00 |
| latest_per_slot | 16 | 0.690 | 0.703 | 1.000 | 14.25 | 52.00 |

All-k filtered dev sweep:

| k | EM | F1 | Answer value present | State acc. | Stale same-slot | Memory size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 2.00 |
| 2 | 0.910 | 0.930 | 0.940 | 1.000 | 1.00 | 5.00 |
| 4 | 0.850 | 0.857 | 0.870 | 1.000 | 3.00 | 12.00 |
| 8 | 0.990 | 0.997 | 1.000 | 1.000 | 6.75 | 25.00 |
| 16 | 0.690 | 0.703 | 0.710 | 1.000 | 14.25 | 52.00 |

### Decision

This issue is now strongly addressed. The result should be a main mechanism finding, with careful wording:

> Much of raw append's slot-prompt collapse comes from exposing stale and non-final slot entries to the answer layer. A slot-aware latest-per-slot answer-time retrieval rewrite dramatically recovers performance without changing stored memory, although it is not a pure top-k stale filter.

## Updated paper-level recommendation

The project is now materially stronger than at the time of `critical_review_v2.md`.

- Long25 reproducibility concern: resolved.
- Stale-intervention concern: strongly addressed.
- Related work: draft addressed.
- External baseline fairness: still the main unresolved issue; the project-local text-only serving path is queryable, but the current Mem0 extraction adapter fails before dev3 completes.

Recommended target remains:

```text
strong workshop / weak-to-moderate Findings empirical diagnostic
```

The paper should not claim to be a broad main-track memory benchmark unless a fair external baseline and multi-model answering replication are completed.

## Immediate next actions

1. Treat the current Mem0 text-backend path as blocked by structured-extraction incompatibility, not by missing model weights or CUDA embeddings.
2. If Mem0 must become a fair main-table row, first fix the adapter/extraction prompt against the Qwen2.5-7B-Instruct endpoint, then rerun k=16 dev20 before scaling to dev100/test200.
3. Keep Memory-R1 as unresolved unless official code/checkpoint becomes available.
4. Use the P7.0 latest-per-slot all-k result as a main mechanism finding, not a new proposed method.
