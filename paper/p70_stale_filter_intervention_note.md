# P7.0 Retrieval-Time Stale Filter Intervention Note

## Motivation

`docs/critical_review_v2.md` argues that the stale-burden mechanism still lacks the most direct causal intervention: on `raw_add`, remove stale same-slot entries from the answer-time retrieval context while leaving the stored memory unchanged. This note records that intervention.

## Implementation

A new answer-time retrieval policy was added to `scripts/eval_evomemory.py`:

```text
--retrieval_policy normal | latest_per_slot
```

`latest_per_slot` keeps the raw memory writes unchanged, retrieves candidate entries, then deduplicates the answer context by `(entity, attribute)` and keeps only the latest entry for each slot according to `event_idx` / update timestamps. This is an answer-time intervention only, not a memory-manager method.

Validation:

```text
python -m py_compile scripts/eval_evomemory.py scripts/smoke_test.py scripts/summarize_update_frequency.py scripts/summarize_stale_filter_intervention.py
python scripts/smoke_test.py
```

Smoke tests now verify that latest-per-slot filtering keeps the final value when the same slot has both stale and current entries.

## Outputs

```text
results/p70_stale_filter_intervention/raw_add_prompt_k16_dev_latest_per_slot/evomemory_results.json
results/p70_stale_filter_intervention/raw_add_prompt_k16_dev_normal_p68_baseline/evomemory_results.json
results/p70_stale_filter_intervention_summary/stale_filter_summary.{json,md}
scripts/summarize_stale_filter_intervention.py
```

The normal baseline is the completed P6.8 raw_add k=16 dev retrieval-topk5 answer-layer diagnostic. A same-code P7.0 normal rerun was attempted on Tang-2 but stopped after GPU OOM; the P6.8 baseline is still the appropriate top-k5 raw_add k=16 dev normal retrieval comparison.

## Results

| Condition | N | EM | F1 | Answer value present | State acc. | Stale same-slot | Same-slot | Gold same-slot | Memory size | Retrieval policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| normal retrieval top-k5 | 100 | 0.140 | 0.173 | 0.160 | 1.000 | 14.250 | 16.000 | 1.750 | 52.000 | normal |
| latest-per-slot answer context | 100 | 0.690 | 0.703 | 0.710 | 1.000 | 14.250 | 16.000 | 1.750 | 52.000 | latest_per_slot |

Effect:

- EM delta: +0.550.
- F1 delta: +0.530.
- Answer-value-present delta: +0.550.
- Memory-size delta: 0.000.
- Stale-burden delta: 0.000.

## Interpretation

This is the strongest causal evidence so far for the stale-competition mechanism. Raw append still stores the same 52 memories on average and still contains the same 14.25 stale same-slot entries on average. The intervention only changes which entries are exposed to the answer prompt. Under that answer-time filter, EM rises from 0.14 to 0.69.

The recovery is large but not complete relative to the gold-context result from P6.8 (EM 0.92). This implies two things:

1. stale same-slot contamination is a major cause of raw append collapse;
2. answer-layer and retrieval-context limitations remain even after stale same-slot entries are filtered from the prompted context.

The manuscript should present this as an intervention/ablation, not as a new proposed method. It answers the reviewer concern that the earlier stale-burden story was only correlational.
