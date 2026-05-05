# P8.0 Expanded Latest-Per-Slot All-k Note

## Purpose

P7.0 found an unexpected non-monotonicity in the P6.3 hard dev latest-per-slot sweep: k=8 reached nearly perfect EM while k=4 and k=16 were lower. This note checks whether that pattern persists on the larger expanded dev split.

## Artifacts

```text
scripts/eval_evomemory.py
scripts/run_p80_expanded_latest_per_slot_sui3.sh
scripts/summarize_context_mechanisms.py
results/p80_expanded_latest_per_slot_allk/
results/p80_expanded_latest_per_slot_summary/expanded_latest_per_slot_summary.{json,csv,md}
```

Remote execution:

```text
node: Sui-3-Wu
model: Qwen/Qwen2.5-7B-Instruct
data: data/evomemory_update_frequency_expanded_k{1,2,4,8,16}_p68_dev.json
mode: raw_add
answer_mode: slot_prompt
retrieval_policy: latest_per_slot
answer_topk: 5
save_answer_traces: true
no_qlora: true
```

## Results

| k | N | EM | F1 | answer value present | gold retrieved | stale retrieved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 0.955 | 0.970 | 1.000 | 1.000 | 0.000 |
| 2 | 200 | 0.940 | 0.954 | 0.980 | 1.000 | 0.000 |
| 4 | 200 | 0.855 | 0.855 | 0.905 | 0.980 | 0.000 |
| 8 | 200 | 0.925 | 0.929 | 0.935 | 0.990 | 0.000 |
| 16 | 200 | 0.750 | 0.764 | 0.775 | 0.860 | 0.000 |

## Interpretation

The expanded split preserves a smaller k=8 local high point, but the extreme P6.3 result no longer holds:

```text
P6.3 latest_per_slot k=8 EM = 0.990
Expanded latest_per_slot k=8 EM = 0.925
```

This suggests the earlier k=8 value was at least partly a sample/attribute-composition artifact. The mitigation remains strong across k, but it is not a monotonic guarantee and it does not eliminate residual clean-state answer/retrieval failures.

The expanded results also clarify the k=16 ceiling: latest_per_slot removes stale same-slot retrieval entirely (`stale retrieved = 0.000`), but gold retrieval is only 0.860 and EM is 0.750. Therefore the remaining errors are not stale contamination; they come from latest value not being surfaced after slot deduplication, distractor/context selection issues, or answer-layer failures.

## Paper implication

The paper should not emphasize the P6.3 k=8 = 0.99 number as a robust phenomenon. A safer statement is:

> Slot-aware latest-per-slot retrieval consistently mitigates stale-context collapse, but performance is not strictly monotonic in update count. The near-perfect P6.3 k=8 result attenuates on the larger expanded split, indicating that it was likely driven by sample or attribute composition rather than a general retrieval sweet spot.

## Caveats

- This run uses expanded dev only, not expanded test.
- The all-k sweep keeps the same answer model and prompt variant as prior Qwen slot-prompt runs.
- Per-attribute breakdown should be added if the k=8 local high point becomes paper-relevant.
