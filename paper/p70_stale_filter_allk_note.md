# P7.0 Stale-Filter All-k Note

## Goal

After the k=16 latest-per-slot result and the partial k=4 extension both showed large recovery, we completed the remaining filtered dev runs on Sui-3 to obtain a full `k=1/2/4/8/16` picture for the slot-aware answer-time retrieval rewrite.

## Setup

Filtered condition:

```text
mode=raw_add
answer_mode=slot_prompt
slot_prompt_variant=v0_current
retrieval_policy=latest_per_slot
```

Important interpretation note:

`latest_per_slot` is not a pure top-k stale filter. It retrieves from the full store, then deduplicates the answer context by `(entity, attribute)` and keeps only the latest entry per slot. Therefore these results reflect a slot-aware answer-time retrieval rewrite that combines stale suppression with recall expansion for the latest slot entry.

## Completed all-k filtered results

| k | EM | F1 | Answer value present | State acc. | Stale same-slot | Memory size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 2.00 |
| 2 | 0.910 | 0.930 | 0.940 | 1.000 | 1.00 | 5.00 |
| 4 | 0.850 | 0.857 | 0.870 | 1.000 | 3.00 | 12.00 |
| 8 | 0.990 | 0.997 | 1.000 | 1.000 | 6.75 | 25.00 |
| 16 | 0.690 | 0.703 | 0.710 | 1.000 | 14.25 | 52.00 |

Artifacts:

```text
results/p70_stale_filter_intervention/raw_add_prompt_k{1,2,4,8,16}_dev_latest_per_slot/evomemory_results.json
```

## Interpretation

Three conclusions are now clear.

1. The k=16 result remains the most important mechanism finding.
   - Normal top-k5 baseline from P6.8: EM/F1 = 0.140 / 0.173.
   - Latest-per-slot retrieval rewrite: EM/F1 = 0.690 / 0.703.
   - Memory size and stale same-slot burden remain unchanged.

2. The retrieval rewrite remains strong at lower and medium k.
   - k=2 and k=4 both stay far above the corresponding raw_add collapse patterns under normal slot-prompt evaluation.
   - k=8 nearly saturates at EM/F1 ≈ 0.99/1.00.

3. The non-monotonic pattern across k confirms that this is not simply “less stale is always better” under the original retrieval budget.
   - Because latest-per-slot retrieves from the full store before slot deduplication, the rewrite is acting as a slot-aware recall mechanism as well as a stale-suppression mechanism.
   - This explains why k=8 can recover even more strongly than k=4 despite larger stale burden.

## Paper implication

The all-k results strengthen the main paper, but they also sharpen the wording discipline:

- Do **not** describe `latest_per_slot` as a pure stale filter.
- Do describe it as a slot-aware answer-time retrieval rewrite.
- The safe claim is:

> Much of raw append's slot-prompt failure comes from exposing stale and non-final slot entries to the answer layer. When answer-time retrieval is rewritten to surface only the latest entry per slot, performance recovers dramatically across the k sweep, even though the underlying memory still contains the same stale entries.

This supports the mechanism story while avoiding overclaiming exact causal isolation.
