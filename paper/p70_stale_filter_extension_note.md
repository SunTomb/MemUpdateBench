# P7.0 Stale-Filter Extension Note

## Goal

After the k=16 latest-per-slot result showed a large recovery, we probed whether the same slot-aware answer-time retrieval rewrite also helps at lower update frequencies.

## Completed extension results

The filtered dev sweep is now complete for `k=1/2/4/8/16` after migrating the remaining runs from Tang to Sui-3.

| k | Condition | EM | F1 | Answer value present | Stale same-slot | Memory size |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | latest_per_slot | 1.000 | 1.000 | 1.000 | 0.00 | 2.00 |
| 2 | latest_per_slot | 0.910 | 0.930 | 0.940 | 1.00 | 5.00 |
| 4 | latest_per_slot | 0.850 | 0.857 | 0.870 | 3.00 | 12.00 |
| 8 | latest_per_slot | 0.990 | 0.997 | 1.000 | 6.75 | 25.00 |
| 16 | latest_per_slot | 0.690 | 0.703 | 0.710 | 14.25 | 52.00 |

Artifacts:

```text
results/p70_stale_filter_intervention/raw_add_prompt_k{1,2,4,8,16}_dev_latest_per_slot/evomemory_results.json
results/p70_stale_filter_intervention_summary/stale_filter_allk_filtered.md
paper/p70_stale_filter_allk_note.md
```

These results strengthen the k=16 headline intervention: the slot-aware latest-per-slot answer-time retrieval rewrite remains highly effective across the full k sweep while leaving write-side burden unchanged.

## Execution note

The sweep was initially attempted on Tang-2, where the remaining k=1/2/8 runs became too slow because of repeated per-example encoder initialization inside fresh `MemoryStore`s. Since Tang GPUs were heavily occupied, the remaining jobs were migrated to Sui-3, which has access to the same NAS-backed project checkout and environment. The final all-k table therefore combines:

- earlier completed Tang results for k=4 and k=16;
- migrated Sui-3 completions for k=1, k=2, and k=8.

## Paper implication

The safest claim is now stronger than before:

1. the k=16 intervention is the main mechanism result;
2. the full all-k filtered sweep shows that slot-aware latest-per-slot answer-time retrieval dramatically recovers raw_add prompting across update frequencies;
3. because this rewrite retrieves from the full store before slot deduplication, it should still be described as a slot-aware retrieval rewrite rather than a pure stale filter.

This makes the mechanism story more complete without pretending stronger causal isolation than the implementation actually provides.
