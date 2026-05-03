# P6.8 Expanded Split Baseline Note

## Motivation

After adding the opt-in expanded update-frequency split, the first required check is not learned-model performance but deterministic sanity: final-state recoverability, stale burden, and memory size should behave as expected across k under inspectable memory managers.

## Setup

Data:

```text
data/evomemory_update_frequency_expanded_k{1,2,4,8,16}_p68_dev.json
```

Modes:

```text
constrained_slot_crud
raw_add
heuristic_crud
```

Answer mode:

```text
slot_direct
```

Outputs:

```text
results/p68_expanded_baselines/
results/p68_expanded_baselines_summary/
```

## Results

| Method | k | EM | F1 | State acc. | Stale same-slot | Same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud | 1 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 2.00 |
| constrained_slot_crud | 2 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 4.00 |
| constrained_slot_crud | 4 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 7.00 |
| constrained_slot_crud | 8 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 12.00 |
| constrained_slot_crud | 16 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 22.00 |
| raw_add | 1 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 2.00 |
| raw_add | 2 | 1.000 | 1.000 | 1.000 | 1.00 | 2.00 | 5.00 |
| raw_add | 4 | 1.000 | 1.000 | 1.000 | 3.00 | 4.00 | 12.00 |
| raw_add | 8 | 1.000 | 1.000 | 1.000 | 6.88 | 8.00 | 25.00 |
| raw_add | 16 | 1.000 | 1.000 | 1.000 | 14.12 | 16.00 | 51.00 |
| heuristic_crud | 1 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 2.00 |
| heuristic_crud | 2 | 1.000 | 1.000 | 1.000 | 1.00 | 2.00 | 5.00 |
| heuristic_crud | 4 | 1.000 | 1.000 | 1.000 | 3.00 | 4.00 | 12.00 |
| heuristic_crud | 8 | 1.000 | 1.000 | 1.000 | 6.88 | 8.00 | 25.00 |
| heuristic_crud | 16 | 1.000 | 1.000 | 1.000 | 14.12 | 16.00 | 51.00 |

## Interpretation

The expanded split preserves the intended slot-state invariant: all three deterministic/inspectable methods retain the final value under oracle `slot_direct` across all k values. The diagnostic burden still appears as stale same-slot growth and memory size, not as final-state loss.

`raw_add` and `heuristic_crud` are identical in this local deterministic run because the local environment uses the zero-vector encoder fallback, making the similarity-threshold heuristic unable to distinguish update candidates. Treat this table as a state/stale/memory sanity check for the expanded split, not as a semantic retrieval result. Model-backed or embedding-backed evaluations should be run on the cluster environment before comparing heuristic retrieval behavior.

## Next step

Run `slot_prompt` or model-backed answer-layer evaluations on the expanded split after choosing a cluster node/environment with the expected embedding/model dependencies. The deterministic oracle result above is the gate that makes those learned/black-box runs interpretable.
