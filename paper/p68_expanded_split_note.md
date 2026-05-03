# P6.8 Expanded Split Note

## Motivation

`docs/critical_review.md` identifies the original paper-facing data scale and attribute diversity as a blocker for stronger benchmark claims. This note records a separate opt-in expanded split that preserves exact `(entity, attribute)` semantics while increasing examples, attribute coverage, names, relations, and explicit update wording.

## Split design

The expanded split does not overwrite P6.3 hard data. It writes new `p68` files:

```text
data/evomemory_update_frequency_expanded_p68_train.json
data/evomemory_update_frequency_expanded_p68_dev.json
data/evomemory_update_frequency_expanded_p68_test.json
data/evomemory_update_frequency_expanded_k{1,2,4,8,16}_p68_{dev,test}.json
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

The split also expands relation/name diversity and uses paraphrased explicit update templates plus semantic near-miss NOOPs. It intentionally avoids implicit, conditional, or ambiguous updates.

## Deterministic oracle sanity

Command pattern:

```bash
PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode constrained_slot_crud \
  --answer_mode slot_direct \
  --data_file data/evomemory_update_frequency_expanded_k${k}_p68_dev.json \
  --output_dir results/p68_expanded_oracle/k${k}_dev
```

Results:

| k | N | EM | F1 | State acc. | Stale same-slot | Memory size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 1.000 | 1.000 | 1.000 | 0.00 | 2.00 |
| 2 | 200 | 1.000 | 1.000 | 1.000 | 0.00 | 4.00 |
| 4 | 200 | 1.000 | 1.000 | 1.000 | 0.00 | 7.00 |
| 8 | 200 | 1.000 | 1.000 | 1.000 | 0.00 | 12.00 |
| 16 | 200 | 1.000 | 1.000 | 1.000 | 0.00 | 22.00 |

## Implementation note

The expanded split required parser support for the new explicit attributes: `timezone`, `hobby`, `instrument`, and `project`. During sanity checking, early templates such as `moved to project Falcon` conflicted with the location parser and near-miss project NOOPs looked too update-like. These were corrected before accepting the split. The accepted split passes deterministic oracle sanity across all k values.

## Interpretation

This split is suitable as an opt-in scale/diversity stressor. It should not replace P6.3 hard as the main historical comparison until baseline rows are rerun, but it directly addresses the reviewer concern that the existing evidence is too small and attribute-limited. Future learned or external baselines on this split should still be interpreted only after deterministic oracle sanity, as above.
