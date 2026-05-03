# P6.9 Expanded Split Slot-Prompt Note

## Motivation

The P6.8 expanded split passed deterministic slot-direct sanity, but that only shows exact final-state recoverability. The next reviewer-risk question is whether the P6.3 mechanism story survives under model-backed slot-conditioned answering on the larger and more diverse expanded split.

## Setup

Data:

```text
data/evomemory_update_frequency_expanded_k16_p68_dev.json
200 examples
8 attributes: location, company, preference, language, timezone, hobby, instrument, project
```

Methods:

```text
constrained_slot_crud
raw_add
```

Evaluation:

```text
answer_mode=slot_prompt
slot_prompt_variant=v0_current
save_answer_traces=true
no_qlora=true
```

Execution:

```text
Tang-2-Wu, GPUs 4/5/7
results/p69_expanded_slot_prompt/
results/p69_expanded_slot_prompt_summary/
```

The run used 20 shards per method and merged them with `scripts/merge_evomemory_shards.py`.

## Main results

| Method | k | N | EM | F1 | State acc. | Gold retrieved | Stale retrieved | Stale same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud | 16 | 200 | 0.675 | 0.688 | 1.000 | 1.000 | 0.000 | 0.00 | 22.00 |
| raw_add | 16 | 200 | 0.140 | 0.163 | 1.000 | 1.000 | 1.000 | 14.12 | 51.00 |

## Attribute breakdown

### Constrained CRUD

| Attribute | N | EM | F1 | Gold retrieved | Stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| company | 25 | 0.280 | 0.280 | 0.320 | 0.000 |
| hobby | 25 | 0.680 | 0.780 | 1.000 | 0.000 |
| instrument | 25 | 1.000 | 1.000 | 1.000 | 0.000 |
| language | 25 | 0.120 | 0.120 | 0.600 | 0.000 |
| location | 25 | 0.960 | 0.960 | 1.000 | 0.000 |
| preference | 25 | 0.960 | 0.960 | 0.960 | 0.000 |
| project | 25 | 0.600 | 0.600 | 1.000 | 0.000 |
| timezone | 25 | 0.800 | 0.800 | 1.000 | 0.000 |

### Raw append

| Attribute | N | EM | F1 | Gold retrieved | Stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| company | 25 | 0.120 | 0.120 | 0.280 | 1.000 |
| hobby | 25 | 0.080 | 0.080 | 0.440 | 1.000 |
| instrument | 25 | 0.080 | 0.080 | 0.600 | 1.000 |
| language | 25 | 0.160 | 0.160 | 0.160 | 1.000 |
| location | 25 | 0.080 | 0.080 | 0.240 | 1.000 |
| preference | 25 | 0.000 | 0.187 | 0.080 | 1.000 |
| project | 25 | 0.280 | 0.280 | 0.480 | 1.000 |
| timezone | 25 | 0.320 | 0.320 | 0.520 | 1.000 |

## Interpretation

The expanded split reproduces the main P6.3 mechanism story at larger scale and higher attribute diversity. Both methods have perfect final-state accuracy under slot-state evaluation, but prompted answering separates clean-state and stale-heavy memory sharply.

Constrained CRUD remains a clean-state upper-bound diagnostic: state accuracy is 1.00, stale same-slot burden is 0.00, and gold retrieval is 1.00 overall. Yet EM is only 0.675, almost identical to the earlier P6.5/P6.8 clean-state gap around 0.67-0.70. The expanded split therefore confirms that clean final memory state is not sufficient for perfect prompted answering. The residual failures are attribute-sensitive, especially for `company`, `language`, `project`, and `hobby`.

Raw append collapses under stale competition. It also has state accuracy 1.00, but stale same-slot retrieval is 1.00 and stale same-slot burden averages 14.12 at k=16. Its EM/F1 fall to 0.140/0.163. This supports the stale-contamination mechanism on the expanded split, not only on the original P6.3 hard data.

## Paper-facing takeaway

The expanded split addresses scale/diversity reviewer risk without changing the core conclusion. Repeated same-slot updates still separate final-state recoverability from answer robustness: clean memory leaves an answer-layer gap, while append-only memory suffers severe stale-contamination collapse.

## Full k-sweep results

After the k=16 endpoint confirmed the mechanism, the run was expanded to all dev k values for the same two methods.

Outputs:

```text
results/p69_expanded_slot_prompt_allk/
results/p69_expanded_slot_prompt_allk_summary/
```

| Method | k | EM | F1 | State acc. | Gold retrieved | Stale retrieved | Stale same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.00 | 2.00 |
| constrained_slot_crud | 2 | 0.965 | 0.965 | 1.000 | 1.000 | 0.000 | 0.00 | 4.00 |
| constrained_slot_crud | 4 | 0.875 | 0.875 | 1.000 | 1.000 | 0.000 | 0.00 | 7.00 |
| constrained_slot_crud | 8 | 0.925 | 0.926 | 1.000 | 1.000 | 0.000 | 0.00 | 12.00 |
| constrained_slot_crud | 16 | 0.675 | 0.688 | 1.000 | 1.000 | 0.000 | 0.00 | 22.00 |
| raw_add | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.00 | 2.00 |
| raw_add | 2 | 0.725 | 0.725 | 1.000 | 1.000 | 1.000 | 1.00 | 5.00 |
| raw_add | 4 | 0.315 | 0.344 | 1.000 | 1.000 | 1.000 | 3.00 | 12.00 |
| raw_add | 8 | 0.095 | 0.096 | 1.000 | 1.000 | 1.000 | 6.88 | 25.00 |
| raw_add | 16 | 0.140 | 0.163 | 1.000 | 1.000 | 1.000 | 14.12 | 51.00 |

The full k sweep strengthens the expanded-split conclusion. Raw append begins to degrade as soon as stale same-slot entries appear: EM drops from 1.000 at k=1 to 0.725 at k=2, 0.315 at k=4, and near-collapse at k=8/16. The exact final state remains recoverable throughout, so the degradation is not a state-availability failure. It is a prompted-answer failure under stale same-slot competition.

Constrained CRUD keeps stale retrieved at 0.000 and state accuracy at 1.000 across all k values, but its EM is not perfectly monotonic because the answer layer is sensitive to attribute and context composition. The high-frequency k=16 endpoint still exposes a clean-state answer-layer gap at 0.675 EM. This supports the paper's separation between memory-state correctness and prompted answer robustness.

## Next step

Use this expanded all-k dev result as the scale/diversity evidence for the current paper package. Running the same full sweep on test is optional; it should be done only if the manuscript needs test-set confirmation after the dev curve is integrated.
