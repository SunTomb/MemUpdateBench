# P6.5 Long25 Seed Stability Note

## Motivation

The learned Long25 row is useful only if its compactness/accuracy point is not a single-seed artifact. This note records the k=16 test slot-prompt stability check across three independently trained Long25 seeds.

## Setup

Seeds:

```text
11, 22, 33
```

Evaluation:

```text
data/evomemory_update_frequency_hard_k16_p63_test.json
mode=learned_constrained_slot
answer_mode=slot_prompt
slot_prompt_variant=v0_current
save_answer_traces=true
no_qlora=true
```

The original Sui-3 runs were kept as backup but were too slow because each process performed thousands of batch-size-1 generations. The completed results below used Tang-3 A40 sharded evaluation:

```text
results/p65_stability_sharded/seed11_merged/evomemory_results.json
results/p65_stability_sharded/seed22_merged/evomemory_results.json
results/p65_stability_sharded/seed33_merged/evomemory_results.json
results/p65_stability/long25_seed_stability_summary.{json,csv,md}
```

Each seed was split into 20 shards of 5 examples and merged after completion.

## Results

| Seed | N | EM | F1 | State acc. | Stale same-slot | Same-slot | Memory size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 100 | 0.870 | 0.903 | 0.940 | 0.10 | 1.04 | 2.06 |
| 22 | 100 | 0.880 | 0.913 | 0.990 | 0.07 | 1.04 | 2.04 |
| 33 | 100 | 0.890 | 0.910 | 0.970 | 0.05 | 1.00 | 2.02 |

Aggregate:

- EM mean/std/range: 0.880 / 0.008 / 0.870-0.890.
- F1 mean/std/range: 0.908 / 0.004 / 0.903-0.913.
- State accuracy mean/std/range: 0.967 / 0.021 / 0.940-0.990.
- Stale same-slot mean/std/range: 0.07 / 0.02 / 0.05-0.10.
- Final memory size mean/std/range: 2.04 / 0.02 / 2.02-2.06.

## Interpretation

The Long25 k=16 slot-prompt row is stable enough to report across these three seeds. The EM best-worst gap is only 0.02, and the state-accuracy standard deviation is about 0.021. This supports treating Long25 as a reproducible learned compact-memory point rather than a one-off run.

The result also changes the paper-facing Long25 story relative to earlier single-run diagnostics: on the k=16 test split, Long25 is much stronger than Raw append and heuristic CRUD under slot-prompt, while remaining highly compact and keeping stale same-slot burden near zero. It is not a perfect upper bound, because state accuracy still varies between 0.94 and 0.99, but it is a stable learned tradeoff point.

## Efficiency note

The initial non-sharded Sui-3 runs reached only 10-20 examples after many hours because the evaluation path performs many small serial generations. Tang-3 sharding completed 60 shards and merged all three seeds in about one hour of wall-clock time, validating sharded evaluation as the preferred path for future learned-manager stability checks.
