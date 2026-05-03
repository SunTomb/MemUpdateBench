# P6.8 Heuristic Threshold Tradeoff Note

## Motivation

`docs/critical_review.md` criticizes the current tradeoff curve as mostly a cross-method scatter plot. This note records a same-method-family threshold sweep for `heuristic_crud`, using the existing `--update_threshold` knob.

## Sweep setup

Thresholds:

```text
0.70, 0.80, 0.85, 0.90, 0.95
```

Splits:

```text
data/evomemory_update_frequency_hard_k8_p63_dev.json
data/evomemory_update_frequency_hard_k16_p63_dev.json
```

Result roots:

```text
results/p68_heuristic_threshold_sweep/
results/p68_heuristic_threshold_summary/
```

## Full threshold sweep results

| Answer | k | Threshold | EM | F1 | State acc. | Stale same-slot | Same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 8 | 0.70 | 1.000 | 1.000 | 1.000 | 3.18 | 4.19 | 9.78 |
| direct | 8 | 0.80 | 1.000 | 1.000 | 1.000 | 4.82 | 5.97 | 15.25 |
| direct | 8 | 0.85 | 1.000 | 1.000 | 1.000 | 5.14 | 6.39 | 18.10 |
| direct | 8 | 0.90 | 1.000 | 1.000 | 1.000 | 5.69 | 6.94 | 20.57 |
| direct | 8 | 0.95 | 1.000 | 1.000 | 1.000 | 6.68 | 7.93 | 23.14 |
| direct | 16 | 0.70 | 1.000 | 1.000 | 1.000 | 4.43 | 5.44 | 11.57 |
| direct | 16 | 0.80 | 1.000 | 1.000 | 1.000 | 6.49 | 7.55 | 20.47 |
| direct | 16 | 0.85 | 1.000 | 1.000 | 1.000 | 7.43 | 8.54 | 26.67 |
| direct | 16 | 0.90 | 1.000 | 1.000 | 1.000 | 9.29 | 10.44 | 32.72 |
| direct | 16 | 0.95 | 1.000 | 1.000 | 1.000 | 13.04 | 14.28 | 42.20 |
| prompt | 8 | 0.70 | 0.290 | 0.295 | 1.000 | 3.18 | 4.19 | 9.78 |
| prompt | 8 | 0.80 | 0.090 | 0.098 | 1.000 | 4.82 | 5.97 | 15.25 |
| prompt | 8 | 0.85 | 0.140 | 0.161 | 1.000 | 5.14 | 6.39 | 18.10 |
| prompt | 8 | 0.90 | 0.140 | 0.180 | 1.000 | 5.69 | 6.94 | 20.57 |
| prompt | 8 | 0.95 | 0.080 | 0.118 | 1.000 | 6.68 | 7.93 | 23.14 |
| prompt | 16 | 0.70 | 0.240 | 0.257 | 1.000 | 4.43 | 5.44 | 11.57 |
| prompt | 16 | 0.80 | 0.150 | 0.210 | 1.000 | 6.49 | 7.55 | 20.47 |
| prompt | 16 | 0.85 | 0.170 | 0.195 | 1.000 | 7.43 | 8.54 | 26.67 |
| prompt | 16 | 0.90 | 0.120 | 0.140 | 1.000 | 9.29 | 10.44 | 32.72 |
| prompt | 16 | 0.95 | 0.070 | 0.098 | 1.000 | 13.04 | 14.28 | 42.20 |

## Interpretation

This sweep gives the within-method tradeoff curve requested by the critical review. The threshold knob changes compactness and stale burden while oracle state accuracy remains 1.00. At both k=8 and k=16, lower threshold 0.70 yields the best slot-prompt EM and the smallest stale burden/memory size. Increasing the threshold generally increases stale same-slot burden and memory size, while slot-prompt EM remains low or declines.

The result is especially clear at k=16: state accuracy is 1.00 for all thresholds, but stale burden rises from 4.43 to 13.04 and slot-prompt EM falls from 0.24 to 0.07. This supports the paper's mechanism claim with a same-method-family parameter curve rather than only cross-method scatter points.

## Paper-facing takeaway

Heuristic CRUD threshold tuning can preserve final-state recoverability across the full threshold range, but answer robustness is sensitive to stale burden. The best threshold in this sweep is not the one with the most retained memories; compactness and stale filtering help prompted answering even when slot-direct state remains perfect.
