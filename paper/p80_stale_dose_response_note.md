# P8.0 Stale Dose-Response Note

## Purpose

`docs/critical_review_v3.md` asks whether stale same-slot burden has a gradual or threshold-like effect on slot-prompt answering. This note records the first-pass dose-response analysis using existing raw_add slot-prompt result files, before adding new controlled mechanism probes.

## Artifacts

```text
scripts/analyze_stale_dose_response.py
results/p80_stale_dose_response/stale_dose_examples.csv
results/p80_stale_dose_response/stale_dose_bins.csv
results/p80_stale_dose_response/retrieved_stale_dose_bins.csv
results/p80_stale_dose_response/stale_dose_logistic.json
results/p80_stale_dose_response/stale_dose_response.md
```

Inputs:

```text
results/update_frequency_p63/raw_add_slot_prompt_k{1,2,4,8,16}/evomemory_results.json
results/p69_expanded_slot_prompt_allk/raw_add_prompt_k{1,2,4,8,16}_dev_merged/evomemory_results.json
```

Total examples analyzed: 1500.

## Main first-pass finding

Stored stale same-slot burden shows a strong dose-response relationship with EM:

| Stored stale same-slot count | N | EM | F1 | Answer value present |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 300 | 0.967 | 0.970 | 1.000 |
| 1 | 300 | 0.743 | 0.748 | 0.780 |
| 3 | 300 | 0.290 | 0.320 | 0.393 |
| 6-7 | 300 | 0.083 | 0.097 | 0.113 |
| 13-15 | 300 | 0.117 | 0.144 | 0.140 |

The key qualitative point is that one stale same-slot entry is already harmful: EM drops from 0.967 at stale=0 to 0.743 at stale=1.

A lightweight logistic fit gives:

```text
EM ~ stored_stale_count: slope = -0.383, ED50 ≈ 3.18 stale entries
```

This ED50 should be treated as an exploratory estimate because the current input combines P6.3 hard and expanded splits, and k/attribute distributions are not controlled.

## Retrieved stale burden appears closer to the answer-time mechanism

For examples with answer traces, retrieved stale same-slot count is also strongly predictive:

| Retrieved stale same-slot count | N | EM | F1 | Answer value present |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 201 | 1.000 | 1.000 | 1.000 |
| 1 | 231 | 0.667 | 0.667 | 0.675 |
| 2 | 92 | 0.174 | 0.196 | 0.261 |
| 3 | 192 | 0.281 | 0.301 | 0.370 |
| 4 | 163 | 0.184 | 0.188 | 0.221 |
| 5 | 121 | 0.000 | 0.034 | 0.008 |

Lightweight logistic fit:

```text
EM ~ retrieved_stale_count: slope = -1.082, ED50 ≈ 1.89 retrieved stale entries
```

This supports the mechanism interpretation that answer-context exposure is more directly relevant than the mere existence of stale entries in the memory store.

## Caveats

- This is not yet the final statistical analysis.
- P6.3 and expanded split examples are pooled.
- Attribute mix is not controlled.
- The stored-stale bins partly reflect k values, so k and stale count are confounded in the first-pass pooled analysis.
- Retrieved-stale counts are available only where answer traces exist.
- Latest-per-slot rows are intentionally excluded from this first pass because that intervention changes retrieval scope.

## P8.1 fixed-k heuristic threshold sweep

`critical_review_v4.md` requested a k-controlled stale/EM curve because the pooled dose-response above mixes k and stale burden. The P8.1 rigor pass addresses this with a fixed `k=16` heuristic CRUD threshold sweep on P6.3 dev.

Artifacts:

```text
scripts/run_p81_heuristic_threshold_k16_sui3.sh
scripts/summarize_heuristic_threshold.py
results/p81_heuristic_threshold_k16_rigor/
results/p81_heuristic_threshold_k16_rigor_summary/heuristic_threshold_k16_summary.{json,csv,md}
```

Results:

| Answer | Threshold | EM | F1 | State acc. | Stale | Same-slot | Mem. size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slot_direct | 0.70 | 1.000 | 1.000 | 1.000 | 4.43 | 5.44 | 11.57 |
| slot_direct | 0.75 | 1.000 | 1.000 | 1.000 | 5.42 | 6.43 | 14.61 |
| slot_direct | 0.80 | 1.000 | 1.000 | 1.000 | 6.49 | 7.55 | 20.47 |
| slot_direct | 0.85 | 1.000 | 1.000 | 1.000 | 7.43 | 8.54 | 26.67 |
| slot_direct | 0.90 | 1.000 | 1.000 | 1.000 | 9.29 | 10.44 | 32.72 |
| slot_direct | 0.95 | 1.000 | 1.000 | 1.000 | 13.04 | 14.28 | 42.20 |
| slot_prompt | 0.70 | 0.220 | 0.247 | 1.000 | 4.43 | 5.44 | 11.57 |
| slot_prompt | 0.75 | 0.190 | 0.218 | 1.000 | 5.42 | 6.43 | 14.61 |
| slot_prompt | 0.80 | 0.120 | 0.158 | 1.000 | 6.49 | 7.55 | 20.47 |
| slot_prompt | 0.85 | 0.140 | 0.164 | 1.000 | 7.43 | 8.54 | 26.67 |
| slot_prompt | 0.90 | 0.100 | 0.120 | 1.000 | 9.29 | 10.44 | 32.72 |
| slot_prompt | 0.95 | 0.100 | 0.134 | 1.000 | 13.04 | 14.28 | 42.20 |

Interpretation:

- At fixed `k=16`, increasing the heuristic threshold increases stale same-slot burden from 4.43 to 13.04 and memory size from 11.57 to 42.20.
- `slot_direct` remains perfect across the entire sweep, so the final slot state is recoverable even as stale burden grows.
- `slot_prompt` degrades overall as stale burden rises: EM falls from 0.220 at threshold 0.70 to 0.100 at thresholds 0.90/0.95.
- The curve is not perfectly monotonic at every adjacent threshold, but the within-k trend supports the same conclusion as the pooled dose-response: stale burden primarily hurts answer-time use, not oracle state resolution.

This closes the main k-confounding caveat for the dose-response section. The pooled raw_add curve shows the broad relationship; the fixed-k heuristic sweep shows that the same direction holds within a single update-frequency endpoint.

## Paper implication

This analysis supports a stronger claim than the previous all-k aggregate tables:

> Stale contamination starts early: even a single stale same-slot entry substantially reduces slot-prompt EM, and the answer-time retrieved stale count has an even sharper relationship with correctness.

The next step is to run controlled mechanism probes that vary stale count, value conflict, order, and labels while holding the rest of the context fixed.
