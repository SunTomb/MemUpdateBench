# P8.0 Paper Table Pack

These tables are generated from result artifacts and are intended as manuscript/appendix source material, not final LaTeX.

## Stored stale dose-response

| stored stale count | n | EM | F1 | answer value present |
| --- | ---: | ---: | ---: | ---: |
| 0.000 | 300 | 0.967 | 0.970 | 1.000 |
| 1.000 | 300 | 0.743 | 0.748 | 0.780 |
| 3.000 | 300 | 0.290 | 0.320 | 0.393 |
| 6.000 | 50 | 0.060 | 0.060 | 0.060 |
| 7.000 | 250 | 0.088 | 0.104 | 0.124 |
| 13.000 | 50 | 0.160 | 0.161 | 0.180 |
| 14.000 | 150 | 0.153 | 0.153 | 0.180 |
| 15.000 | 100 | 0.040 | 0.120 | 0.060 |

## Retrieved stale dose-response

| retrieved stale count | n | EM | F1 | answer value present |
| --- | ---: | ---: | ---: | ---: |
| 0.000 | 201 | 1.000 | 1.000 | 1.000 |
| 1.000 | 231 | 0.667 | 0.667 | 0.675 |
| 2.000 | 92 | 0.174 | 0.196 | 0.261 |
| 3.000 | 192 | 0.281 | 0.301 | 0.370 |
| 4.000 | 163 | 0.184 | 0.188 | 0.221 |
| 5.000 | 121 | 0.000 | 0.034 | 0.008 |

## Logistic dose-response fit

| predictor | n | slope | ED50 |
| --- | ---: | ---: | ---: |
| stored_stale_same_slot_count | 1500 | -0.383 | 3.176 |
| retrieved_stale_same_slot_count | 1000 | -1.082 | 1.895 |

## Real-context mechanism probe

| condition | EM | F1 | answer value present | gold retrieved | stale retrieved avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal_order_none | 0.110 | 0.136 | 0.140 | 0.360 | 4.040 |
| chronological_none | 0.230 | 0.275 | 0.320 | 0.360 | 4.040 |
| reverse_chronological_none | 0.010 | 0.050 | 0.010 | 0.360 | 4.040 |
| normal_timestamp | 0.150 | 0.200 | 0.230 | 0.360 | 4.040 |
| normal_latest_outdated_label | 0.260 | 0.298 | 0.350 | 0.360 | 4.040 |
| current_first_none | 0.020 | 0.060 | 0.020 | 0.360 | 4.040 |
| current_last_none | 0.040 | 0.080 | 0.040 | 0.360 | 4.040 |

## Llama3.1-8B stale susceptibility

| condition | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.060 | 0.062 | 0.120 | 0.360 | 1.000 |
| latest_per_slot | 0.290 | 0.341 | 0.750 | 0.860 | 0.000 |
| latest/outdated labels | 0.080 | 0.105 | 0.350 | 0.360 | 1.000 |
| chronological | 0.020 | 0.039 | 0.140 | 0.360 | 1.000 |
| reverse chronological | 0.050 | 0.050 | 0.110 | 0.360 | 1.000 |

## Llama3.1-8B constrained zero-stale control

| answer | EM | F1 | value EM | answer value present | state | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slot_prompt | 0.270 | 0.321 | 0.660 | 0.730 | 1.000 | 0.000 |
| slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Mistral-7B third-model stale susceptibility

| condition | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.080 | 0.177 | 0.140 | 0.360 | 1.000 |
| latest_per_slot | 0.720 | 0.735 | 0.750 | 0.860 | 0.000 |
| latest/outdated labels | 0.300 | 0.332 | 0.350 | 0.360 | 1.000 |
| chronological | 0.150 | 0.182 | 0.190 | 0.360 | 1.000 |
| reverse chronological | 0.040 | 0.117 | 0.060 | 0.360 | 1.000 |

## Mistral-7B constrained zero-stale control

| answer | EM | F1 | value EM | answer value present | state | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slot_prompt | 0.720 | 0.735 | 0.750 | 0.750 | 1.000 | 0.000 |
| slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Ceiling-recovery comparison

| model | normal raw_add EM | latest_per_slot EM | zero-stale EM | latest - ceiling |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | 0.110 | 0.690 | 0.700 | -0.010 |
| Llama3.1-8B | 0.060 | 0.290 | 0.270 | +0.020 |
| Mistral-7B | 0.080 | 0.720 | 0.720 | 0.000 |

## P8.1 expanded synthetic same-slot diagnostic subset

| condition | stale | n | EM | F1 | answer value present |
| --- | ---: | ---: | ---: | ---: | ---: |
| conflict + chronological + none | 1 | 64 | 1.000 | 1.000 | 1.000 |
| conflict + chronological + none | 2 | 64 | 0.938 | 0.959 | 1.000 |
| conflict + chronological + none | 4 | 64 | 0.906 | 0.935 | 1.000 |
| conflict + chronological + none | 8 | 64 | 0.875 | 0.903 | 0.969 |
| conflict + chronological + none | 16 | 64 | 0.797 | 0.838 | 0.984 |
| conflict + reverse_chronological + latest_outdated_label | 1 | 64 | 0.969 | 0.970 | 1.000 |
| conflict + reverse_chronological + latest_outdated_label | 2 | 64 | 0.969 | 0.983 | 1.000 |
| conflict + reverse_chronological + latest_outdated_label | 4 | 64 | 0.891 | 0.902 | 1.000 |
| conflict + reverse_chronological + latest_outdated_label | 8 | 64 | 1.000 | 1.000 | 1.000 |
| conflict + reverse_chronological + latest_outdated_label | 16 | 64 | 1.000 | 1.000 | 1.000 |
| conflict + reverse_chronological + none | 1 | 64 | 0.234 | 0.288 | 0.422 |
| conflict + reverse_chronological + none | 2 | 64 | 0.094 | 0.095 | 0.109 |
| conflict + reverse_chronological + none | 4 | 64 | 0.156 | 0.158 | 0.172 |
| conflict + reverse_chronological + none | 8 | 64 | 0.000 | 0.000 | 0.000 |
| conflict + reverse_chronological + none | 16 | 64 | 0.031 | 0.031 | 0.031 |
| same_as_current + chronological + none | 4 | 64 | 0.688 | 0.723 | 1.000 |
| same_as_current + chronological + none | 8 | 64 | 0.547 | 0.624 | 1.000 |
| same_as_current + chronological + none | 16 | 64 | 0.531 | 0.597 | 1.000 |
| same_as_current + reverse_chronological + none | 4 | 64 | 0.906 | 0.916 | 1.000 |
| same_as_current + reverse_chronological + none | 8 | 64 | 0.812 | 0.857 | 1.000 |
| same_as_current + reverse_chronological + none | 16 | 64 | 0.781 | 0.821 | 1.000 |

## P8.1 fixed-k heuristic threshold sweep

| answer | threshold | EM | F1 | state | stale | same-slot | memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 0.7 | 1.000 | 1.000 | 1.000 | 4.430 | 5.440 | 11.570 |
| direct | 0.75 | 1.000 | 1.000 | 1.000 | 5.420 | 6.430 | 14.610 |
| direct | 0.8 | 1.000 | 1.000 | 1.000 | 6.490 | 7.550 | 20.470 |
| direct | 0.85 | 1.000 | 1.000 | 1.000 | 7.430 | 8.540 | 26.670 |
| direct | 0.9 | 1.000 | 1.000 | 1.000 | 9.290 | 10.440 | 32.720 |
| direct | 0.95 | 1.000 | 1.000 | 1.000 | 13.040 | 14.280 | 42.200 |
| prompt | 0.7 | 0.220 | 0.247 | 1.000 | 4.430 | 5.440 | 11.570 |
| prompt | 0.75 | 0.190 | 0.218 | 1.000 | 5.420 | 6.430 | 14.610 |
| prompt | 0.8 | 0.120 | 0.158 | 1.000 | 6.490 | 7.550 | 20.470 |
| prompt | 0.85 | 0.140 | 0.164 | 1.000 | 7.430 | 8.540 | 26.670 |
| prompt | 0.9 | 0.100 | 0.120 | 1.000 | 9.290 | 10.440 | 32.720 |
| prompt | 0.95 | 0.100 | 0.134 | 1.000 | 13.040 | 14.280 | 42.200 |

## Simple extract-then-store external pipeline

| run | update | answer | EM | F1 | state | stale same-slot | memory size |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| append_parsed_only_k16_dev | append | slot_direct | 1.000 | 1.000 | 1.000 | 14.250 | 31.000 |
| append_parsed_only_k16_dev_slot_prompt | append | slot_prompt | 0.140 | 0.177 | 1.000 | 14.250 | 31.000 |
| slot_update_parsed_only_k16_dev | slot_update | slot_direct | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 |
| slot_update_parsed_only_k16_dev_slot_prompt | slot_update | slot_prompt | 0.910 | 0.926 | 1.000 | 0.000 | 2.000 |
| slot_update_parsed_only_k16_test | slot_update | slot_direct | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 |

## Lost-in-the-Middle gold-position probe

| run | gold position | distractors | EM | F1 | answer value present |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen25_k16_dev | beginning | 8 | 0.010 | 0.073 | 0.040 |
| qwen25_k16_dev | middle | 8 | 0.090 | 0.183 | 0.190 |
| qwen25_k16_dev | end | 8 | 0.630 | 0.654 | 0.720 |

## Expanded latest-per-slot all-k

| k | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.955 | 0.970 | 1.000 | 1.000 | 0.000 |
| 2 | 0.940 | 0.954 | 0.980 | 1.000 | 0.000 |
| 4 | 0.855 | 0.855 | 0.905 | 0.980 | 0.000 |
| 8 | 0.925 | 0.929 | 0.935 | 0.990 | 0.000 |
| 16 | 0.750 | 0.764 | 0.775 | 0.860 | 0.000 |
