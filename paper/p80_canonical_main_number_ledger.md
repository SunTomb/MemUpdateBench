# P8.0 Canonical Main-Number Ledger

## Purpose

This ledger lists the current main-paper candidate numbers and their exact source artifacts. Use this as the final consistency checklist before LaTeX conversion.

## P6.3 hard k=16 main tradeoff table

| Claim row | Split | Key numbers | Source |
| --- | --- | --- | --- |
| Raw append slot_direct | P6.3 test k=16 | state acc 1.00; stale same-slot 14.25; memory size 52.00 | `results/update_frequency_p63/raw_add_slot_direct_k16/evomemory_results.json` |
| Raw append slot_prompt | P6.3 test k=16 | EM/F1 0.07/0.10; stale same-slot 14.25; memory size 52.00 | `results/update_frequency_p63/raw_add_slot_prompt_k16/evomemory_results.json` |
| Constrained CRUD slot_prompt | P6.3 test k=16 | EM/F1 0.70/0.70; state acc 1.00; stale 0.00 | `results/update_frequency_p63/constrained_slot_crud_slot_prompt_k16/evomemory_results.json` |
| Heuristic CRUD slot_prompt | P6.3 test k=16 | EM/F1 0.10/0.13; stale same-slot 7.44; memory size 26.73 | `results/update_frequency_p63/heuristic_crud_slot_prompt_k16/evomemory_results.json` |
| Long25 original slot_prompt | P6.3 test k=16 | EM/F1 0.48/0.53; state acc 0.91; stale 1.13; memory 9.43 | `results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json` |

## Latest-per-slot intervention

| Claim row | Split | Key numbers | Source |
| --- | --- | --- | --- |
| Raw append normal top-k5 | P6.3 dev k=16 | EM/F1 0.140/0.173; state acc 1.00; stale 14.25; memory 52.00 | `results/p68_answer_layer_diagnostics/raw_add_normal_topk5/evomemory_results.json` or P6.8 diagnostic note |
| Raw append latest_per_slot | P6.3 dev k=16 | EM/F1 0.690/0.703; stale burden unchanged; stale retrieved removed | `results/p70_stale_filter_intervention/raw_add_latest_per_slot_k16_dev/evomemory_results.json` |
| P6.3 latest_per_slot all-k | P6.3 dev | k=1/2/4/8/16 EM 1.000/0.910/0.850/0.990/0.690 | `results/p70_stale_filter_intervention_summary/stale_filter_allk_filtered.md` |
| Expanded latest_per_slot all-k | expanded dev | k=1/2/4/8/16 EM 0.955/0.940/0.855/0.925/0.750 | `results/p80_expanded_latest_per_slot_summary/expanded_latest_per_slot_summary.md` |

## Dose-response

| Claim | Split/source | Key numbers | Source |
| --- | --- | --- | --- |
| Stored stale count dose | pooled raw_add slot_prompt | stale 0/1/3 EM 0.967/0.743/0.290 | `results/p80_stale_dose_response/stale_dose_response.md` |
| Retrieved stale count dose | pooled raw_add slot_prompt | retrieved stale 0/1/2 EM 1.000/0.667/0.174 | `results/p80_stale_dose_response/stale_dose_response.md` |
| Logistic ED50 | pooled raw_add slot_prompt | stored ED50 3.18; retrieved ED50 1.89 | `results/p80_stale_dose_response/stale_dose_logistic.json` |

## Real-context mechanism probe

| Condition | Split | EM/F1 | Source |
| --- | --- | ---: | --- |
| normal none | P6.3 dev k=16 | 0.110/0.136 | `results/p80_mechanism_probe_summary/context_mechanism_summary.md` |
| chronological none | P6.3 dev k=16 | 0.230/0.275 | `results/p80_mechanism_probe_summary/context_mechanism_summary.md` |
| reverse chronological none | P6.3 dev k=16 | 0.010/0.050 | `results/p80_mechanism_probe_summary/context_mechanism_summary.md` |
| timestamp | P6.3 dev k=16 | 0.150/0.200 | `results/p80_mechanism_probe_summary/context_mechanism_summary.md` |
| latest/outdated label | P6.3 dev k=16 | 0.260/0.298 | `results/p80_mechanism_probe_summary/context_mechanism_summary.md` |

Fixed retrieval composition across rows:

```text
gold retrieved = 0.360
stale retrieved = 1.000
stale retrieved avg = 4.040
```

## Synthetic same-slot probe

| Condition | Key numbers | Source |
| --- | --- | --- |
| conflict + reverse chronological + no labels | pilot: stale=1 EM 0.188; stale=2 EM 0.062; stale=8/16 EM 0.000 | `results/p80_synthetic_same_slot_probe_analysis/synthetic_same_slot_grouped_summary.md` |
| conflict + reverse chronological + labels | pilot: nearly repaired; stale=1/2/8/16 EM mostly 1.000, stale=4 EM 0.875 | same |
| conflict + chronological + no labels | pilot: stale=16 EM 0.750 | same |
| same_as_current + chronological + no labels | pilot: stale=4/8/16 EM 0.562/0.500/0.562; answer-value-present 1.000 | same |
| P8.1 expanded conflict + reverse chronological + no labels | n=64/cell; stale=1/2/8/16 EM 0.234/0.094/0.000/0.031 | `results/p81_synthetic_same_slot_probe_expanded_analysis/synthetic_same_slot_grouped_summary.md` |
| P8.1 expanded conflict + reverse chronological + labels | n=64/cell; stale=1/2/8/16 EM 0.969/0.969/1.000/1.000 | same |
| P8.1 expanded conflict + chronological + no labels | n=64/cell; stale=16 EM 0.797 | same |
| P8.1 expanded same_as_current + chronological + no labels | n=64/cell; stale=4/8/16 EM 0.688/0.547/0.531; answer-value-present 1.000 | same |
| P8.1 same_as_current failure classification | 76 existing-pilot EM-fail / answer-value-present cases classified; non-exact answer surface rather than stale-value selection | `results/p81_same_as_current_failure_analysis/same_as_current_failure_analysis.md` |

## P8.1 k-controlled heuristic threshold sweep

| Condition | Split | Key numbers | Source |
| --- | --- | --- | --- |
| heuristic slot_prompt threshold 0.70 | P6.3 dev k=16 | EM/F1 0.220/0.247; state 1.000; stale 4.43; memory 11.57 | `results/p81_heuristic_threshold_k16_rigor_summary/heuristic_threshold_k16_summary.md` |
| heuristic slot_prompt threshold 0.85 | P6.3 dev k=16 | EM/F1 0.140/0.164; state 1.000; stale 7.43; memory 26.67 | same |
| heuristic slot_prompt threshold 0.95 | P6.3 dev k=16 | EM/F1 0.100/0.134; state 1.000; stale 13.04; memory 42.20 | same |
| heuristic slot_direct threshold sweep | P6.3 dev k=16 | all thresholds state/EM/F1 1.000 while stale rises 4.43 -> 13.04 | same |

## Lost-in-the-Middle gold-position probe

| Condition | Split | Key numbers | Source |
| --- | --- | --- | --- |
| gold at beginning | P6.3 dev k=16 | EM/F1 0.010/0.073; answer value present 0.040 | `results/p80_lost_in_middle_probe_summary/lost_in_middle_summary.md` |
| gold in middle | P6.3 dev k=16 | EM/F1 0.090/0.183; answer value present 0.190 | same |
| gold at end | P6.3 dev k=16 | EM/F1 0.630/0.654; answer value present 0.720 | same |

## Llama3.1-8B dev replication

| Condition | Split | Key numbers | Source |
| --- | --- | --- | --- |
| raw_add normal top-k5 | P6.3 dev k=16 | EM/F1 0.060/0.062; stale retrieved 1.000 | `results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.md` |
| raw_add latest_per_slot | P6.3 dev k=16 | EM/F1 0.290/0.341; stale retrieved 0.000 | same |
| raw_add latest/outdated labels | P6.3 dev k=16 | EM/F1 0.080/0.105 | same |
| raw_add chronological | P6.3 dev k=16 | EM/F1 0.020/0.039 | same |
| raw_add reverse chronological | P6.3 dev k=16 | EM/F1 0.050/0.050 | same |

Matching test-split confirmation:

| Condition | Split | Key numbers | Source |
| --- | --- | --- | --- |
| raw_add normal top-k5 | P6.3 test k=16 | EM/F1 0.040/0.042; stale retrieved 1.000 | `results/p80_multimodel_stale_susceptibility_summary/llama31_8b_test_context_summary.md` |
| raw_add latest_per_slot | P6.3 test k=16 | EM/F1 0.290/0.345; stale retrieved 0.000 | same |
| raw_add latest/outdated labels | P6.3 test k=16 | EM/F1 0.100/0.125 | same |
| raw_add chronological | P6.3 test k=16 | EM/F1 0.050/0.057 | same |
| raw_add reverse chronological | P6.3 test k=16 | EM/F1 0.040/0.040 | same |

## P8.1 Llama zero-stale control

| Condition | Split | Key numbers | Source |
| --- | --- | --- | --- |
| Llama constrained_slot_crud slot_prompt | P6.3 dev k=16 | EM/F1 0.270/0.321; value EM 0.660; answer value present 0.730; state 1.000; stale 0.000 | `results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json` |
| Llama constrained_slot_crud slot_direct | P6.3 dev k=16 | EM/F1 1.000/1.000; state 1.000; stale 0.000 | `results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json` |

## Mistral-7B dev replication

| Condition | Split | Key numbers | Source |
| --- | --- | --- | --- |
| raw_add normal top-k5 | P6.3 dev k=16 | EM/F1 0.080/0.177; stale retrieved 1.000 | `results/p80_multimodel_stale_susceptibility_summary/mistral7b_context_summary.md` |
| raw_add latest_per_slot | P6.3 dev k=16 | EM/F1 0.720/0.735; stale retrieved 0.000 | same |
| raw_add latest/outdated labels | P6.3 dev k=16 | EM/F1 0.300/0.332 | same |
| raw_add chronological | P6.3 dev k=16 | EM/F1 0.150/0.182 | same |
| raw_add reverse chronological | P6.3 dev k=16 | EM/F1 0.040/0.117 | same |

## P8.2 Mistral zero-stale control

| Condition | Split | Key numbers | Source |
| --- | --- | --- | --- |
| Mistral constrained_slot_crud slot_prompt | P6.3 dev k=16 | EM/F1 0.720/0.735; value EM 0.750; answer value present 0.750; state 1.000; stale 0.000 | `results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json` |
| Mistral constrained_slot_crud slot_direct | P6.3 dev k=16 | EM/F1 1.000/1.000; state 1.000; stale 0.000 | `results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json` |

## Ceiling-recovery headline comparison

| Model | Split | Normal raw_add EM | latest_per_slot EM | zero-stale constrained EM | Source |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B-Instruct | P6.3 dev k=16 | 0.110 | 0.690 | 0.700 | `results/p80_mechanism_probe_summary/context_mechanism_summary.md`; `results/p70_stale_filter_intervention/raw_add_latest_per_slot_k16_dev/evomemory_results.json`; `results/update_frequency_p63/constrained_slot_crud_slot_prompt_k16/evomemory_results.json` |
| Llama3.1-8B-Instruct | P6.3 dev k=16 | 0.060 | 0.290 | 0.270 | `results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.md`; `results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json` |
| Mistral-7B-Instruct | P6.3 dev k=16 | 0.080 | 0.720 | 0.720 | `results/p80_multimodel_stale_susceptibility_summary/mistral7b_context_summary.md`; `results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json` |

## Long25 provenance

| Family | Split | Key numbers | Source |
| --- | --- | --- | --- |
| P6.3 original | P6.3 test k=16 | EM/F1 0.48/0.53; state 0.91; stale 1.13; memory 9.43 | `results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json` |
| P6.5 seed 11 | P6.3 test k=16 | EM/F1 0.870/0.903; state 0.940; stale 0.10; memory 2.06 | `results/p65_stability_sharded/seed11_merged/evomemory_results.json` |
| P6.5 seed 22 | P6.3 test k=16 | EM/F1 0.880/0.913; state 0.990; stale 0.07; memory 2.04 | `results/p65_stability_sharded/seed22_merged/evomemory_results.json` |
| P6.5 seed 33 | P6.3 test k=16 | EM/F1 0.890/0.910; state 0.970; stale 0.05; memory 2.02 | `results/p65_stability_sharded/seed33_merged/evomemory_results.json` |

Paper wording: report as separate checkpoint families; do not call the gap pure seed sensitivity.

## External baseline status

| Baseline | Status | Source |
| --- | --- | --- |
| Mem0 Qwen2.5-VL dev20 | runnable but qualitative/unfair; EM/F1 0.00/0.05 after value extraction | `paper/p70_external_baseline_fairness_note.md` |
| Mem0 Qwen2.5-7B text backend | queryable server and CPU MiniLM work, but structured extraction fails before dev3 completes | `paper/p70_external_baseline_text_backend_probe.md` |
| Original Memory-R1 | unavailable; project-local approximation must not be reported as original | `paper/p70_external_baseline_fairness_note.md` |
