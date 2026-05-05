# P8.0 Multi-Model Stale Susceptibility Note

## Purpose

`docs/critical_review_v3.md` raised the risk that stale-context collapse and the mechanism probes might be specific to the Qwen answer model. This note records the first Llama3.1-8B-Instruct replication of the P8.0 stale susceptibility matrix.

## Artifacts

```text
scripts/eval_evomemory.py
scripts/run_p80_llama_replication_sui3.sh
scripts/summarize_context_mechanisms.py
results/p80_multimodel_stale_susceptibility/llama31_8b/
results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.{json,csv,md}
```

Remote execution:

```text
node: Sui-3-Wu
model: /NAS/HuggingFaceModels/Llama3.1-8B-Instruct
data: data/evomemory_update_frequency_hard_k16_p63_dev.json
mode: raw_add
answer_mode: slot_prompt
answer_topk: 5
save_answer_traces: true
no_qlora: true
```

## Conditions

| Condition | Purpose |
| --- | --- |
| raw_add normal top-k5 | Baseline stale-contaminated retrieval context |
| raw_add latest_per_slot top-k5 | Slot-aware retrieval-time stale filtering |
| raw_add latest/outdated label top-k5 | Semantic version disambiguation under the same retrieved set |
| raw_add chronological top-k5 | Order-sensitivity probe with stale entries before newer ones when timestamps are available |
| raw_add reverse chronological top-k5 | Opposite order-sensitivity probe |

## Results

| Condition | EM | F1 | answer value present | gold retrieved | stale retrieved | stale retrieved avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.060 | 0.062 | 0.120 | 0.360 | 1.000 | 4.040 |
| latest_per_slot top-k5 | 0.290 | 0.341 | 0.750 | 0.860 | 0.000 | 0.000 |
| latest/outdated labels | 0.080 | 0.105 | 0.350 | 0.360 | 1.000 | 4.040 |
| chronological | 0.020 | 0.039 | 0.140 | 0.360 | 1.000 | 4.040 |
| reverse chronological | 0.050 | 0.050 | 0.110 | 0.360 | 1.000 | 4.040 |

For comparison, Qwen and Mistral on the same P6.3 hard k=16 dev setup found:

| Condition | Qwen EM | Llama EM | Mistral EM |
| --- | ---: | ---: | ---: |
| normal top-k5 | 0.110 | 0.060 | 0.080 |
| latest/outdated labels | 0.260 | 0.080 | 0.300 |
| chronological | 0.230 | 0.020 | 0.150 |
| reverse chronological | 0.010 | 0.050 | 0.040 |
| latest_per_slot | 0.690 | 0.290 | 0.720 |

## Interpretation

The Llama and Mistral replications support the broad model-agnostic stale-contamination claim: normal raw_add top-k5 collapses for all three answer models while every example retrieves stale same-slot context. The slot-aware latest_per_slot intervention removes stale retrieval and improves EM for both additional models, so stale context remains a causal answer-time problem beyond Qwen.

After adding zero-stale constrained CRUD controls, the multi-model story is sharper than a generic "model-dependent mitigation magnitude" claim. Latest-per-slot recovers each tested model to approximately its own zero-stale slot-prompt ceiling: Qwen reaches 0.690 vs a constrained CRUD ceiling around 0.700, Llama reaches 0.290 vs a constrained ceiling 0.270, and Mistral reaches 0.720 vs a constrained ceiling 0.720. The absolute recovery differs because each model has a different slot-prompt / instruction-following ceiling under clean memory, not because stale filtering solves a fundamentally different fraction of stale-specific loss across models.

Labels and ordering interventions should therefore be interpreted relative to both retrieval composition and each model's clean-context ceiling. Llama labels remain weak in absolute EM, but Llama's zero-stale ceiling is also low; Mistral and Qwen have higher clean-context ceilings and show stronger absolute label/order gains.

## P8.1 Llama zero-stale control

The P8.1 rigor pass adds the missing control requested in `critical_review_v4.md`: a Llama3.1-8B-Instruct `constrained_slot_crud` k=16 dev run, where state accuracy is perfect and stale same-slot burden is forced to zero.

Artifacts:

```text
scripts/run_p81_llama_constrained_zero_stale_sui3.sh
results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json
results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json
```

Key numbers:

| Condition | EM | F1 | value EM | answer value present | state acc. | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama raw_add latest_per_slot | 0.290 | 0.341 | — | 0.750 | 1.000 | 0.000 |
| Llama constrained_slot_crud slot_prompt | 0.270 | 0.321 | 0.660 | 0.730 | 1.000 | 0.000 |
| Llama constrained_slot_crud slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Interpretation:

- The weak Llama recovery is not only a stale-retrieval problem.
- Even after moving from append-only retrieval filtering to a true zero-stale clean-state baseline, Llama slot-prompt EM remains only 0.270.
- Therefore the residual Llama gap is largely an answer-layer / instruction-following / answer-surface weakness under this slot-conditioned prompting setup, not just failure to remove stale same-slot context.
- This makes Llama a useful counterexample: stale susceptibility itself generalizes, but mitigation transfer can be bottlenecked by model-specific answer behavior even when state and retrieval are already clean.

This control closes the main interpretive hole in the earlier multi-model section. We no longer need to infer Llama's answer-layer weakness indirectly from low latest_per_slot EM; we now have an explicit zero-stale measurement.

## P8.2 Mistral zero-stale control

The P8.2 follow-up requested by `critical_review_v5.md` adds the corresponding Mistral-7B-Instruct constrained CRUD k=16 dev zero-stale baseline.

Artifacts:

```text
scripts/run_p82_mistral_constrained_zero_stale_sui3.sh
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json
```

Key numbers:

| Condition | EM | F1 | value EM | answer value present | state acc. | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral raw_add latest_per_slot | 0.720 | 0.735 | — | 0.750 | 1.000 | 0.000 |
| Mistral constrained_slot_crud slot_prompt | 0.720 | 0.735 | 0.750 | 0.750 | 1.000 | 0.000 |
| Mistral constrained_slot_crud slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

This closes the final missing cell in the three-model ceiling-recovery story.

## Ceiling-recovery comparison

| Model | normal raw_add EM | latest_per_slot EM | zero-stale constrained EM | Gap: latest_per_slot - ceiling |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 0.110 | 0.690 | 0.700 | -0.010 |
| Llama3.1-8B-Instruct | 0.060 | 0.290 | 0.270 | +0.020 |
| Mistral-7B-Instruct | 0.080 | 0.720 | 0.720 | 0.000 |

The paper can now state the stronger v5 conclusion:

> Across Qwen, Llama, and Mistral, raw append collapses under stale-contaminated top-k retrieval, and retrieval-time stale filtering recovers performance to approximately each model's own zero-stale slot-prompt ceiling. Absolute differences across models therefore reflect model-specific clean-context / instruction-following ceilings rather than a fundamentally model-specific stale mechanism.

This strengthens external validity while keeping the interpretation disciplined: latest_per_slot is still a diagnostic retrieval intervention, not a proposed deployable memory manager.

## Test-split confirmation

A matching P6.3 hard k=16 **test** rerun preserves the same qualitative picture:

| Condition | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.040 | 0.042 | 0.140 | 0.360 | 1.000 |
| latest_per_slot top-k5 | 0.290 | 0.345 | 0.760 | 0.870 | 0.000 |
| latest/outdated labels | 0.100 | 0.125 | 0.360 | 0.360 | 1.000 |
| chronological | 0.050 | 0.057 | 0.170 | 0.360 | 1.000 |
| reverse chronological | 0.040 | 0.040 | 0.100 | 0.360 | 1.000 |

The test split therefore confirms the main dev claim: stale-context collapse generalizes to Llama, latest-per-slot helps after removing stale retrieved entries, and the recovery remains much weaker than Qwen's.

## Caveats

- The Llama and Mistral matrices are diagnostic rather than exhaustive.
- The one-model-family concern is now addressed with two non-Qwen answer-model families, but test-split confirmation is currently complete for Llama and not yet for Mistral.
- The real-context label and ordering interventions remain low-accuracy settings for Llama, so they should be interpreted comparatively rather than as standalone strong baselines.
