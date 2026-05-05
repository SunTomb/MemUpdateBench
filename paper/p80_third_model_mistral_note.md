# P8.0 Third-Model Mistral Replication Note

## Purpose

导师建议在 Qwen + Llama 之外再补第三个 answer model，例如 Mistral-7B 或 Phi-3。Sui-3 的 `/NAS/HuggingFaceModels` 中已经存在：

```text
/NAS/HuggingFaceModels/Mistral-7B-Instruct-v0.1
/NAS/HuggingFaceModels/Phi-3-mini-4k-instruct
```

本轮优先选择 Mistral-7B-Instruct-v0.1，因为它更接近已有 Qwen/Llama 的 7B/8B 量级。

## Artifacts

```text
scripts/run_p80_third_model_replication_sui3.sh
results/p80_multimodel_stale_susceptibility/mistral7b_smoke/
results/p80_multimodel_stale_susceptibility/mistral7b/
results/p80_multimodel_stale_susceptibility_summary/mistral7b_smoke_context_summary.{json,csv,md}
results/p80_multimodel_stale_susceptibility_summary/mistral7b_context_summary.{json,csv,md}
```

## Smoke result

A dev3 smoke test completed successfully across the same five-condition matrix used for Llama:

| Condition | EM | F1 | answer value present | gold retrieved | stale retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.000 | 0.167 | 0.000 | 0.000 | 1.000 |
| latest_per_slot | 0.667 | 0.667 | 0.667 | 0.667 | 0.000 |
| latest/outdated labels | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| chronological | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| reverse chronological | 0.000 | 0.167 | 0.000 | 0.000 | 1.000 |

The smoke run is too small for paper claims, but it verifies that Mistral loads, answers, saves traces, and follows the expected qualitative direction: stale-contaminated normal retrieval is poor, while latest_per_slot can help after removing stale retrieved entries.

## Full dev result

The full P6.3 hard k=16 dev matrix completed on Sui-3.

| Condition | EM | F1 | answer value present | gold retrieved | stale retrieved | stale retrieved avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal top-k5 | 0.080 | 0.177 | 0.140 | 0.360 | 1.000 | 4.040 |
| latest_per_slot | 0.720 | 0.735 | 0.750 | 0.860 | 0.000 | 0.000 |
| latest/outdated labels | 0.300 | 0.332 | 0.350 | 0.360 | 1.000 | 4.040 |
| chronological | 0.150 | 0.182 | 0.190 | 0.360 | 1.000 | 4.040 |
| reverse chronological | 0.040 | 0.117 | 0.060 | 0.360 | 1.000 | 4.040 |

## P8.2 zero-stale control

`critical_review_v5.md` pointed out that the Mistral row needed one additional control: constrained CRUD under the same k=16 dev split, so that Mistral's latest_per_slot recovery could be compared to its own zero-stale slot-prompt ceiling.

Artifacts:

```text
scripts/run_p82_mistral_constrained_zero_stale_sui3.sh
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json
```

Results:

| Condition | EM | F1 | value EM | answer value present | state | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud slot_prompt | 0.720 | 0.735 | 0.750 | 0.750 | 1.000 | 0.000 |
| constrained_slot_crud slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Interpretation

Mistral provides the requested third-model evidence. It reproduces the same stale-contamination failure pattern: normal top-k5 retrieval has stale retrieved rate 1.000 and collapses to EM 0.080. The latest_per_slot rewrite removes stale retrieved entries and recovers to EM/F1 0.720/0.735.

The P8.2 zero-stale control changes the interpretation from a loose model-dependent mitigation story to a cleaner ceiling-recovery story: Mistral latest_per_slot EM/F1 0.720/0.735 exactly matches Mistral constrained-slot zero-stale EM/F1 0.720/0.735. Together with Qwen and Llama, this shows that stale filtering recovers each model to approximately its own clean-memory slot-prompt ceiling.

The context-presentation interventions also behave more like Qwen than Llama: latest/outdated labels improve from 0.080 to 0.300, and chronological order improves to 0.150, while reverse chronological remains poor at 0.040. These absolute gains should be interpreted relative to each model's clean-context ceiling.

Paper-level implication:

> A third-model Mistral-7B replication confirms that stale same-slot context collapse is not a Qwen/Llama pair artifact. More importantly, Mistral completes the ceiling-recovery pattern: raw append collapses under stale-contaminated retrieval, and latest-per-slot filtering recovers exactly to Mistral's own zero-stale constrained-slot ceiling.
