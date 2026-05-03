# P6.7 k=16 Error Analysis

## Purpose

This note explains the two key high-frequency failure questions:

1. Why can Constrained CRUD have perfect slot-state accuracy but imperfect slot-conditioned answering?
2. What does the Long25 k=16 spot-check reveal about the compactness/reliability tradeoff?

## Inputs

Canonical local inputs:

```text
results/update_frequency_p63/constrained_slot_crud_slot_prompt_k16/evomemory_results.json
results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json
```

Remote verification inputs, when available:

```text
results/remote_sanity_oracle_k16/evomemory_results.json
results/remote_long25_slot_direct_k16/evomemory_results.json
results/remote_long25_slot_prompt_k16/evomemory_results.json
```

## Constrained CRUD: clean state does not guarantee perfect prompted answering

Canonical k=16 facts:

```text
slot-direct state_acc = 1.00
slot-prompt EM/F1 = 0.70 / 0.70
stale_same_slot = 0.00
```

Interpretation:

Constrained CRUD removes stale same-slot memory by construction, so its remaining slot-prompt errors are not stale-retention errors. They should be treated as answer-layer or prompt-conditioning failures under hard same-name distractors and near-miss context.

## Long25: compact but imperfectly reliable

Canonical k=16 facts:

```text
slot-direct state_acc = 0.91
slot-prompt EM/F1 = 0.48 / 0.53
stale_same_slot = 1.13
final_memory_size = 9.43
```

Interpretation:

Long25 substantially reduces stale burden and memory size, but the state accuracy gap means some final values are missed or incorrectly represented. This supports the compactness/reliability tradeoff framing.

## Analysis commands

Use the OOD error analyzer on available result files:

```bash
python scripts/analyze_ood_errors.py \
  --inputs results/update_frequency_p63/constrained_slot_crud_slot_prompt_k16/evomemory_results.json \
           results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json \
  --output paper/error_analysis_k16_local.json
```

If remote Long25 reruns complete:

```bash
python scripts/analyze_ood_errors.py \
  --inputs results/remote_long25_slot_direct_k16/evomemory_results.json \
           results/remote_long25_slot_prompt_k16/evomemory_results.json \
  --output paper/error_analysis_k16_remote_long25.json
```

## Local analyzer summary

Command:

```bash
python scripts/analyze_ood_errors.py \
  --inputs results/update_frequency_p63/constrained_slot_crud_slot_prompt_k16/evomemory_results.json \
           results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json \
  --output paper/error_analysis_k16_local.json
```

State-level analyzer results:

| Input | State correct | State error pattern | Stale avg | Memory size avg |
| --- | ---: | --- | ---: | ---: |
| constrained CRUD slot_prompt k=16 | 100 / 100 | none | 0.00 | 23.00 |
| Long25 slot_prompt k=16 | 91 / 100 | 9 wrong_value | 1.13 | 9.43 |

For constrained CRUD, the state-level analyzer finds no state errors. Its slot-prompt EM/F1 gap therefore comes from answer-layer/prompt-conditioned failures rather than stale same-slot state errors.

For Long25, all 9 state errors are `wrong_value`; there are no wrong-entity or wrong-attribute state errors in this analyzer output. The errors concentrate mostly in company slots: `company` has 8 wrong-value cases, while `language` has 1. This supports a reliability-cost framing: Long25 usually resolves the right slot but sometimes stores or preserves the wrong final value.

## Remote Long25 spot-check summary

Sui-3-Wu completed both k=16 Long25 spot-checks under the expected NAS `gmsra` environment.

Observed metrics:

| Remote input | State correct | EM/F1 | State error pattern | Stale avg | Memory size avg |
| --- | ---: | ---: | --- | ---: | ---: |
| Long25 slot_direct k=16 | 92 / 100 | 0.92 / 0.92 | 8 wrong_value | 1.13 | 9.28 |
| Long25 slot_prompt k=16 | 92 / 100 | 0.49 / 0.5467 | 8 wrong_value | 1.13 | 9.28 |

Remote analysis file:

```text
paper/error_analysis_k16_remote_long25.json
```

The remote rerun closely confirms the canonical ledger. The small metric differences do not change the paper claim: Long25 remains much more compact and lower-stale than append-only baselines, but still has wrong-value final-state failures under hard k=16 distractors.

Remote state errors are again concentrated in `company` slots: 8 wrong-value company errors, with no wrong-entity or wrong-attribute state errors. This strengthens the interpretation that Long25 usually identifies the right slot but can fail to preserve the final value.

## Paper implication

The most precise wording is:

> Long25 reduces stale same-slot burden and memory size, but its residual k=16 failures are primarily wrong-value final-state errors, especially on company slots. This supports the compactness/reliability tradeoff rather than a simple learned-manager win.
