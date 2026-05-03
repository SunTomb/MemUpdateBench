# P6.7 Diagnostic and Ablation Plan

## Purpose

This plan adds research depth without launching new training or generating new splits prematurely. The goal is to explain the current P6.3 tradeoff more sharply and define future ablations behind deterministic-oracle gates.

## Diagnostic questions using existing results

### Q1. Why does Constrained CRUD drop under slot-conditioned answering?

Observation:

- k=16 Constrained CRUD slot-direct state accuracy: 1.00
- k=16 Constrained CRUD stale same-slot entries: 0.00
- k=16 Constrained CRUD slot-prompt EM/F1: 0.70 / 0.70

Hypothesis:

Even clean final memory state does not eliminate answer-layer failures under hard distractors. The model may still fail to bind the prompted slot, copy the exact value, or ignore distractor content.

Analysis hook:

```bash
python scripts/analyze_ood_errors.py \
  --inputs results/update_frequency_p63/constrained_slot_crud_slot_prompt_k16/evomemory_results.json
```

Expected categories:

- answer-layer extraction failure,
- wrong entity grounding,
- wrong attribute grounding,
- value formatting mismatch.

### Q2. What kind of error dominates Long25 at k=16?

Observation:

- k=16 Long25 slot-direct state accuracy: 0.91
- k=16 Long25 slot-prompt EM/F1: 0.48 / 0.53
- k=16 Long25 stale same-slot entries: 1.13
- k=16 Long25 final memory size: 9.43

Hypothesis:

Long25 improves compactness but sometimes misses the final update or compresses incompletely. Its remaining prompt errors may mix state-level misses and answer-layer failures.

Analysis hook:

```bash
python scripts/analyze_ood_errors.py \
  --inputs results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json
```

Expected categories:

- missed final update,
- stale retained value,
- wrong entity grounding,
- wrong attribute parsing,
- answer extraction failure.

### Q3. Does stale burden grow linearly while answer quality collapses nonlinearly?

Observation:

Raw append stale same-slot burden grows from 0.00 at k=1 to 14.25 at k=16, while slot-prompt EM falls from 0.90 to 0.07.

Hypothesis:

Stale burden grows approximately with repeated updates, while answer robustness collapses once multiple conflicting same-slot values are visible.

Analysis hook:

Use the generated derived figures:

```text
paper/p63_gap_slot_direct_vs_prompt.pdf
paper/p63_stale_vs_prompt_em_k16.pdf
paper/p63_memory_size_vs_prompt_em_k16.pdf
```

## Future ablation candidates

These are design candidates only. Do not implement until the deterministic oracle plan is approved.

### Candidate A: late-final-update hard

Motivation:

Stress whether managers correctly preserve the final update when distractors and stale values appear immediately before the query.

Deterministic oracle expectation:

`constrained_slot_crud` should achieve slot-direct state accuracy 1.00 and stale same-slot burden 0.00.

Generation constraints:

- Same `(entity, attribute)` receives many updates.
- The final update appears late, after at least one same-name distractor event.
- NOOP events must not alter the target slot.

Stop condition:

Do not evaluate learned managers unless deterministic oracle is perfect or any oracle imperfection is explained.

### Candidate B: same-name-only hard

Motivation:

Isolate entity grounding when many entities share surface names but differ in attributes or context.

Deterministic oracle expectation:

Exact `(entity, attribute)` resolution should recover the target final value.

Generation constraints:

- Multiple entities share name tokens.
- Attribute names are unambiguous.
- Distractor updates target non-gold entities.

Stop condition:

If oracle fails, inspect entity ID construction before any learned eval.

### Candidate C: attribute-near-miss hard

Motivation:

Stress attribute parsing with semantically close but distinct attributes.

Deterministic oracle expectation:

Exact attribute matching should separate target and near-miss attributes.

Generation constraints:

- Same entity, multiple attributes with near-miss names.
- NOOP and distractor events mention related attributes.
- Gold query names the exact target attribute.

Stop condition:

If oracle fails, fix parser/data generation first.

## Non-goals for this diagnostic phase

- No repair training.
- No k=32 generation by default.
- No new split implementation without oracle-first approval.
- No external baseline integration.
