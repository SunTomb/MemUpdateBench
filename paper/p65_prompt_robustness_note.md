# P6.5 Prompt Robustness Note

## Purpose

This note records whether the P6.3 slot-prompt collapse is stable under mild prompt wording changes. The goal is to distinguish a robust stale-memory phenomenon from a single-template prompt artifact.

## Prompt variants

- `v0_current`: the original slot-conditioned answer prompt.
- `v1_value_only`: stricter value-only response instruction.
- `v2_ignore_distractors`: value-only response plus explicit instruction to ignore distractors, older values, other entities, and other attributes.

## Result roots

```text
results/p65_prompt_robustness/
results/p65_prompt_robustness_summary/
```

## Summary command

```bash
python scripts/summarize_prompt_robustness.py \
  --result_root results/p65_prompt_robustness \
  --output_dir results/p65_prompt_robustness_summary
```

## Interpretation rules

- If variants change absolute EM/F1 but preserve the main ordering and high-k collapse pattern, the paper can claim the phenomenon is prompt-robust within these mild template changes.
- If stronger prompts substantially improve all methods, the manuscript should report prompt sensitivity and avoid treating one slot-prompt template as definitive.
- If stronger prompts help Constrained CRUD but not Raw append/Heuristic CRUD, that supports the interpretation that clean-state answer-layer failures are prompt-sensitive while stale-burdened memory remains brittle.

## Results

The cluster sweep completed 45/45 cells on `Sui-3-Wu` and wrote summaries under `results/p65_prompt_robustness_summary/`.

At k=16, the main ordering is stable across mild prompt variants:

| Method | Variant | EM | F1 | State acc. | Stale same-slot | Gold retrieved | Stale retrieved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | `v0_current` | 0.69 | 0.70 | 1.00 | 0.00 | 0.86 | 0.00 |
| Constrained CRUD | `v1_value_only` | 0.69 | 0.69 | 1.00 | 0.00 | 0.86 | 0.00 |
| Constrained CRUD | `v2_ignore_distractors` | 0.68 | 0.68 | 1.00 | 0.00 | 0.86 | 0.00 |
| Long25 | `v0_current` | 0.45 | 0.50 | 0.92 | 1.07 | 0.86 | 0.57 |
| Long25 | `v1_value_only` | 0.42 | 0.46 | 0.92 | 1.07 | 0.86 | 0.57 |
| Long25 | `v2_ignore_distractors` | 0.48 | 0.51 | 0.92 | 1.07 | 0.86 | 0.57 |
| Raw append | `v0_current` | 0.11 | 0.14 | 1.00 | 14.25 | 0.36 | 1.00 |
| Raw append | `v1_value_only` | 0.09 | 0.14 | 1.00 | 14.25 | 0.36 | 1.00 |
| Raw append | `v2_ignore_distractors` | 0.09 | 0.14 | 1.00 | 14.25 | 0.36 | 1.00 |

## Interpretation

The high-k pattern is not explained by one fragile answer-template wording. Stronger value-only or ignore-distractor instructions shift absolute EM/F1 slightly, but they do not rescue Raw append from stale-burden collapse and do not remove the Constrained CRUD answer-layer gap. Long25 remains between the clean-state oracle and stale append baseline, with better compactness and lower stale burden than Raw append but residual state and answer-layer failures.

The answer-trace rates also separate failure modes: Raw append has perfect state accuracy but low gold retrieval at k=16 and near-universal stale retrieval; Constrained CRUD has no stale retrieval but still misses gold in the retrieved context for 14% of examples; Long25 has comparable gold retrieval to Constrained CRUD but also retrieves stale same-slot entries in 57% of examples.
