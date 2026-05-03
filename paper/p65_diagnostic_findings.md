# P6.5 Diagnostic Findings

## Purpose

This note records the first answer-trace diagnostics from the P6.5 prompt robustness sweep. It focuses on k=16 because that is the hardest existing P6.3 endpoint and the clearest reviewer-risk case for prompt-conditioned failure.

## Inputs

```text
results/p65_prompt_robustness/*_k16/evomemory_results.json
results/p65_diagnostics/k16_prompt_diagnostics.json
```

The diagnostics cover Constrained CRUD, Raw append, and Long25 across `v0_current`, `v1_value_only`, and `v2_ignore_distractors`.

## k=16 answer-layer decomposition

### Constrained CRUD

Constrained CRUD keeps perfect state accuracy and zero stale same-slot burden across variants, so its residual failures are answer-layer or retrieval-context failures rather than state-update failures.

| Variant | State acc. | Answer correct | Gold not retrieved | Gold retrieved but wrong | Format only | Distractor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v0_current` | 1.00 | 69 | 14 | 13 | 2 | 2 |
| `v1_value_only` | 1.00 | 69 | 14 | 14 | 1 | 2 |
| `v2_ignore_distractors` | 1.00 | 68 | 14 | 15 | 0 | 3 |

Interpretation: stronger prompt wording does not close the clean-state answer gap. About 14% of examples miss the gold value in the retrieved context even with compact final state, and another 13-15% have the gold retrieved but the answer layer still returns the wrong value.

### Raw append

Raw append keeps perfect state accuracy but collapses under slot-prompt because stale same-slot burden dominates retrieval.

| Variant | State acc. | Answer correct | Gold not retrieved | Stale contamination | Multiple values | Format only | Distractor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v0_current` | 1.00 | 11 | 61 | 18 | 5 | 3 | 2 |
| `v1_value_only` | 1.00 | 9 | 60 | 16 | 0 | 13 | 2 |
| `v2_ignore_distractors` | 1.00 | 9 | 60 | 20 | 0 | 9 | 2 |

Interpretation: prompt wording does not rescue append-only memory. At k=16, stale same-slot entries crowd retrieval so the gold value is absent from the answer context in about 60% of examples, while stale contamination accounts for another large share.

### Long25

Long25 is compact and has much lower stale burden than Raw append, but still mixes state errors with retrieval and answer-layer failures.

| Variant | State acc. | Answer correct | State wrong | Gold not retrieved | Distractor | Stale | Format only | Gold retrieved but wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v0_current` | 0.92 | 45 | 8 | 7 | 24 | 9 | 5 | 2 |
| `v1_value_only` | 0.92 | 42 | 8 | 7 | 27 | 10 | 4 | 2 |
| `v2_ignore_distractors` | 0.92 | 48 | 8 | 7 | 25 | 10 | 0 | 2 |

Interpretation: Long25's prompt-conditioned gap is not only final-state error. The fixed 8 state-wrong examples explain part of the deficit, but many failures occur when state is correct and the retrieved context contains distractors or stale values. `v2_ignore_distractors` improves answer correctness slightly at k=16, but does not remove the mixed failure mode.

## Long25 action-pathology by k

A first action-level pass compares Long25 `v0_current` predictions with reconstructed gold constrained actions from the dev split.

| k | Full action acc. | Op acc. | Invalid | Wrong op | Wrong entity | Wrong attr. | Wrong value | Unnecessary NOOP | Missed NOOP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.200 | 0.200 | 0.000 | 0.800 | 0.010 | 0.010 | 0.010 | 0.010 | 0.460 |
| 2 | 0.552 | 0.552 | 0.000 | 0.448 | 0.016 | 0.016 | 0.016 | 0.016 | 0.350 |
| 4 | 0.654 | 0.654 | 0.003 | 0.346 | 0.037 | 0.038 | 0.037 | 0.037 | 0.274 |
| 8 | 0.655 | 0.662 | 0.002 | 0.338 | 0.049 | 0.051 | 0.054 | 0.049 | 0.271 |
| 16 | 0.676 | 0.684 | 0.001 | 0.316 | 0.046 | 0.051 | 0.050 | 0.046 | 0.262 |

Interpretation: Long25 is not primarily failing through malformed actions; invalid action rate is near zero. The dominant action-level issue is structured decision error, especially wrong operation and missed `NOOP`. Entity, attribute, and value field errors remain smaller but non-trivial at high k. This supports repair-training or action-objective diagnostics more than output-format cleanup.

## Main conclusion

The new traces support the tradeoff framing. Clean state is necessary but not sufficient for robust prompted answering; stale append systems fail largely through retrieval-context pollution; and learned compact managers reduce stale burden but still need better structured action reliability and answer-context robustness.

## Next diagnostic step

The next step is to run Long25 stability checks or a targeted repair diagnostic. Because invalid outputs are already rare, repair should focus on operation selection and NOOP discrimination rather than only stricter output formatting.
