# P6.8 Answer-Layer Mechanism Note

## Motivation

`docs/critical_review.md` identifies the Constrained CRUD k=16 gap as one of the most interesting under-analyzed findings: even with perfect state accuracy and zero stale same-slot burden, slot-prompt EM is only about 0.70. This note records the first controlled answer-layer mechanism probe.

## Probe design

The probe compares normal retrieval contexts at different `answer_topk` values with an oracle `gold_context` that contains only the final gold same-slot event.

Current scope:

```text
data/evomemory_update_frequency_hard_k16_p63_dev.json
100 examples
slot_prompt_variant: v0_current
answer_topk: 1, 3, 5, 10
```

Outputs:

```text
results/p68_answer_layer_diagnostics/constrained_k16_dev10/
results/p68_answer_layer_diagnostics/raw_add_k16_dev10/
scripts/analyze_answer_layer_mechanism.py
```

## Pilot results

### Constrained CRUD

| Context | N | EM | F1 | Value EM | Answer value present | State acc. | Gold retrieved | Stale retrieved | Distractor retrieved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gold_context` | 100 | 0.920 | 0.927 | 0.920 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `retrieval_topk_1` | 100 | 0.510 | 0.548 | 0.510 | 0.600 | 1.000 | 0.600 | 0.000 | 0.030 |
| `retrieval_topk_3` | 100 | 0.670 | 0.674 | 0.680 | 0.700 | 1.000 | 0.770 | 0.000 | 0.260 |
| `retrieval_topk_5` | 100 | 0.670 | 0.672 | 0.670 | 0.710 | 1.000 | 0.860 | 0.000 | 0.330 |
| `retrieval_topk_10` | 100 | 0.760 | 0.760 | 0.760 | 0.760 | 1.000 | 0.950 | 0.000 | 0.620 |

Interpretation: the clean-state gap is largely a retrieval-context issue. State accuracy remains 1.00 throughout, and oracle gold context reaches 0.92 EM. Increasing top-k improves gold retrieval, but top-k 10 still trails gold-only context because it admits many distractors.

### Raw append

| Context | N | EM | F1 | Value EM | Answer value present | State acc. | Gold retrieved | Stale retrieved | Distractor retrieved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gold_context` | 100 | 0.920 | 0.927 | 0.920 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `retrieval_topk_1` | 100 | 0.050 | 0.071 | 0.050 | 0.060 | 1.000 | 0.060 | 0.880 | 0.060 |
| `retrieval_topk_3` | 100 | 0.060 | 0.111 | 0.060 | 0.150 | 1.000 | 0.270 | 1.000 | 0.210 |
| `retrieval_topk_5` | 100 | 0.140 | 0.173 | 0.140 | 0.160 | 1.000 | 0.360 | 1.000 | 0.300 |
| `retrieval_topk_10` | 100 | 0.070 | 0.145 | 0.070 | 0.130 | 1.000 | 0.620 | 1.000 | 0.410 |

Interpretation: Raw append failure is much more directly stale-burden driven. Normal retrieval includes stale same-slot entries in 88-100% of examples and answers poorly, while gold-only context recovers to 0.92 EM. Increasing top-k improves gold retrieval but does not improve EM when stale entries remain present.

## Current conclusion

This pilot supports a sharper mechanism story:

1. Clean final state is not sufficient for prompted answer robustness.
2. Constrained CRUD's residual gap is primarily retrieval-context/gold-recall limited, not stale-state burden.
3. Raw append's collapse is dominated by stale same-slot retrieval contamination.
4. Gold-only context recovers both methods to high EM, so the answer model can often produce the final value when the context is controlled.

## Next step

Scale this pilot from 10 examples to the full k=16 dev/test split, then add a stale-filtering intervention for Raw append to test whether removing stale same-slot entries from retrieval restores slot-prompt EM under non-oracle context.
