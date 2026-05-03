# P6.8 Stale-Burden Intervention Note

## Motivation

`docs/critical_review.md` argues that the tradeoff claim risks sounding like a truism unless the project tests the mechanism behind stale burden. This note records the first intervention-style analysis using Raw append k=16 dev answer-layer traces.

## Inputs

```text
results/p68_answer_layer_diagnostics/raw_add_k16_dev/answer_layer_mechanism.json
scripts/analyze_stale_intervention.py
```

The analysis compares normal Raw append retrieval at `retrieval_topk_5` with:

- `gold_context`: only the final gold same-slot event;
- `retrieval_topk_10`: more retrieved context without removing stale entries;
- `retrieval_topk_1`: less retrieved context.

## Context summaries

| Context | N | EM | F1 | Value EM | Answer value present | Gold retrieved | Stale retrieved | Distractor retrieved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gold_context` | 100 | 0.920 | 0.927 | 0.920 | 1.000 | 1.000 | 0.000 | 0.000 |
| `retrieval_topk_1` | 100 | 0.050 | 0.071 | 0.050 | 0.060 | 0.060 | 0.880 | 0.060 |
| `retrieval_topk_3` | 100 | 0.060 | 0.111 | 0.060 | 0.150 | 0.270 | 1.000 | 0.210 |
| `retrieval_topk_5` | 100 | 0.140 | 0.173 | 0.140 | 0.160 | 0.360 | 1.000 | 0.300 |
| `retrieval_topk_10` | 100 | 0.070 | 0.145 | 0.070 | 0.130 | 0.620 | 1.000 | 0.410 |

## Paired intervention effects

| Baseline | Target | Pairs | ΔEM | ΔF1 | ΔValue EM | ΔGold retrieved | ΔStale retrieved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `retrieval_topk_5` | `gold_context` | 100 | +0.780 | +0.754 | +0.780 | +0.640 | -1.000 |
| `retrieval_topk_5` | `retrieval_topk_10` | 100 | -0.070 | -0.029 | -0.070 | +0.260 | 0.000 |
| `retrieval_topk_5` | `retrieval_topk_1` | 100 | -0.090 | -0.103 | -0.090 | -0.300 | -0.120 |

## Interpretation

This is strong evidence that Raw append collapse is not only caused by failure to store the final value. The final value is recoverable under a controlled gold context, but normal retrieval is dominated by stale same-slot entries.

The key contrast is `retrieval_topk_10`: it increases gold retrieval by 0.26 relative to top-k 5, but EM decreases by 0.07 because stale entries remain present. This supports a competition/contamination mechanism rather than a simple "retrieve more memories" fix.

## Next step

Implement a true retrieval-time stale filter that keeps only the latest `(entity, attribute)` memory for Raw append, then evaluate it without oracle gold context. That will separate an oracle intervention from a realistic stale-filtering method.
