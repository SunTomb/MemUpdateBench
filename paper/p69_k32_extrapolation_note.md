# P6.9 k=32 Extrapolation Note

## Motivation

k=32 was previously deferred because k=16 already exposed the main mechanism. After the expanded split and all-k dev slot-prompt sweep confirmed the P6.3 story at larger scale, k=32 was added as an opt-in extrapolation point rather than a new default benchmark axis.

## Data

New opt-in variant:

```text
update_frequency_expanded_k32
```

Files:

```text
data/evomemory_update_frequency_expanded_k32_p69k32_dev.json
data/evomemory_update_frequency_expanded_k32_p69k32_test.json
```

The k=32 dev/test files each contain 200 examples, balanced across the eight expanded attributes:

```text
location, company, preference, language, timezone, hobby, instrument, project
```

The variant writes with suffix `p69k32` and does not overwrite the P6.8 `p68` expanded split.

## Slot-direct sanity

| Method | k | N | EM | F1 | State acc. | Stale same-slot | Same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud | 32 | 200 | 1.000 | 1.000 | 1.000 | 0.00 | 1.00 | 42.00 |
| raw_add | 32 | 200 | 1.000 | 1.000 | 1.000 | 28.50 | 32.00 | 103.00 |

Deterministic oracle sanity passes at k=32. Raw append still keeps the final state recoverable under exact slot lookup, but stale same-slot burden grows to 28.50 and final memory size grows to 103.00.

## Slot-prompt endpoint

Execution:

```text
Tang-2-Wu, GPUs 3/4
results/p69_k32_slot_prompt/
results/p69_k32_slot_prompt_summary/
```

| Method | k | N | EM | F1 | State acc. | Gold retrieved | Stale retrieved | Stale same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud | 32 | 200 | 0.655 | 0.655 | 1.000 | 1.000 | 0.000 | 0.00 | 42.00 |
| raw_add | 32 | 200 | 0.155 | 0.172 | 1.000 | 1.000 | 1.000 | 28.50 | 103.00 |

## Interpretation

k=32 is useful as extrapolation but does not change the core story. Raw append's prompted-answer collapse has already saturated by k=8/k=16 on the expanded split: k=32 keeps state accuracy at 1.00 but retains 28.50 stale same-slot entries and remains near collapse at 0.155 EM. This supports the claim that exact final-state recoverability is insufficient once stale same-slot competition dominates the answer context.

Constrained CRUD remains clean-state but not perfect-answer: state accuracy is 1.00, stale retrieved is 0.00, and gold retrieval is 1.00, yet EM is 0.655. This is close to the k=16 expanded endpoint (0.675), reinforcing that the clean-state answer-layer gap is stable under higher update frequency and is not simply caused by stale retention.

## Paper-facing takeaway

k=32 can be reported as an appendix extrapolation/saturation point. It should not replace k=16 as the main figure endpoint: k=16 already demonstrates the tradeoff clearly, while k=32 confirms saturation and increases compute cost.
