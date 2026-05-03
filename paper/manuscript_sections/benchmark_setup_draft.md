# Benchmark Setup Draft

## Task format

MemUpdateBench represents memory updates with three actions:

```text
ADD <entity>.<attribute> = <value>
UPDATE <entity>.<attribute> = <value>
NOOP
```

The central invariant is exact slot resolution by `(entity, attribute)`. Each example defines a target slot, a final gold value, distractor entities or attributes, and events that should either update the slot, add other facts, or be ignored as NOOPs. This structure lets the benchmark distinguish state-management failures from answer-generation failures.

## Update-frequency stressor

The main P6.3 hard split varies update frequency with `k in {1, 2, 4, 8, 16}`. Larger `k` means more repeated updates to the same target slot. This controlled axis directly stresses whether memory systems overwrite stale values, append new evidence without compacting old evidence, or lose track of the final value.

The hard split also includes same-name multi-entity distractors and semantic near-miss NOOP events. These are included to test whether a manager can resolve the correct entity and attribute rather than simply react to surface lexical overlap.

## Evaluation modes

We report two complementary answer modes.

- `slot_direct` is oracle-like slot-state evaluation. It checks whether the final value can be resolved for the target `(entity, attribute)` slot. This mode is useful as a state-management diagnostic, but it is not a realistic end-user answering setup.
- `slot_prompt` is slot-conditioned answering. It asks the model to answer under a slot-conditioned prompt using the memory contents. This is the more realistic robustness test because stale same-slot entries remain visible to the answer layer.

The gap between these modes is central to the paper. If a method scores well under `slot_direct` but poorly under `slot_prompt`, the final value may be recoverable in principle while stale evidence still causes answer collapse.

## Metrics

We report four metrics together:

1. state accuracy under `slot_direct`, measuring final-state reliability;
2. EM/F1 under `slot_prompt`, measuring answer robustness;
3. stale same-slot entry count, measuring obsolete same-slot burden;
4. final memory size, measuring compactness.

No single metric is sufficient. State accuracy alone can hide stale evidence; memory size alone can reward overly aggressive deletion; answer EM alone can conflate state errors and answer-layer failures.

## Compared systems

The main P6.3 comparison includes four systems:

- Constrained CRUD: an oracle-like diagnostic upper bound with exact slot updates and no stale same-slot values by construction.
- Raw append: an append-only baseline that keeps all observed memory evidence.
- Heuristic CRUD: a partial-compaction baseline using rule-based update behavior.
- Long25: a learned compact manager checkpoint trained under the previous constrained-slot curriculum.

Constrained CRUD should not be read as a deployable external-memory method. Its role is to anchor what perfect slot resolution and stale-value deletion can achieve.
