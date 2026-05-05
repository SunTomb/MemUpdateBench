# P8.0 Context Mechanism Probe Result Note

## Purpose

`docs/critical_review_v3.md` asked why stale same-slot context causes answer collapse. This note records the first controlled context-order and context-annotation probe on raw_add k=16 dev.

## Artifacts

```text
scripts/eval_evomemory.py
scripts/run_p80_mechanism_probe_batch_sui3.sh
scripts/summarize_context_mechanisms.py
results/p80_mechanism_probes/
results/p80_mechanism_probe_summary/context_mechanism_summary.{json,csv,md}
paper/p80_mechanism_probe_setup_note.md
```

Remote execution:

```text
node: Sui-3-Wu
project path: /NAS/yesh/MemUpdateBench
data: data/evomemory_update_frequency_hard_k16_p63_dev.json
mode: raw_add
answer_mode: slot_prompt
answer_topk: 5
save_answer_traces: true
no_qlora: true
```

## Results

| Condition | EM | F1 | Answer value present | Gold retrieved | Stale retrieved | Stale retrieved avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal order, no annotation | 0.110 | 0.136 | 0.140 | 0.360 | 1.000 | 4.040 |
| chronological, no annotation | 0.230 | 0.275 | 0.320 | 0.360 | 1.000 | 4.040 |
| reverse chronological, no annotation | 0.010 | 0.050 | 0.010 | 0.360 | 1.000 | 4.040 |
| current first, no annotation | 0.020 | 0.060 | 0.020 | 0.360 | 1.000 | 4.040 |
| current last, no annotation | 0.040 | 0.080 | 0.040 | 0.360 | 1.000 | 4.040 |
| normal order + timestamp | 0.150 | 0.200 | 0.230 | 0.360 | 1.000 | 4.040 |
| normal order + latest/outdated label | 0.260 | 0.298 | 0.350 | 0.360 | 1.000 | 4.040 |

All formal conditions have the same gold retrieval rate (0.360), stale retrieved rate (1.000), and retrieved stale count average (4.040). Therefore the EM differences come from answer-layer behavior under different context presentation, not from changing retrieval composition.

## Interpretation

### 1. Explicit version labels help most

Adding `[latest]` / `[outdated]` labels improves EM from 0.110 to 0.260 and answer-value-present from 0.140 to 0.350.

This supports the semantic/version-disambiguation hypothesis: when stale entries remain in context, the model benefits from explicit cues about which same-slot entry is current.

### 2. Timestamps help, but less than semantic labels

Timestamp annotation improves EM from 0.110 to 0.150. This suggests that raw temporal metadata is somewhat useful, but less directly usable than explicit latest/outdated labels.

### 3. Order effects are real but not a simple current-first story

Chronological ordering improves EM to 0.230, while reverse chronological falls to 0.010. This indicates a strong order sensitivity.

However, the synthetic `current_first` and `current_last` reorderings are both worse than normal order. A likely reason is that these reorderings group all same-slot entries together and disrupt the broader retrieval ranking/context structure. Therefore the current result supports positional/context-order sensitivity, but does not support the simple claim that putting the latest value first is sufficient.

### 4. Retrieval composition remains the dominant bottleneck

Gold retrieval is only 0.360 in all conditions. Even the best annotation condition reaches EM 0.260, still far below gold-context EM 0.92 and latest_per_slot EM 0.69. This means labels help the answer model use a contaminated context, but they do not solve the missing-gold/latest-recall problem.

## Paper-level claim supported

A careful claim now supported by the probe is:

> Stale same-slot collapse is not only a retrieval problem. Holding retrieved entries fixed, answer accuracy changes substantially with context presentation. Explicit version labels and chronological order improve performance, indicating that part of the failure is semantic/version disambiguation under stale context, although retrieval of the latest slot value remains the larger bottleneck.

## Caveats

- This is a first mechanism probe on P6.3 hard k=16 dev only.
- `current_first` / `current_last` are coarse reorderings and may disrupt retrieval ranking in unnatural ways.
- The probe does not yet isolate majority voting or attention dilution; those require synthetic same-slot contexts with controlled stale counts and conflict/no-conflict values.
- The gold retrieval rate is low, so this probe primarily tests behavior under contaminated/weak retrieval, not gold-present-only answer behavior.

## Next steps

1. Add controlled synthetic same-slot probes where gold is always present and stale count is varied.
2. Add conflict/no-conflict controls to distinguish value conflict from attention dilution.
3. Repeat the most informative conditions on the expanded split after the k=8 latest_per_slot anomaly is resolved.
