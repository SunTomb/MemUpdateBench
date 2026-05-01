# PROJECT_WORKFLOW6.0.1 — P6.0 Critique and Survival-Space Reassessment

## Context

After `PROJECT_WORKFLOW6.0.0.md` reframed G-MSRA as a structured diagnostic framework, `LITERATURE_REVIEW_P6_20260501.md` raised a second-order critique: even the diagnostic/benchmark space is already partially occupied.

The main threats are:

1. **AMemGym** — interactive memory benchmarking with structured user profiles, state-dependent questions, and state evolution trajectories.
2. **UMA / Ledger-QA** — continuous state tracking where answers derive from latent values accumulated through updates.
3. **MemEvoBench** — memory misevolution benchmark focused on safety drift under adversarial/noisy memory updates.
4. **Beyond pass@1** — long-horizon reliability science framework with decay curves, variance amplification, meltdown onset, and claims that memory scaffolds can hurt long-horizon performance.

This means the broad P6.0 claim — "structured diagnostic framework for memory update failures" — is also too broad. The surviving space must be narrower.

## Web Verification Summary

I cross-checked the report's claims with web/arXiv fetches:

- `arXiv:2603.01966` confirms **AMemGym** is an interactive long-horizon memory benchmark using structured data sampling, user profiles, state-dependent questions, and state-evolution paths.
- `arXiv:2602.18493` confirms **UMA** introduces **Ledger-QA**, a diagnostic benchmark for continuous state tracking where answers come from latent values derived from accumulated updates, not local span retrieval.
- `arXiv:2508.19828` confirms **Memory-R1** directly uses ADD / UPDATE / DELETE / NOOP with PPO/GRPO for a Memory Manager plus Answer Agent.
- `arXiv:2604.15774` confirms **MemEvoBench** focuses on memory misevolution, with 7 domains and 36 risk types.
- `arXiv:2603.29231` confirms the long-horizon reliability framework uses Reliability Decay Curve, Variance Amplification, Graceful Degradation, and Meltdown Onset; its abstract also reports memory scaffolds hurting long-horizon performance across models.

## Revised Survival Space

The viable contribution is no longer:

```text
A general diagnostic framework for memory agents.
```

The more defensible niche is:

```text
Repeated-update stress testing for memory agents:
How state accuracy decays as the same slot is overwritten many times.
```

Possible title:

```text
How Well Do Memory Agents Handle Repeated Knowledge Updates?
A Stress-Test Diagnostic of Stale-Value Failures
```

Core claim:

> Existing memory-agent benchmarks include long-horizon state tracking, but they do not isolate update frequency as an independent stress variable. G-MSRA studies how memory systems degrade as the same `(entity, attribute)` slot is overwritten 1, 2, 4, 8, 16, ... times, and diagnoses whether failures are stale-value retention, missed update, wrong entity, wrong attribute, or output-format artifacts.

This is narrower but more scientifically testable than P6.0.

## What Must Be True for This Direction to Survive

The direction is viable only if experiments show a clear scaling law or reliability pattern:

1. State accuracy decreases monotonically or sharply as update frequency increases.
2. Failure type shifts toward `wrong_value` / stale-value retention at high update counts.
3. The degradation appears across multiple memory strategies, not only our own model.
4. Deterministic oracle remains near 1.0, so the benchmark itself is not the bottleneck.
5. At least one real external baseline or clearly labeled project-local approximation is evaluated.

If these conditions fail, this direction should be downgraded to an internal engineering benchmark rather than a paper thesis.

## Contribution Reframing

### Unsafe broad claims

Avoid:

- "First diagnostic benchmark for memory agents."
- "First structured state-tracking memory benchmark."
- "First RL CRUD memory manager."
- "Constrained SFT is the main novelty."

### Defensible narrow claims

Potentially safe:

- "We isolate update frequency as a controlled stress variable."
- "We measure stale-value reliability decay under repeated overwrites of the same slot."
- "We show QA metrics can hide whether systems store the latest value or a stale value."
- "We provide oracle-ceiling checks to separate benchmark/parser artifacts from model failures."
- "We compare internal and project-local external baselines under the same state-level diagnostic interface."

## Required Benchmark Redesign

The current long-horizon split mixes several stressors:

- many updates;
- distractors;
- NOOP markers;
- multiple entities;
- multiple attributes;
- occasional parser artifacts.

For the new direction, we need controlled splits where update frequency is the main independent variable.

Recommended split family:

```text
eovmemory_updatefreq_k1_test.json
eovmemory_updatefreq_k2_test.json
eovmemory_updatefreq_k4_test.json
eovmemory_updatefreq_k8_test.json
eovmemory_updatefreq_k16_test.json
```

Use the corrected prefix in implementation:

```text
eovmemory -> evomemory
```

Stress dimensions:

1. `k_updates`: number of overwrites of target slot.
2. `distractor_level`: none / same-attribute / same-name / mixed.
3. `noop_level`: none / marker events / semantic near-miss events.
4. `position`: final update early / middle / late.
5. `attribute`: location / company / preference / language.

Minimum first pass:

```text
k = 1, 2, 4, 8, 16
examples per k = 100 test + 100 dev
attributes balanced
oracle must be >= 0.98 for every k
```

## Baseline Requirements

Baseline hierarchy:

1. Deterministic oracle — mandatory ceiling.
2. raw ADD — stale-value lower baseline.
3. heuristic CRUD — simple update baseline.
4. current long25 parser-fix checkpoint — strongest internal learned baseline.
5. project-local Memory-R1 approximation — useful but must be labeled honestly.
6. original Memory-R1 / Mem0 / other external methods — only if real code/checkpoints can be obtained.

The paper cannot rely on "Memory-R1" unless it is either:

- original code/checkpoint, or
- explicitly labeled as `project-local aligned Memory-R1 approximation`.

## 10-Hour Immediate Plan

### Phase 1 — Feasibility and naming correction (0.5h)

- Confirm exact file naming convention for a new `update_frequency` EvoMemory variant.
- Avoid the typo shown above (`eovmemory`).
- Decide whether to implement as:
  - `--evomemory_variant update_frequency`, or
  - a separate script for diagnostic generation.

Recommended: add `update_frequency` as a new `prepare_data.py` variant so it inherits seed/suffix conventions.

### Phase 2 — Implement controlled update-frequency generator (2h)

Modify:

```text
scripts/prepare_data.py
```

Generate balanced dev/test splits for:

```text
k_updates = [1, 2, 4, 8, 16]
```

Each example should include:

- explicit `stress_type = update_frequency`;
- `k_updates`;
- `distractor_level`;
- `attribute`;
- `entity`;
- `latest_event_idx`;
- exact `answer`.

Start with clean controlled setting:

- one target entity;
- optional same-attribute distractors;
- no ambiguous same-name entity in the first version;
- no arbitrary natural-language templates beyond parser-supported forms.

### Phase 3 — Oracle gate (1h)

Run deterministic oracle for every k split:

```bash
python scripts/eval_evomemory.py \
  --mode constrained_slot_crud \
  --answer_mode slot_direct \
  --data_file data/evomemory_update_frequency_k<k>_test.json \
  --output_dir results/update_frequency/oracle_k<k>
```

Gate:

```text
oracle_state_accuracy >= 0.98 for every k
```

If oracle fails, fix generator/parser before any learned eval.

### Phase 4 — Internal baseline decay curves (2h)

Run:

1. raw ADD / heuristic CRUD if available in `eval_evomemory.py`, otherwise add only minimal support.
2. long25 parser-fix learned constrained manager.

Metrics per k:

```text
state_accuracy
state_resolve_rate
avg_em
avg_f1
wrong_value rate
stale_value_present_same_slot
```

Main plot/table:

```text
Update Frequency Reliability Curve
x-axis: k updates
 y-axis: state_accuracy
```

### Phase 5 — Project-local Memory-R1 diagnostic adapter (2h)

Do not train full Memory-R1 yet.

First implement/evaluate a diagnostic adapter that runs the existing project-local `memory_r1` event processor and converts final memory contents to slot-state predictions.

Important: label this as:

```text
project-local Memory-R1 approximation
```

not original Memory-R1.

If adapter is too slow, run only k=1/4/16 first.

### Phase 6 — Root-cause analysis (1h)

For each method and k, compute:

- `wrong_value` count;
- stale value count;
- whether gold appears anywhere in memory;
- whether stale same-slot value appears;
- failure overlap across methods.

This upgrades the taxonomy from shallow error labels to a more defensible diagnostic analysis.

### Phase 7 — Write workflow and decision memo (1.5h)

Create:

```text
PROJECT_WORKFLOW6.1.0.md
```

Contents:

1. why P6.0 is too broad after AMemGym/Ledger-QA;
2. why repeated-update stress is the narrower survival space;
3. update-frequency benchmark design;
4. oracle results;
5. first reliability curves;
6. baseline feasibility;
7. decision: continue, pivot again, or stop paper ambitions.

## Stop Conditions

Stop and reassess if any of these happens:

1. Oracle cannot reach >= 0.98 on controlled update-frequency data.
2. All methods remain near ceiling for k=16, meaning no stress signal.
3. Only our own model shows the effect but external/project-local baselines do not.
4. Memory-R1 or comparable external baseline cannot be credibly included.
5. The result is merely parser cleanup rather than a memory update phenomenon.

## Current Recommendation

Proceed only with the repeated-update stress-test niche, not generic diagnostic framework.

The next implementation should be the controlled `update_frequency` split family and oracle gate. Do **not** start new training until update-frequency reliability curves show a real and reproducible degradation pattern.
