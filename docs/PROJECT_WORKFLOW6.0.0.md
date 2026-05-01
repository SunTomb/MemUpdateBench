# PROJECT_WORKFLOW6.0.0 — Literature-Driven Research Narrative Pivot

## Context

`LITERATURE_REVIEW_20260501.md` shows that the broad idea of training LLM agents to manage external memory with CRUD-style actions is now heavily occupied by recent 2025–2026 work:

- Memory-R1: RL/PPO/GRPO for ADD/UPDATE/DELETE/NOOP memory management.
- UMA: end-to-end memory-agent training for CRUD + QA, with Ledger-QA for continuous state tracking.
- AgeMem: progressive RL for unified LTM/STM management across long-horizon benchmarks.
- Inside Out: structured persona memory trees with RL-trained memory listeners.
- MemFactory and related frameworks: unified training/inference infrastructure for memory agents.

This means the old framing — "G-MSRA proposes a new RL/SFT memory CRUD manager" — is no longer safe as a main research contribution. The Phase 4.0 move toward constrained slot-action SFT was technically useful, but as a paper thesis it risks looking like an engineering simplification of already-published RL memory-manager work.

## What Phase 4 Actually Contributed

P4.0–P4.3.1 should be reinterpreted as evidence for a diagnostic framework, not as the main method contribution.

Important assets now available:

```text
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
scripts/prepare_data.py
gmsra/manager/memory_manager.py
```

These provide:

1. Exact structured state tracking over `(entity, attribute, value)`.
2. Executable memory action traces.
3. Deterministic oracle ceilings.
4. Fine-grained error categories:
   - `wrong_value`
   - `missing_state`
   - `wrong_entity`
   - `wrong_attribute`
5. Stress splits covering schema randomization, multi-entity distractors, OOD templates, hard stale cases, and long-horizon frequent updates.

The strongest candidate checkpoint remains:

```text
outputs/constrained_sft_curriculum_long25/best
```

After the parser artifact fix in `PROJECT_WORKFLOW5.3.1.md`, its long-horizon behavior is:

| Split | state_accuracy | Notes |
|---|---:|---|
| long_horizon_dev | 0.9833 | parser-fix result |
| long_horizon_test | 0.9500 | parser-fix result |
| long_horizon_seed101_test | 0.9833 | matches deterministic oracle |
| long_horizon_seed202_test | 0.9667 | matches deterministic oracle |

This result is strong, but it should support a diagnostic framework story rather than a claim that constrained SFT is a novel memory-agent method.

## Unsafe Claims to Avoid

Avoid these claims in paper/project framing:

1. "We are the first to train LLMs for memory CRUD."
2. "Constrained SFT is the main methodological innovation."
3. "RL memory CRUD is an open niche untouched by prior work."
4. "Long-horizon language failures prove model weakness" without checking parser/gold/oracle ceilings.
5. "EvoMemory alone is a sufficient community benchmark" without external baseline comparisons.

## Recommended New Thesis

Recommended title-style framing:

```text
Diagnosing Memory Update Failures in Long-Horizon Dialogue Agents:
A Structured State-Tracking Framework
```

Thesis:

> Existing memory-agent work often reports downstream QA or task metrics, but those metrics hide *where* memory management fails. G-MSRA provides a structured state-tracking diagnostic framework that exposes memory update failures at the slot/action level, enabling controlled long-horizon stress tests and failure-mode analysis across memory managers.

Core contribution should be:

1. A diagnostic evaluation framework, not just a trained manager.
2. A failure taxonomy for evolving memory state.
3. Long-horizon stress tests that isolate frequent update, stale value, entity grounding, and output-format failures.
4. A comparison methodology for existing memory managers under exact state tracking.
5. A small candidate repair only if diagnosis reveals a real gap beyond oracle ceilings.

## Evidence Table From P4

| Stage | Key result | Interpretation |
|---|---|---|
| P4.0 long-horizon stress | curriculum baseline: dev 0.9333, test 0.8500 | long-horizon frequent updates expose failures hidden by short splits |
| P4.1 full long curriculum | test improved to 0.9500 but advanced collapsed to 0.7037 | broad long replay causes negative transfer |
| P4.2 long25 replay | long test 0.9500, advanced 1.0000, schema 0.9875 | low-ratio replay preserves short splits while improving long test |
| P4.3 seed stress | initial seed202 0.8833 | apparent language weakness required deeper diagnosis |
| P4.3.1 parser fix | seed101/202 match oracle ceilings | diagnostic framework prevented false model-failure conclusion |

The most scientifically valuable lesson is the last one: fine-grained trace/state analysis found that many apparent model failures were actually parser artifacts.

## Baseline Plan

Priority baselines for the new framing:

| Baseline | Priority | Role |
|---|---:|---|
| raw ADD | mandatory | shows stale-value failure under no update policy |
| heuristic CRUD | mandatory | simple non-learned update baseline |
| deterministic constrained oracle | mandatory | establishes parser/data ceiling |
| constrained SFT curriculum | mandatory | internal learned baseline before long replay |
| long25 parser-fix | mandatory | strongest internal candidate |
| Memory-R1 | highest external priority | direct competitor; should be run or feasibility-blocked explicitly |
| Mem0 / MemGPT / A-MEM-style methods | medium | external memory baselines if easier than Memory-R1 |
| UMA / AgeMem | literature comparison minimum | include as related work even if reproduction is infeasible |

## Immediate 10-Hour Execution Plan

### Phase 1: Baseline infrastructure audit

Inspect:

```text
gmsra/baselines/
baselines/eval_baselines.py
```

Questions:

1. Which baselines are already registered?
2. Is there any Memory-R1 / Mem0 / MemGPT adapter stub?
3. What input/output interface would a new external baseline adapter need?
4. Can current `evomemory_results.json` and `analyze_ood_errors.py` be reused for external baselines?

### Phase 2: Memory-R1 feasibility check

Goal: determine whether Memory-R1 can be run on EvoMemory long-horizon splits in this repo/cluster environment.

Need to answer:

1. Is Memory-R1 code/checkpoint already present locally?
2. If not, is it available through an accessible repository or package?
3. What data format does it require?
4. Can EvoMemory episodes be exported without changing the core benchmark?
5. What minimal adapter would produce comparable memory state traces?

If direct execution is not feasible quickly, document the blocker and implement only an export/adapter plan.

### Phase 3: Diagnostic benchmark card

Organize existing EvoMemory variants as a benchmark suite:

| Split | Stress factor | Expected failure mode |
|---|---|---|
| clean | basic user facts | sanity check |
| advanced | multi-entity distractors | wrong_entity / missing_state |
| hard | stale-value distractors | wrong_value |
| OOD | template shift | parser/prompt robustness |
| schema_random | dynamic entity schema | entity normalization |
| long_horizon | frequent repeated updates | stale value / late update / wrong NOOP |

For each split, require:

1. deterministic oracle ceiling;
2. learned manager result;
3. `by_error` distribution;
4. state metrics plus answer metrics.

### Phase 4: First diagnostic comparison table

Build a unified table from existing cached results where possible:

```text
raw_add
heuristic_crud
constrained_slot_crud oracle
constrained_sft_curriculum
constrained_sft_curriculum_long25 parser-fix
external baseline if feasible
```

Target splits:

```text
advanced_test
hard_test
OOD_test
schema_random_test
long_horizon_test
long_horizon_seed101_test
long_horizon_seed202_test
```

### Phase 5: Decide whether new training is justified

Do not run more SFT/RL unless the comparison reveals a real gap:

- below oracle ceiling;
- stable across seeds;
- attributable to an interpretable error class;
- not fixable by parser/output cleanup.

If a method repair is needed, prefer targeted repair such as:

1. stale-value penalty;
2. update-vs-NOOP contrast;
3. constrained decoding/output cleanup;
4. value-focused replay.

## Risks and Mitigations

### Risk 1: External baselines are hard to reproduce

Mitigation:

- Start with feasibility and adapter export.
- Clearly report unavailable code/checkpoints as a limitation.
- Use internal baselines to demonstrate diagnostic framework value while preparing external integration.

### Risk 2: EvoMemory remains too synthetic

Mitigation:

- Position it as a diagnostic stress suite, not a full real-world benchmark replacement.
- Add controlled stress-family descriptions.
- Report oracle ceilings and confidence/seed variance.

### Risk 3: The strongest internal model is near oracle ceiling, leaving little room for method improvement

Mitigation:

- Treat this as evidence that the framework can separate system bugs from model weaknesses.
- Use external baselines to show diagnostic utility.
- Only propose targeted repair if failures remain beyond oracle.

### Risk 4: Paper lacks method novelty

Mitigation:

- Emphasize measurement, diagnosis, and failure taxonomy.
- If needed, add a small targeted repair as secondary contribution, not the central claim.

## Baseline Infrastructure Audit

Existing baseline infrastructure is split across two paths:

```text
gmsra/baselines.py
scripts/run_baselines.py
baselines/eval_baselines.py
baselines/train_and_eval_rl_baselines.py
baselines/*_agent.py
```

Findings:

1. `gmsra/baselines.py` registers five project-local aligned reproductions:
   - `memory_r1`
   - `mem0_memory_r1`
   - `reflexion`
   - `evolver`
   - `self_consolidation`
2. `baselines/eval_baselines.py` has a separate dynamic `AGENT_REGISTRY` for similarly named baseline agents.
3. `baselines/memory_r1_agent.py` is a **project-local approximation**, not the original authors' code. It uses the repo's `MemoryStore` and `MemoryManager`, with QA-F1 reward style logic.
4. `baselines/train_and_eval_rl_baselines.py` can train Memory-R1-like and Mem0+Memory-R1-like baselines, but its training loop caps events heavily and evaluates mainly QA F1/EM.
5. Neither baseline harness currently emits EvoMemory-style diagnostic fields:
   - `gold_state`
   - `predicted_state`
   - `state_accuracy`
   - `state_resolve_rate`
   - `slot_actions`
   - `by_error` categories consumed by `scripts/analyze_ood_errors.py`

Conclusion: the repo has useful baseline scaffolding, but it is not yet suitable for the new paper thesis without a diagnostic adapter. The next technical step should not be to train these baselines as-is; it should be to add or design an EvoMemory diagnostic adapter that converts baseline memory contents or operations into comparable slot-state predictions.

Recommended minimal adapter design:

1. Reuse baseline `process_event(...)`, `answer_question(...)`, and `get_memory_contents()`.
2. For each EvoMemory episode, run the baseline over events.
3. Convert final baseline memory contents to a slot state using the existing `parse_event_slot(...)` / `EpisodeEntityResolver` logic where possible.
4. Emit an `evomemory_results.json`-compatible structure so `scripts/analyze_ood_errors.py` can run unchanged.
5. Keep the result clearly labeled as `project_local_aligned`, not original Memory-R1 reproduction.

## Diagnostic Benchmark Organization

Existing EvoMemory variants can be reframed as a diagnostic suite rather than a single synthetic benchmark:

| Variant | Current files | Current scale | Primary stress factor | Primary expected failures |
|---|---|---:|---|---|
| clean | `evomemory_{train,dev,test}.json` | 125/28/27 | basic user fact evolution | sanity / answer extraction |
| advanced | `evomemory_advanced_*` | 125/28/27 | multi-entity state tracking | `wrong_entity`, `missing_state` |
| hard | `evomemory_hard_*` | 160-ish / 40-ish / 36 | stale and distractor memory | `wrong_value` |
| OOD | `evomemory_ood_*` | train/dev/test around 40 | template shift | parser/prompt brittleness |
| schema_random | `evomemory_schema_random_*` | 240/60/80 | dynamic relation-name schema | entity normalization failures |
| long_horizon | `evomemory_long_horizon_*` | 240/60/60 | repeated slot updates + distractors | stale value, late update, wrong NOOP |

The current `prepare_long_horizon_evomemory(...)` generator in `scripts/prepare_data.py` combines several stressors in one split:

- 6 updates to a target slot;
- 4 distractor entities;
- interleaved NOOP-like marker events;
- same family of attributes: `location`, `company`, `preference`, `language`;
- random seeds and suffixes are already supported.

For the diagnostic-paper framing, this should be presented as **Long-Horizon Mixed Stress**. Future expansion should parameterize this into separate stress families rather than only increasing size:

1. `long_late_update`: final answer depends on the last 1–2 events.
2. `long_same_slot_repeated`: many repeated updates to the same `(entity, attribute)`.
3. `long_same_name_multi_entity`: same bare name appears under different relations.
4. `long_same_attribute_distractor`: distractors share target attribute but different entity.
5. `long_noop_contrast`: semantically similar events where only some require UPDATE.
6. `long_format_contamination`: explicitly tests multi-action / trailing-op output artifacts.

Each diagnostic split should report:

```text
oracle_state_accuracy
learned_state_accuracy
state_resolve_rate
avg_em
avg_f1
by_error
by_attribute
```

No learned comparison should be interpreted without first reporting oracle ceiling.

## Next Immediate Action

The next implementation step should be a minimal EvoMemory diagnostic adapter for project-local baselines. It should emit `evomemory_results.json`-compatible outputs for `raw_add`, `heuristic_crud`, and project-local `memory_r1` before attempting original Memory-R1 reproduction or new training.
