# External Baseline Feasibility Note

## Current decision

No-go for external-baseline implementation in the current P6.7 manuscript block. The current P6.3/P6.7 assets already support the main diagnostic claim: repeated same-slot updates create a tradeoff among final-state reliability, stale same-slot burden, memory compactness, and slot-conditioned answer robustness.

External baselines should be added only if the paper draft needs ecosystem grounding beyond controlled diagnostic baselines.

## Why not now

1. The main claim is already visible without an external system: raw append and heuristic CRUD preserve `slot_direct` recoverability but collapse under `slot_prompt`, while Long25 trades compactness for imperfect final-state reliability.
2. The existing `baselines/` directory contains older agent-interface code and is not aligned with the current clean MemUpdateBench manuscript path.
3. Mem0/Letta/MemGPT feasibility was previously checked at the package/environment level; they were not installed in the current cluster environment.
4. Installing external systems into the main environment would add dependency and integration risk without first proving that the paper needs those rows.

## If an external baseline becomes necessary

Use an isolated environment and start with Mem0 only.

Recommended scope:

- one small feasibility script or notebook outside the main evaluation path,
- no changes to the core benchmark semantics,
- no reuse of old G-MSRA Phase 1-5 agent pipeline,
- no package installation into the main `gmsra` environment,
- no learned repair training in the same phase.

Recommended evaluation target:

- only P6.3 hard k=16 first,
- use deterministic oracle/constrained CRUD as the sanity anchor,
- report whether Mem0 can represent exact `(entity, attribute)` slot updates or whether it degrades into append/retrieval behavior,
- measure the same four quantities where possible: final-state reliability, stale same-slot burden, memory size, and slot-prompt answer quality.

## Decision criteria

Add an external baseline only if at least one is true:

1. The paper draft reads too self-contained and needs a recognizable external-memory system for positioning.
2. The venue/advisor expectation requires comparison to an existing memory framework.
3. A small isolated feasibility run can produce a clean row without compromising the controlled benchmark framing.

Do not add it if:

1. it requires changing the benchmark semantics,
2. it requires heavy server/agent infrastructure,
3. it cannot expose memory entries well enough to compute stale same-slot burden,
4. it delays the main figure/table/narrative integration.

## Recommended next step

First stabilize `paper/manuscript_draft.md`, `paper/p63_metric_ledger.md`, `paper/p63_error_analysis_k16.md`, and `paper/remote_verification_log.md` as the primary manuscript-support package. Revisit Mem0 only after that package makes clear whether an external row is necessary.
