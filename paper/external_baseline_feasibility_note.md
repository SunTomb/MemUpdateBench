# External Baseline Feasibility Note

## Current decision

Bounded external-system evidence now exists, but no fair native-SDK leaderboard row is claimed. The LangMem-labeled result is a LangGraph `InMemoryStore` plus custom MemUpdateBench adapter pipeline with the LangMem package used as an identity gate only. Letta has an authenticated native block runtime qualification and a 52/80 supported-scope prompted-answer matrix, both explicitly reported as a joint Qwen-extraction/controller/Letta pipeline. Mem0 has capability/admission evidence only, not native benchmark accuracy.

These rows supplement rather than replace the controlled diagnostic baselines. Broader external validity still requires comparable native integrations, wider data and domains, and a representative answer-model panel.

## Why not now

1. The main claim remains visible in the controlled baselines: raw append and heuristic CRUD preserve `slot_direct` recoverability but collapse under `slot_prompt`, while Long25 trades compactness for imperfect final-state reliability.
2. The existing `baselines/` directory contains older agent-interface code and is not aligned with the current clean MemUpdateBench manuscript path.
3. The completed external rows have deliberately bounded identities: LangGraph Store/custom adapter with a LangMem package identity gate; Letta native block state/retrieval combined with source-bound Qwen extraction, controller reconciliation, and Qwen answering; and Mem0 capability admission without benchmark accuracy.
4. A fair native-SDK comparison would still add dependency, integration, and answer-model comparability requirements; those gaps should not be hidden by relabeling the bounded rows.

## If further external-baseline work becomes necessary

Use an isolated environment and distinguish native framework execution from custom adapter pipelines.

Recommended scope:

- one small feasibility script or notebook outside the main evaluation path,
- no changes to the core benchmark semantics,
- no reuse of old G-MSRA Phase 1-5 agent pipeline,
- no package installation into the main `gmsra` environment,
- no relabeling of the existing LangGraph-store/custom-adapter or Letta joint-pipeline rows as native framework-only accuracy,
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

Use the completed LangGraph-store/custom-adapter and Letta joint-pipeline artifacts as bounded manuscript-supporting evidence, and report Mem0 only at its admitted capability scope rather than as native benchmark accuracy. Further work should target comparable native integrations and broader data/model coverage only if the venue requires a true external-system leaderboard.
