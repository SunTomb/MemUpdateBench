# Claim–Evidence Map and Paper Boundary

## Recommended paper argument

**Working claim:** MemUpdateBench isolates repeated updates to the same external-memory slot and shows that final-state recoverability, stale-value retention, compactness and prompted-answer robustness can move in different directions. The benchmark is a diagnostic instrument, not a universal leaderboard or a claim that every external-memory system fails under repeated updates.

**Evidence ladder:**

1. **Controlled internal result:** P6.3 hard update-frequency experiments establish the main tradeoff among state accuracy, stale burden, compactness and slot-conditioned answering.
2. **Mechanism diagnostics:** P8.0/P8.3/P8.4 show that stale-context effects depend on retrieval exposure, ordering, version metadata and answer-model behavior; the Qwen3.5 boundary prevents universalizing the earlier answer-model effect.
3. **External-validity probes:** Letta and the LangGraph Store/custom-adapter pipeline demonstrate supported-scope end-to-end operation, while Mem0 supplies a preliminary off-the-shelf external probe. These rows are coverage- and configuration-qualified, not a fair cross-system leaderboard.
4. **Native-system challenge:** The Graphiti-only development screen is INCONCLUSIVE for current-value degradation. Its historical-serving follow-up completed one full runtime retry but remained BLOCKED at conservative semantic attribution because native relation facts were not covered by the frozen evaluator grammar. It produces no additional mechanism lead.
5. **Boundary:** The current evidence does not establish broad external validity, universal Graphiti behavior, a mechanism GO, or justification for held-out validation.

## Claim–evidence table

| Claim | Evidence | Status | Required wording |
|---|---|---|---|
| Repeated same-slot updates expose a recoverability/contamination tradeoff | P6.3 hard: Raw append and Heuristic CRUD state accuracy 1.00 at k=16 but slot-prompt EM/F1 0.07/0.10 and 0.10/0.13; stale burden 14.25 and 7.44 | Supported in the tested controlled setting | “In the P6.3 hard setting…” |
| Compact learned management changes the tradeoff rather than dominating all axes | Long25 k=16 stale burden 1.13, memory size 9.43, state accuracy 0.91, slot-prompt EM/F1 0.48/0.53 | Supported in the tested setting | “Long25 occupies a different point on the tradeoff curve…” |
| Clean state is not sufficient for prompted answering | Constrained CRUD has state accuracy 1.00 and zero stale burden but k=16 slot-prompt EM/F1 0.70/0.70 | Supported as an answer-layer diagnostic | “This exposes a separate reader/answer-layer gap…” |
| Stale-context mechanisms are answer-model dependent | Earlier Qwen/Llama/Mistral controls show stale susceptibility; qualified Qwen3.5 matrix is nearly perfect, EM 0.9979167 overall | Supported as a boundary, not a universal law | “The tested answer models do not share one invariant response…” |
| External systems are operationally testable under explicit support envelopes | Letta and LangGraph Store/custom adapter each support 52/80 Family-A tasks; unsupported multi-object tasks remain typed | Supported as joint-pipeline evidence | “On the supported subset of the disclosed pipeline…” |
| Mem0 is a preliminary external probe, not a main-table baseline | Real vLLM/Qdrant/MiniLM dev20; inspectable memory, EM/F1 0.00/0.00 initially and 0.00/0.05 after extraction adjustment | Supported but preliminary | “The current off-the-shelf probe is badly aligned with exact-slot scoring…” |
| Native Graphiti has no observed current-value degradation in the bounded dev screen | Current answers 8/8 at k=1/4/16 in native and reversed conditions; formal decision INCONCLUSIVE | Supported as a scoped null observation | “We did not observe the prespecified signal…” |
| Historical-serving/cross-entity mechanism is established | First attempt startup-blocked; second retry complete but semantic attribution BLOCKED because native facts were Chinese and the frozen grammar admitted only an exact English clause | **Not established** | Do not claim a mechanism lead or Graphiti failure |
| MemUpdateBench is a broad external-memory benchmark | Core/Pilot are controlled releases; external rows have typed support boundaries; Graphiti is inconclusive | Not supported | Call it a “controlled diagnostic benchmark” |

## Evidence classes to keep separate in the manuscript

- **Engineering release:** authenticated tasks, schemas, manifests, reproducibility and cleanup. This is not scientific accuracy.
- **Capability/preflight:** model, SDK, database, source or transport loadability. This is not benchmark performance.
- **Internal benchmark accuracy:** P6.3/P8 results with the named methods and answer conditions.
- **External joint-pipeline performance:** Letta and LangGraph custom-adapter rows on supported subsets. Do not relabel them as native LangMem/Letta-only accuracy.
- **Preliminary external probe:** Mem0 dev20, whose semantic alignment is not yet fair enough for the main table.
- **Native-system development diagnosis:** Graphiti current-value INCONCLUSIVE and historical follow-up BLOCKED. Neither is a zero-accuracy result.

## Recommended section order

1. Introduction: repeated updates are the narrow unresolved stressor; state and answer quality must be separated.
2. Benchmark setup: slot contract, update-frequency axis, stale burden, compactness and answer modes.
3. Main internal results: P6.3 tradeoff curve and clean-state/reader gap.
4. Mechanism diagnostics: P8 stale exposure, version arbitration and model-dependent boundary.
5. External-validity probes: supported-scope joint pipelines and the preliminary Mem0 probe, with typed unsupported rows.
6. Native-system challenge: Graphiti INCONCLUSIVE current screen and BLOCKED historical-serving follow-up.
7. Discussion/limitations: controlled synthetic scope, semantic alignment, system configuration and answer-model dependence.

## Claims to remove or weaken

- “Repeated updates are poorly characterized” → “Repeated same-slot updates remain insufficiently isolated in broad evaluations.”
- “The benchmark demonstrates a universal tradeoff” → “The tested methods trace a tradeoff curve in the controlled P6.3 setting.”
- “Graphiti fails at historical serving” → “The historical-serving protocol was completed only as a BLOCKED attribution outcome.”
- “Letta/LangMem achieve 1.0 accuracy” → “The disclosed supported-scope joint pipelines achieve 1.0 on their evaluated supported rows.”
- “Mem0 is a baseline” → “Mem0 is a preliminary external probe unless a fair text-only configuration and larger split are frozen.”
