# Version-transfer development findings

Status: audited **single-system development diagnostic**, not confirmatory validation or a final benchmark release.

## Decision

**Protocol status: INCONCLUSIVE. Investment recommendation: pivot or stop the current-value degradation claim before spending validation budget.**

The first qualified native-system development sample did not exhibit the pre-specified primary signals: current-answer deterioration from one to sixteen target updates, or a current-answer order sensitivity that increases with update depth. This is not evidence that repeated updates are universally solved. It is a reason not to automatically run the held-out validation set to support the current claim. The validation set remains unconsumed and no fresh-ingestion replication was performed.

Historical-query failures are an exploratory lead, not a replacement success criterion. They require separate state/retrieval diagnosis and a separately specified hypothesis before further scientific experimentation.

## What was evaluated

- Eight declared synthetic development trajectories: four user-state predicates and four project-configuration predicates.
- Three paired update counts per trajectory: 1, 4 and 16 target replacements; 17 public events in every condition, with fixed final target time and matched raw-event token lengths.
- Value labels are deterministically permuted independently of chronological numbering, removing the initial fixed-largest-value shortcut. This does not establish population independence or ecological validity.
- Real Graphiti `5c27edfa8d3e0f49b2571bcebb4261bcc89ee037` / package 0.31.2, Neo4j 5.26.0, Qwen2.5-7B-Instruct BF16, MiniLM embeddings, explicit native BM25/cosine/RRF retrieval and public-query-derived validity filters.
- Schema-constrained decoding uses lm-format-enforcer 0.11.3. All source/model/configuration boundaries are recorded in the run admission. This is not Graphiti's default provider/reranker configuration.
- Current and historical-as-of queries, each answered from native-order evidence, the same evidence in reverse order, and separately labeled privileged oracle evidence.
- Mem0 was blocked during qualification and is **not** a comparison row. Its failure is not accuracy zero.

The native query interval is `valid_at <= public_query_time` and `(invalid_at IS NULL OR invalid_at > public_query_time)`. Correctly retained history is not scored as an error. The reader receives time metadata, not evaluator-only latest/outdated labels.

## Results

Every cell contains eight queries, paired by semantic trajectory. These are not 144 independent samples.

| Query / evidence condition | 1 update | 4 updates | 16 updates |
|---|---:|---:|---:|
| Current, native order | 8/8 | 8/8 | 8/8 |
| Current, reversed order | 8/8 | 8/8 | 8/8 |
| Current, oracle evidence | 8/8 | 8/8 | 8/8 |
| Historical, native order | 2/8 | 3/8 | 0/8 |
| Historical, reversed order | 2/8 | 3/8 | 0/8 |
| Historical, oracle evidence | 8/8 | 6/8 | 8/8 |

All 144 readouts were format-valid. Current correctness was 24/24 in both native and reversed conditions. Every core's primary current-depth contrast and current order-by-depth contrast was zero. There were zero candidate deteriorating cores, below the pre-specified screening requirement; no scientific GO is issued.

### Manipulation checks matter

Reversal changed 20/24 current-query prompts (7/8, 6/8 and 7/8 by depth). Thus current robustness is not entirely explained by an inert reversal. Four current prompts nevertheless had only one entry and were unchanged.

All historical contexts had at most one entry: nine were empty and fifteen had one. **Historical reversal changed no prompt**, so the identical historical scores do not measure historical order robustness. At depth 16, seven of eight historical contexts were empty. Of 19 incorrect native historical answers, 17 were abstentions and two were answered incorrectly.

Oracle historical correctness was 22/24, not perfect. It changes evidence membership, content and length; its advantage over native retrieval is not a pure causal estimate of retrieval loss and must not be presented as a guaranteed answer upper bound.

## What can and cannot be inferred

Supported within this setting:

- The tested native time-aware pipeline maintained current-answer correctness across the sampled update counts.
- Preserving temporal metadata and reversing the native current evidence did not create an observed current-answer error in this sample.
- Current-query success and historical-query success diverged substantially at the end-to-end level.
- Historical misses were mostly abstentions, often with no retrieved evidence, rather than observed copying of a stale value under competing evidence.

Not established:

- Perfect underlying graph state, correct every-step updates, historical fact deletion, or a general temporal-memory design trade-off. Canonical state/object alignment was explicitly unavailable and state accuracy stayed null.
- A complete causal explanation of the historical gap. Saved mutation outputs sometimes contain time-compatible historical facts, and one low-depth case shows an earlier target fact invalidated after a non-target update; these are diagnostic clues, not complete final-state snapshots or a proven general failure mechanism.
- A universal Graphiti result, an external-system ranking, a multi-model effect, or a statistically confirmed absence of degradation.
- A novel mechanism already distinguished from LongMemEval, STALE, MINTEval or LAPSE.

The next useful question, if a follow-up is selected, is whether historical serving fails because of entity/version interpretation, persisted-state behavior, native retrieval/candidate thresholds, or reader interpretation. First rule out an interface mistake or a version-specific SDK defect; neither alone establishes a broad paper contribution. Do not automatically enlarge the dataset or reuse the held-out validation set as a search space for failures.

## Verification and resource closure

A separate post-run audit checked all 972 terminal-indexed artifacts, 408 native episode files with unique IDs/public-time/namespace bindings, 48 retrieval files, 144 raw response/request/prediction triplets, actual call-ID membership, and both canonical result sets. It rebuilt prompts and typed predictions and recomputed all canonical v3 scores. Every planned task was a development task, and all 24 cases had complete 17-event/2-query/6-readout coverage and namespace cleanup.

- LLM attempts: 1,747, all settled; 1,603 native Graphiti, 96 native/reversed readouts, 48 oracle readouts.
- LLM tokens: 2,457,599 input and 118,502 output.
- Embeddings: 1,994 calls, 2,888 items, 30,727 input tokens.
- Elapsed time: 8,301.18 seconds (about 2.31 hours), below the six-hour job ceiling.
- All known prerequisite plus dev LLM usage: 1,831 attempts, 2,586,044 input tokens, 124,788 output tokens.
- Owned processes/ports were checked absent after termination; no paid API was used.
- Native qualification attempts remain 6/9; the three remaining slots were not spent on retries.

Internal integrity checks and a separate audit pass are not an independent human review or a cryptographic signature. Private raw provider/prompt material remains on the authorized cluster; only compact decision metadata is tracked here.

## Offline mechanism follow-up

A read-only diagnosis over the same 408 saved mutation outputs and 48 saved historical retrievals was run after the dev decision. It does not open the graph, rerun retrieval, call a model, or reconstruct a final graph state.

The diagnosis found **7 observed target-edge invalidation transitions on non-target events**: four at u=1, three at u=4, and zero at u=16. All seven transitions had an exact observed endpoint-name witness and the same fact text before/after invalidation. A non-target event means the task-level target object was not scheduled for update at that sequence position; it does not prove why the Graphiti resolver made that transition. The pattern is consistent with a possible cross-entity/relationship conflict-resolution issue, but it is not yet a causal or universal Graphiti result. Because the number of non-target events changes as target-update density changes, this observation is also not an isolated estimate of update depth.

Historical retrieval diagnosis found time-compatible target mentions in 5/8, 5/8 and 1/8 queries at u=1/4/16, with exact lexical gold-candidate mentions in 4/8, 4/8 and 0/8. At u=16, the last time-compatible candidate seen in mutation observations was absent from all eight historical retrieval result sets, but these are observations from mutation-returned edges, not final-store snapshots; they do not prove deletion. Nine historical retrievals were empty and 17/24 historical answers abstained. This supports a follow-up hypothesis about historical serving and evidence exposure, not a replacement primary claim about current-value degradation.

The compact redacted artifact is `results/vnext/version_transfer_mechanism_diagnostic_20261030_v1/diagnostic.json`, SHA-256 `84d0300a5b0f9e96973756e62964cd1fd5cbfc5a8740e1212c00cfb1718a6d0e`; its index is `version-transfer-mechanism-index.v1`. It contains IDs, times, hashes and counts, not raw prompts/provider payloads or private paths. The prior INCONCLUSIVE development decision remains unchanged.

- Execution revision: `1218c766f034f8a2e0268c90f6ea008442ee5c2e`.
- Task packet index: `d84fcdc7eb7d09ff393b8e279d9c987252c033073b19ab950feb8b9d94490341`.
- Terminal receipt: `84e75a7352e48ecff9cadd8f01fb81da303ecc89253b15f88e1084e95c34b1c7`.
- Independent audit: `25774fde2cd8f4440e19ea3f3893b6ae06736a37d5071f1f0da2d5fd0af2a879`.
- Compact decision: `results/vnext/version_transfer_development_decision_20261028_v1/decision.json`, SHA-256 `514036ee90b795e5d89ce3da39d95376acff972458560314c81969773be0534a`.
- Decision index: `22e81e99941e4075cc3dc13198f64ecbedb00d358249b855575cff6477592ae5`.
