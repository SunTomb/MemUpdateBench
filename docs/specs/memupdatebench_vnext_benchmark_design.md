# MemUpdateBench vNext Benchmark Design

> Status: Draft for design review
> Date: 2026-07-20
> Scope: Contract-first, phased benchmark suite
> Compatibility target: Existing P6.x/P8.x MemUpdateBench assets remain usable through a legacy compiler
> Implementation status: Design only; this document does not claim that vNext components or new datasets have been implemented

## 1. Executive Summary

MemUpdateBench vNext is a phased benchmark suite for evaluating how external memory systems interpret, maintain, retrieve, and use evolving information over time.

The existing repeated same-slot update study remains the benchmark's anchor task family and canonical stress test. It has already exposed a valuable separation between final-state recoverability, stale same-slot burden, memory compactness, retrieval exposure, version arbitration, and prompted answer correctness. vNext turns that diagnostic line into a broader and implementable benchmark engineering project without discarding the current P6/P8 evidence.

The benchmark covers the full dynamic-memory path:

```text
Source event
-> Event interpretation
-> Memory action
-> Stored state
-> Retrieval
-> Version arbitration
-> Final answer
```

vNext is contract-first. It defines canonical schemas for tasks, events, actions, memory snapshots, retrieval traces, answers, scores, adapters, and run manifests before expanding the data. Existing P6/P8 data and results are preserved as `legacy_p63`/legacy-analysis assets and enter the new system through compatibility compilers. New benchmark families are introduced through Pilot, Core, and Full gates rather than through an immediate rewrite or one-shot data expansion.

The benchmark reports decomposed metrics by layer. A system may answer correctly while writing the wrong memory, preserve the final value while retaining many stale versions, retrieve the correct entry but copy an obsolete value, or compact aggressively while deleting the current state. vNext treats these as distinct outcomes instead of collapsing them into a single leaderboard score.

## 2. Context and Motivation

### 2.1 What the current project established

The current MemUpdateBench line established several important findings:

1. Append-only memory can retain the final value under oracle-like exact-slot lookup while accumulating stale same-slot records.
2. Prompted answering can collapse even when the final state remains recoverable.
3. Clean target-slot state does not guarantee answer correctness because retrieval and answer generation remain separate failure layers.
4. Learned or heuristic compaction reduces stale burden but can introduce missed updates, incorrect state, or incomplete compaction.
5. Stale same-slot conflict is an order- and metadata-sensitive version-arbitration failure, not a universal claim that stale same-slot distractors are always harder than generic distractors.
6. Latest-per-slot retrieval, context order, version labels, position, stale count, and prompt variants provide useful mechanism controls.

These findings are valuable but currently live across scripts, result directories, manuscript ledgers, temporary probes, and several historical data formats. The current benchmark also has known engineering and scientific limitations, including legacy split leakage, weak artifact provenance, inconsistent run conditions in some historical cells, incomplete external-system integration, and metrics embedded directly in evaluator scripts.

### 2.2 Why vNext is needed

A deployable benchmark requires more than additional experiments. It needs stable contracts and quality gates so that new task families, realistic data, external memory SDKs, learned managers, answer models, and future training methods can be compared without changing the meaning of the metrics.

vNext therefore prioritizes:

- canonical schemas;
- deterministic validation;
- leakage-resistant splits;
- capability-aware external adapters;
- one scorer shared by evaluation, filtering, training, and visualization;
- incremental, resumable artifacts;
- explicit task-family and difficulty profiles;
- source provenance and licensing/privacy metadata;
- case-level auditability;
- phased scale-up.

## 3. Goals and Non-Goals

### 3.1 Goals

vNext must:

1. Preserve repeated same-slot updates as a first-class benchmark family.
2. Expand evaluation to other dynamic-memory lifecycle behaviors, including multi-slot updates, grounding, write discipline, deletion, historical queries, and long-horizon memory use.
3. Make every scored task programmatically auditable through structured gold actions, states, event roles, and answers.
4. Separate event understanding, memory action, state maintenance, store quality, retrieval, answer behavior, and systems performance.
5. Support built-in managers and external memory systems through a common adapter contract.
6. Report only the metrics supported by an adapter's observable capabilities.
7. Prevent semantic trajectory leakage across train/dev/test in new benchmark data.
8. Preserve and label legacy P6/P8 results without presenting them as newly leakage-free vNext results.
9. Support deterministic oracle and corrupted smoke baselines before model evaluation.
10. Produce artifacts that can be resumed, re-scored, inspected, and packaged independently of the original run environment.
11. Permit future SFT or RL-style method learning while keeping the benchmark scorer as the source of truth.
12. Scale from Pilot to Core to Full only after explicit acceptance gates pass.

### 3.2 Non-goals

The initial vNext implementation does not require:

- completing all planned task families at once;
- replacing all legacy P6/P8 scripts in one change;
- publishing a public leaderboard immediately;
- performing full RLVR/GRPO training;
- using LLM judges as the main online reward or scoring mechanism;
- forcing black-box external systems into metrics they cannot support;
- treating a single composite score as the primary scientific result;
- broad crawling of private, copyrighted, or license-uncertain sources;
- claiming ecological coverage of every long-term-memory use case;
- discarding existing manuscript, figure, or result assets.

## 4. Design Principles

### 4.1 Verifiability first

Every task must include enough structured gold information to replay the intended memory lifecycle and evaluate the relevant outputs. Natural language alone is not the gold contract.

### 4.2 Decomposed diagnosis

Action correctness, final state, stale burden, compactness, retrieval, version arbitration, answer correctness, and system cost remain separate dimensions. Composite scores are optional secondary summaries.

### 4.3 Contract-first, compatibility-preserving migration

The project first freezes canonical interfaces. Existing data and results are imported through compatibility layers. New components consume the canonical interfaces; they do not infer semantics from directory names or ad hoc dictionaries.

### 4.4 Exactness where exactness is meaningful

The current `(entity, attribute)` invariant remains the default object identity for slot-based tasks. The schema supports namespaces and typed objects for future families, but vNext does not replace exact identities with unbounded semantic similarity.

### 4.5 Controlled expansion

Each new family must preserve diagnostic attribution. Increasing realism must not make it impossible to determine whether a failure came from event interpretation, state corruption, retrieval, or answering.

### 4.6 Capability-aware fairness

External systems differ in state export, timestamps, retrieval traces, update APIs, and deletion support. vNext declares these differences and limits comparisons to observable capabilities.

### 4.7 Artifact truth over narrative truth

Every reported number must resolve to a task manifest, run manifest, per-example outputs, and a scorer version. Notes or manuscripts do not substitute for missing raw artifacts.

### 4.8 Pilot before scale

New data sources and families enter a small pilot, pass validator/oracle/error-analysis gates, and only then enter Core or Full.

## 5. System Architecture

```text
Raw or Synthetic Sources
        |
        v
Source Normalization and Provenance
        |
        v
Family-Specific Task Compilers
        |
        v
Canonical MemUpdateTask
        |
        +--> Validator
        +--> Difficulty/Profile Resolver
        +--> Stratified Split Builder
        |
        v
MemoryAdapter Contract
        |
        +--> Built-in managers
        +--> External memory systems
        |
        v
Evaluation Runtime
        |
        +--> Parsed manager actions
        +--> Memory snapshots
        +--> Retrieval traces
        +--> Answer predictions
        +--> System measurements
        |
        v
Canonical Scorer
        |
        +--> Per-layer metrics
        +--> Failure taxonomy
        +--> Optional diagnostic composite
        |
        v
JSONL Artifacts and Run Manifest
        |
        +--> Slice summaries and figures
        +--> Case audit viewer
        +--> SFT filtering
        +--> Future reward callbacks
```

### 5.1 Component boundaries

Each component has one primary responsibility:

- **Source adapters** normalize raw sources and preserve provenance.
- **Task compilers** create benchmark semantics for one task family.
- **Validators** reject invalid or ambiguous tasks before release.
- **Profile resolvers** map difficulty names to explicit family parameters.
- **Split builders** enforce group isolation and joint stratification.
- **Memory adapters** expose a common execution and introspection interface.
- **Evaluation runtime** executes tasks and records traces without defining metric semantics.
- **Scorer** converts canonical task and runtime records into canonical scores.
- **Summarizers** aggregate existing score records without reinterpreting raw outputs.
- **Case viewer** renders canonical artifacts without recomputing scores.
- **Training interfaces** reuse canonical scorer fields and never redefine benchmark correctness.

## 6. Task-Family Taxonomy

### 6.1 Family A: Repeated Same-Slot Update

This is the anchor family and direct successor to current P6/P8 work.

Variables include:

- target-slot update depth;
- stale same-slot count;
- same-name other-entity distractors;
- same-entity other-attribute distractors;
- semantic near-miss NOOP events;
- duplicate-current values;
- context length;
- event and retrieval order;
- version metadata;
- answer prompt variant.

Primary questions:

- Is the final value stored?
- How many obsolete same-slot versions remain?
- Is the current version retrieved?
- Does the answer model copy a stale value?
- Which order or metadata conditions repair the failure?

### 6.2 Family B: Interleaved Multi-Slot Update

Multiple entity/attribute objects are updated in one episode.

This family measures:

- target-slot update accuracy;
- non-target-slot preservation;
- cross-slot contamination;
- global memory pressure;
- interference under interleaved event order;
- retrieval competition across slots.

The generator must independently control target update depth, number of active slots, interleaving pattern, and cross-slot distractor density.

### 6.3 Family C: Entity and Attribute Grounding

This family formalizes grounding errors that currently appear as stressors or post-hoc diagnostics.

Conditions include:

- same surface name, different entity namespace;
- same entity, different attribute;
- aliases and qualified/unqualified mentions;
- attribute paraphrases;
- entity and attribute near matches;
- namespace collisions;
- unresolved or ambiguous references.

Primary metrics include wrong-entity action rate, wrong-attribute action rate, cross-object corruption, and `reference_resolution_accuracy`. Ambiguous and no-match references require explicit typed abstention gold; they must never be assigned a guessed value.

### 6.4 Family D: NOOP and Write Discipline

This family tests whether systems know when not to modify memory.

Events include:

- transient observations;
- hypothetical or conditional statements;
- negated facts;
- uncertain statements;
- semantically related but non-updating statements;
- repeated identical current facts;
- irrelevant events;
- unsupported inferences.

Primary metrics include false-write rate, missed-write rate, duplicate-current burden, unnecessary update rate, and action calibration.

### 6.5 Family E: Deletion and Forgetting

This family extends the action contract with `DELETE` for systems that support lifecycle removal.

Conditions include:

- explicit forget requests;
- attribute-level deletion;
- entity-level deletion;
- correction versus deletion;
- expiration/TTL;
- privacy-sensitive removal;
- deletion followed by semantically similar retrieval queries;
- delete-and-relearn sequences.

Primary metrics include deletion compliance, residual forgotten-value exposure, collateral deletion, state recovery after relearning, and answer leakage after deletion.

### 6.6 Family F: Current-State and Historical Queries

This family separates current-state memory from version-history use.

Query types include:

- current value;
- immediately previous value;
- value at a specified event/time;
- transition description;
- ordered version history.

This family measures the trade-off between aggressive stale removal and historical-query utility. A method must not receive credit for current-state compactness when the task explicitly requires history.

### 6.7 Family G: Long-Horizon Memory Synthesis

This family requires reasoning over multiple memory objects or updates while retaining explicit gold provenance.

Conditions include:

- multi-object aggregation;
- update-sensitive multi-hop reasoning;
- stale-value propagation through a reasoning chain;
- long-history retrieval with a short answer;
- current-state consistency across several related slots.

This family borrows the verifiable evidence discipline of VeriLong-RL but remains a memory-lifecycle task rather than generic long-context QA.

### 6.8 Family H: Realistic Source Updates

This family compiles public or user-authorized sources into canonical tasks.

Candidate source classes include:

- user-profile dialogues;
- project status logs and changelogs;
- calendars and deadline histories;
- issue/ticket lifecycles;
- technical report revisions and errata;
- public documentation version histories.

All realistic sources must preserve source-level provenance, legal/privacy metadata, normalization hashes, and stable event anchors.

## 7. Canonical Data Model

Phase 0 uses Pydantic v2 models as the normative in-repository implementation and exports versioned JSON Schema for external consumers. Compatibility is defined by the exported JSON Schema and documented semantics, not by Python import paths.

The top-level artifact models are:

```text
MemUpdateTask
TaskRunRecord
ScoreRecord
RunManifest
TaskManifest
```

Every top-level artifact contains `schema_version`. Nested records are validated as part of their owning artifact and do not independently require a schema version.

### 7.0.1 v2 reference-resolution and answer-disposition contract

Family C requires a first-class unresolved-reference path rather than a sentinel answer convention. Contract v2 therefore defines:

```text
AnswerDisposition = answered | abstained | unavailable
ReferenceResolutionStatus = unique | ambiguous | no_match
QueryType = ... | unresolved_reference
```

`ReferenceCandidate` stores a stable candidate ID and an exact four-part `MemoryObjectKey`; `SurfaceReference` links visible entity/attribute wording to an ordered candidate-ID set; `CanonicalAnswer` stores disposition, resolution status, selected candidate IDs, abstention reason, and an optional value. Identity remains exactly `(namespace, entity, attribute, subkey)`; `object_type` is classification metadata and never distinguishes candidates.

Gold may contain `ANSWERED` or `ABSTAINED`, but never `UNAVAILABLE`. Runtime `UNAVAILABLE` and `parsed_answer=None` mean missing or unavailable evidence, not intentional abstention. Ambiguous/no-match queries require explicit `ABSTAINED` gold, while a unique answered query is linked to the selected candidate's replayed current value. Ordinary absent targets, deletion results, and NOOP events do not become unresolved references.

The semantic task projection includes candidate identities/order, resolution status, canonical disposition/value, and semantically exposed reference evidence. It excludes surface prose, linked IDs, query metadata, difficulty/split/compiler fields, and `object_type`. Scoring adds `reference_resolution_accuracy`; `wrong_reference_guess` marks an answer where abstention was required, and `unjustified_abstention` marks abstention where a unique answer was required. No adapter capability bit is added for abstention.

The v2 task/runtime/score/manifest schemas and compiler/scorer/metric/profile identities use `2.0.0`. The published Phase 0 v1 release remains immutable historical evidence. v1 artifacts are interpreted only by v1 semantics; there is no silent migration of `None` or missing answers into v2 abstention records.

### 7.1 `SourceRecord`

```text
source_id: str
source_type: synthetic | dialogue | changelog | calendar | issue | report_revision | other
source_uri: str | null
license_or_privacy: str
raw_hash: str | null
normalized_hash: str
normalization_version: str
provenance: object
generator: object | null
```

For synthetic data, `generator` records generator name, seed, config hash, code revision, and family compiler version.

### 7.2 `SplitKey`

```text
semantic_core_id: str
source_group_id: str
trajectory_id: str
paraphrase_group_id: str | null
source_document_id: str | null
version_group_id: str | null
split_exception_id: str | null
split_policy_version: str
```

The split builder groups by `source_group_id` and `trajectory_id` before stratification. `semantic_core_id` prevents semantically identical trajectories from crossing splits. A non-null `split_exception_id` is allowed only for a versioned evaluation-only robustness pair and is never used to justify train/test overlap.

### 7.3 `TaskMetadata`

```text
split: train | dev | test | evaluation_only
split_key: SplitKey
profile_name: easy | medium | hard | challenge
resolved_profile: object
generation_config_hash: str
compiler_version: str
tags: list[str]
legacy_provenance: LegacyProvenance | null
extra: object
```

`resolved_profile` stores the actual family-specific parameters used to build the task; downstream code does not infer them from filenames.

### 7.4 `MemoryObjectKey`

```text
namespace: str
entity: str
attribute: str
subkey: str | null
object_type: str
```

For initial slot tasks, `namespace="default"`, `subkey=null`, and identity is exact `(namespace, entity, attribute, subkey)` equality.

### 7.5 `GoldAction`

```text
action_id: str
event_id: str
operation: ADD | UPDATE | NOOP | DELETE
scope: object | attribute | entity | namespace | ttl
target_object_keys: list[MemoryObjectKey]
value: structured value | null
effective_at: str | null
expected_effect: object
```

An event may compile to one or more atomic gold actions. Existing ADD/UPDATE/NOOP tasks compile to exactly one action per event. `NOOP` has an empty target list and no value. Attribute-, entity-, namespace-, and TTL-level deletion is represented by `scope`, while `target_object_keys` explicitly enumerates the objects that the deterministic reference executor must affect. This avoids wildcard semantics during scoring.

### 7.6 `MemoryEvent`

```text
event_id: str
sequence_index: int
timestamp: str | null
raw_text: str
normalized_text: str
speaker: str | null
gold_action_ids: list[str]
role: EventRole
source_anchor: object
metadata: object
```

Canonical `EventRole` values:

```text
latest_gold
stale_same_slot
duplicate_current
same_entity_other_attribute
same_name_other_entity
noop_near_miss
neutral
deletion
historical_support
```

A compiler may add family-specific role details in metadata, but the canonical role set remains stable within a schema version.

### 7.7 `MemoryQuery`

```text
query_id: str
query_type: current_state | historical_state | transition | multi_object | deletion_compliance
text: str
target_object_keys: list[MemoryObjectKey]
answer_schema: string | number | boolean | list | object
evaluation_mode: state_direct | retrieved_prompt | native_system
metadata: object
```

### 7.8 `GoldRecord`

```text
actions: list[GoldAction]
action_sequence: list[str]
final_state: object
version_history: object
expected_present_objects: list[MemoryObjectKey]
expected_absent_objects: list[MemoryObjectKey]
gold_source_event_ids: list[str]
gold_answers: object
acceptable_answers: object
```

`action_sequence` is an ordered list of `action_id` values and must contain every action exactly once. `MemoryEvent.gold_action_ids` references this same action set. `acceptable_answers` contains explicit aliases or structured equivalents; it is not an unrestricted semantic-judge escape hatch.

### 7.9 `LegacyProvenance`

```text
legacy_family_id: str
legacy_phase: str
legacy_dataset_id: str
legacy_split_id: str
legacy_metric_namespace: str
legacy_run_condition_id: str | null
checkpoint_family: str | null
training_seed: int | null
answer_mode: str | null
memory_trajectory_id: str | null
source_artifact_path: str
source_artifact_hash: str
known_caveats: list[str]
```

### 7.10 `MemUpdateTask`

```text
task_id: str
schema_version: str
task_family: str
difficulty: easy | medium | hard | challenge
source: SourceRecord
events: list[MemoryEvent]
target_objects: list[MemoryObjectKey]
queries: list[MemoryQuery]
gold: GoldRecord
metadata: TaskMetadata
```

### 7.11 Stable identity and hashing

A task's semantic identity hash is computed from canonical family semantics, normalized source anchors, object keys, gold actions/values/roles, query types, and gold state. Surface paraphrases may have different task IDs but share a `semantic_core_id`.

Family compilers define the semantic-core projection explicitly:

- Family A removes surface wording while preserving target object, ordered version values, update depth, distractor roles, and query semantics.
- Family B additionally preserves the multi-object trajectories and interleaving relation.
- Family C preserves the grounding graph, alias/namespace relation, target action, and ambiguity condition.
- Family D preserves the state before the event, write-trap type, intended NOOP/write action, and resulting state.
- Families E/F preserve deletion/version-history scope and query time.
- Families G/H preserve the compiled object/action graph and source-group identity.

`SplitKey` fields are part of `TaskMetadata`; they are not reconstructed after generation. These fields are used for leakage detection, grouped analysis, and deterministic split reproduction.

### 7.12 `TaskManifest`

```text
schema_version
task_manifest_version
data_release_id
split_policy_version
task_schema_version
compiler_versions
source_manifest_paths_and_hashes
generation_configs_and_hashes
split_counts
family_difficulty_counts
semantic_core_counts
task_file_paths_and_hashes
leakage_check_summary
human_audit_artifacts
created_at
code_revision
```

A task file is release-valid only when its hash and counts match the manifest. Per-slice exports reference the same task IDs and are verified as deterministic filters of the canonical aggregate task set.

## 8. Runtime and Prediction Schemas

### 8.1 `ParsedManagerAction`

```text
event_id
operation
target_object_key
value
format_valid
execution_status
fallback_used
error_flags
raw_output
latency_ms
```

### 8.2 `MemoryEntryRecord`

```text
entry_id
content
object_key_candidate
value_candidate
created_at
updated_at
source_event_ids
version_index
raw_metadata
```

### 8.3 `MemorySnapshot`

```text
after_event_id
entries
state_by_object
store_size
raw_adapter_state
snapshot_hash
```

Snapshots may be full or delta-compressed, but the artifact reader must expose the same logical representation.

### 8.4 `RetrievalTrace`

```text
query_id
retrieved_entries
scores
ranks
gold_in_context
stale_in_context
distractor_in_context
retrieval_policy
context_order
version_metadata
prompt_hash
```

### 8.5 `AnswerPrediction`

```text
query_id
raw_output
parsed_answer
cited_event_ids
cited_entry_ids
format_valid
error_flags
latency_ms
usage
```

### 8.6 `ParserExtractorProvenance`

```text
action_parser_version
answer_parser_version
memory_entry_extractor_version
object_value_extractor_config_hash
redaction_policy_version
raw_provider_artifact_path
raw_provider_artifact_hash
raw_adapter_state_path
raw_adapter_state_hash
```

A parser or extractor change creates a new normalized runtime artifact. It does not overwrite the earlier `TaskRunRecord`.

### 8.7 `TaskRunRecord`

```text
schema_version
runtime_record_version
task_id
adapter_id
run_id
parsed_actions
memory_snapshots
retrieval_traces
answer_predictions
system_events
parser_extractor_provenance
exceptions
completion_status
```

A task-level exception is recorded in the run record and does not discard previously completed tasks. `completion_status` is one of `completed`, `failed`, `partial`, or `not_supported`.

### 8.8 `ScoreRecord`

```text
schema_version
scorer_version
task_id
run_id
adapter_id
task_family
difficulty
completion_status
supported_metric_fields
protocol_scores
action_scores
state_scores
store_scores
retrieval_scores
answer_scores
system_scores
audit_scores
failure_flags
primary_failure
legacy_metrics
```

Each score layer is a typed Pydantic model. Pilot requires the following minimum fields:

```text
protocol_scores:
  action_parse_valid
  answer_parse_valid
  execution_success_rate
  unsupported_operation_rate
  fallback_rate

action_scores:
  operation_accuracy
  full_action_exact_match
  object_key_accuracy
  entity_accuracy
  attribute_accuracy
  value_accuracy
  false_write_rate
  missed_write_rate
  wrong_object_write_rate

state_scores:
  final_state_accuracy
  state_precision
  state_recall
  state_f1
  state_resolve_rate
  collateral_corruption_rate
  expected_absence_accuracy

store_scores:
  obsolete_version_count
  stale_conflicting_value_count
  duplicate_current_count
  final_memory_size
  compaction_ratio
  write_amplification

retrieval_scores:
  current_recall_at_k
  current_mrr
  stale_exposure_rate
  stale_count_in_context
  distractor_exposure_rate

answer_scores:
  exact_match
  normalized_match
  token_f1
  structured_field_accuracy
  stale_copied
  distractor_copied
  gold_retrieved_wrong_answer
  answer_state_consistency

system_scores:
  ingest_latency_ms
  retrieval_latency_ms
  answer_latency_ms
  token_usage
  api_cost
  error_rate

audit_scores:
  action_trace_available
  state_export_available
  retrieval_trace_available
  source_provenance_coverage
  manifest_completeness
```

A field that does not apply to the task or cannot be observed under the adapter capability contract is serialized as `null` and excluded from its mean denominator. The reason is recorded as `not_applicable`, `not_supported`, `runtime_failed`, or `missing_artifact` in the metric-support map. Summarizers must not need to re-read raw model text or infer the experimental condition from directory names.

### 8.9 Metric registry

A versioned machine-readable metric registry defines every `ScoreRecord` field with:

```text
field_name
layer
value_type
numerator_definition
denominator_definition
aggregation_rule
applicable_task_families
required_adapter_capabilities
unsupported_value_policy
runtime_failure_policy
legacy_aliases
introduced_in_scorer_version
```

Core/Full may add registered fields, but an existing field's meaning cannot change within a scorer major version.

## 9. Difficulty and Profile System

Difficulty is family-specific and explicitly resolved from profiles.

Canonical control dimensions include:

```text
update_depth
active_object_count
entity_ambiguity
attribute_ambiguity
noop_density
cross_slot_interleaving
stale_count
context_length
context_order
version_metadata
query_type
source_naturalness
```

Profiles:

- **easy**: direct wording, low ambiguity, short histories, favorable order;
- **medium**: moderate interleaving and distractors, mixed order;
- **hard**: adversarial grounding, high stale burden, long histories, unfavorable order;
- **challenge**: compositional combinations intended for hidden or specialist evaluation.

Each family provides canonical defaults for these profile names. YAML configuration may override allowed parameters, but:

1. unknown keys fail validation;
2. canonical family and difficulty labels cannot be overridden;
3. resolved parameters are stored in task metadata;
4. hard and challenge defaults are locked by regression tests;
5. aggregate reports use resolved parameters rather than filenames.

## 10. Mechanism Factor Matrix

Existing P8-style mechanism probes become a formal diagnostic factor matrix.

```yaml
storage_policy:
  - append_only
  - overwrite_exact_object
  - heuristic_compaction
  - learned_compaction

retrieval_policy:
  - semantic_topk
  - latest_per_object
  - target_object
  - recency_weighted

context_order:
  - retrieval_score
  - chronological
  - reverse_chronological
  - current_first
  - current_last
  - random

version_metadata:
  - none
  - timestamp
  - event_index
  - latest_outdated

prompt_variant:
  - current_value
  - value_only
  - ignore_distractors
  - historical_value

stale_count:
  - 0
  - 1
  - 2
  - 4
  - 8
  - 16
```

Not every Cartesian-product cell is required. A versioned matrix manifest lists the supported cells, sample count, seed, answer model, retrieval composition, and expected comparison group.

The matrix distinguishes:

- storage intervention;
- retrieval intervention;
- presentation intervention;
- prompt intervention;
- answer-model replacement.

For example, latest-per-object over a full store is labeled a retrieval rewrite that combines stale suppression and candidate recall expansion. It is not described as a pure deletion from the original top-k context.

## 11. Metric Framework

### 11.1 Layer 0: Protocol and execution

- action parse validity;
- answer parse validity;
- execution success rate;
- unsupported-operation rate;
- fallback rate;
- task completion rate;
- exception rate.

### 11.2 Layer 1: Event interpretation and action

- operation accuracy;
- full action exact match;
- object-key accuracy;
- entity accuracy;
- attribute accuracy;
- value accuracy;
- false-write rate;
- missed-write rate;
- unnecessary-update rate;
- wrong-object write rate;
- deletion-action accuracy.

### 11.3 Layer 2: Memory state

- final-state exact accuracy;
- per-object state precision/recall/F1;
- state resolve rate;
- missed-final-update rate;
- wrong-final-value rate;
- collateral-object corruption rate;
- expected-absence accuracy;
- historical-state accuracy;
- deletion compliance;
- relearning accuracy.

### 11.4 Layer 3: Store quality

- obsolete-version count;
- stale conflicting-value count;
- duplicate-current count;
- same-name distractor count;
- unrelated-entry burden;
- final memory size;
- compaction ratio;
- write amplification;
- delete amplification;
- provenance coverage;
- entry-to-source traceability.

An obsolete record is defined by version/order semantics. A stale conflicting-value record is an obsolete record whose value conflicts with the current value. An old record whose value happens to equal the current value is obsolete but not conflicting. These are separate metrics.

### 11.5 Layer 4: Retrieval

- current evidence recall@k;
- current object recall@k;
- slot/object precision@k;
- MRR/current-version rank;
- stale exposure rate;
- stale count in context;
- distractor exposure rate;
- gold-present-but-not-selected rate;
- retrieval composition entropy;
- target-object coverage;
- historical-evidence recall for historical queries.

### 11.6 Layer 5: Answer and version arbitration

- exact match;
- normalized match;
- token F1;
- structured field accuracy;
- answer-value-present;
- stale copied rate;
- distractor copied rate;
- multiple-version answer rate;
- gold-retrieved-but-wrong-answer rate;
- prompt sensitivity;
- order sensitivity;
- version-label repair delta;
- repeated-decoding consistency;
- answer/state consistency.

### 11.7 Layer 6: Systems metrics

- ingest latency;
- retrieval latency;
- answer latency;
- end-to-end latency;
- throughput;
- token usage;
- API cost;
- storage growth;
- cache-hit rate;
- retry/error rate;
- deterministic reset success;
- run reproducibility status.

### 11.8 Layer 7: Auditability

- action-trace availability;
- state-export availability;
- retrieval-trace availability;
- source-event provenance coverage;
- config/model/code/data hash completeness;
- failure-taxonomy coverage;
- case-export completeness.

### 11.9 Optional diagnostic composite

A composite score may be provided for engineering optimization, but it is secondary. It must:

- be versioned;
- expose all components and weights;
- never replace per-layer reporting;
- exclude unsupported capability metrics rather than treating them as zero;
- not be used to claim a universal system ranking.

## 12. Failure Taxonomy

The scorer assigns both independent flags and an optional primary failure label.

Independent flags include:

```text
invalid_action_format
unsupported_action
wrong_operation
wrong_entity
wrong_attribute
wrong_value
false_write
missed_update
collateral_corruption
deletion_failure
current_state_missing
stale_retained
current_not_retrieved
stale_retrieved
stale_copied
distractor_retrieved
distractor_copied
gold_retrieved_wrong_answer
answer_format_only
system_exception
```

The primary label is derived through a versioned precedence rule, but reports retain all flags so that a single-label taxonomy does not hide overlapping causes.

Primary layer precedence:

```text
execution/protocol
-> action/grounding
-> state
-> retrieval
-> answer/version arbitration
-> format-only
-> correct
```

## 13. Canonical Scorer

The scorer is the single source of metric truth.

### 13.1 `ScorerConfig`

```text
scorer_version
metric_registry_version
value_normalization_profile
answer_normalization_profile
primary_failure_precedence_version
requested_metric_fields
legacy_compatibility_mode
strict_capability_check
```

`legacy_compatibility_mode` is disabled for vNext-native results and names the exact legacy metric namespace when importing historical results.

### 13.2 Inputs and output

Inputs:

```text
MemUpdateTask
TaskRunRecord
AdapterCapabilities
ScorerConfig
```

Output:

```text
ScoreRecord
```

### 13.3 Scoring rules

1. The scorer does not call a model or external service.
2. The metric registry supplies the denominator, aggregation rule, task applicability, required capabilities, unsupported policy, and runtime-failure policy for every field.
3. Before scoring, strict mode verifies that requested metrics are compatible with declared adapter capabilities and available parser/extractor provenance.
4. State comparisons use canonical object identity and typed value normalization.
5. Retrieval metrics use canonical entry/event linkage, not unrestricted substring matching when structured linkage is available.
6. Answer metrics report exact, normalized, and structured matches separately.
7. Unsupported or non-applicable metrics remain `null` with an explicit support reason; they are never converted to zero.
8. Runtime failures are reported separately from model errors and do not silently enter accuracy denominators.
9. Legacy metrics are preserved under explicit legacy field names when their semantics differ.
10. Summarizers aggregate `ScoreRecord` fields and never independently reclassify failures.
11. Training filters and future reward callbacks call the same scorer or a documented registered subset of its fields.
12. The scorer reads normalized `TaskRunRecord` artifacts by default. Raw provider or SDK payloads are audit inputs only; reparsing raw payloads produces a new runtime artifact with new parser/extractor provenance before rescoring.

## 14. External Memory Adapter Contract

### 14.1 Supporting adapter types

#### `AdapterInfo`

```text
adapter_id
adapter_version
system_name
system_version
sdk_version
configuration_hash
extractor_id
extractor_version
```

#### `AdapterCapabilities`

Capabilities are an explicit bitset rather than an inference from a single level:

```text
supports_isolated_reset
supports_event_ingest
supports_add
supports_update
supports_noop
supports_delete
supports_ttl
supports_native_answer
exports_entries
exports_raw_state
exports_source_event_ids
exports_timestamps_or_order
exports_object_keys
exports_values
exports_retrieval_ids
exports_retrieval_scores
exports_action_trace
reports_latency
reports_token_usage
reports_cost
requires_evaluation_extractor
extractor_version
```

#### Adapter result records

```text
ResetResult:
  success
  namespace
  error

AdapterActionLog:
  event_id
  requested_operation
  effective_operation
  affected_entry_ids
  raw_action
  latency_ms
  error

RetrievalResult:
  query_id
  entries
  scores
  raw_result
  latency_ms
  error

AnswerResult:
  query_id
  raw_output
  usage
  cost
  latency_ms
  error
```

All result records are JSON-serializable typed models. Unsupported operations are declared through capabilities before execution and return a typed `not_supported` status if invoked by a diagnostic probe.

### 14.2 Interface

```python
class MemoryAdapter:
    def adapter_info(self) -> AdapterInfo: ...
    def capabilities(self) -> AdapterCapabilities: ...
    def reset(self, namespace: str, config: dict) -> ResetResult: ...
    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog: ...
    def export_entries(self) -> list[MemoryEntryRecord]: ...
    def export_raw_state(self) -> object: ...
    def retrieve(self, query: MemoryQuery, k: int) -> RetrievalResult: ...
    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult: ...
    def close(self) -> None: ...
```

Capabilities are authoritative. The runtime must not infer support from a method's existence or from failure after evaluation begins.

### 14.3 Derived capability levels

Levels are presentation shortcuts derived from the bitset; the bitset remains the scoring authority.

#### Level 0: Answer-only

Requires `supports_native_answer`.

Supported reporting:

- answer metrics;
- end-to-end latency/cost when reported;
- protocol errors.

It cannot enter state/stale/compactness comparisons.

#### Level 1: Retrieval trace

Requires retrieval IDs or exported retrieved entries. Retrieval-score metrics additionally require `exports_retrieval_scores`.

Additional reporting may include:

- current recall;
- stale/distractor exposure;
- retrieval rank and composition.

#### Level 2: Memory export

Requires `exports_entries`, isolated reset, and enough entry content to run a versioned evaluation extractor. Individual metrics add finer requirements:

- final-state metrics require structured object/value export or a versioned object/value extractor;
- obsolete/stale metrics require timestamp/order or source-event linkage;
- provenance coverage requires source-event IDs;
- memory size requires only entry export;
- deletion compliance requires delete support and post-delete state export.

Level 2 is the minimum category for the main memory-maintenance table, but each cell still follows metric-specific capability requirements.

#### Level 3: Full action trace

Requires `exports_action_trace` and state transition linkage.

Additional reporting:

- action correctness;
- grounding errors;
- fallback behavior;
- write/delete amplification;
- full failure decomposition.

### 14.4 Extractor provenance

When an SDK exports only text entries, object/value parsing is performed by a benchmark evaluation extractor, not attributed as a native SDK capability. The run manifest records:

- extractor ID/version;
- extractor configuration hash;
- whether each metric used native structured fields or extracted fields;
- extraction failures and coverage.

A changed extractor creates new normalized runtime and score artifacts.

### 14.5 Metric capability gating

The metric registry lists required capability bits. Examples:

```text
final_memory_size:
  requires: [exports_entries]

stale_conflicting_value_count:
  requires: [exports_entries, exports_timestamps_or_order]
  plus_one_of: [exports_object_keys, requires_evaluation_extractor]

current_mrr:
  requires: [exports_retrieval_ids, exports_retrieval_scores]

action_operation_accuracy:
  requires: [exports_action_trace]

deletion_compliance:
  requires: [supports_delete, exports_entries]
```

### 14.6 Fairness requirements

1. Every task uses an isolated namespace or verified reset.
2. Adapter configuration and SDK version are recorded.
3. The benchmark does not simulate unsupported state export and report it as native capability.
4. Black-box systems remain eligible for supplementary answer-only evaluation.
5. External entry parsing is separated from the adapter and versioned as an evaluation extractor.
6. Costs, retries, and provider failures are reported rather than silently dropped.
7. Tables disclose whether a metric used native structured fields or benchmark extraction.
8. Capability verification runs before benchmark execution and becomes part of the run manifest.

## 15. Source, Provenance, and Data Governance

### 15.1 Source-to-task compilation

```text
Raw source
-> deterministic normalization
-> anchored source events
-> family-specific semantic annotation
-> canonical MemUpdateTask
```

Raw and compiled data remain separate. The task contains source anchors and hashes but does not require redistribution of private or license-restricted raw material.

### 15.2 Source policy

Pilot realistic sources must come from:

- permissively licensed public sources;
- public technical documentation with recorded use terms;
- synthetic sources;
- user-provided or institution-authorized sources.

Broad crawling and private-user memory collection are excluded from the initial design.

### 15.3 Required provenance

Every non-synthetic source records:

- source URI or internal reference;
- license/privacy classification;
- retrieval date;
- raw and normalized hashes where permitted;
- section/turn/record anchors;
- normalization version;
- compiler version.

### 15.4 Privacy and deletion tasks

Deletion/forgetting tasks use synthetic or explicitly authorized data. The benchmark does not retain real sensitive personal data solely to test forgetting.

## 16. Split and Leakage Prevention

### 16.1 Joint stratification

At minimum, splits are stratified by:

```text
(task_family, difficulty, update_depth_bucket)
```

Core/Full may additionally stratify by source type, query type, and challenge condition.

### 16.2 Group isolation and split keys

Group-level isolation precedes stratification. The split builder consumes the `SplitKey` stored in every task:

- `source_group_id` isolates realistic source documents, users, projects, and synthetic generator groups;
- `trajectory_id` keeps all queries and surface variants from one memory lifecycle in one split;
- `semantic_core_id` keeps semantically equivalent trajectories together;
- `paraphrase_group_id` keeps paraphrases together;
- `version_group_id` keeps revisions of one real source together;
- `split_policy_version` identifies the deterministic grouping/stratification implementation.

Evaluation-only robustness pairs may span named evaluation slices only when they share a non-null `split_exception_id`; they never cross from training into dev/test and are excluded from held-out generalization claims.

### 16.3 Leakage checks

The build must fail release validation when:

- `semantic_core_id` overlaps across train/dev/test;
- `trajectory_id`, `paraphrase_group_id`, or protected `source_group_id` crosses train/dev/test;
- exact task hashes overlap across splits;
- a source/version group appears in multiple splits without an allowed evaluation-only exception;
- per-slice exports differ from aggregate filtering;
- expected family-specific strata are missing;
- task counts do not match the split manifest;
- a declared split exception crosses into training data.

Reports publish entity, attribute, value, template, source, paraphrase, and version-group overlap statistics even when those overlaps are allowed. The split validator reports both unique-task and semantic-core counts.

### 16.4 Family-specific stratification

The common stratum begins with `(task_family, difficulty)`, then adds family-specific axes:

- Families A/B: `update_depth_bucket`, active-object count, and interleaving level;
- Family C: entity ambiguity, attribute ambiguity, alias/namespace condition;
- Family D: NOOP/write-trap type, NOOP density, duplicate-current condition;
- Family E: deletion scope and relearning condition;
- Family F: query type and requested version distance;
- Family G: reasoning depth and active-object count;
- Family H: source type and provenance class.

The split builder balances these axes within available counts and records unavoidable small-cell deviations in the task manifest.

### 16.5 Legacy P6/P8 policy

Existing historical assets use explicit legacy identities rather than one generic label:

```text
legacy_p63
legacy_p65
legacy_p68_p70
legacy_p80_p82
legacy_p83
legacy_p84
legacy_p85_api_replacement
```

`LegacyProvenance` records dataset/split/run condition, source path/hash, answer mode, memory trajectory, checkpoint family, and known caveats.

Mandatory caveats include, where applicable:

- P6.3 semantic-core split leakage;
- original P6.3 Long25 versus P6.5 checkpoint-family differences;
- historical runs whose nominal answer modes used different memory trajectories;
- P8 pilot versus expanded/current summary identities;
- trace-composition stale-removal analyses versus actual answer-model reruns;
- API availability or pending cells that must not be represented as completed results.

Legacy assets remain useful for historical mechanism analyses and regression checks, but they are not described as leakage-free vNext held-out generalization. Legacy fields with non-equivalent semantics remain in their named `legacy_metric_namespace`.

## 17. Validation Pipeline

Validation occurs before model evaluation.

### 17.1 Structural validation

- schema version supported;
- IDs non-empty and unique;
- sequence indices ordered;
- referenced object/event/source IDs exist;
- required family metadata present;
- actions and values type-compatible;
- query targets exist or are expected absent for deletion tasks.

### 17.2 Semantic replay validation

A deterministic reference executor replays gold actions and verifies:

- final state equals `GoldRecord.final_state`;
- version history matches gold;
- expected present/absent objects match;
- NOOP does not mutate state;
- DELETE removes only the intended object;
- historical queries resolve to the expected version.

### 17.3 Distractor and ambiguity validation

- distractor events do not independently establish the current gold state;
- stale events are obsolete by order/version semantics;
- duplicate-current is not mislabeled as conflicting stale;
- simple retrieval distractors do not contain an accepted answer unless the condition explicitly tests ambiguity;
- query answer is unique under the task contract.

### 17.4 Split validation

- joint-stratum coverage;
- semantic-core isolation;
- source-group isolation;
- deterministic split reproduction;
- manifest count/hash agreement.

### 17.5 Human audit

Pilot and each new realistic-source compiler require a stratified human audit sample covering every family, difficulty, and source type. Human audit records are artifacts, not informal notes.

## 18. Evaluation Runtime and Artifact Protocol

### 18.1 Incremental execution

Evaluation writes one `TaskRunRecord` JSONL row per completed or failed task and flushes after each row. A late failure must not discard earlier work.

### 18.2 Cache and normalized-artifact identity

Raw response cache keys include:

```text
adapter/version
model/provider/revision
prompt version and hash
task semantic hash
runtime generation config hash
scorer-independent decoding parameters
```

A raw response cache record is reused only when all raw-generation identity fields match.

Normalized runtime-artifact identity additionally includes:

```text
action parser version
answer parser version
memory-entry extractor version
object/value extractor config hash
redaction policy version
```

This separation permits a raw response to be reparsed without another provider call while ensuring that reparsing creates a new `TaskRunRecord` rather than mutating an old record.

### 18.3 Resume semantics

Resume operates at task level:

- completed matching rows are skipped;
- failed rows may be retried through an explicit flag;
- hash mismatches force re-execution;
- duplicate task IDs are rejected;
- missing expected task IDs are reported before summary generation.

### 18.4 `RunManifest`

Every run writes a top-level `RunManifest` containing:

```text
schema_version
run_manifest_version
run_id and timestamp
code revision and dirty-state flag
task manifest path/hash
schema/scorer/metric-registry/profile versions
adapter information and capability bitset
capability-verification artifact path/hash
model/provider/revision
prompt and decoding config
seed information
action parser version
answer parser version
memory-entry extractor version and config hash
redaction policy version
environment/package summary
expected/completed/failed/not-supported task counts
raw provider response paths/hashes
raw adapter state paths/hashes
normalized runtime artifact paths/hashes
score artifact paths/hashes
```

The manifest distinguishes native structured fields from values produced by a benchmark extractor.

### 18.5 Artifact layout

Proposed layout:

```text
data/vnext/
  sources/
  tasks/
  manifests/
  audit/

results/vnext/
  runs/<run_id>/
    run_manifest.json
    task_runs.jsonl
    scores.jsonl
    summary.json
    slices/
    cases/

results/raw/vnext/
  provider_cache/
  adapter_raw/
```

Private and license-uncertain raw sources do not enter redistributable artifacts.

### 18.6 Raw, normalized, and scored artifact boundary

Artifacts form an immutable derivation chain:

```text
raw provider/SDK artifact
-> parser/extractor provenance
-> normalized TaskRunRecord
-> scorer config and metric registry
-> ScoreRecord
-> summary/ledger/case export
```

The canonical scorer reads `TaskRunRecord`, not raw provider payloads. Raw payloads are retained for audit and optional reparsing. Reparsing or changing an extractor creates new normalized artifacts and hashes; rescoring with a new scorer creates new score artifacts. Neither operation overwrites the prior derivation chain.

## 19. Baselines and Controls

### 19.1 Built-in methods

Initial compatibility baselines:

- raw append;
- heuristic CRUD;
- deterministic exact-object CRUD;
- learned constrained manager/Long25 legacy checkpoint;
- latest-per-object answer-time retrieval rewrite.

Their names and true semantics are defined in the benchmark documentation. For example, raw append with benchmark-injected slot metadata is distinguished from a completely unstructured text store.

### 19.2 Oracle controls

- gold action executor;
- exact-object state readout;
- gold retrieval context;
- clean-current-only answer context;
- historical oracle for history queries.

### 19.3 Corrupted smoke controls

- always ADD;
- always NOOP;
- stale-value copier;
- wrong-entity writer;
- wrong-attribute writer;
- invalid action formatter;
- current-not-retrieved context;
- gold-retrieved wrong-answer output;
- deletion ignored;
- collateral deletion.

Each corrupted control must trigger the intended scorer fields in regression tests.

## 20. Reporting and Statistical Protocol

### 20.1 Required reporting

Every benchmark table or figure states:

- task family and split;
- number of tasks and semantic cores;
- difficulty/profile;
- adapter capability level;
- answer model and decoding;
- retrieval policy and top-k;
- context order and version metadata;
- metric denominator;
- run/artifact identifier.

### 20.2 Slice reports

Minimum slices:

- family;
- difficulty;
- update depth;
- entity/attribute ambiguity;
- query type;
- source type;
- context order;
- version metadata;
- adapter/method;
- answer model.

### 20.3 Uncertainty

Core/Full reports include confidence intervals or bootstrap intervals for principal rate metrics. Repeated stochastic runs are required when decoding or system behavior is stochastic. API alias and provider-version limitations are recorded explicitly.

### 20.4 Missing capabilities

Unsupported metrics are reported as `not_supported`, not zero. Cross-system tables indicate capability levels and avoid averaging over incomparable metric sets.

### 20.5 Canonical ledger

A generated canonical ledger maps each claim/table cell to:

- run ID;
- score field;
- task slice;
- sample count;
- artifact path/hash.

The ledger is generated from manifests and score records rather than maintained only by manual prose.

## 21. Training and Method-Learning Interfaces

### 21.1 Gold-locked SFT

For manager training, programmatic gold controls:

- operation;
- target object key;
- value;
- memory delta;
- final-state target;
- answer target where applicable.

Teacher models may supply optional rationale phrasing, but cannot overwrite these critical fields.

### 21.2 Distillation filter gate

A distilled example is accepted only when:

- action matches gold;
- target object and value match gold;
- resulting memory delta/state match gold;
- final answer matches accepted answer semantics;
- no stale/distractor source is treated as current support;
- output structure is valid.

Failed examples fall back to programmatic gold targets or are excluded.

### 21.3 Training loss boundaries

- prompt/history tokens are masked from supervised loss;
- critical target fields are not truncated;
- over-length examples are bucketed or rejected rather than silently losing supervision;
- data/template smoke and GPU training smoke are separate gates.

### 21.4 Future reward learning

Future RL-style callbacks consume canonical `ScoreRecord` components. Training reward and reported benchmark metrics remain aligned. Full RLVR is not a prerequisite for Pilot or Core benchmark completion.

### 21.5 LLM judge role

LLM judges may support:

- realistic-source candidate filtering;
- ambiguity review;
- paraphrase quality checks;
- manual-audit prioritization;
- semantic equivalence research.

They are not the default online reward and do not replace deterministic gold where deterministic verification is available.

## 22. Offline Case Audit Viewer

The viewer is an audit tool, not a live-model leaderboard.

A case page displays:

1. task metadata and source provenance;
2. event timeline with canonical roles;
3. gold action sequence;
4. predicted actions and execution outcomes;
5. memory snapshots or state deltas;
6. final memory entries with stale/current/distractor highlighting;
7. retrieved context with scores and ranks;
8. answer output and parser result;
9. metric breakdown and failure flags;
10. prompt/config/run-manifest references.

The initial viewer is offline-first and reads exported canonical case JSON. Live API execution is optional and later; it must preserve environment-only secrets, timeout/rate-limit/cache/fallback behavior, and explicit source labels.

Case exports are stratified by family, difficulty, update depth, method, and failure type rather than selected from the head of a result file.

## 23. Phased Delivery

### 23.1 Phase 0: Contract and Legacy Bridge

Deliverables:

- canonical schema definitions;
- scorer field definitions;
- failure taxonomy;
- profile and split specifications;
- adapter capability contract;
- run-manifest specification;
- legacy P6/P8 task/result compiler design;
- regression fixtures for current canonical results.

Acceptance criteria:

- every canonical field has one documented meaning;
- current P6/P8 examples can be represented without losing their original fields;
- legacy caveats are preserved;
- no new benchmark performance claim is made.

### 23.2 Pilot

Recommended scale: 500-2,000 tasks.

Initial implemented families:

- Family A: repeated same-slot;
- Family B: interleaved multi-slot;
- Family C: entity/attribute grounding;
- Family D: NOOP/write discipline.

Pilot balancing is family-specific:

- Families A/B balance `update_depth={1,4,16}`, active-object count, and easy/medium/hard profiles;
- Family C balances entity ambiguity, attribute ambiguity, aliasing, and namespace-collision conditions;
- Family D balances NOOP density, write-trap type, duplicate-current conditions, and easy/medium/hard profiles.

A selected subset of order and version-metadata mechanism conditions is included without requiring the full Cartesian matrix.

Required methods:

- deterministic oracle;
- raw append;
- heuristic CRUD;
- deterministic exact-object CRUD.

Optional feasibility:

- one external adapter that supports at least Level 2 state export.

Pilot acceptance criteria:

1. all tasks pass structural and semantic replay validation;
2. oracle final-state, expected-absence, non-mutation, and query metrics are perfect for valid tasks;
3. Families C/D pass family-specific grounding and false-write oracle checks rather than being judged only by update depth;
4. semantic-core overlap across train/dev/test is zero;
5. split and artifact manifests reproduce exactly;
6. corrupted controls activate expected score fields;
7. every family/difficulty and family-specific challenge cell has audited cases;
8. baseline trends can be explained through canonical traces;
9. no result depends on undocumented directory-name parsing.

### 23.3 Core

Recommended scale: 10,000-20,000 tasks after Pilot acceptance.

Core includes:

- Families A-D at larger scale;
- Family E deletion/forgetting;
- Family F current/history queries;
- selected Family G synthesis tasks;
- paraphrase/naturalness variants;
- at least one qualifying external Level 2/3 adapter;
- multiple answer models for answer-layer robustness;
- a versioned hard suite;
- confidence intervals and generated canonical ledgers.

Core acceptance criteria:

1. every family has validated train/dev/test or evaluation-only splits as appropriate;
2. principal metrics have confidence intervals;
3. adapter capability declarations are verified;
4. failure taxonomy explains at least 90% of non-system failures through one or more flags;
5. all main result cells resolve to raw artifacts and manifests;
6. a public scorer CLI and data card are usable from a clean environment;
7. case viewer artifacts reproduce benchmark scores exactly.

### 23.4 Full/Public Benchmark

Full scale may exceed 50,000 tasks only after Core is stable.

Full may include:

- Family H realistic-source tasks;
- multiple qualifying external memory systems;
- hidden evaluation sets;
- public adapter examples;
- expanded natural-language and source diversity;
- public case viewer;
- capability-aware leaderboard views.

Full acceptance criteria:

- source licenses/privacy policies documented;
- public and hidden split governance established;
- release artifacts reproducible from manifests;
- scorer and adapter contracts versioned;
- no single total score replaces per-layer reporting;
- benchmark governance defines schema/metric deprecation and submission rules.

## 24. Legacy Migration Strategy

### 24.1 Preserve historical assets

Current P6/P8 results remain read-only historical artifacts. Migration does not rewrite them in place.

### 24.2 Legacy task compiler

A legacy compiler maps current episode JSON fields into `MemUpdateTask` and adds:

- stable event IDs;
- family-specific `SplitKey` fields and semantic-core IDs;
- explicit event roles;
- source/generator provenance where recoverable;
- `LegacyProvenance` with phase, dataset, split, metric namespace, source path/hash, and caveats;
- action/value semantics without renaming `num_events` as update count;
- a legacy schema/version label that identifies the compiler.

### 24.3 Legacy result importer

A result importer maps historical evaluator outputs into canonical runtime/score fields when semantics match. The importer requires a `LegacyProvenance` record and preserves:

- legacy phase/dataset/split identity;
- answer mode and retrieval/prompt condition;
- memory trajectory ID;
- checkpoint family and training seed when known;
- source artifact path/hash;
- parser/extractor assumptions;
- known completeness and availability caveats.

Fields with different semantics remain under their named `legacy_metric_namespace`; unknown fields are retained as raw legacy payload rather than guessed into canonical metrics.

### 24.4 Regression use

Legacy canonical cells are used to verify compatibility, not to certify new vNext held-out generalization. The importer must reproduce selected historical headline values from the same raw artifacts before it is accepted.

### 24.5 Historical inconsistencies

Known inconsistencies are recorded as separate run conditions and never silently merged. Required explicit cases include:

- nominally similar answer modes with different memory trajectories;
- original Long25 versus later checkpoint families;
- P8 pilot, expanded, and current summaries;
- stale-removal trace composition versus actual answer reruns;
- pending or capacity-failed API cells;
- figures/tables whose sample count or source artifact differs from the claimed condition.

The importer rejects a completed canonical result row when the source artifact is missing, partial, pending, or condition-incompatible.

## 25. Testing Strategy

### 25.1 Schema tests

- valid/invalid objects;
- schema-version dispatch;
- stable serialization and hashing;
- object-key equality;
- answer-schema validation.

### 25.2 Generator/compiler tests

- family-specific gold semantics;
- profile overrides and unknown-key rejection;
- deterministic seed behavior;
- distractor non-leakage;
- duplicate-current versus stale-conflict distinction;
- realistic-source anchor stability.

### 25.3 Validator tests

- final-state replay;
- NOOP immutability;
- DELETE semantics;
- version-history correctness;
- unique answer support;
- split leakage rejection;
- manifest count/hash agreement.

### 25.4 Adapter contract tests

- namespace reset;
- capability declarations;
- deterministic fake adapter;
- state export mapping;
- retrieval trace mapping;
- unsupported-method behavior;
- exception and retry recording.

### 25.5 Scorer tests

- one fixture for every metric;
- denominator and missing-capability policy;
- overlapping failure flags;
- primary-label precedence;
- legacy metric compatibility;
- scorer invariance under artifact relocation.

### 25.6 End-to-end smoke tests

- schema/parser-only smoke;
- oracle task execution;
- each corrupted baseline;
- incremental run interruption/resume;
- cache hash invalidation;
- summary completeness rejection;
- offline case export and viewer load.

### 25.7 Regression tests from current project bugs

At minimum, tests prevent recurrence of:

- semantic-core split leakage;
- `num_updates` versus event-count confusion;
- aggregate/per-slice double counting;
- P8 pilot/current summary mixing;
- latest-per-object being mislabeled as a pure top-k filter;
- heuristic runs with inconsistent memory trajectories being merged;
- unsupported API cells appearing as completed manuscript rows;
- stale removal trace analyses being presented as answer reruns;
- image/table sample counts differing from source artifacts.

## 26. Error Handling

1. Invalid tasks fail before evaluation and enter a rejected-task audit artifact.
2. Per-task runtime failures are recorded and do not terminate the full run by default.
3. Repeated systemic failures may trigger a configurable fail-fast threshold.
4. Partial runs cannot produce a complete summary unless the manifest explicitly marks them partial.
5. Missing expected tasks, duplicate task IDs, or mixed run hashes cause summary validation failure.
6. Provider capacity failures are reported as availability failures, not model scores.
7. Adapter state-export failures invalidate state/store metrics for affected tasks and are not converted to zeros.
8. Raw provider/SDK responses remain separate from normalized runtime records.

## 27. Security, Privacy, and Operational Constraints

- API keys come only from environment variables or approved secret stores.
- Keys are never written to artifacts, logs, prompts, or browser responses.
- External adapters run in isolated namespaces and, where needed, isolated environments.
- The evaluator only stops processes it owns.
- Realistic-source data follows license/privacy policy and avoids unauthorized personal data.
- Deletion tasks do not preserve sensitive deleted content in public case exports.
- Raw external-system payloads may contain sensitive metadata and remain in restricted artifacts.

## 28. Reporting and Benchmark Governance

### 28.1 Versioning

The following are independently versioned:

- task schema;
- task-family compiler;
- difficulty profile;
- scorer;
- adapter contract;
- mechanism matrix;
- data release.

### 28.2 Backward compatibility

A new version may read older artifacts only through an explicit, separately tested migration. Contract v2 does not silently reinterpret v1 `None`, missing-answer, or raw answer-map records as abstention; v1 artifacts remain tied to the published v1 release semantics. Metric meaning must never change silently under an existing field name.

### 28.3 Deprecation

Deprecated fields remain readable for at least one release cycle and are documented with replacement semantics.

### 28.4 Submission/leaderboard policy

A future public submission must include:

- adapter version and capability declaration;
- run manifest;
- required raw/normalized outputs;
- no hidden manual correction;
- disclosure of external APIs/models and costs;
- separate reporting by task family and metric layer.

## 29. Alternatives Considered

### 29.1 Compatibility-layer-only design

This option would add validators, manifests, and external adapters around the current evaluator without defining a new core contract.

It was rejected as the primary architecture because metric semantics, task roles, run conditions, and result identity would remain distributed across scripts and file names. Compatibility wrappers remain part of the migration strategy, not the long-term core.

### 29.2 Clean-room rewrite

This option would build a completely new package and treat P6/P8 as historical only.

It was rejected because it risks losing working evaluators, mechanism evidence, and result continuity. vNext instead uses a new canonical contract with legacy compilers.

### 29.3 Single broad end-to-end score

This option would rank systems by one answer or reward score.

It was rejected because it hides the central MemUpdateBench insight: state success, stale burden, compactness, retrieval, and answer correctness can diverge sharply.

### 29.4 Immediate full external-system leaderboard

This option was rejected because systems expose different levels of memory state and traces. Capability-aware adapters and a Pilot feasibility gate are required first.

### 29.5 Immediate RLVR-first development

This option was rejected because benchmark validity, data splits, scorer contracts, and external evaluation are prerequisites. Future method learning remains supported but does not block benchmark delivery.

## 30. Risks and Mitigations

### 30.1 Scope expansion dilutes the central contribution

**Mitigation:** repeated same-slot remains the anchor family; each added family must test a distinct memory-lifecycle capability and preserve verifiable attribution.

### 30.2 Schema over-design delays usable results

**Mitigation:** Phase 0 implements only fields required by legacy migration and Pilot families; optional future fields remain metadata until exercised.

### 30.3 Realistic data becomes ambiguous

**Mitigation:** source-to-task compilation, deterministic checks, LLM-assisted filtering only, human audit, and rejection rather than forced labeling.

### 30.4 External systems lack introspection

**Mitigation:** capability levels; answer-only systems remain supplementary, while Level 2 is required for the main maintenance table.

### 30.5 Metric proliferation becomes unreadable

**Mitigation:** stable metric layers, required headline fields per family, slice-specific reports, and optional secondary composite only.

### 30.6 Legacy results conflict with new protocol

**Mitigation:** explicit legacy namespace, immutable source artifacts, separate run conditions, and no claim that legacy split results are vNext held-out results.

### 30.7 API/provider drift breaks reproducibility

**Mitigation:** model/provider metadata, prompt/config hashes, raw response metadata, timestamps, repeated runs where needed, and explicit alias caveats.

### 30.8 Engineering work outruns scientific questions

**Mitigation:** every Core/Full feature requires a task-family question, metric definition, oracle gate, and acceptance criterion.

## 31. Design Acceptance Checklist

This design is ready for implementation planning when all statements below are accepted:

- [x] MemUpdateBench vNext is a phased dynamic external-memory benchmark suite.
- [x] Repeated same-slot update remains the anchor family but is not the only task family.
- [x] The architecture is contract-first and preserves legacy P6/P8 assets through compilers.
- [x] Metrics remain decomposed by action, state, store, retrieval, answer, system, and audit layers.
- [x] New Core data must eliminate semantic-core split leakage.
- [x] External systems are compared according to declared capabilities.
- [x] Level 2 state export is required for the main memory-maintenance table.
- [x] Mechanism experiments become a versioned factor matrix.
- [x] The canonical scorer is shared across evaluation, filtering, visualization, and future reward callbacks.
- [x] Evaluation is incremental, resumable, and manifest-backed.
- [x] Realistic data uses source provenance, licensing/privacy controls, and compilation into canonical tasks.
- [x] Pilot, Core, and Full each have explicit acceptance gates.
- [x] RLVR and live API demos are not prerequisites for benchmark completion.
- [x] A single total score does not replace per-layer reporting.

## 32. Recommended Next Document

After this design is reviewed and approved, the next artifact should be a separate implementation plan. The implementation plan should decompose work into independently verifiable units, beginning with:

1. canonical schema package;
2. validator and deterministic reference executor;
3. scorer core and regression fixtures;
4. legacy P6/P8 compiler/importer;
5. profile/split builder;
6. built-in adapter wrappers;
7. incremental runtime and run manifest;
8. Pilot family compilers;
9. case export/viewer contract;
10. external Level 2 adapter feasibility.

No implementation is authorized by this design document alone; implementation begins only after the written design is reviewed and an implementation plan is approved.
