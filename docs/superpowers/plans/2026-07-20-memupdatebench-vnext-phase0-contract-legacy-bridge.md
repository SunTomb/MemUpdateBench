# MemUpdateBench vNext Phase 0 Contract and Legacy Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the versioned vNext contract, validation, scoring, adapter, manifest, and read-only legacy bridge needed to represent current P6/P8 artifacts without changing their historical meaning or making new benchmark claims.

**Architecture:** Add an isolated `mub.vnext` package beside the existing evaluator rather than refactoring `scripts/eval_evomemory.py`, `MemoryStore`, or `MemoryManager`. Pydantic v2 models are the in-repository normative contract; deterministic JSON Schema, canonical JSON hashing, JSONL records, and manifests are the external compatibility layer. Legacy loaders remain read-only, preserve raw fields and caveats, and compile only semantically equivalent fields into canonical records.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, PyYAML, JSON/JSONL, hashlib, pathlib, existing MemUpdateBench parsers and metrics where semantics match.

---

## 0. Execution boundaries and milestone gate

This plan implements **Phase 0 only**. It does not generate the vNext Pilot, evaluate new models, call external APIs, alter paper numbers, or claim leakage-free vNext performance.

Before implementation:

1. Work on a dedicated branch or isolated execution worktree selected by the user at execution handoff.
2. Preserve the current dirty workspace. Do not restore, stage, overwrite, or commit unrelated files.
3. Treat `data/`, `results/`, and `paper/` as immutable legacy inputs.
4. Write generated test outputs only under pytest temporary directories.
5. Never read API keys from transcripts or write secrets into artifacts.
6. Do not route vNext through obsolete `mub/baselines.py` or restore `agent.py`, `reward/`, `consolidation/`, `train_phase*.py`, `eval_locomo.py`, `run_ablations.py`, or `run_baselines.py`.
7. Keep Phase 0 read-only with respect to legacy source artifacts. Compatibility outputs are new derived artifacts, not rewrites of historical files.

Phase 0 is accepted only when:

- canonical schemas and JSON Schemas are deterministic;
- task replay and split validation are enforced;
- unsupported metrics remain `null` with a SupportReason and are excluded from denominators;
- representative P6.3/P6.5/P8.3/P8.4 fixtures import without semantic invention;
- missing, pending, partial, or capacity-failed artifacts cannot be reported as completed runs;
- legacy source files remain byte-identical;
- compatibility outputs are labeled as regression artifacts, not new benchmark results.

## 1. File structure map

### Dependency and test configuration

- Modify `requirements.txt`: add Pydantic v2 and pytest.
- Create `pytest.ini`: restrict discovery to `tests/` and define concise defaults.
- Create `tests/vnext/conftest.py`: fixture paths and immutable-source hash helper.

### Versioned contracts

- Create `mub/vnext/__init__.py`: stable public version exports.
- Create `mub/vnext/version.py`: schema/scorer/registry/compiler/profile version constants.
- Create `mub/vnext/contracts/enums.py`: controlled vocabularies whose exported schema allows future `task_family` strings.
- Create `mub/vnext/contracts/common.py`: source, generator, object-key, support-reason, and extension records.
- Create `mub/vnext/contracts/task.py`: `MemUpdateTask` and nested gold/split/metadata records.
- Create `mub/vnext/contracts/runtime.py`: normalized action/state/retrieval/answer runtime records.
- Create `mub/vnext/contracts/score.py`: score layers, failure fields, and `ScoreRecord`.
- Create `mub/vnext/contracts/manifest.py`: task/run manifests and `ScorerConfig`.
- Create `mub/vnext/contracts/adapter.py`: adapter information, capability bitset, typed call results, and protocol.
- Create `mub/vnext/contracts/__init__.py`: supported public contract exports.

### Serialization, schemas, validation, and scoring

- Create `mub/vnext/io/canonical.py`: canonical JSON bytes and SHA-256 identities.
- Create `mub/vnext/io/jsonl.py`: typed incremental JSONL reader/writer.
- Create `mub/vnext/schema_export.py`: deterministic top-level JSON Schema export.
- Create `mub/vnext/validation/issues.py`: typed validation issues and reports.
- Create `mub/vnext/validation/task.py`: structural, semantic, and distractor task validation.
- Create `mub/vnext/validation/replay.py`: deterministic gold-action executor.
- Create `mub/vnext/validation/split.py`: leakage, split exception, and manifest validation.
- Create `mub/vnext/scoring/registry.py`: machine-readable metric definitions.
- Create `mub/vnext/scoring/failures.py`: independent flags and primary precedence.
- Create `mub/vnext/scoring/scorer.py`: capability-aware canonical scorer.

### Read-only legacy bridge

- Create `mub/vnext/legacy/caveats.py`: explicit phase/run caveat registry.
- Create `mub/vnext/legacy/names.py`: isolated fallback parsing of legacy directory names.
- Create `mub/vnext/legacy/loaders.py`: typed read-only JSON/CSV loaders.
- Create `mub/vnext/legacy/dataset.py`: P6/P8 episode-to-task compiler.
- Create `mub/vnext/legacy/results.py`: legacy result-to-runtime/score importer.
- Create `mub/vnext/legacy/mechanisms.py`: P8.3/P8.4 probe importers.
- Create `mub/vnext/legacy/ledger.py`: paper-ledger path audit with unresolved references.
- Create `mub/vnext/legacy/__init__.py`: public bridge entry points.

### CLIs, schemas, fixtures, and documentation

- Create `scripts/vnext_export_schemas.py`: JSON Schema export CLI.
- Create `scripts/vnext_compile_legacy.py`: read-only compiler/import CLI.
- Create `scripts/vnext_validate_artifacts.py`: task/run/score/manifest validation CLI.
- Create `schemas/vnext/`: committed generated JSON Schemas.
- Create `tests/vnext/fixtures/legacy/`: small hand-curated immutable fixtures, never copied wholesale from private/raw stores.
- Create focused `tests/vnext/test_*.py` files named in the tasks below.
- Create `docs/vnext/legacy_bridge.md`: compatibility boundaries and caveats.
- Modify `scripts/smoke_test.py`: append a no-network vNext import/round-trip smoke section only after the pytest suite is green.
- Modify `WORKFLOW.md`: append implementation commands and validation status only after Phase 0 is actually implemented.

Existing `mub/memory/entry.py`, `mub/memory/store.py`, `mub/manager/memory_manager.py`, `scripts/prepare_data.py`, and `scripts/eval_evomemory.py` remain unchanged in Phase 0.

## 2. Task breakdown

### Task 1: Add the Phase 0 dependency and test scaffold

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`
- Create: `tests/vnext/__init__.py`
- Create: `tests/vnext/conftest.py`
- Create: `tests/vnext/test_environment.py`

- [ ] **Step 1: Write the failing dependency/version test**

Create `tests/vnext/test_environment.py`:

```python
from pydantic import VERSION


def test_pydantic_v2_is_active() -> None:
    assert int(VERSION.split(".", 1)[0]) == 2
```

- [ ] **Step 2: Run the test and verify the missing test dependency or unsupported Pydantic version is visible**

Run:

```bash
python -m pytest tests/vnext/test_environment.py -v
```

Expected before installation: pytest/Pydantic import failure, or a failing major-version assertion.

- [ ] **Step 3: Add bounded development dependencies**

Append to `requirements.txt`:

```text
pydantic>=2.7.0,<3.0.0
pytest>=8.2.0,<9.0.0
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -ra
```

Create `tests/vnext/__init__.py` as an empty file.

Create `tests/vnext/conftest.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_root() -> Path:
    return FIXTURE_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 4: Install dependencies and rerun the test**

Run:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/vnext/test_environment.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Record an isolated checkpoint**

If execution-time commit permission is active, stage only the files named in this task and commit:

```bash
git add requirements.txt pytest.ini tests/vnext/__init__.py tests/vnext/conftest.py tests/vnext/test_environment.py
git commit -m "test: add vnext contract test scaffold"
```

Otherwise record `git diff -- requirements.txt pytest.ini tests/vnext` in the execution log and continue without committing.

### Task 2: Define versions, enums, and shared records

**Files:**
- Create: `mub/vnext/__init__.py`
- Create: `mub/vnext/version.py`
- Create: `mub/vnext/contracts/__init__.py`
- Create: `mub/vnext/contracts/enums.py`
- Create: `mub/vnext/contracts/common.py`
- Test: `tests/vnext/test_common_contracts.py`

- [ ] **Step 1: Write failing enum and object-key tests**

Create `tests/vnext/test_common_contracts.py` with tests that assert:

```python
from pydantic import ValidationError
import pytest

from mub.vnext.contracts.common import MemoryObjectKey, SourceRecord
from mub.vnext.contracts.enums import Difficulty, EvaluationMode, QueryType, SourceType, Split, SupportReason


def test_design_vocabularies_are_exact() -> None:
    assert [item.value for item in SourceType] == ["synthetic", "dialogue", "changelog", "calendar", "issue", "report_revision", "other"]
    assert [item.value for item in Split] == ["train", "dev", "test", "evaluation_only"]
    assert [item.value for item in Difficulty] == ["easy", "medium", "hard", "challenge"]
    assert [item.value for item in QueryType] == ["current_state", "historical_state", "transition", "multi_object", "deletion_compliance"]
    assert [item.value for item in EvaluationMode] == ["state_direct", "retrieved_prompt", "native_system"]
    assert [item.value for item in SupportReason] == ["not_applicable", "not_supported", "runtime_failed", "missing_artifact"]


def test_source_record_uses_design_fields() -> None:
    record = SourceRecord(
        source_id="source_fixture_001",
        source_type=SourceType.SYNTHETIC,
        source_uri=None,
        license_or_privacy="synthetic-fixture",
        raw_hash=None,
        normalized_hash="0" * 64,
        normalization_version="norm-v1",
        provenance={"source": "fixture"},
        generator={"name": "fixture", "seed": 7},
    )
    assert set(record.model_dump(mode="json")) == {
        "source_id", "source_type", "source_uri", "license_or_privacy", "raw_hash", "normalized_hash", "normalization_version", "provenance", "generator",
    }


def test_object_key_fields_and_delimiter_safe_canonical_id() -> None:
    key = MemoryObjectKey(namespace="default", entity="friend|alex", attribute="location", subkey=None, object_type="slot")
    assert set(key.model_dump(mode="json")) == {"namespace", "entity", "attribute", "subkey", "object_type"}
    assert key.canonical_id == "default|friend%7Calex|location|"


def test_object_key_rejects_blank_identity_parts() -> None:
    with pytest.raises(ValidationError):
        MemoryObjectKey(namespace="default", entity=" ", attribute="location", subkey=None, object_type="slot")
```

- [ ] **Step 2: Run the test and verify imports fail**

Run:

```bash
python -m pytest tests/vnext/test_common_contracts.py -v
```

Expected: collection fails because `mub.vnext` does not exist.

- [ ] **Step 3: Add version constants and controlled enums**

Create `mub/vnext/version.py` with `SCHEMA_VERSION`, `RUNTIME_RECORD_VERSION`, `SCORER_VERSION`, `METRIC_REGISTRY_VERSION`, `COMPILER_VERSION`, `PROFILE_VERSION`, `TASK_MANIFEST_VERSION`, `RUN_MANIFEST_VERSION`, and `PRIMARY_FAILURE_PRECEDENCE_VERSION`, all initially set to `"1.0.0"`.

Create `mub/vnext/contracts/enums.py` with `Operation`, `ActionScope`, `EventRole`, `SourceType`, `Split`, `Difficulty`, `QueryType`, `AnswerSchema`, `EvaluationMode`, `CompletionStatus`, and `SupportReason`. Exact values are:

```text
Operation: ADD, UPDATE, NOOP, DELETE
ActionScope: object, attribute, entity, namespace, ttl
EventRole: latest_gold, stale_same_slot, duplicate_current, same_entity_other_attribute, same_name_other_entity, noop_near_miss, neutral, deletion, historical_support
SourceType: synthetic, dialogue, changelog, calendar, issue, report_revision, other
Split: train, dev, test, evaluation_only
Difficulty: easy, medium, hard, challenge
QueryType: current_state, historical_state, transition, multi_object, deletion_compliance
AnswerSchema: string, number, boolean, list, object
EvaluationMode: state_direct, retrieved_prompt, native_system
CompletionStatus: completed, failed, partial, not_supported
SupportReason: not_applicable, not_supported, runtime_failed, missing_artifact
```

`task_family` is not a closed exported enum field; a helper controlled list may exist only if `MemUpdateTask` exports `task_family` as `str` and accepts future family names.

- [ ] **Step 4: Implement shared Pydantic records**

Create `mub/vnext/contracts/common.py` with `ContractModel(extra="forbid")`, `SourceRecord`, `MemoryObjectKey`, `MetricFieldSupport`, and `RawExtension`. Exact public fields:

```text
SourceRecord: source_id, source_type, source_uri, license_or_privacy, raw_hash, normalized_hash, normalization_version, provenance, generator
MemoryObjectKey: namespace, entity, attribute, subkey, object_type
MetricFieldSupport: reason, null_policy, detail
RawExtension: namespace, payload
```

`MemoryObjectKey.canonical_id` is a non-serialized convenience property. It must percent-escape delimiter characters before joining `(namespace, entity, attribute, subkey or "")` with `|`. Serialization for canonical artifacts must exclude computed fields.

Create `mub/vnext/__init__.py` and `mub/vnext/contracts/__init__.py` to export version constants and contract classes without importing legacy loaders.

- [ ] **Step 5: Run the contract tests**

Run:

```bash
python -m pytest tests/vnext/test_common_contracts.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext tests/vnext/test_common_contracts.py
git commit -m "feat: define vnext shared contracts"
```

Commit only when execution-time permission is active; otherwise record the scoped diff.

### Task 3: Define canonical task contracts

**Files:**
- Create: `mub/vnext/contracts/task.py`
- Create: `tests/vnext/factories.py`
- Modify: `mub/vnext/contracts/__init__.py`
- Modify: `tests/vnext/conftest.py`
- Test: `tests/vnext/test_task_contract.py`

- [ ] **Step 1: Write failing round-trip and referential-integrity tests**

Create `tests/vnext/test_task_contract.py` with fixture-backed tests for canonical task fields:

```python
from pydantic import ValidationError
import pytest

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import AnswerSchema, Difficulty, EvaluationMode, Operation, QueryType, Split
from mub.vnext.contracts.task import MemUpdateTask


def test_task_round_trip_preserves_design_fields(make_task) -> None:
    task = make_task()
    restored = MemUpdateTask.model_validate_json(task.model_dump_json())
    assert restored == task
    assert restored.task_family == "repeated_same_slot_update"
    assert restored.gold.action_sequence == ["action_0", "action_1"]
    assert restored.gold.actions[1].operation is Operation.UPDATE


def test_future_task_family_string_is_accepted(make_task) -> None:
    payload = make_task().model_dump(mode="json")
    payload["task_family"] = "future_family_from_next_release"
    assert MemUpdateTask.model_validate(payload).task_family == "future_family_from_next_release"


def test_memory_query_has_no_inline_gold_answer(make_task) -> None:
    payload = make_task().model_dump(mode="json")
    payload["queries"][0]["inline_gold_answer"] = "Qingdao"
    with pytest.raises(ValidationError):
        MemUpdateTask.model_validate(payload)


def test_action_sequence_covers_every_action_once(make_task) -> None:
    payload = make_task().model_dump(mode="json")
    payload["gold"]["action_sequence"] = ["action_0", "action_0"]
    with pytest.raises(ValidationError, match="action_sequence"):
        MemUpdateTask.model_validate(payload)


def test_event_action_and_gold_source_references_must_exist(make_task) -> None:
    payload = make_task().model_dump(mode="json")
    payload["events"][0]["gold_action_ids"] = ["missing"]
    with pytest.raises(ValidationError, match="missing gold action"):
        MemUpdateTask.model_validate(payload)
    payload = make_task().model_dump(mode="json")
    payload["gold"]["gold_source_event_ids"] = ["missing_event"]
    with pytest.raises(ValidationError, match="gold_source_event_ids"):
        MemUpdateTask.model_validate(payload)


def test_declared_targets_cover_actions_queries_and_expected_absence(make_task) -> None:
    payload = make_task().model_dump(mode="json")
    payload["queries"][0]["target_object_keys"] = [
        MemoryObjectKey(namespace="default", entity="friend:alex", attribute="pet", subkey=None, object_type="slot").model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match="query target"):
        MemUpdateTask.model_validate(payload)


def test_canonical_query_and_profile_values(make_task) -> None:
    task = make_task()
    assert task.difficulty is Difficulty.EASY
    assert task.metadata.profile_name is Difficulty.EASY
    assert task.metadata.split is Split.TEST
    assert task.queries[0].query_type is QueryType.CURRENT_STATE
    assert task.queries[0].answer_schema is AnswerSchema.STRING
    assert task.queries[0].evaluation_mode is EvaluationMode.RETRIEVED_PROMPT
```

Create `tests/vnext/factories.py` with a deterministic ADD then UPDATE fixture. The fixture must populate `source`, `events`, `target_objects`, `queries`, `gold.actions`, `gold.action_sequence`, `gold.final_state`, `gold.version_history`, `gold.expected_present_objects`, `gold.expected_absent_objects`, `gold.gold_source_event_ids`, `gold.gold_answers`, `gold.acceptable_answers`, and `metadata.split_key` with required `source_group_id` and `trajectory_id`.

Append to `tests/vnext/conftest.py`:

```python
from tests.vnext.factories import build_task


@pytest.fixture
def make_task():
    return build_task
```

- [ ] **Step 2: Run the tests and verify the task module is missing**

Run:

```bash
python -m pytest tests/vnext/test_task_contract.py -v
```

Expected: import failure for `mub.vnext.contracts.task`.

- [ ] **Step 3: Implement exact task models**

Create `mub/vnext/contracts/task.py` with these fields:

```text
SplitKey: semantic_core_id, source_group_id, trajectory_id, paraphrase_group_id, source_document_id, version_group_id, split_exception_id, split_policy_version
LegacyProvenance: legacy_family_id, legacy_phase, legacy_dataset_id, legacy_split_id, legacy_metric_namespace, legacy_run_condition_id, checkpoint_family, training_seed, answer_mode, memory_trajectory_id, source_artifact_path, source_artifact_hash, known_caveats
TaskMetadata: split, split_key, profile_name, resolved_profile, generation_config_hash, compiler_version, tags, legacy_provenance, extra
GoldAction: action_id, event_id, operation, scope, target_object_keys, value, effective_at, expected_effect
MemoryEvent: event_id, sequence_index, timestamp, raw_text, normalized_text, speaker, gold_action_ids, role, source_anchor, metadata
MemoryQuery: query_id, query_type, text, target_object_keys, answer_schema, evaluation_mode, metadata
GoldRecord: actions, action_sequence, final_state, version_history, expected_present_objects, expected_absent_objects, gold_source_event_ids, gold_answers, acceptable_answers
MemUpdateTask: task_id, schema_version, task_family, difficulty, source, events, target_objects, queries, gold, metadata
```

`SplitKey.source_group_id` and `SplitKey.trajectory_id` are required strings. The other documented group fields are nullable. `TaskMetadata.profile_name` uses the `easy`/`medium`/`hard`/`challenge` difficulty values. `GoldAction.effective_at` is `str | None`. `MemoryEvent.source_anchor` is `dict[str, Any]` and must not be narrowed by a helper that rejects arbitrary valid anchor metadata.

Add validators enforcing operation shape, unique IDs, ordered sequence indices, action/event references, complete action sequence coverage, declared object targets, query target coverage, gold source-event references, and query references in `gold_answers` and `acceptable_answers`.

- [ ] **Step 4: Run task contract tests**

Run:

```bash
python -m pytest tests/vnext/test_task_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Record an isolated checkpoint**

```bash
git add mub/vnext/contracts/task.py mub/vnext/contracts/__init__.py tests/vnext/conftest.py tests/vnext/factories.py tests/vnext/test_task_contract.py
git commit -m "feat: define canonical vnext task schema"
```

### Task 4: Define runtime and adapter contracts

**Files:**
- Create: `mub/vnext/contracts/runtime.py`
- Create: `mub/vnext/contracts/adapter.py`
- Modify: `mub/vnext/contracts/__init__.py`
- Modify: `tests/vnext/factories.py`
- Modify: `tests/vnext/conftest.py`
- Test: `tests/vnext/test_runtime_adapter_contracts.py`

- [ ] **Step 1: Write failing capability and runtime tests**

Create `tests/vnext/test_runtime_adapter_contracts.py`:

```python
from mub.vnext.contracts.adapter import AdapterCapabilities, AnswerResult, ResetResult
from mub.vnext.contracts.enums import CompletionStatus, Operation
from mub.vnext.contracts.runtime import ParsedManagerAction, TaskRunRecord


def test_runtime_record_top_level_fields_match_design(make_task_run) -> None:
    run = make_task_run(status=CompletionStatus.FAILED, exception_type="timeout")
    assert set(run.model_dump(mode="json")) == {
        "schema_version", "runtime_record_version", "task_id", "adapter_id", "run_id",
        "parsed_actions", "memory_snapshots", "retrieval_traces", "answer_predictions",
        "system_events", "parser_extractor_provenance", "exceptions", "completion_status",
    }
    assert run.exceptions[0]["type"] == "timeout"


def test_parsed_manager_action_uses_design_fields(make_object_key) -> None:
    action = ParsedManagerAction(
        event_id="event_0",
        operation=Operation.UPDATE,
        target_object_key=make_object_key(),
        value="Qingdao",
        format_valid=True,
        execution_status="succeeded",
        fallback_used=False,
        error_flags=[],
        raw_output="UPDATE friend:alex.location = Qingdao",
        latency_ms=1.5,
    )
    assert action.target_object_key.attribute == "location"


def test_capability_levels_are_derived_shortcuts() -> None:
    l0 = AdapterCapabilities(supports_native_answer=True)
    l1 = l0.model_copy(update={"exports_retrieval_ids": True})
    l2 = l1.model_copy(update={"supports_isolated_reset": True, "exports_entries": True, "requires_evaluation_extractor": True})
    l3 = l2.model_copy(update={"exports_action_trace": True})
    assert [cap.presentation_level(state_transition_linkage_available=(cap is l3)) for cap in [l0, l1, l2, l3]] == [0, 1, 2, 3]


def test_adapter_result_records_are_direct_records() -> None:
    reset = ResetResult(success=True, namespace="fixture", error=None)
    answer = AnswerResult(query_id="query_0", raw_output="Qingdao", usage={"output_tokens": 3}, cost=0.01, latency_ms=4.0, error=None)
    assert reset.success is True
    assert answer.raw_output == "Qingdao"
```

Append task-run and object-key builders to `tests/vnext/factories.py` and expose `make_object_key` and `make_task_run` fixtures in `tests/vnext/conftest.py`.

- [ ] **Step 2: Run and verify imports fail**

```bash
python -m pytest tests/vnext/test_runtime_adapter_contracts.py -v
```

Expected: missing runtime/adapter module error.

- [ ] **Step 3: Implement normalized runtime records with design names**

`mub/vnext/contracts/runtime.py` must define exactly these public model names and fields:

```text
ParsedManagerAction: event_id, operation, target_object_key, value, format_valid, execution_status, fallback_used, error_flags, raw_output, latency_ms
MemoryEntryRecord: entry_id, content, object_key_candidate, value_candidate, created_at, updated_at, source_event_ids, version_index, raw_metadata
MemorySnapshot: after_event_id, entries, state_by_object, store_size, raw_adapter_state, snapshot_hash
RetrievalTrace: query_id, retrieved_entries, scores, ranks, gold_in_context, stale_in_context, distractor_in_context, retrieval_policy, context_order, version_metadata, prompt_hash
AnswerPrediction: query_id, raw_output, parsed_answer, cited_event_ids, cited_entry_ids, format_valid, error_flags, latency_ms, usage
ParserExtractorProvenance: action_parser_version, answer_parser_version, memory_entry_extractor_version, object_value_extractor_config_hash, redaction_policy_version, raw_provider_artifact_path, raw_provider_artifact_hash, raw_adapter_state_path, raw_adapter_state_hash
TaskRunRecord: schema_version, runtime_record_version, task_id, adapter_id, run_id, parsed_actions, memory_snapshots, retrieval_traces, answer_predictions, system_events, parser_extractor_provenance, exceptions, completion_status
```

`ParsedManagerAction.execution_status` is `str` in Phase 0 because the approved design names the field but does not define a controlled vocabulary. Do not add an `ExecutionStatus` enum until a future schema version explicitly defines its values.

Use `Field(default_factory=list)` or `Field(default_factory=dict)` for every mutable default. Do not define alternate public wrapper models for actions, retrievals, or answers.

- [ ] **Step 4: Implement the adapter bitset, result records, and protocol**

`mub/vnext/contracts/adapter.py` must define `AdapterInfo`, `AdapterCapabilities`, `ResetResult`, `AdapterActionLog`, `RetrievalResult`, `AnswerResult`, and a `MemoryAdapter` `Protocol`.

`AdapterInfo` fields are exactly `adapter_id`, `adapter_version`, `system_name`, `system_version`, `sdk_version`, `configuration_hash`, `extractor_id`, and `extractor_version`.

`AdapterCapabilities` fields are the exact design bitset: `supports_isolated_reset`, `supports_event_ingest`, `supports_add`, `supports_update`, `supports_noop`, `supports_delete`, `supports_ttl`, `supports_native_answer`, `exports_entries`, `exports_raw_state`, `exports_source_event_ids`, `exports_timestamps_or_order`, `exports_object_keys`, `exports_values`, `exports_retrieval_ids`, `exports_retrieval_scores`, `exports_action_trace`, `reports_latency`, `reports_token_usage`, `reports_cost`, `requires_evaluation_extractor`, and `extractor_version`. The metric bitset remains authoritative. Level 3 additionally requires verified state-transition linkage from the capability-verification artifact or run traces; do not add a non-design bit to `AdapterCapabilities`.

Adapter result fields are exactly:

```text
ResetResult: success, namespace, error
AdapterActionLog: event_id, requested_operation, effective_operation, affected_entry_ids, raw_action, latency_ms, error
RetrievalResult: query_id, entries, scores, raw_result, latency_ms, error
AnswerResult: query_id, raw_output, usage, cost, latency_ms, error
```

`MemoryAdapter` protocol method signatures are exactly:

```python
class MemoryAdapter(Protocol):
    def adapter_info(self) -> AdapterInfo:
        raise NotImplementedError

    def capabilities(self) -> AdapterCapabilities:
        raise NotImplementedError

    def reset(self, namespace: str, config: dict) -> ResetResult:
        raise NotImplementedError

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        raise NotImplementedError

    def export_entries(self) -> list[MemoryEntryRecord]:
        raise NotImplementedError

    def export_raw_state(self) -> object:
        raise NotImplementedError

    def retrieve(self, query: MemoryQuery, k: int) -> RetrievalResult:
        raise NotImplementedError

    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

Capability levels are display shortcuts: L0 requires native answer; L1 retrieval export; L2 entries plus isolated reset plus extractable content; L3 action trace plus verified state-transition linkage.

- [ ] **Step 5: Run runtime/adapter tests**

```bash
python -m pytest tests/vnext/test_runtime_adapter_contracts.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/contracts tests/vnext/test_runtime_adapter_contracts.py tests/vnext/conftest.py tests/vnext/factories.py
git commit -m "feat: define vnext runtime and adapter contracts"
```

### Task 5: Define score layers, manifests, and scorer configuration

**Files:**
- Create: `mub/vnext/contracts/score.py`
- Create: `mub/vnext/contracts/manifest.py`
- Modify: `mub/vnext/contracts/__init__.py`
- Modify: `tests/vnext/factories.py`
- Modify: `tests/vnext/conftest.py`
- Test: `tests/vnext/test_score_manifest_contracts.py`

- [ ] **Step 1: Write failing unsupported-metric and manifest tests**

Create tests that assert unsupported values are `None`, reasons use exact SupportReason values, top-level score fields match the design, and run-manifest counts reconcile:

```python
from mub.vnext.contracts.common import MetricFieldSupport
from mub.vnext.contracts.enums import SupportReason
from mub.vnext.contracts.score import ScoreRecord


def test_unsupported_metric_is_null_with_reason(make_score_record) -> None:
    score = make_score_record(
        retrieval_scores={"current_mrr": None},
        supported_metric_fields={"retrieval_scores.current_mrr": MetricFieldSupport(reason=SupportReason.NOT_SUPPORTED, null_policy="exclude_from_mean")},
    )
    assert score.retrieval_scores.current_mrr is None
    assert score.supported_metric_fields["retrieval_scores.current_mrr"].reason is SupportReason.NOT_SUPPORTED


def test_score_record_top_level_fields_match_design(make_score_record) -> None:
    score = make_score_record()
    assert isinstance(score, ScoreRecord)
    assert set(score.model_dump(mode="json")) == {
        "schema_version", "scorer_version", "task_id", "run_id", "adapter_id", "task_family", "difficulty", "completion_status",
        "supported_metric_fields", "protocol_scores", "action_scores", "state_scores", "store_scores", "retrieval_scores",
        "answer_scores", "system_scores", "audit_scores", "failure_flags", "primary_failure", "legacy_metrics",
    }


def test_run_manifest_counts_cover_expected_tasks(make_run_manifest) -> None:
    manifest = make_run_manifest(expected=10, completed=7, failed=2, not_supported=1)
    assert manifest.completed_task_count + manifest.failed_task_count + manifest.not_supported_task_count == manifest.expected_task_count
```

Append concrete score and manifest builders to `tests/vnext/factories.py`, and expose `make_score_record` and `make_run_manifest` in `tests/vnext/conftest.py`.

- [ ] **Step 2: Run and verify imports fail**

```bash
python -m pytest tests/vnext/test_score_manifest_contracts.py -v
```

Expected: missing score/manifest modules.

- [ ] **Step 3: Implement typed score layers and `ScoreRecord`**

Create score-layer models with optional values so capability gating can use `null`. The fields must be exactly these canonical metric fields:

```text
protocol_scores: action_parse_valid, answer_parse_valid, execution_success_rate, unsupported_operation_rate, fallback_rate
action_scores: operation_accuracy, full_action_exact_match, object_key_accuracy, entity_accuracy, attribute_accuracy, value_accuracy, false_write_rate, missed_write_rate, wrong_object_write_rate
state_scores: final_state_accuracy, state_precision, state_recall, state_f1, state_resolve_rate, collateral_corruption_rate, expected_absence_accuracy
store_scores: obsolete_version_count, stale_conflicting_value_count, duplicate_current_count, final_memory_size, compaction_ratio, write_amplification
retrieval_scores: current_recall_at_k, current_mrr, stale_exposure_rate, stale_count_in_context, distractor_exposure_rate
answer_scores: exact_match, normalized_match, token_f1, structured_field_accuracy, stale_copied, distractor_copied, gold_retrieved_wrong_answer, answer_state_consistency
system_scores: ingest_latency_ms, retrieval_latency_ms, answer_latency_ms, token_usage, api_cost, error_rate
audit_scores: action_trace_available, state_export_available, retrieval_trace_available, source_provenance_coverage, manifest_completeness
```

`ScoreRecord` fields are exactly `schema_version`, `scorer_version`, `task_id`, `run_id`, `adapter_id`, `task_family`, `difficulty`, `completion_status`, `supported_metric_fields`, all eight score layers, `failure_flags`, `primary_failure`, and `legacy_metrics`. `supported_metric_fields` maps fully qualified score field names to a reason/null-policy record using only `not_applicable`, `not_supported`, `runtime_failed`, and `missing_artifact`.

- [ ] **Step 4: Implement `TaskManifest`, `RunManifest`, and `ScorerConfig`**

`TaskManifest` fields are exactly design section 7.12: `schema_version`, `task_manifest_version`, `data_release_id`, `split_policy_version`, `task_schema_version`, `compiler_versions`, `source_manifest_paths_and_hashes`, `generation_configs_and_hashes`, `split_counts`, `family_difficulty_counts`, `semantic_core_counts`, `task_file_paths_and_hashes`, `leakage_check_summary`, `human_audit_artifacts`, `created_at`, and `code_revision`.

`RunManifest` must represent all design section 18.1 identities: schema and run-manifest versions; run ID and timestamp; code revision and dirty-state flag; task manifest path/hash; schema, scorer, metric-registry, and profile versions; adapter information and capability bitset; capability-verification artifact path/hash; model/provider/revision; prompt and decoding config; seed information; action parser version; answer parser version; memory-entry extractor version and config hash; redaction policy version; environment/package summary; expected/completed/failed/not-supported task counts; raw provider response paths/hashes; raw adapter state paths/hashes; normalized runtime artifact paths/hashes; score artifact paths/hashes; and native-versus-extracted field summary.

`ScorerConfig` fields are exactly `scorer_version`, `metric_registry_version`, `value_normalization_profile`, `answer_normalization_profile`, `primary_failure_precedence_version`, `requested_metric_fields`, `legacy_compatibility_mode`, and `strict_capability_check`.

- [ ] **Step 5: Run score/manifest tests**

```bash
python -m pytest tests/vnext/test_score_manifest_contracts.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/contracts tests/vnext/test_score_manifest_contracts.py tests/vnext/conftest.py tests/vnext/factories.py
git commit -m "feat: define vnext score and manifest contracts"
```

### Task 6: Add canonical JSON, JSONL, hashing, and schema export

**Contract correction:** Serialization uses the corrected top-level schemas from Tasks 2-5. Semantic task hashing follows design section 7.11: family semantics, normalized source anchors, object keys, gold actions/values/roles, query types, and gold state are included; split labels, run identity, and surface-only paraphrase text are excluded.

**Files:**
- Create: `mub/vnext/io/__init__.py`
- Create: `mub/vnext/io/canonical.py`
- Create: `mub/vnext/io/jsonl.py`
- Create: `mub/vnext/schema_export.py`
- Create: `scripts/vnext_export_schemas.py`
- Create: `schemas/vnext/.gitkeep`
- Test: `tests/vnext/test_serialization.py`
- Test: `tests/vnext/test_schema_export.py`

- [ ] **Step 1: Write failing deterministic serialization tests**

Tests must prove canonical JSON ignores dictionary insertion order, excludes non-serialized computed convenience properties, preserves Unicode, and writes stable JSONL rows. A semantic-hash test must prove that a surface paraphrase with unchanged source anchors, objects, gold, roles, query types, and gold state keeps the same semantic hash.

- [ ] **Step 2: Run and verify imports fail**

```bash
python -m pytest tests/vnext/test_serialization.py -v
```

Expected: missing serialization modules.

- [ ] **Step 3: Implement canonical JSON and incremental JSONL**

`canonical_json_bytes` must compute `payload = model.model_dump(mode="json", exclude_none=False, exclude_computed_fields=True)`, then return `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`. `sha256_model` hashes those bytes. `semantic_task_hash(task)` computes the design semantic identity and documents its projection. `write_models` and `read_models` operate one row at a time and report duplicate IDs or line-numbered validation errors.

- [ ] **Step 4: Write and implement schema-export tests**

Test that schemas for `MemUpdateTask`, `TaskRunRecord`, `ScoreRecord`, `TaskManifest`, and `RunManifest` export twice with identical bytes and each has a title and schema-version property. `scripts/vnext_export_schemas.py` accepts `--output-dir` and performs no network or source-artifact access.

- [ ] **Step 5: Run serialization and schema tests**

```bash
python -m pytest tests/vnext/test_serialization.py tests/vnext/test_schema_export.py -v
python scripts/vnext_export_schemas.py --output-dir schemas/vnext
```

Expected: tests pass and five deterministic schema files are generated.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/io mub/vnext/schema_export.py scripts/vnext_export_schemas.py schemas/vnext tests/vnext/test_serialization.py tests/vnext/test_schema_export.py
git commit -m "feat: add canonical vnext serialization and schemas"
```

### Task 7: Implement structural validation and deterministic gold replay

**Files:**
- Create: `mub/vnext/validation/__init__.py`
- Create: `mub/vnext/validation/issues.py`
- Create: `mub/vnext/validation/task.py`
- Create: `mub/vnext/validation/replay.py`
- Test: `tests/vnext/test_task_validation.py`
- Test: `tests/vnext/test_gold_replay.py`

- [ ] **Step 1: Write failing structural validation tests**

Tests must reject unsupported schema version, duplicate IDs, unordered sequence indices, missing event/action/source references, missing required family metadata in `resolved_profile`, action/value type incompatibility, current-state queries without targets, deletion-compliance targets that are neither declared nor expected absent, a declared target absent from `target_objects`, and an expected-absent object that appears in final state. A valid task returns `ValidationReport(valid=True, issues=[])`.

- [ ] **Step 2: Implement typed issues and structural validator**

Use:

```python
class ValidationIssue(ContractModel):
    code: str
    message: str
    path: str
    severity: Literal["error", "warning"]


class ValidationReport(ContractModel):
    valid: bool
    issues: list[ValidationIssue]
```

`validate_task(task)` implements design section 17.1 plus target/expected-absence rules. It performs cross-field checks outside Pydantic construction and never mutates the task.

- [ ] **Step 3: Write failing replay and distractor tests**

Cover ADD to UPDATE, NOOP non-mutation, duplicate-current update history, DELETE of exactly enumerated objects, expected-present and expected-absent checks, action-sequence replay, historical query resolution, stale event obsolescence, duplicate-current not counted as conflicting stale, distractor event not independently establishing the accepted current answer, and unique query answer support.

- [ ] **Step 4: Implement the pure reference executor and semantic validators**

Define `ReplayResult(final_state, version_history, mutation_count)` and `replay_actions(actions)` as a pure function over ordered `GoldAction` records. `validate_gold_replay(task)` compares replay output to `task.gold.final_state`, `task.gold.version_history`, `expected_present_objects`, and `expected_absent_objects`; verifies historical query answers against version history; and emits field-specific issues. `validate_distractors(task)` enforces design section 17.3 stale/duplicate/distractor uniqueness semantics.

- [ ] **Step 5: Run validation/replay tests**

```bash
python -m pytest tests/vnext/test_task_validation.py tests/vnext/test_gold_replay.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/validation tests/vnext/test_task_validation.py tests/vnext/test_gold_replay.py
git commit -m "feat: validate and replay vnext tasks"
```

### Task 8: Implement profile and split validation contracts

**Files:**
- Create: `mub/vnext/profiles.py`
- Create: `mub/vnext/validation/split.py`
- Test: `tests/vnext/test_profiles.py`
- Test: `tests/vnext/test_split_validation.py`

- [ ] **Step 1: Write failing strict-profile tests**

Define fixture profiles for `easy`, `medium`, `hard`, and `challenge` with canonical controls `update_depth`, `active_object_count`, `entity_ambiguity`, `attribute_ambiguity`, `noop_density`, `cross_slot_interleaving`, `stale_count`, `context_length`, `context_order`, `version_metadata`, `query_type`, and `source_naturalness`. Assert unknown overrides fail, allowed overrides are reflected in `resolved_profile`, canonical family/difficulty labels cannot be changed by overrides, and hard/challenge defaults are locked by regression assertions.

- [ ] **Step 2: Implement strict profile models and resolver**

Create `ProfileSpec(name, version, task_family, difficulty, parameters, allowed_overrides)` and `resolve_profile(profile, overrides)`. The resolver rejects unknown keys, rejects attempts to override `task_family`, `difficulty`, or `profile_name`, copies parameters deterministically, and records `task_family`, `difficulty`, `profile_name`, and `profile_version` in the resolved profile.

- [ ] **Step 3: Write failing split-leakage tests**

Construct tasks with semantic-core, trajectory, paraphrase, protected source-group, source-document, version-group, and exact-hash overlaps across train/dev/test. Assert group isolation is checked before stratification. Assert the minimum split stratum is exactly `(task_family, difficulty, update_depth_bucket)`. Assert Phase 0 recognizes the family-specific stratification axes for all design families, even though Pilot implements A-D first:

```text
Families A/B: update_depth_bucket, active-object count, interleaving level
Family C: entity ambiguity, attribute ambiguity, alias/namespace condition
Family D: NOOP/write-trap type, NOOP density, duplicate-current condition
Family E: deletion scope, relearning condition
Family F: query type, requested version distance
Family G: reasoning depth, active-object count
Family H: source type, provenance class
```

Assert `evaluation_only` pairs may share a declared non-null `split_exception_id` only when both tasks are outside training, and that exceptions have a version, rationale, allowed group IDs, and reviewer. Assert missing minimum strata, mismatched exact hashes/counts/manifests, and non-deterministic slices fail.

- [ ] **Step 4: Implement `validate_splits` and deterministic slice checks**

The function accepts tasks, declared exceptions, a task manifest, and deterministic slice definitions. It indexes every non-null `SplitKey` group, rejects train/dev/test leakage, applies evaluation-only exception semantics, verifies exact task hashes, verifies split counts and family-difficulty counts, enforces the minimum `(task_family, difficulty, update_depth_bucket)` stratum, records family-specific A-H strata coverage, and confirms per-slice exports are deterministic filters of the canonical aggregate set. It must not infer grouping from task IDs or paths.

- [ ] **Step 5: Run profile/split tests**

```bash
python -m pytest tests/vnext/test_profiles.py tests/vnext/test_split_validation.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/profiles.py mub/vnext/validation/split.py tests/vnext/test_profiles.py tests/vnext/test_split_validation.py
git commit -m "feat: enforce vnext profiles and split isolation"
```

### Task 9: Implement the metric registry, failure taxonomy, and scorer skeleton

**Files:**
- Create: `mub/vnext/scoring/__init__.py`
- Create: `mub/vnext/scoring/registry.py`
- Create: `mub/vnext/scoring/failures.py`
- Create: `mub/vnext/scoring/scorer.py`
- Test: `tests/vnext/test_metric_registry.py`
- Test: `tests/vnext/test_failure_taxonomy.py`
- Test: `tests/vnext/test_scorer_capabilities.py`

- [ ] **Step 1: Write failing registry completeness tests**

Flatten all optional fields from the eight score-layer models and assert each appears exactly once in the metric registry. Assert no unregistered metric field appears. Assert each definition provides layer, value type, numerator, denominator, aggregation, applicable task families, required capabilities, unsupported policy, runtime-failure policy, legacy aliases, and introduced version.

- [ ] **Step 2: Implement `MetricDefinition` and registry entries**

Use:

```python
class MetricDefinition(ContractModel):
    field_name: str
    layer: str
    value_type: str
    numerator_definition: str
    denominator_definition: str
    aggregation_rule: str
    applicable_task_families: list[str]
    required_adapter_capabilities: list[str]
    unsupported_value_policy: str
    runtime_failure_policy: str
    legacy_aliases: list[str]
    introduced_in_scorer_version: str
```

Define entries only for the canonical fields in Task 5. Legacy aliases are allowed only where semantics are identical; trace-composition stale-removal summaries and answer reruns must not share aliases. The registry is structurally aligned with design section 8.9 and rejects unregistered metric names requested by `ScorerConfig` in strict mode.

- [ ] **Step 3: Write and implement failure-precedence tests**

Independent failure flags are exactly:

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

`primary_failure(flags)` follows design precedence: execution/protocol, action/grounding, state, retrieval, answer/version arbitration, format-only, then the correct outcome when no independent failure flag is present. Tests must prove overlapping flags remain preserved while the primary label is stable.

- [ ] **Step 4: Write and implement the capability-aware scorer skeleton**

`score_task(task, run, capabilities, config)` must always fill protocol and audit layers when the run row is available; fill a metric only when its registry capabilities are true and required artifacts exist; set unsupported or non-applicable values to `None` and add `supported_metric_fields` entries with exact SupportReason values; keep failed tasks as score rows but mark affected accuracy metrics with `runtime_failed`; compare deterministic current-state answers without any model call; preserve unknown legacy values only under `legacy_metrics`; and derive independent failure flags without storing the correct outcome as a flag.

- [ ] **Step 5: Run registry/taxonomy/scorer tests**

```bash
python -m pytest tests/vnext/test_metric_registry.py tests/vnext/test_failure_taxonomy.py tests/vnext/test_scorer_capabilities.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/scoring tests/vnext/test_metric_registry.py tests/vnext/test_failure_taxonomy.py tests/vnext/test_scorer_capabilities.py
git commit -m "feat: add capability-aware vnext scorer contracts"
```

### Task 10: Add read-only legacy loaders and immutable golden fixtures

**Files:**
- Create: `mub/vnext/legacy/__init__.py`
- Create: `mub/vnext/legacy/loaders.py`
- Create: `tests/vnext/fixtures/legacy/p63_dataset_minimal.json`
- Create: `tests/vnext/fixtures/legacy/evomemory_results_old.json`
- Create: `tests/vnext/fixtures/legacy/evomemory_results_traced.json`
- Create: `tests/vnext/fixtures/legacy/p65_prompt_summary_minimal.json`
- Create: `tests/vnext/fixtures/legacy/p83_conflict_rows.csv`
- Create: `tests/vnext/fixtures/legacy/p83_synthetic_dose_rows.csv`
- Create: `tests/vnext/fixtures/legacy/p84_api_rows.csv`
- Test: `tests/vnext/test_legacy_loaders.py`

- [ ] **Step 1: Create minimal fixtures with explicit provenance**

Fixtures contain only synthetic two-to-four-row records needed for schema drift tests:

- P6.3: one k=1 and one k=2 episode, same-name distractor, semantic near-miss, and explicit `num_events`, `num_target_updates`, legacy `num_updates`.
- Old result: no answer top-k, context order, or answer trace.
- Traced result: one answer trace with retrieved entries and source event IDs.
- P6.5: summary-only retrieval rates.
- P8.3: CSV booleans represented as `True`, `False`, `1.0`, and `0.0`.
- P8.4: one clean row, one truncated/empty-response caveat row, and one capacity-failed row.
- Source/provenance: no fixture uses a legacy source type; legacy identity is represented by `LegacyProvenance` or legacy-analysis metadata.

Each fixture starts from hand-written synthetic content; do not copy API keys, private raw payloads, or large source artifacts.

- [ ] **Step 2: Write failing loader tests**

Assert loaders preserve raw payloads, distinguish absent fields from zero, normalize supported CSV booleans, reject malformed top-level types, and never modify the source file hash.

- [ ] **Step 3: Implement typed read-only loaders**

`mub/vnext/legacy/loaders.py` exposes:

```python
load_evomemory_dataset(path: Path) -> list[dict[str, Any]]
load_evomemory_results(path: Path) -> dict[str, Any]
load_json_summary(path: Path) -> dict[str, Any]
load_csv_rows(path: Path) -> list[dict[str, str]]
parse_legacy_bool(value: str | bool | int | float | None) -> bool | None
```

Every loader computes source SHA-256 before and after reading and raises if it changes. It validates required top-level shape but does not backfill missing optional fields.

- [ ] **Step 4: Run loader tests**

```bash
python -m pytest tests/vnext/test_legacy_loaders.py -v
```

Expected: all tests pass and fixture hashes remain unchanged.

- [ ] **Step 5: Record an isolated checkpoint**

```bash
git add mub/vnext/legacy tests/vnext/fixtures/legacy tests/vnext/test_legacy_loaders.py
git commit -m "feat: add immutable legacy artifact loaders"
```

### Task 11: Compile legacy P6 episodes into canonical tasks

**Files:**
- Create: `mub/vnext/legacy/caveats.py`
- Create: `mub/vnext/legacy/dataset.py`
- Modify: `mub/vnext/legacy/__init__.py`
- Test: `tests/vnext/test_legacy_dataset_compiler.py`

- [ ] **Step 1: Write failing compiler tests**

For the two P6.3 fixture episodes, assert:

- stable task/event/action/query IDs across two compiles;
- exact `(entity, attribute)` object identity under `MemoryObjectKey(namespace="default", subkey=None, object_type="slot")`;
- SourceType is `synthetic`, `dialogue`, `changelog`, `calendar`, `issue`, `report_revision`, or `other`; legacy identity is stored only in `LegacyProvenance`;
- `num_events` and `num_target_updates` remain distinct;
- legacy `num_updates` is stored only under `metadata.extra.legacy_num_updates`;
- semantic-core ID excludes split and surface-only IDs;
- `LegacyProvenance.known_caveats` includes P6.3 split leakage;
- `MemoryQuery` has no inline gold answer; gold answers live in `GoldRecord.gold_answers` and `GoldRecord.acceptable_answers`;
- replayed final state matches the gold answer.

- [ ] **Step 2: Implement explicit legacy namespace and caveat registry**

`caveats.py` defines:

```python
LEGACY_NAMESPACES = {
    "p63": "legacy_p63",
    "p65": "legacy_p65",
    "p68_p70": "legacy_p68_p70",
    "p80_p82": "legacy_p80_p82",
    "p83": "legacy_p83",
    "p84": "legacy_p84",
    "p85_api_replacement": "legacy_p85_api_replacement",
}

LEGACY_CAVEATS = {
    "p63_split_leakage": "P6.3 semantic cores overlap across historical train/dev/test splits; compatibility only.",
    "state_direct_oracle": "Legacy slot_direct is an oracle-like structured state readout and maps to state_direct only when exact slot state semantics match.",
    "retrieved_prompt_legacy": "Legacy slot_prompt is a prompted answer condition and maps to retrieved_prompt only when retrieved-context semantics match.",
    "latest_per_slot_rewrite": "latest_per_slot scans the full store and rewrites retrieval; it is not a pure deletion from original top-k.",
    "p83_order_metadata": "Stale same-slot conflict is order- and metadata-sensitive and is not universally the strongest distractor.",
    "p84_answer_layer_only": "P8.4 probes the answer layer and is not a full external-memory baseline.",
}
```

- [ ] **Step 3: Implement the P6 episode compiler**

Expose:

```python
def compile_legacy_episode(
    episode: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    split: Split,
    example_index: int,
    legacy_phase: str,
) -> MemUpdateTask:
```

The compiler must use the existing constrained-slot parser only as an explicit legacy parsing dependency, emit stable IDs from canonical hashes, assign event roles from exact target identity and latest-event metadata, emit ordered ADD/UPDATE/NOOP gold actions, populate `target_objects`, `gold.action_sequence`, `gold.gold_answers`, `gold.acceptable_answers`, and source anchors, and fail rather than guess when entity/attribute/latest-event information is missing.

- [ ] **Step 4: Run compiler and replay tests**

```bash
python -m pytest tests/vnext/test_legacy_dataset_compiler.py tests/vnext/test_gold_replay.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Record an isolated checkpoint**

```bash
git add mub/vnext/legacy tests/vnext/test_legacy_dataset_compiler.py
git commit -m "feat: compile legacy p6 episodes into vnext tasks"
```

### Task 12: Import legacy result runs without semantic invention

**Files:**
- Create: `mub/vnext/legacy/names.py`
- Create: `mub/vnext/legacy/results.py`
- Modify: `mub/vnext/legacy/__init__.py`
- Test: `tests/vnext/test_legacy_result_importer.py`

- [ ] **Step 1: Write failing identity and caveat tests**

Assert explicit summary fields win over directory-name parsing, fallback parsing is isolated and emits a warning, old results keep missing fields `None`, trace dialects remain distinguishable, incompatible memory trajectories cannot be merged, unsupported/pending/capacity-failed source rows cannot become completed canonical runs, and legacy direct/prompt answer modes appear only as legacy run identity metadata.

- [ ] **Step 2: Implement isolated fallback name parsing**

`parse_legacy_run_name(name)` recognizes only documented forms such as `raw_add_slot_prompt_k16`, `long25_slot_direct_k8`, and `oracle_slot_direct_k1`. It returns parsed legacy fields plus warning `legacy_directory_name_inference`; unknown forms return no inferred identity. The parsed answer modes are legacy strings and are mapped to `state_direct` or `retrieved_prompt` only after the importer verifies semantic compatibility.

- [ ] **Step 3: Implement result importer**

Expose:

```python
def import_evomemory_results(
    payload: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    run_name: str | None,
    task_by_legacy_index: dict[int, MemUpdateTask],
) -> tuple[RunManifest, list[TaskRunRecord], list[ScoreRecord], list[str]]:
```

Rules:

1. Preserve raw summary and per-example fields under namespaced metadata.
2. Map a metric into typed score fields only when the registry declares an exact legacy alias.
3. Store all other values in `legacy_metrics`.
4. Keep legacy answer mode, retrieval policy, context order, annotation, checkpoint, seed, and memory trajectory in run identity; map legacy direct/prompt modes to `state_direct` or `retrieved_prompt` only after semantic compatibility is verified.
5. Reject a completed manifest when expected rows are missing or the source says failed, pending, partial, or capacity-exhausted.
6. Preserve `gold_value_in_retrieved` and `gold_retrieved` as separate raw dialect fields while exposing normalized aliases only when definitions match.
7. Emit canonical `TaskRunRecord` runtime names and `ScoreRecord.supported_metric_fields` rather than alternate support maps.

- [ ] **Step 4: Run importer tests**

```bash
python -m pytest tests/vnext/test_legacy_result_importer.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Record an isolated checkpoint**

```bash
git add mub/vnext/legacy tests/vnext/test_legacy_result_importer.py
git commit -m "feat: import legacy evaluation runs safely"
```

### Task 13: Import P8.3/P8.4 mechanism artifacts and ledger warnings

**Files:**
- Create: `mub/vnext/legacy/mechanisms.py`
- Create: `mub/vnext/legacy/ledger.py`
- Test: `tests/vnext/test_legacy_mechanism_importers.py`
- Test: `tests/vnext/test_legacy_ledger.py`

- [ ] **Step 1: Write failing mechanism-boundary tests**

Assert:

- P8.3 condition/order/annotation/stale-count axes remain separate;
- stale-specific removal `original_em_avg` is labeled trace composition, not answer rerun;
- P8.4 rows preserve prompt SHA and raw response separately;
- `gemini-2.5-flash` rows remain imported with a format caveat;
- capacity-failed `gemini-2.5-pro` cannot become a completed cell;
- clean-subset selection filters views without deleting caveat evidence;
- no mechanism artifact invents a new canonical source type or task family when it is only a legacy-analysis row.

- [ ] **Step 2: Implement mechanism import records and functions**

Create typed records `ConflictProbeCell`, `SyntheticDoseCell`, `StaleRemovalTraceCell`, and `ApiProbeCell`. Implement:

```python
import_conflict_probe(path: Path) -> list[ConflictProbeCell]
import_synthetic_dose(path: Path) -> list[SyntheticDoseCell]
import_stale_removal_trace(path: Path) -> list[StaleRemovalTraceCell]
import_api_probe(path: Path) -> list[ApiProbeCell]
select_clean_api_cells(cells: list[ApiProbeCell]) -> list[ApiProbeCell]
```

Every record includes source path/hash, legacy namespace, surface condition, sample count, prompt/config hash when available, raw response path/hash when available, and caveats. Mechanism importers create compatibility analysis records unless a source row has full task/run/score provenance.

- [ ] **Step 3: Write failing ledger-reference tests**

Use a fixture ledger containing one existing path and one absent path. The audit must emit a warning for the missing path and must not silently substitute a likely alias.

- [ ] **Step 4: Implement ledger path audit**

`audit_ledger_references(ledger_path, project_root)` extracts repository-relative artifact references, resolves them without glob substitution, and returns `resolved` and `unresolved` records. An optional `candidate_aliases` field may report suggestions but never changes resolution status.

- [ ] **Step 5: Run mechanism and ledger tests**

```bash
python -m pytest tests/vnext/test_legacy_mechanism_importers.py tests/vnext/test_legacy_ledger.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/legacy tests/vnext/test_legacy_mechanism_importers.py tests/vnext/test_legacy_ledger.py tests/vnext/fixtures/legacy
git commit -m "feat: preserve p8 mechanism and api provenance"
```

### Task 14: Add manifests and read-only Phase 0 CLIs

**Files:**
- Create: `scripts/vnext_compile_legacy.py`
- Create: `scripts/vnext_validate_artifacts.py`
- Test: `tests/vnext/test_phase0_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Using `subprocess.run` and temporary directories, test:

1. `vnext_compile_legacy.py dataset` writes canonical tasks and a `TaskManifest` with exact design fields;
2. `vnext_compile_legacy.py results` writes `TaskRunRecord`, `ScoreRecord`, and `RunManifest` rows with canonical field names;
3. source hashes before and after are equal;
4. missing inputs exit nonzero without creating partial final artifacts;
5. `vnext_validate_artifacts.py` rejects a tampered hash;
6. legacy direct/prompt identities are not accepted as canonical evaluation modes unless semantic mapping has been verified.

- [ ] **Step 2: Implement atomic output helpers**

All CLIs write to `<name>.tmp`, flush and close, validate the finished temporary file, then replace the requested destination. Existing destinations require `--overwrite`; source and destination paths may not resolve to the same file.

- [ ] **Step 3: Implement `vnext_compile_legacy.py`**

CLI subcommands:

```text
dataset --input --split --legacy-phase --output-dir
results --input --tasks --output-dir
mechanism --kind {conflict,dose,stale-removal,api} --input --output-dir
ledger --input --project-root --output-dir
```

Every output directory contains a manifest with input/output hashes, compiler version, row counts, warnings, and `compatibility_only=true`. Task and run artifacts use the canonical `TaskManifest` and `RunManifest` fields; mechanism-only compatibility manifests are separate typed legacy-analysis manifests.

- [ ] **Step 4: Implement `vnext_validate_artifacts.py`**

CLI arguments:

```text
--kind {tasks,task-runs,scores,task-manifest,run-manifest}
--input PATH
--manifest PATH
```

It validates JSON/JSONL types, record uniqueness, declared hashes/counts, schema versions, cross-record task/run IDs, canonical top-level fields, and `supported_metric_fields` reasons. It prints a JSON validation report and exits 0 only when valid.

- [ ] **Step 5: Run CLI tests**

```bash
python -m pytest tests/vnext/test_phase0_cli.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add scripts/vnext_compile_legacy.py scripts/vnext_validate_artifacts.py tests/vnext/test_phase0_cli.py
git commit -m "feat: add vnext legacy bridge cli"
```

### Task 15: Integrate no-network smoke coverage and document the bridge

**Files:**
- Modify: `scripts/smoke_test.py`
- Create: `docs/vnext/legacy_bridge.md`
- Modify: `WORKFLOW.md`
- Test: full Phase 0 suite

- [ ] **Step 1: Add a no-network vNext smoke function**

Append one focused function to `scripts/smoke_test.py` that:

1. imports the five top-level models;
2. constructs one fixed ADD to UPDATE task in memory using `target_objects`, `gold.action_sequence`, `gold.gold_answers`, `gold.acceptable_answers`, and `MemoryQuery.answer_schema`;
3. validates and replays it;
4. serializes and restores it in a temporary directory;
5. validates a capability-gated score using `supported_metric_fields` and the exact SupportReason values;
6. reports pass/fail through the existing `SmokeTestResult` mechanism.

Do not make the smoke script read `data/`, `results/`, API environment variables, or model checkpoints.

- [ ] **Step 2: Document bridge semantics and caveats**

`docs/vnext/legacy_bridge.md` must state:

- canonical object identity and schema versions;
- immutable input roots;
- exact legacy namespaces and why legacy identity stays in `LegacyProvenance`;
- exact `LegacyProvenance` fields: `legacy_family_id`, `legacy_phase`, `legacy_dataset_id`, `legacy_split_id`, `legacy_metric_namespace`, `legacy_run_condition_id`, `checkpoint_family`, `training_seed`, `answer_mode`, `memory_trajectory_id`, `source_artifact_path`, `source_artifact_hash`, and `known_caveats`;
- canonical SourceType values;
- `num_events` versus `num_target_updates` versus legacy `num_updates`;
- direct/prompt trajectory separation;
- legacy `slot_direct` and `slot_prompt` mapping to `state_direct` and `retrieved_prompt` only where semantics match;
- `latest_per_slot`, P8.3 order/metadata, stale-removal trace, and P8.4 answer-layer caveats;
- missing/pending/capacity-failed cell policy;
- compatibility-only claim boundary;
- commands to export schemas, compile fixtures, and validate artifacts.

- [ ] **Step 3: Run the complete Phase 0 test suite**

```bash
python -m pytest tests/vnext -q
python scripts/smoke_test.py
```

Expected: all vNext tests pass and the existing smoke suite remains green.

- [ ] **Step 4: Run compile checks**

```bash
python -m py_compile mub/vnext/version.py mub/vnext/profiles.py mub/vnext/schema_export.py mub/vnext/contracts/enums.py mub/vnext/contracts/common.py mub/vnext/contracts/task.py mub/vnext/contracts/runtime.py mub/vnext/contracts/score.py mub/vnext/contracts/manifest.py mub/vnext/contracts/adapter.py mub/vnext/io/canonical.py mub/vnext/io/jsonl.py mub/vnext/validation/issues.py mub/vnext/validation/task.py mub/vnext/validation/replay.py mub/vnext/validation/split.py mub/vnext/scoring/registry.py mub/vnext/scoring/failures.py mub/vnext/scoring/scorer.py mub/vnext/legacy/caveats.py mub/vnext/legacy/names.py mub/vnext/legacy/loaders.py mub/vnext/legacy/dataset.py mub/vnext/legacy/results.py mub/vnext/legacy/mechanisms.py mub/vnext/legacy/ledger.py scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py scripts/vnext_validate_artifacts.py
```

Expected: command exits 0 with no output.

- [ ] **Step 5: Verify deterministic schemas and immutable sources**

Export schemas twice to two temporary directories and compare their SHA-256 maps. Run fixture compilation twice and compare task, runtime, score, and manifest hashes. Re-hash every legacy source used by the regression run and confirm no input changed.

- [ ] **Step 6: Append actual implementation evidence to `WORKFLOW.md`**

Record only commands actually run, current test counts, generated schema hashes, fixture counts, known warnings, and the statement that Phase 0 creates compatibility/regression artifacts only. Do not add projected Pilot metrics.

- [ ] **Step 7: Review the scoped diff**

Run:

```bash
git diff --check
git status --short
git diff -- requirements.txt pytest.ini mub/vnext scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py scripts/vnext_validate_artifacts.py scripts/smoke_test.py schemas/vnext tests/vnext docs/vnext/legacy_bridge.md WORKFLOW.md
```

Expected: no whitespace errors; no legacy source/result/paper artifact modified; no unrelated dirty file staged.

- [ ] **Step 8: Record the Phase 0 checkpoint**

If commit permission is active:

```bash
git add requirements.txt pytest.ini mub/vnext scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py scripts/vnext_validate_artifacts.py scripts/smoke_test.py schemas/vnext tests/vnext docs/vnext/legacy_bridge.md WORKFLOW.md
git commit -m "feat: establish vnext contract and legacy bridge"
```

Otherwise leave changes uncommitted and report the exact validation results.

## 3. Phase 0 acceptance checklist

- [ ] Five top-level artifacts validate and export deterministic JSON Schema.
- [ ] Canonical task/action/runtime/score/manifest/adapter types match the approved design names.
- [ ] `SourceType` is limited to `synthetic`, `dialogue`, `changelog`, `calendar`, `issue`, `report_revision`, and `other`; legacy identity stays in `LegacyProvenance`.
- [ ] Exact object identity is `(namespace, entity, attribute, subkey)`; `object_type` is classification metadata excluded from identity, and current legacy slots map without invented subkeys.
- [ ] `task_family` is exported as a string and does not reject future family names.
- [ ] Gold replay validates ADD, UPDATE, NOOP, and explicitly enumerated DELETE.
- [ ] Structural validation covers schema version, references, family metadata, type compatibility, targets, expected absence, action-sequence replay, historical queries, stale/duplicate/distractor uniqueness, and answer uniqueness.
- [ ] Split validation rejects semantic-core, trajectory, paraphrase, protected source, version-group, and exact-hash leakage, while enforcing `evaluation_only` exception semantics.
- [ ] Every canonical score field appears exactly once in the metric registry.
- [ ] Unsupported and failed metrics are represented through `supported_metric_fields` without silently entering denominators.
- [ ] Independent failure flags and primary precedence are regression-tested; the correct case is an outcome rather than an independent flag.
- [ ] Adapter support is derived from the declared capability bitset, never method-name inference.
- [ ] Exact `LegacyProvenance` fields match the approved design.
- [ ] P6.3/P6.5/P8.3/P8.4 fixtures preserve raw fields, hashes, identities, and caveats.
- [ ] P6.3 split leakage is explicit and prevents new held-out-generalization claims.
- [ ] Split validation locks minimum stratum `(task_family, difficulty, update_depth_bucket)` and family-specific A-H axes from the design.
- [ ] Legacy direct/prompt, checkpoint-family, and memory-trajectory identities remain separate.
- [ ] Legacy `slot_direct` and `slot_prompt` map to `state_direct` and `retrieved_prompt` only where semantics match.
- [ ] Stale-removal trace composition is not imported as an answer rerun.
- [ ] P8.4 format/capacity caveats and incomplete cells remain visible.
- [ ] Legacy directory parsing is fallback-only and emits warnings.
- [ ] CLIs are atomic, read-only with respect to source artifacts, and manifest-backed.
- [ ] `python -m pytest tests/vnext -q`, `python scripts/smoke_test.py`, compile checks, deterministic-hash checks, and `git diff --check` pass.
- [ ] No new benchmark metric or paper claim is reported from Phase 0.

## 4. Handoff to the Pilot plan

Do not start the Pilot implementation until all Phase 0 acceptance items pass and the generated contract artifacts are reviewed. The Pilot plan is `docs/superpowers/plans/2026-07-20-memupdatebench-vnext-pilot.md`; it consumes the versioned schemas, validators, scorer registry, adapter protocol, and manifests created here rather than introducing parallel contracts.
