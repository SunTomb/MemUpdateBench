from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from pydantic import field_validator, model_validator

from mub.vnext.contracts.common import ArtifactRef, ImmutableContractModel, MetricFieldSupport
from mub.vnext.contracts.enums import CompletionStatus, Operation, SupportReason
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.common import StrictIdentifier, object_identity
from mub.vnext.contracts.v3.enums import ExecutionStatusV3, LedgerEntryStatus, QueryTypeV3
from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
from mub.vnext.contracts.v3.runtime import TaskRunRecordV3
from mub.vnext.contracts.v3.score import CORE_SCORE_LAYER_TYPES, ScoreRecordV3, ScorerConfigV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.scoring.registry import METRIC_REGISTRY as METRIC_REGISTRY_V2, missing_capabilities as missing_capabilities_v2
from mub.vnext.scoring.registry_v3 import CORE_METRIC_REGISTRY_V3, metric_applies_v3, missing_capabilities_v3
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3, resolve_query_v3

_NULL_POLICY = "serialize_null_exclude_from_aggregation"


def _contract_hash(value) -> str:
    raw = json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


class VerifiedScoringContextV3(ImmutableContractModel):
    task_manifest: TaskManifestV3
    run_manifest: RunManifestV3
    task_artifact: ArtifactRef
    run_artifact: ArtifactRef
    authenticated_task_manifest_sha256: str
    authenticated_run_manifest_sha256: str
    task_manifest_payload_sha256: str
    run_manifest_payload_sha256: str
    task_payload_sha256: str
    run_payload_sha256: str
    capabilities_sha256: str
    adapter_configuration_hash: str
    scorer_configuration_hash: str
    task_id: StrictIdentifier
    run_id: StrictIdentifier

    @property
    def adapter_info(self) -> AdapterInfoV3:
        return self.run_manifest.adapter_info

    @property
    def capabilities(self) -> AdapterCapabilitiesV3:
        return self.run_manifest.adapter_capabilities

    @property
    def scorer_config(self) -> ScorerConfigV3:
        return self.run_manifest.scorer_config

    @staticmethod
    def capability_hash(capabilities: AdapterCapabilitiesV3) -> str:
        return _contract_hash(capabilities)

    @classmethod
    def create_verified(cls, **kwargs):
        raise ValueError("unauthenticated scoring contexts are forbidden; use from_authenticated_manifests")

    @classmethod
    def from_authenticated_manifests(
        cls, *, task, run, task_manifest, run_manifest, task_artifact, run_artifact,
        authenticated_task_manifest_sha256, authenticated_run_manifest_sha256,
    ):
        task = MemUpdateTaskV3.model_validate(task.model_dump(mode="python") if isinstance(task, MemUpdateTaskV3) else task)
        run = TaskRunRecordV3.model_validate(run.model_dump(mode="python") if isinstance(run, TaskRunRecordV3) else run)
        task_manifest = TaskManifestV3.model_validate(task_manifest.model_dump(mode="python") if isinstance(task_manifest, TaskManifestV3) else task_manifest)
        run_manifest = RunManifestV3.model_validate(run_manifest.model_dump(mode="python") if isinstance(run_manifest, RunManifestV3) else run_manifest)
        task_artifact = ArtifactRef.model_validate(task_artifact.model_dump(mode="python") if isinstance(task_artifact, ArtifactRef) else task_artifact)
        run_artifact = ArtifactRef.model_validate(run_artifact.model_dump(mode="python") if isinstance(run_artifact, ArtifactRef) else run_artifact)
        if task.task_id != run.task_id:
            raise ValueError("task_id mismatch between authenticated task and run")
        if run.run_id != run_manifest.run_id:
            raise ValueError("run_id mismatch with authenticated run manifest")
        if run.adapter_id != run_manifest.adapter_info.adapter_id:
            raise ValueError("adapter_id mismatch with authenticated run manifest")
        if authenticated_task_manifest_sha256 != _contract_hash(task_manifest):
            raise ValueError("authenticated task manifest hash does not match canonical manifest payload")
        if authenticated_run_manifest_sha256 != _contract_hash(run_manifest):
            raise ValueError("authenticated run manifest hash does not match canonical manifest payload")
        return cls(
            task_manifest=task_manifest, run_manifest=run_manifest,
            task_artifact=task_artifact, run_artifact=run_artifact,
            authenticated_task_manifest_sha256=authenticated_task_manifest_sha256,
            authenticated_run_manifest_sha256=authenticated_run_manifest_sha256,
            task_manifest_payload_sha256=_contract_hash(task_manifest),
            run_manifest_payload_sha256=_contract_hash(run_manifest),
            task_payload_sha256=_contract_hash(task), run_payload_sha256=_contract_hash(run),
            capabilities_sha256=cls.capability_hash(run_manifest.adapter_capabilities),
            adapter_configuration_hash=run_manifest.adapter_info.configuration_hash,
            scorer_configuration_hash=run_manifest.scorer_config.configuration_hash,
            task_id=task.task_id, run_id=run.run_id,
        )

    @field_validator(
        "authenticated_task_manifest_sha256", "authenticated_run_manifest_sha256",
        "task_manifest_payload_sha256", "run_manifest_payload_sha256",
        "task_payload_sha256", "run_payload_sha256", "capabilities_sha256",
        "adapter_configuration_hash", "scorer_configuration_hash",
    )
    @classmethod
    def _hash(cls, value):
        if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("authenticated hashes must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def _verified(self):
        if self.run_manifest.task_manifest.sha256 != self.authenticated_task_manifest_sha256:
            raise ValueError("run manifest task manifest binding mismatch")
        if self.task_artifact not in self.task_manifest.task_file_paths_and_hashes:
            raise ValueError("task artifact is not authenticated by task manifest")
        if self.run_artifact not in self.run_manifest.normalized_runtime_artifacts:
            raise ValueError("run artifact is not authenticated by run manifest")
        if self.run_manifest.capability_verification_artifact is None:
            raise ValueError("capability verification artifact is required")
        if self.task_manifest_payload_sha256 != _contract_hash(self.task_manifest):
            raise ValueError("authenticated task manifest substitution detected")
        if self.run_manifest_payload_sha256 != _contract_hash(self.run_manifest):
            raise ValueError("authenticated run manifest/config/capability substitution detected")
        if self.capabilities_sha256 != self.capability_hash(self.run_manifest.adapter_capabilities):
            raise ValueError("authenticated capability hash mismatch")
        if self.adapter_configuration_hash != self.run_manifest.adapter_info.configuration_hash:
            raise ValueError("authenticated adapter configuration mismatch")
        if self.scorer_configuration_hash != self.run_manifest.scorer_config.configuration_hash:
            raise ValueError("authenticated scorer configuration mismatch")
        if self.run_id != self.run_manifest.run_id:
            raise ValueError("run identity does not match authenticated run manifest")
        return self


def _support(reason: SupportReason, detail: str) -> MetricFieldSupport:
    return MetricFieldSupport(reason=reason, null_policy=_NULL_POLICY, detail=detail)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _same(left, right) -> bool:
    left, right = _plain(left), _plain(right)
    return type(left) is type(right) and left == right


def _mean(values):
    return sum(values) / len(values) if values else None


def _normalized(value):
    if type(value) is str:
        return " ".join(value.casefold().split())
    if isinstance(value, Mapping):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    return value


def _answer_contains(output, candidate):
    if _same(output, candidate):
        return True
    output = _plain(output)
    candidate = _plain(candidate)
    if isinstance(output, str) and isinstance(candidate, str):
        return bool(candidate) and candidate.casefold() in output.casefold()
    if isinstance(output, Mapping):
        return any(_answer_contains(value, candidate) for value in output.values())
    if isinstance(output, list):
        return any(_answer_contains(value, candidate) for value in output)
    return False


def _distractor_candidates(trace):
    return tuple(
        candidate
        for entry in trace.retrieved_entries
        if entry.raw_metadata.get("is_distractor") is True or entry.raw_metadata.get("role") == "distractor"
        for candidate in (entry.value_candidate, entry.content)
        if candidate is not None
    )


def _token_f1(predicted, gold):
    predicted_tokens = str(_plain(predicted)).casefold().split()
    gold_tokens = str(_plain(gold)).casefold().split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _structured_accuracy(predicted, gold):
    predicted, gold = _plain(predicted), _plain(gold)
    if isinstance(gold, Mapping) and isinstance(predicted, Mapping):
        return _mean([float(key in predicted and _same(predicted[key], value)) for key, value in gold.items()]) if gold else float(not predicted)
    if isinstance(gold, list) and isinstance(predicted, list):
        size = max(len(gold), len(predicted))
        return 1.0 if size == 0 else sum(index < len(gold) and index < len(predicted) and _same(predicted[index], gold[index]) for index in range(size)) / size
    return float(_same(predicted, gold))


def _identity(key):
    return object_identity(key)


def _final_snapshot(run, task=None):
    if not run.memory_snapshots:
        return None
    if task is None:
        return run.memory_snapshots[-1]
    positions = {event.event_id: event.sequence_index for event in task.events}
    anchored = [snapshot for snapshot in run.memory_snapshots if snapshot.after_event_id in positions]
    return max(anchored, key=lambda snapshot: positions[snapshot.after_event_id]) if anchored else run.memory_snapshots[-1]


def _snapshot_state(run, task=None):
    snapshot = _final_snapshot(run, task)
    return {} if snapshot is None else dict(snapshot.state_by_object)


def _prediction_maps(task, run):
    query_ids = [query.query_id for query in task.queries]
    predictions = {item.query_id: item for item in run.answer_predictions}
    traces = {item.query_id: item for item in run.retrieval_traces}
    if len(traces) != len(run.retrieval_traces):
        raise ValueError("duplicate retrieval query rows")
    if run.completion_status == CompletionStatus.COMPLETED and set(predictions) != set(query_ids):
        raise ValueError("missing or extra answer prediction query rows")
    if set(predictions) - set(query_ids) or set(traces) - set(query_ids):
        raise ValueError("runtime rows reference unknown query IDs")
    return predictions, traces


def _validate_bindings(task, run, context):
    if task.task_id != run.task_id:
        raise ValueError("task_id mismatch between task and run")
    if run.run_id != context.run_id:
        raise ValueError("run_id mismatch between run and verified context")
    if context.task_id != task.task_id or context.task_payload_sha256 != _contract_hash(task):
        raise ValueError("authenticated task substitution detected")
    if context.run_payload_sha256 != _contract_hash(run):
        raise ValueError("authenticated run substitution detected")
    if run.adapter_id != context.adapter_info.adapter_id:
        raise ValueError("adapter_id mismatch between run and verified context")
    provenance = run.parser_extractor_provenance
    manifest = context.run_manifest
    if provenance.action_parser_version != manifest.action_parser_version or provenance.answer_parser_version != manifest.answer_parser_version:
        raise ValueError("parser config identity mismatch")
    if provenance.memory_entry_extractor_version != manifest.memory_entry_extractor_version:
        raise ValueError("extractor config identity mismatch")
    if provenance.redaction_policy_version != manifest.redaction_policy_version:
        raise ValueError("redaction config identity mismatch")
    if context.adapter_info.extractor_version is not None and provenance.memory_entry_extractor_version != context.adapter_info.extractor_version:
        raise ValueError("extractor/config version mismatch")


def _action_facts(task, run):
    by_event = {action.event_id: action for action in run.parsed_actions}
    gold = list(task.actions)
    rows = [(action, by_event.get(action.event_id)) for action in gold]
    def exact_target(left, right):
        return tuple(_identity(key) for key in left) == tuple(_identity(key) for key in right)
    facts = []
    for expected, observed in rows:
        op = observed is not None and observed.operation == expected.operation
        scope = observed is not None and observed.observed_scope == expected.scope
        targets = observed is not None and exact_target(observed.target_object_keys, expected.target_object_keys)
        value = observed is not None and _same(observed.value, expected.value)
        executed = observed is not None and observed.execution_status == ExecutionStatusV3.EXECUTED
        facts.append((expected, observed, op, scope, targets, value, executed))
    return facts


def _forgotten_values(replay):
    values = []
    for ledger in replay.ledgers:
        for index, version in enumerate(ledger.versions):
            if version.status == LedgerEntryStatus.PRESENT and any(later.status == LedgerEntryStatus.TOMBSTONE for later in ledger.versions[index + 1:]):
                if not any(later.status == LedgerEntryStatus.PRESENT and _same(later.value, version.value) for later in ledger.versions[index + 1:]):
                    values.append(version.value)
    return values


def _entry_matches_version(entry, version):
    if entry.object_key_candidate is None or _identity(entry.object_key_candidate) != _identity(version.object_key):
        return False
    if entry.version_index is not None:
        return entry.version_index == version.version_index
    if entry.source_event_ids:
        return bool(set(entry.source_event_ids) & set(version.source_event_ids))
    return _same(entry.value_candidate, version.value)


def _entry_current_match_status(entry, version, replay):
    if entry.version_index is not None or entry.source_event_ids:
        return _entry_matches_version(entry, version)
    if entry.object_key_candidate is None or _identity(entry.object_key_candidate) != _identity(version.object_key):
        return False
    ledger = replay.ledger_by_identity.get(_identity(version.object_key))
    same_value_versions = [] if ledger is None else [candidate for candidate in ledger.versions if candidate.status == LedgerEntryStatus.PRESENT and _same(candidate.value, entry.value_candidate)]
    if len(same_value_versions) > 1:
        return None
    return _same(entry.value_candidate, version.value)


def _entry_obsolete_status(entry, replay):
    if entry.object_key_candidate is None:
        return False
    ledger = replay.ledger_by_identity.get(_identity(entry.object_key_candidate))
    if ledger is None or not ledger.versions:
        return False
    if entry.version_index is not None:
        return entry.version_index < ledger.versions[-1].version_index
    if entry.source_event_ids:
        current_sources = set(ledger.versions[-1].source_event_ids)
        if set(entry.source_event_ids) & current_sources:
            return False
        return any(set(entry.source_event_ids) & set(version.source_event_ids) for version in ledger.versions[:-1])
    current = ledger.versions[-1]
    obsolete = [version for version in ledger.versions[:-1] if version.status == LedgerEntryStatus.PRESENT and _same(entry.value_candidate, version.value)]
    if obsolete and current.status == LedgerEntryStatus.PRESENT and _same(entry.value_candidate, current.value):
        return None
    return bool(obsolete)


def _entry_forgotten_status(entry, replay):
    if entry.object_key_candidate is None:
        return False
    ledger = replay.ledger_by_identity.get(_identity(entry.object_key_candidate))
    if ledger is None:
        return False
    matched = None
    if entry.version_index is not None and entry.version_index < len(ledger.versions):
        matched = ledger.versions[entry.version_index]
    elif entry.source_event_ids:
        matched = next((version for version in ledger.versions if set(entry.source_event_ids) & set(version.source_event_ids)), None)
    if matched is None:
        return None
    return any(version.status == LedgerEntryStatus.TOMBSTONE for version in ledger.versions[matched.version_index + 1:])


def _metric_value(path, task, run, context, replay, resolutions, evidence, predictions, traces, action_facts):
    layer, leaf = path.split(".", 1)
    snapshot = _final_snapshot(run, task)
    state = _snapshot_state(run, task)
    expected_state = {key: version.value for key, version in replay.current_state.items()}
    answers = list(predictions.values())
    actions = [fact[1] for fact in action_facts if fact[1] is not None]

    if layer == "protocol_scores":
        if leaf == "action_parse_valid": return (all(action.format_valid for action in actions), None) if task.actions and len(actions) == len(task.actions) else (None, "missing action rows")
        if leaf == "answer_parse_valid": return (all(item.format_valid for item in answers), None) if answers else (None, "missing answer rows")
        if leaf == "execution_success_rate": return (_mean([float(fact[6]) for fact in action_facts]), None) if action_facts else (None, "no actions")
        if leaf == "unsupported_operation_rate": return (_mean([float(fact[1] is not None and fact[1].execution_status == ExecutionStatusV3.NOT_SUPPORTED) for fact in action_facts]), None) if action_facts else (None, "no actions")
        if leaf == "fallback_rate": return (_mean([float(fact[1].fallback_used) for fact in action_facts if fact[1] is not None]), None) if actions else (None, "no action rows")
    if layer == "action_scores":
        if not action_facts: return None, "no gold actions"
        indexes = {"operation_accuracy": 2, "object_key_accuracy": 4, "value_accuracy": 5}
        if leaf in indexes: return _mean([float(fact[indexes[leaf]]) for fact in action_facts]), None
        if leaf == "full_action_exact_match": return _mean([float(all(fact[index] for index in (2, 3, 4, 5)) and fact[6]) for fact in action_facts]), None
        if leaf == "entity_accuracy": return _mean([float(fact[1] is not None and tuple(key.entity for key in fact[1].target_object_keys) == tuple(key.entity for key in fact[0].target_object_keys)) for fact in action_facts]), None
        if leaf == "attribute_accuracy": return _mean([float(fact[1] is not None and tuple(key.attribute for key in fact[1].target_object_keys) == tuple(key.attribute for key in fact[0].target_object_keys)) for fact in action_facts]), None
        if leaf == "false_write_rate": return _mean([float(fact[0].operation == Operation.NOOP and fact[1] is not None and fact[1].operation != Operation.NOOP) for fact in action_facts]), None
        if leaf == "missed_write_rate": return _mean([float(fact[0].operation != Operation.NOOP and not fact[6]) for fact in action_facts]), None
        if leaf == "wrong_object_write_rate": return _mean([float(fact[1] is not None and fact[1].operation != Operation.NOOP and not fact[4]) for fact in action_facts]), None
    if layer == "state_scores":
        if snapshot is None: return None, "final memory snapshot is missing"
        exact = sum(1 for key, value in state.items() if key in expected_state and _same(value, expected_state[key]))
        if leaf == "final_state_accuracy": return float(set(state) == set(expected_state) and exact == len(expected_state)), None
        if leaf == "state_precision": return (exact / len(state) if state else float(not expected_state)), None
        if leaf == "state_recall": return (exact / len(expected_state) if expected_state else float(not state)), None
        if leaf == "state_f1":
            p = exact / len(state) if state else float(not expected_state); r = exact / len(expected_state) if expected_state else float(not state)
            return (0.0 if p + r == 0 else 2 * p * r / (p + r)), None
        if leaf == "state_resolve_rate": return (exact / len(expected_state) if expected_state else 1.0), None
        if leaf == "collateral_corruption_rate": return (sum(key not in expected_state for key in state) / len(state) if state else 0.0), None
        if leaf == "expected_absence_accuracy":
            absent = [key.canonical_id for key in replay.expected_absent]
            return (_mean([float(key not in state) for key in absent]), None) if absent else (None, "no expected-absent objects")
    if layer == "store_scores":
        if snapshot is None: return None, "final memory snapshot is missing"
        entries = snapshot.entries
        if leaf == "final_memory_size": return snapshot.store_size, None
        mutations = max(1, sum(action.operation != Operation.NOOP for action in task.actions))
        if leaf == "compaction_ratio": return snapshot.store_size / mutations, None
        if leaf == "write_amplification": return sum(action.operation not in {None, Operation.NOOP} for action in actions) / mutations, None
        current_versions = tuple(replay.current_state.values())
        statuses = tuple(_entry_obsolete_status(entry, replay) for entry in entries)
        if any(status is None for status in statuses) and leaf in {"obsolete_version_count", "stale_conflicting_value_count", "duplicate_current_count"}:
            return None, "version identity is ambiguous for repeated-value store entries"
        stale = [entry for entry, status in zip(entries, statuses) if status is True]
        if leaf == "obsolete_version_count": return len(stale), None
        if leaf == "stale_conflicting_value_count": return len(stale), None
        if leaf == "duplicate_current_count": return sum(max(0, sum(_entry_matches_version(entry, version) for entry in entries) - 1) for version in current_versions), None
    if layer == "retrieval_scores":
        if not traces: return None, "retrieval traces are missing"
        current_queries = [query for query in task.queries if query.query_type in {QueryTypeV3.CURRENT, QueryTypeV3.MULTI_OBJECT_CURRENT}]
        current_rows = [(query, traces.get(query.query_id)) for query in current_queries if traces.get(query.query_id) is not None]
        current_statuses = {
            query.query_id: tuple(
                any(_entry_current_match_status(entry, version, replay) is True for version in resolutions[query.query_id].selected_versions)
                if not any(_entry_current_match_status(entry, version, replay) is None for version in resolutions[query.query_id].selected_versions)
                else None
                for entry in trace.retrieved_entries
            )
            for query, trace in current_rows
        }
        if leaf in {"current_recall_at_k", "current_mrr"} and any(status is None for statuses in current_statuses.values() for status in statuses):
            return None, "version identity is ambiguous for repeated-value current retrieval entries"
        if leaf == "current_recall_at_k": return (_mean([float(any(status is True for status in current_statuses[q.query_id])) for q, trace in current_rows]), None) if current_rows else (None, "no current retrieval rows")
        if leaf == "current_mrr":
            reciprocal = []
            for query, trace in current_rows:
                hit = next((index + 1 for index, status in enumerate(current_statuses[query.query_id]) if status is True), None)
                reciprocal.append(0.0 if hit is None else 1.0 / hit)
            return (_mean(reciprocal), None) if reciprocal else (None, "no ranked current rows")
        classified = {
            trace.query_id: tuple(_entry_obsolete_status(entry, replay) for entry in trace.retrieved_entries)
            for trace in traces.values()
        }
        if any(status is None for statuses in classified.values() for status in statuses):
            return None, "version identity is ambiguous for repeated-value retrieval entries"
        exposed = [any(status is True for status in classified[trace.query_id]) for trace in traces.values()]
        if leaf == "stale_exposure_rate": return _mean([float(value) for value in exposed]), None
        if leaf == "stale_count_in_context": return sum(sum(status is True for status in classified[trace.query_id]) for trace in traces.values()), None
        if leaf == "distractor_exposure_rate": return _mean([float(trace.distractor_in_context is True) for trace in traces.values()]), None
    if layer == "answer_scores":
        if not predictions: return None, "answer predictions are missing"
        exacts = [float(predictions[q.query_id].format_valid and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in task.queries if q.query_id in predictions]
        if leaf == "exact_match": return _mean(exacts), None
        if leaf == "normalized_match": return _mean([float(item.format_valid and _normalized(item.parsed_answer) == _normalized(evidence[item.query_id].answer)) for item in predictions.values()]), None
        if leaf == "token_f1": return _mean([_token_f1(item.parsed_answer, evidence[item.query_id].answer) if item.format_valid else 0.0 for item in predictions.values()]), None
        if leaf == "structured_field_accuracy": return _mean([_structured_accuracy(item.parsed_answer, evidence[item.query_id].answer) if item.format_valid else 0.0 for item in predictions.values()]), None
        if leaf == "answer_state_consistency": return _mean([float(item.format_valid and _same(item.parsed_answer, resolutions[item.query_id].answer)) for item in predictions.values()]), None
        obsolete = replay.obsolete_present_values
        if leaf == "stale_copied": return _mean([float(not _same(item.parsed_answer, evidence[item.query_id].answer) and any(_same(item.parsed_answer, value) for value in obsolete)) for item in predictions.values()]), None
        if leaf == "distractor_copied": return _mean([float(traces.get(item.query_id) is not None and traces[item.query_id].distractor_in_context is True and any(_answer_contains(item.parsed_answer, candidate) for candidate in _distractor_candidates(traces[item.query_id]))) for item in predictions.values()]), None
        if leaf == "gold_retrieved_wrong_answer": return _mean([float(traces.get(item.query_id) is not None and traces[item.query_id].gold_in_context is True and not _same(item.parsed_answer, evidence[item.query_id].answer)) for item in predictions.values()]), None
        if leaf == "reference_resolution_accuracy": return None, "no v3 unresolved-reference query kind"
    if layer == "system_scores":
        if leaf == "error_rate": return float(run.completion_status != CompletionStatus.COMPLETED or bool(run.exceptions)), None
        if leaf == "ingest_latency_ms": return (_mean([item.latency_ms for item in actions if item.latency_ms is not None]), None) if any(item.latency_ms is not None for item in actions) else (None, "ingest latency missing")
        if leaf == "answer_latency_ms": return (_mean([item.latency_ms for item in answers if item.latency_ms is not None]), None) if any(item.latency_ms is not None for item in answers) else (None, "answer latency missing")
        if leaf == "retrieval_latency_ms": return None, "retrieval trace has no latency field"
        if leaf == "token_usage": return (sum(sum(item.usage.values()) for item in answers), None) if any(item.usage for item in answers) else (None, "usage missing")
        if leaf == "api_cost": return None, "cost artifact missing"
    if layer == "audit_scores":
        if leaf == "action_trace_available": return bool(run.parsed_actions), None
        if leaf == "state_export_available": return snapshot is not None, None
        if leaf == "retrieval_trace_available": return bool(run.retrieval_traces), None
        if leaf == "source_provenance_coverage":
            if snapshot is None or not snapshot.entries: return None, "entry provenance missing"
            return _mean([float(bool(entry.source_event_ids)) for entry in snapshot.entries]), None
        if leaf == "manifest_completeness": return 1.0, None
    if layer == "deletion_scores":
        deletes = [fact for fact in action_facts if fact[0].operation == Operation.DELETE]
        if not deletes: return None, "no DELETE actions"
        if leaf == "deletion_accuracy": return _mean([float(all(fact[i] for i in (2, 3, 4)) and fact[6]) for fact in deletes]), None
        if leaf == "delete_scope_accuracy": return _mean([float(fact[3] and fact[4]) for fact in deletes]), None
        if leaf == "collateral_damage_rate":
            if snapshot is None: return None, "final state missing"
            protected = [key for key in replay.protected_collateral if key.canonical_id in replay.current_state]
            return (_mean([float(key.canonical_id not in state or not _same(state[key.canonical_id], replay.current_state[key.canonical_id].value)) for key in protected]), None) if protected else (0.0, None)
        if leaf == "ttl_compliance_rate":
            ttl = [fact for fact in deletes if fact[0].scope.value == "ttl"]
            if not ttl: return None, "no TTL actions"
            observations = []
            for fact in ttl:
                target_ids = [key.canonical_id for key in fact[0].target_object_keys]
                event_times = {event.event_id: event.timestamp for event in task.events if event.timestamp is not None}
                candidates = [
                    item for item in run.memory_snapshots
                    if item.after_event_id in event_times
                    and fact[0].effective_at is not None
                    and event_times[item.after_event_id] >= fact[0].effective_at
                ]
                observed_snapshot = min(candidates, key=lambda item: event_times[item.after_event_id]) if candidates else None
                if observed_snapshot is None:
                    return None, "TTL expiry-time state snapshot missing"
                observed_state = dict(observed_snapshot.state_by_object)
                observations.append(float(fact[6] and fact[3] and fact[4] and all(target not in observed_state for target in target_ids)))
            return _mean(observations), None
        if leaf == "relearn_accuracy":
            relearn = [ledger for ledger in replay.ledgers if any(v.status == LedgerEntryStatus.TOMBSTONE for v in ledger.versions[:-1]) and ledger.versions[-1].status == LedgerEntryStatus.PRESENT]
            return (_mean([float(ledger.object_key.canonical_id in state and _same(state[ledger.object_key.canonical_id], ledger.versions[-1].value)) for ledger in relearn]), None) if relearn and snapshot is not None else (None, "no observable relearn sequence")
        forgotten = _forgotten_values(replay)
        if leaf == "forgotten_exposure_rate":
            if not traces: return None, "retrieval traces missing"
            classified = [tuple(_entry_forgotten_status(entry, replay) for entry in trace.retrieved_entries) for trace in traces.values()]
            if any(status is None for statuses in classified for status in statuses):
                return None, "version identity is ambiguous for forgotten-value retrieval entries"
            return _mean([float(any(status is True for status in statuses)) for statuses in classified]), None
        if leaf == "forgotten_value_leakage_rate":
            if not predictions: return None, "answer predictions missing"
            return _mean([float(not _same(item.parsed_answer, evidence[item.query_id].answer) and any(_same(item.parsed_answer, value) for value in forgotten)) for item in predictions.values()]), None
    if layer == "historical_scores":
        kinds = {
            "current_state_accuracy": {QueryTypeV3.CURRENT}, "previous_state_accuracy": {QueryTypeV3.PREVIOUS},
            "point_in_time_accuracy": {QueryTypeV3.POINT_IN_TIME}, "transition_accuracy": {QueryTypeV3.TRANSITION},
            "ordered_history_accuracy": {QueryTypeV3.ORDERED_HISTORY},
        }
        if leaf in kinds:
            rows = [q for q in task.queries if q.query_type in kinds[leaf]]
            if not rows: return None, "query kind not present"
            if not all(q.query_id in predictions for q in rows): return None, "answer prediction missing"
            return _mean([float(predictions[q.query_id].format_valid and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in rows]), None
        historical = [q for q in task.queries if q.query_type in {QueryTypeV3.PREVIOUS, QueryTypeV3.POINT_IN_TIME, QueryTypeV3.TRANSITION, QueryTypeV3.ORDERED_HISTORY}]
        if not historical: return None, "no historical query"
        if leaf == "version_confusion_rate": return _mean([float(q.query_id in predictions and any(_same(predictions[q.query_id].parsed_answer, replay.current_state.get(key.canonical_id).value if replay.current_state.get(key.canonical_id) else None) for key in q.target_object_keys) and not _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in historical]), None
        if leaf == "historical_support_recall":
            rows = [q for q in historical if q.query_id in traces]
            return (_mean([len(set(evidence[q.query_id].supporting_event_ids) & {event for entry in traces[q.query_id].retrieved_entries for event in entry.source_event_ids}) / len(evidence[q.query_id].supporting_event_ids) for q in rows]), None) if rows else (None, "historical retrieval support missing")
        if leaf == "historical_distance_accuracy": return _mean([float(q.query_id in predictions and predictions[q.query_id].format_valid and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in historical]), None
    if layer == "synthesis_scores":
        multi_hop = [q for q in task.queries if q.query_type == QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP]
        multi_object = [q for q in task.queries if q.query_type in {QueryTypeV3.MULTI_OBJECT_CURRENT, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY}]
        if leaf == "multi_hop_accuracy": return (_mean([float(q.query_id in predictions and predictions[q.query_id].format_valid and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in multi_hop]), None) if multi_hop else (None, "no multi-hop query")
        if leaf == "multi_object_accuracy": return (_mean([float(q.query_id in predictions and predictions[q.query_id].format_valid and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in multi_object]), None) if multi_object else (None, "no multi-object query")
        g = [q for q in task.queries if q.query_type in {QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY}]
        cited = [(q, predictions[q.query_id]) for q in g if q.query_id in predictions]
        if leaf in {"evidence_precision", "evidence_recall", "evidence_f1"}:
            if not cited: return None, "evidence citations missing"
            ps, rs = [], []
            for q, prediction in cited:
                required = set(evidence[q.query_id].supporting_event_ids) | {key.canonical_id for key in evidence[q.query_id].supporting_object_keys} | {step.step_id for step in evidence[q.query_id].derivation_steps}
                observed = set(prediction.cited_event_ids) | {key.canonical_id for key in prediction.cited_object_keys} | set(prediction.cited_derivation_step_ids)
                overlap = len(required & observed); ps.append(overlap / len(observed) if observed else 0.0); rs.append(overlap / len(required))
            p, r = _mean(ps), _mean(rs)
            return ({"evidence_precision": p, "evidence_recall": r, "evidence_f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}[leaf], None)
        if leaf == "reasoning_support_accuracy":
            rows = [(q, traces[q.query_id]) for q in g if q.query_id in traces]
            return (_mean([len(set(evidence[q.query_id].supporting_event_ids) & {event for entry in trace.retrieved_entries for event in entry.source_event_ids}) / len(evidence[q.query_id].supporting_event_ids) for q, trace in rows]), None) if rows else (None, "reasoning retrieval support missing")
        if leaf == "stale_propagation_rate":
            rows = [predictions[q.query_id] for q in g if q.query_id in predictions and evidence[q.query_id].stale_alternative is not None]
            return (_mean([float(not _same(item.parsed_answer, evidence[item.query_id].answer) and _same(item.parsed_answer, evidence[item.query_id].stale_alternative.answer)) for item in rows]), None) if rows else (None, "registered stale-alternative prediction rows missing")
    return None, "metric artifact is unavailable"


def _metric_applies_to_task_v3(path, task, replay):
    if path.startswith("deletion_scores."):
        deletes = [action for action in task.actions if action.operation == Operation.DELETE]
        if not deletes:
            return False
        if path == "deletion_scores.ttl_compliance_rate":
            return any(action.scope is not None and action.scope.value == "ttl" for action in deletes)
        if path == "deletion_scores.relearn_accuracy":
            return any(
                any(version.status == LedgerEntryStatus.TOMBSTONE for version in ledger.versions[:-1])
                and ledger.versions[-1].status == LedgerEntryStatus.PRESENT
                for ledger in replay.ledgers
            )
        if path in {"deletion_scores.forgotten_exposure_rate", "deletion_scores.forgotten_value_leakage_rate"}:
            return any(any(version.status == LedgerEntryStatus.TOMBSTONE for version in ledger.versions) for ledger in replay.ledgers)
    if path == "synthesis_scores.stale_propagation_rate":
        return any(evidence.stale_alternative is not None for evidence in task.gold_evidence)
    return True

def score_task_v3(task: MemUpdateTaskV3, run: TaskRunRecordV3, context: VerifiedScoringContextV3) -> ScoreRecordV3:
    task = MemUpdateTaskV3.model_validate(task.model_dump(mode="python"))
    run = TaskRunRecordV3.model_validate(run.model_dump(mode="python"))
    context = VerifiedScoringContextV3.model_validate(context.model_dump(mode="python"))
    _validate_bindings(task, run, context)
    predictions, traces = _prediction_maps(task, run)
    replay = replay_task_v3(task)
    if replay.issues:
        raise ValueError(f"authenticated task gold replay failed: {replay.issues[0].code}")
    resolutions = {query.query_id: resolve_query_v3(query, replay, task.events) for query in task.queries}
    if any(result.issues for result in resolutions.values()):
        raise ValueError("typed query resolution failed")
    evidence = {item.query_id: item for item in task.gold_evidence}
    for item in task.gold_evidence:
        evaluated = evaluate_evidence_v3(item, replay, item.stale_alternative)
        # Non-G derivation operation vocabularies are descriptive and need not execute.
        if any(query.query_id == item.query_id and query.query_type in {QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY} for query in task.queries) and evaluated.issues:
            raise ValueError(f"G evidence replay failed: {evaluated.issues[0].code}")
    action_facts = _action_facts(task, run)
    query_kinds = {query.query_type.value for query in task.queries}
    requested = set(context.scorer_config.requested_metric_fields or CORE_METRIC_REGISTRY_V3)
    layers = {layer: {field: None for field in model.model_fields} for layer, model in CORE_SCORE_LAYER_TYPES.items()}
    support = {}
    runtime_failed = run.completion_status in {CompletionStatus.FAILED, CompletionStatus.PARTIAL} or bool(run.exceptions)
    for path, descriptor in CORE_METRIC_REGISTRY_V3.items():
        layer, leaf = path.split(".", 1)
        if not metric_applies_v3(descriptor, task.task_family, query_kinds) or not _metric_applies_to_task_v3(path, task, replay):
            support[path] = _support(SupportReason.NOT_APPLICABLE, "Metric does not apply to this family/query kind.")
            continue
        if path not in requested:
            support[path] = _support(SupportReason.NOT_APPLICABLE, "Metric was not requested.")
            continue
        # A runtime failure dominates missing capabilities for every otherwise-applicable metric.
        if runtime_failed and layer not in {"protocol_scores", "system_scores", "audit_scores"}:
            support[path] = _support(SupportReason.RUNTIME_FAILED, f"Task completion status is {run.completion_status.value}.")
            continue
        if run.completion_status == CompletionStatus.NOT_SUPPORTED:
            support[path] = _support(SupportReason.NOT_SUPPORTED, "Run completion status is not_supported.")
            continue
        missing = (
            missing_capabilities_v2(METRIC_REGISTRY_V2[path], context.capabilities)
            if path in METRIC_REGISTRY_V2
            else missing_capabilities_v3(descriptor, context.capabilities)
        )
        if missing:
            support[path] = _support(SupportReason.NOT_SUPPORTED, f"Verified adapter lacks capabilities: {', '.join(missing)}.")
            continue
        value, detail = _metric_value(path, task, run, context, replay, resolutions, evidence, predictions, traces, action_facts)
        if value is None:
            support[path] = _support(SupportReason.MISSING_ARTIFACT, detail or "Required artifact is missing.")
        else:
            layers[layer][leaf] = value
    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3
    flags = derive_failure_flags_v3(task=task, run=run, replay=replay, layer_values=layers, predictions=predictions, traces=traces, evidence=evidence)
    return ScoreRecordV3.empty(
        task_id=task.task_id, run_id=run.run_id, adapter_id=run.adapter_id,
        task_family=task.task_family, difficulty=task.difficulty,
        completion_status=run.completion_status, supported_metric_fields=support,
        **layers, failure_flags=flags,
    )


__all__ = ["VerifiedScoringContextV3", "score_task_v3"]
