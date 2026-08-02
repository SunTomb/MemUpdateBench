from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import field_validator, model_validator

from mub.vnext.contracts.common import ImmutableContractModel, MetricFieldSupport
from mub.vnext.contracts.enums import CompletionStatus, Operation, SupportReason
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.common import StrictIdentifier, object_identity
from mub.vnext.contracts.v3.enums import ExecutionStatusV3, LedgerEntryStatus, QueryTypeV3
from mub.vnext.contracts.v3.runtime import TaskRunRecordV3
from mub.vnext.contracts.v3.score import CORE_SCORE_LAYER_TYPES, ScoreRecordV3, ScorerConfigV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.scoring.registry_v3 import CORE_METRIC_REGISTRY_V3, metric_applies_v3, missing_capabilities_v3
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3, resolve_query_v3

_NULL_POLICY = "serialize_null_exclude_from_aggregation"


class VerifiedScoringContextV3(ImmutableContractModel):
    adapter_info: AdapterInfoV3
    capabilities: AdapterCapabilitiesV3
    scorer_config: ScorerConfigV3
    run_id: StrictIdentifier
    verified_capabilities_sha256: str

    @staticmethod
    def capability_hash(capabilities: AdapterCapabilitiesV3) -> str:
        raw = json.dumps(capabilities.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def create_verified(cls, *, adapter_info, capabilities, scorer_config, run_id):
        capabilities = AdapterCapabilitiesV3.model_validate(capabilities.model_dump(mode="python") if isinstance(capabilities, AdapterCapabilitiesV3) else capabilities)
        return cls(adapter_info=adapter_info, capabilities=capabilities, scorer_config=scorer_config, run_id=run_id, verified_capabilities_sha256=cls.capability_hash(capabilities))

    @field_validator("verified_capabilities_sha256")
    @classmethod
    def _hash(cls, value):
        if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("verified capability hash must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def _verified(self):
        if self.verified_capabilities_sha256 != self.capability_hash(self.capabilities):
            raise ValueError("forged capabilities: verification hash mismatch")
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


def _identity(key):
    return object_identity(key)


def _final_snapshot(run):
    return run.memory_snapshots[-1] if run.memory_snapshots else None


def _snapshot_state(run):
    snapshot = _final_snapshot(run)
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
    if run.adapter_id != context.adapter_info.adapter_id:
        raise ValueError("adapter_id mismatch between run and verified context")
    if context.verified_capabilities_sha256 != context.capability_hash(context.capabilities):
        raise ValueError("forged capabilities")
    provenance = run.parser_extractor_provenance
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
    return entry.object_key_candidate is not None and _identity(entry.object_key_candidate) == _identity(version.object_key) and _same(entry.value_candidate, version.value)


def _metric_value(path, task, run, context, replay, resolutions, evidence, predictions, traces, action_facts):
    layer, leaf = path.split(".", 1)
    snapshot = _final_snapshot(run)
    state = _snapshot_state(run)
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
        stale = [entry for entry in entries if entry.object_key_candidate is not None and not any(_entry_matches_version(entry, version) for version in current_versions)]
        if leaf == "obsolete_version_count": return len(stale), None
        if leaf == "stale_conflicting_value_count": return len(stale), None
        if leaf == "duplicate_current_count": return sum(max(0, sum(_entry_matches_version(entry, version) for entry in entries) - 1) for version in current_versions), None
    if layer == "retrieval_scores":
        if not traces: return None, "retrieval traces are missing"
        current_queries = [query for query in task.queries if query.query_type in {QueryTypeV3.CURRENT, QueryTypeV3.MULTI_OBJECT_CURRENT}]
        current_rows = [(query, traces.get(query.query_id)) for query in current_queries if traces.get(query.query_id) is not None]
        if leaf == "current_recall_at_k": return (_mean([float(any(any(_entry_matches_version(entry, version) for version in resolutions[q.query_id].selected_versions) for entry in trace.retrieved_entries)) for q, trace in current_rows]), None) if current_rows else (None, "no current retrieval rows")
        if leaf == "current_mrr":
            reciprocal = []
            for query, trace in current_rows:
                hit = next((index + 1 for index, entry in enumerate(trace.retrieved_entries) if any(_entry_matches_version(entry, version) for version in resolutions[query.query_id].selected_versions)), None)
                reciprocal.append(0.0 if hit is None else 1.0 / hit)
            return (_mean(reciprocal), None) if reciprocal else (None, "no ranked current rows")
        forgotten = _forgotten_values(replay)
        exposed = [any(any(_same(entry.value_candidate, value) for value in forgotten) for entry in trace.retrieved_entries) for trace in traces.values()]
        if leaf == "stale_exposure_rate": return _mean([float(value) for value in exposed]), None
        if leaf == "stale_count_in_context": return sum(sum(any(_same(entry.value_candidate, value) for value in forgotten) for entry in trace.retrieved_entries) for trace in traces.values()), None
        if leaf == "distractor_exposure_rate": return _mean([float(trace.distractor_in_context is True) for trace in traces.values()]), None
    if layer == "answer_scores":
        if not predictions: return None, "answer predictions are missing"
        exacts = [float(predictions[q.query_id].format_valid and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in task.queries if q.query_id in predictions]
        if leaf in {"exact_match", "normalized_match", "structured_field_accuracy", "answer_state_consistency"}: return _mean(exacts), None
        if leaf == "token_f1": return _mean(exacts), None
        forgotten = _forgotten_values(replay)
        if leaf == "stale_copied": return _mean([float(any(_same(item.parsed_answer, value) for value in forgotten)) for item in predictions.values()]), None
        if leaf == "distractor_copied": return _mean([float(traces.get(item.query_id) is not None and traces[item.query_id].distractor_in_context is True and not _same(item.parsed_answer, evidence[item.query_id].answer)) for item in predictions.values()]), None
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
            protected = [key.canonical_id for key in replay.protected_collateral]
            return (_mean([float(key not in state) for key in protected]), None) if protected else (0.0, None)
        if leaf == "ttl_compliance_rate":
            ttl = [fact for fact in deletes if fact[0].scope.value == "ttl"]
            return (_mean([float(fact[6] and fact[3] and fact[4]) for fact in ttl]), None) if ttl else (None, "no TTL actions")
        if leaf == "relearn_accuracy":
            relearn = [ledger for ledger in replay.ledgers if any(v.status == LedgerEntryStatus.TOMBSTONE for v in ledger.versions[:-1]) and ledger.versions[-1].status == LedgerEntryStatus.PRESENT]
            return (_mean([float(ledger.object_key.canonical_id in state and _same(state[ledger.object_key.canonical_id], ledger.versions[-1].value)) for ledger in relearn]), None) if relearn and snapshot is not None else (None, "no observable relearn sequence")
        forgotten = _forgotten_values(replay)
        if leaf == "forgotten_exposure_rate":
            if not traces: return None, "retrieval traces missing"
            return _mean([float(any(any(_same(entry.value_candidate, value) for value in forgotten) for entry in trace.retrieved_entries)) for trace in traces.values()]), None
        if leaf == "forgotten_value_leakage_rate":
            if not predictions: return None, "answer predictions missing"
            return _mean([float(any(_same(item.parsed_answer, value) for value in forgotten)) for item in predictions.values()]), None
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
            return _mean([float(_same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in rows]), None
        historical = [q for q in task.queries if q.query_type in {QueryTypeV3.PREVIOUS, QueryTypeV3.POINT_IN_TIME, QueryTypeV3.TRANSITION, QueryTypeV3.ORDERED_HISTORY}]
        if not historical: return None, "no historical query"
        if leaf == "version_confusion_rate": return _mean([float(q.query_id in predictions and any(_same(predictions[q.query_id].parsed_answer, replay.current_state.get(key.canonical_id).value if replay.current_state.get(key.canonical_id) else None) for key in q.target_object_keys) and not _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in historical]), None
        if leaf == "historical_support_recall":
            rows = [q for q in historical if q.query_id in traces]
            return (_mean([len(set(evidence[q.query_id].supporting_event_ids) & {event for entry in traces[q.query_id].retrieved_entries for event in entry.source_event_ids}) / len(evidence[q.query_id].supporting_event_ids) for q in rows]), None) if rows else (None, "historical retrieval support missing")
        if leaf == "historical_distance_accuracy": return _mean([float(q.query_id in predictions and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in historical]), None
    if layer == "synthesis_scores":
        multi_hop = [q for q in task.queries if q.query_type == QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP]
        multi_object = [q for q in task.queries if q.query_type in {QueryTypeV3.MULTI_OBJECT_CURRENT, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY}]
        if leaf == "multi_hop_accuracy": return (_mean([float(q.query_id in predictions and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in multi_hop]), None) if multi_hop else (None, "no multi-hop query")
        if leaf == "multi_object_accuracy": return (_mean([float(q.query_id in predictions and _same(predictions[q.query_id].parsed_answer, evidence[q.query_id].answer)) for q in multi_object]), None) if multi_object else (None, "no multi-object query")
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
        if leaf == "stale_propagation_rate": return 0.0, None
    return None, "metric artifact is unavailable"


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
        evaluated = evaluate_evidence_v3(item, replay)
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
        if not metric_applies_v3(descriptor, task.task_family, query_kinds):
            support[path] = _support(SupportReason.NOT_APPLICABLE, "Metric does not apply to this family/query kind.")
            continue
        if path not in requested:
            support[path] = _support(SupportReason.NOT_APPLICABLE, "Metric was not requested.")
            continue
        # A runtime failure dominates missing capabilities for every otherwise-applicable metric.
        if runtime_failed and layer not in {"system_scores", "audit_scores"}:
            support[path] = _support(SupportReason.RUNTIME_FAILED, f"Task completion status is {run.completion_status.value}.")
            continue
        if run.completion_status == CompletionStatus.NOT_SUPPORTED:
            support[path] = _support(SupportReason.NOT_SUPPORTED, "Run completion status is not_supported.")
            continue
        missing = missing_capabilities_v3(descriptor, context.capabilities)
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
