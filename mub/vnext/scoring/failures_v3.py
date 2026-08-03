from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import field_validator

from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel, StrictNonnegativeFloat, StrictNonnegativeInt, freeze_mapping
from mub.vnext.contracts.enums import CompletionStatus, Operation, TaskFamily
from mub.vnext.contracts.v3.enums import ExecutionStatusV3, LedgerEntryStatus, QueryTypeV3
from mub.vnext.contracts.v3.score import ScoreRecordV3, V3_FAILURE_FLAGS
from mub.vnext.scoring.registry_v3 import CORE_METRIC_REGISTRY_V3


def _plain(value):
    if isinstance(value, Mapping): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple): return [_plain(item) for item in value]
    return value


def _same(left, right):
    left, right = _plain(left), _plain(right)
    return type(left) is type(right) and left == right


def _contains_value(output, candidate):
    if _same(output, candidate):
        return True
    output = _plain(output)
    candidate = _plain(candidate)
    if isinstance(output, str) and isinstance(candidate, str):
        return bool(candidate) and candidate.casefold() in output.casefold()
    if isinstance(output, Mapping):
        return any(_contains_value(value, candidate) for value in output.values())
    if isinstance(output, list):
        return any(_contains_value(value, candidate) for value in output)
    return False


def _forgotten_values(replay):
    values = []
    for ledger in replay.ledgers:
        for index, version in enumerate(ledger.versions):
            if version.status == LedgerEntryStatus.PRESENT and any(later.status == LedgerEntryStatus.TOMBSTONE for later in ledger.versions[index + 1:]):
                values.append(version.value)
    return values


def _entry_version_status(entry, replay):
    if entry.object_key_candidate is None:
        return False, False
    identity = (entry.object_key_candidate.namespace, entry.object_key_candidate.entity, entry.object_key_candidate.attribute, entry.object_key_candidate.subkey)
    ledger = replay.ledger_by_identity.get(identity)
    if ledger is None or not ledger.versions:
        return False, False
    matched = None
    if entry.version_index is not None and entry.version_index < len(ledger.versions):
        matched = ledger.versions[entry.version_index]
    elif entry.source_event_ids:
        matched = next((version for version in ledger.versions if set(entry.source_event_ids) & set(version.source_event_ids)), None)
    if matched is None:
        return None, None
    obsolete = matched.version_index < ledger.versions[-1].version_index
    forgotten = any(version.status == LedgerEntryStatus.TOMBSTONE for version in ledger.versions[matched.version_index + 1:])
    return obsolete, forgotten


def derive_failure_flags_v3(*, task, run, replay, layer_values, predictions, traces, evidence) -> tuple[str, ...]:
    flags: set[str] = set()
    if run.exceptions or run.completion_status in {CompletionStatus.FAILED, CompletionStatus.PARTIAL}:
        flags.add("system_exception")
    by_event = {item.event_id: item for item in run.parsed_actions}
    for gold in task.actions:
        observed = by_event.get(gold.event_id)
        if observed is None:
            if gold.operation != Operation.NOOP:
                flags.add("missed_update")
                if gold.operation == Operation.DELETE: flags.add("deletion_failure")
            continue
        if not observed.format_valid: flags.add("invalid_action_format")
        if observed.execution_status == ExecutionStatusV3.NOT_SUPPORTED: flags.add("unsupported_action")
        if observed.operation != gold.operation:
            flags.add("wrong_operation")
            if gold.operation == Operation.DELETE: flags.add("deletion_failure")
        if observed.observed_scope != gold.scope and gold.operation == Operation.DELETE: flags.add("wrong_delete_scope")
        gold_ids = [key.canonical_id for key in gold.target_object_keys]
        observed_ids = [key.canonical_id for key in observed.target_object_keys]
        if observed_ids != gold_ids:
            if tuple(key.entity for key in observed.target_object_keys) != tuple(key.entity for key in gold.target_object_keys): flags.add("wrong_entity")
            if tuple(key.attribute for key in observed.target_object_keys) != tuple(key.attribute for key in gold.target_object_keys): flags.add("wrong_attribute")
            if gold.operation == Operation.DELETE: flags.add("wrong_delete_scope")
        if not _same(observed.value, gold.value): flags.add("wrong_value")
        if gold.operation == Operation.NOOP and observed.operation not in {None, Operation.NOOP}: flags.add("false_write")
        if gold.operation != Operation.NOOP and observed.execution_status != ExecutionStatusV3.EXECUTED:
            flags.add("missed_update")
            if gold.operation == Operation.DELETE: flags.add("deletion_failure")
    deletion = layer_values["deletion_scores"]
    historical = layer_values["historical_scores"]
    synthesis = layer_values["synthesis_scores"]
    if deletion.get("collateral_damage_rate") not in {None, 0.0}: flags.update(("collateral_mutation", "collateral_corruption"))
    if deletion.get("ttl_compliance_rate") not in {None, 1.0}: flags.add("ttl_violation")
    if deletion.get("relearn_accuracy") not in {None, 1.0}: flags.add("current_state_missing")
    forgotten = _forgotten_values(replay)
    obsolete = replay.obsolete_present_values
    event_positions = {event.event_id: event.sequence_index for event in task.events}
    snapshots = [snapshot for snapshot in run.memory_snapshots if snapshot.after_event_id in event_positions]
    final_snapshot = max(snapshots, key=lambda snapshot: event_positions[snapshot.after_event_id]) if snapshots else (run.memory_snapshots[-1] if run.memory_snapshots else None)
    if final_snapshot and any(_entry_version_status(entry, replay)[0] is True for entry in final_snapshot.entries): flags.add("stale_retained")
    if any(trace.stale_in_context is True or any(_entry_version_status(entry, replay)[0] is True for entry in trace.retrieved_entries) for trace in traces.values()): flags.add("stale_retrieved")
    if any(any(_entry_version_status(entry, replay)[1] is True for entry in trace.retrieved_entries) for trace in traces.values()): flags.add("forgotten_value_exposed")
    if any(not _same(prediction.parsed_answer, evidence[prediction.query_id].answer) and any(_same(prediction.parsed_answer, value) for value in obsolete) for prediction in predictions.values()): flags.add("stale_copied")
    if any(not _same(prediction.parsed_answer, evidence[prediction.query_id].answer) and any(_same(prediction.parsed_answer, value) for value in forgotten) for prediction in predictions.values()): flags.add("forgotten_value_exposed")
    for query_id, prediction in predictions.items():
        trace = traces.get(query_id)
        gold_answer = evidence[query_id].answer
        semantic_wrong = not _same(prediction.parsed_answer, gold_answer)
        wrong = not prediction.format_valid or semantic_wrong
        if not prediction.format_valid and _same(prediction.parsed_answer, gold_answer):
            flags.add("answer_format_only")
        if trace is not None:
            if trace.gold_in_context is False:
                flags.add("current_not_retrieved")
            if trace.gold_in_context is True and wrong:
                flags.add("gold_retrieved_wrong_answer")
            if trace.distractor_in_context is True:
                flags.add("distractor_retrieved")
                candidates = tuple(
                    candidate
                    for entry in trace.retrieved_entries
                    if entry.raw_metadata.get("is_distractor") is True or entry.raw_metadata.get("role") == "distractor"
                    for candidate in (entry.value_candidate, entry.content)
                    if candidate is not None
                )
                if semantic_wrong and any(_contains_value(prediction.parsed_answer, candidate) for candidate in candidates):
                    flags.add("distractor_copied")
    if historical.get("version_confusion_rate") not in {None, 0.0}: flags.add("version_confusion")
    if any(historical.get(field) not in {None, 1.0} for field in ("previous_state_accuracy", "point_in_time_accuracy", "transition_accuracy", "ordered_history_accuracy", "historical_distance_accuracy")): flags.add("version_confusion")
    if historical.get("historical_support_recall") not in {None, 1.0}: flags.update(("evidence_linkage_error", "current_not_retrieved"))
    for query in task.queries:
        if query.query_id not in predictions: continue
        prediction = predictions[query.query_id]
        gold = evidence[query.query_id]
        if query.query_type in {QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY}:
            required_events = set(gold.supporting_event_ids); observed_events = set(prediction.cited_event_ids)
            required_objects = {key.canonical_id for key in gold.supporting_object_keys}; observed_objects = {key.canonical_id for key in prediction.cited_object_keys}
            required_steps = {step.step_id for step in gold.derivation_steps}; observed_steps = set(prediction.cited_derivation_step_ids)
            if not required_events <= observed_events or not required_objects <= observed_objects or not required_steps <= observed_steps: flags.add("evidence_linkage_error")
            if observed_events - required_events or observed_objects - required_objects or observed_steps - required_steps: flags.add("evidence_linkage_error")
    if synthesis.get("stale_propagation_rate") not in {None, 0.0}: flags.add("stale_propagation")
    if synthesis.get("multi_object_accuracy") not in {None, 1.0}: flags.add("wrong_value")
    ordered = [flag for flag in V3_FAILURE_FLAGS if flag in flags]
    return tuple(ordered)


class CoverageCellV3(ImmutableContractModel):
    numerator: StrictNonnegativeInt
    denominator: StrictNonnegativeInt
    coverage: StrictNonnegativeFloat | None


class FailureTaxonomyCoverageV3(ImmutableContractModel):
    overall: CoverageCellV3
    by_family: Mapping[str, CoverageCellV3]

    @field_validator("by_family")
    @classmethod
    def _freeze(cls, value): return freeze_mapping(value)


def failure_taxonomy_coverage_v3(scores: Iterable[ScoreRecordV3], *, families: Iterable[str] | None = None) -> FailureTaxonomyCoverageV3:
    rows = tuple(scores)
    requested_families = tuple(dict.fromkeys(families or (family.value for family in TaskFamily)))
    all_families = tuple(dict.fromkeys((*requested_families, *(row.task_family for row in rows))))
    non_system = set(V3_FAILURE_FLAGS) - {"system_exception"}

    def qualifies(row):
        if row.completion_status != CompletionStatus.COMPLETED: return False
        row_flags = {flag.value if hasattr(flag, "value") else flag for flag in row.failure_flags}
        if row_flags == {"system_exception"}:
            return False
        principal_values = []
        for path, descriptor in CORE_METRIC_REGISTRY_V3.items():
            if not descriptor.principal: continue
            layer, leaf = path.split(".", 1); value = getattr(getattr(row, layer), leaf)
            if value is not None: principal_values.append((descriptor.direction, value))
        if not principal_values: return False
        incorrect = any((direction == "higher" and value != 1.0) or (direction == "lower" and value != 0.0) for direction, value in principal_values)
        return incorrect

    def cell(subset):
        denominator_rows = [row for row in subset if qualifies(row)]
        numerator = sum(bool(set(flag.value if hasattr(flag, "value") else flag for flag in row.failure_flags) & non_system) for row in denominator_rows)
        denominator = len(denominator_rows)
        return CoverageCellV3(numerator=numerator, denominator=denominator, coverage=None if denominator == 0 else numerator / denominator)

    return FailureTaxonomyCoverageV3(overall=cell(rows), by_family={family: cell([row for row in rows if row.task_family == family]) for family in all_families})


__all__ = ["CoverageCellV3", "FailureTaxonomyCoverageV3", "derive_failure_flags_v3", "failure_taxonomy_coverage_v3"]
