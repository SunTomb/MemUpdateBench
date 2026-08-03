from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.enums import ActionScope, AnswerSchema, Difficulty, EvaluationMode, EventRole, Operation, SourceType, Split
from mub.vnext.contracts.v3.common import FrozenJsonObjectV3, FrozenJsonValue, MemoryObjectKeyV3, StrictIdentifier, object_identity, validate_action_coherence
from mub.vnext.contracts.v3.enums import LedgerEntryStatus, QueryTypeV3, SynthesisKindV3
from mub.vnext.contracts.v3.version import SCHEMA_VERSION_V3

StrictString = StrictIdentifier
StrictIndex = Annotated[int, Field(strict=True, ge=0)]
HashString = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
_DERIVATION_READ_OPERATIONS = frozenset({"read", "read_current", "read_version"})
_DERIVATION_SEED_OPERATIONS = frozenset({"seed0", "seed1"})
_DERIVATION_COLLECTION_OPERATIONS = frozenset({
    "list", "ordered_history", "collect", "combine", "merge",
    "object", "multi_object", "consistency",
})


def _derivation_step_reads_support(step) -> bool:
    if step.operation in _DERIVATION_READ_OPERATIONS:
        return True
    if step.operation in _DERIVATION_SEED_OPERATIONS:
        return len(step.input_step_ids) != 1
    return step.operation in _DERIVATION_COLLECTION_OPERATIONS and not step.input_step_ids


class GeneratorProvenanceV3(ImmutableContractModel):
    generator_name: StrictString
    seed: Annotated[int, Field(strict=True)]
    config_sha256: HashString
    code_revision: StrictString
    compiler_version: StrictString


class SourceRecordV3(ImmutableContractModel):
    source_id: StrictString
    source_type: SourceType
    source_uri: str | None
    license_or_privacy: StrictString
    raw_hash: HashString | None
    normalized_hash: HashString
    normalization_version: StrictString
    provenance: FrozenJsonObjectV3 = Field(default_factory=dict)
    generator: GeneratorProvenanceV3 | None = None

    @model_validator(mode="after")
    def _synthetic_generator(self) -> Self:
        if self.source_type == SourceType.SYNTHETIC and self.generator is None:
            raise ValueError("synthetic sources require generator provenance")
        return self


class MemoryEventV3(ImmutableContractModel):
    event_id: StrictString
    sequence_index: StrictIndex
    timestamp: str | None = Field(default=None, strict=True)
    raw_text: str = Field(strict=True)
    normalized_text: str = Field(strict=True)
    speaker: str | None = Field(default=None, strict=True)
    gold_action_ids: tuple[StrictString, ...] = ()
    role: EventRole
    source_anchor: FrozenJsonObjectV3 = Field(default_factory=dict)
    metadata: FrozenJsonObjectV3 = Field(default_factory=dict)


class SplitKeyV3(ImmutableContractModel):
    semantic_core_id: StrictString
    source_group_id: StrictString
    trajectory_id: StrictString
    paraphrase_group_id: StrictIdentifier | None = None
    source_document_id: StrictIdentifier | None = None
    version_group_id: StrictIdentifier | None = None
    split_exception_id: StrictIdentifier | None = None
    split_policy_version: StrictString


class LegacyProvenanceV3(ImmutableContractModel):
    legacy_family_id: StrictString
    legacy_phase: StrictString
    legacy_dataset_id: StrictString
    legacy_split_id: StrictString
    legacy_metric_namespace: StrictString
    legacy_run_condition_id: StrictIdentifier | None = None
    checkpoint_family: str | None = None
    training_seed: int | None = Field(default=None, strict=True)
    answer_mode: str | None = None
    memory_trajectory_id: StrictIdentifier | None = None
    source_artifact_path: StrictString
    source_artifact_hash: HashString
    known_caveats: tuple[str, ...] = ()


class TaskMetadataV3(ImmutableContractModel):
    split: Split
    split_key: SplitKeyV3
    profile_name: Difficulty
    resolved_profile: FrozenJsonObjectV3 = Field(default_factory=dict)
    generation_config_hash: HashString
    compiler_version: StrictString
    tags: tuple[str, ...] = ()
    legacy_provenance: LegacyProvenanceV3 | None = None
    extra: FrozenJsonObjectV3 = Field(default_factory=dict)


class GoldActionV3(ImmutableContractModel):
    action_id: StrictString
    event_id: StrictString
    operation: Operation
    scope: ActionScope | None = None
    target_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    value: FrozenJsonValue | None = None
    effective_at: str | None = Field(default=None, strict=True)
    expected_effect: FrozenJsonObjectV3 = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        validate_action_coherence(operation=self.operation, scope=self.scope, targets=self.target_object_keys, value=self.value)
        return self


class CurrentSelector(ImmutableContractModel):
    kind: Literal["current"] = "current"


class PreviousSelector(ImmutableContractModel):
    kind: Literal["previous"] = "previous"


class ExactVersionSelector(ImmutableContractModel):
    kind: Literal["exact_version"] = "exact_version"
    version_index: StrictIndex


class EventAnchorSelector(ImmutableContractModel):
    kind: Literal["event_anchor"] = "event_anchor"
    event_id: StrictString


class LogicalTimeAnchorSelector(ImmutableContractModel):
    kind: Literal["logical_time_anchor"] = "logical_time_anchor"
    logical_time: StrictString


class TransitionSelector(ImmutableContractModel):
    kind: Literal["transition"] = "transition"
    from_version_index: StrictIndex
    to_version_index: StrictIndex

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.from_version_index >= self.to_version_index:
            raise ValueError("transition versions must be strictly ordered")
        return self


class OrderedHistorySelector(ImmutableContractModel):
    kind: Literal["ordered_history"] = "ordered_history"
    start_version_index: StrictIndex | None = None
    end_version_index: StrictIndex | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start_version_index is not None and self.end_version_index is not None and self.start_version_index > self.end_version_index:
            raise ValueError("history range must be ordered")
        return self


class MultiObjectCurrentSelector(ImmutableContractModel):
    kind: Literal["multi_object_current"] = "multi_object_current"
    object_keys: tuple[MemoryObjectKeyV3, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        _require_unique_objects(self.object_keys, "multi-object selector")
        return self


SelectorV3 = Annotated[
    CurrentSelector | PreviousSelector | ExactVersionSelector | EventAnchorSelector |
    LogicalTimeAnchorSelector | TransitionSelector | OrderedHistorySelector | MultiObjectCurrentSelector,
    Field(discriminator="kind"),
]


class UpdateSensitiveMultiHopSynthesis(ImmutableContractModel):
    kind: Literal["update_sensitive_multi_hop"] = "update_sensitive_multi_hop"
    minimum_hops: Annotated[int, Field(strict=True, ge=2)] = 2


class MultiObjectCurrentConsistencySynthesis(ImmutableContractModel):
    kind: Literal["multi_object_current_consistency"] = "multi_object_current_consistency"
    minimum_objects: Annotated[int, Field(strict=True, ge=2)] = 2


SynthesisSpecV3 = Annotated[
    UpdateSensitiveMultiHopSynthesis | MultiObjectCurrentConsistencySynthesis,
    Field(discriminator="kind"),
]


class MemoryQueryV3(ImmutableContractModel):
    query_id: StrictString
    query_type: QueryTypeV3
    text: Annotated[str, Field(strict=True)]
    selector: SelectorV3
    target_object_keys: tuple[MemoryObjectKeyV3, ...] = Field(min_length=1)
    answer_schema: AnswerSchema
    evaluation_mode: EvaluationMode
    synthesis: SynthesisSpecV3 | None = None

    @model_validator(mode="after")
    def _selector_matches_query(self) -> Self:
        allowed = {
            QueryTypeV3.CURRENT: {"current"},
            QueryTypeV3.PREVIOUS: {"previous"},
            QueryTypeV3.POINT_IN_TIME: {"exact_version", "event_anchor", "logical_time_anchor"},
            QueryTypeV3.TRANSITION: {"transition"},
            QueryTypeV3.ORDERED_HISTORY: {"ordered_history"},
            QueryTypeV3.MULTI_OBJECT_CURRENT: {"multi_object_current"},
            QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP: {"current", "previous", "exact_version", "event_anchor", "logical_time_anchor", "transition", "ordered_history", "multi_object_current"},
            QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY: {"multi_object_current"},
        }
        if self.selector.kind not in allowed[self.query_type]:
            raise ValueError("selector kind does not match query_type")
        expected_synthesis = {
            QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP: SynthesisKindV3.UPDATE_SENSITIVE_MULTI_HOP,
            QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY: SynthesisKindV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
        }.get(self.query_type)
        if expected_synthesis is None and self.synthesis is not None:
            raise ValueError("non-G query cannot carry synthesis")
        if expected_synthesis is not None and (self.synthesis is None or self.synthesis.kind != expected_synthesis.value):
            raise ValueError("G query requires matching synthesis kind")
        if isinstance(self.selector, MultiObjectCurrentSelector):
            if {_identity(k) for k in self.selector.object_keys} != {_identity(k) for k in self.target_object_keys}:
                raise ValueError("multi-object selector scope must match query targets")
        _require_unique_objects(self.target_object_keys, "query targets")
        return self


class VersionHistoryEntry(ImmutableContractModel):
    version_index: StrictIndex
    status: LedgerEntryStatus
    value: FrozenJsonValue | None = None
    valid_from_event_id: StrictString | None = None
    valid_until_event_id: StrictString | None = None
    logical_time: StrictString | None = None
    source_event_ids: tuple[StrictString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_status(self) -> Self:
        if self.status == LedgerEntryStatus.PRESENT and self.value is None:
            raise ValueError("present ledger entries require value")
        if self.status == LedgerEntryStatus.TOMBSTONE and self.value is not None:
            raise ValueError("tombstones cannot carry value")
        if self.valid_from_event_id is None and self.logical_time is None:
            raise ValueError("ledger entries require an event or logical-time anchor")
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("source_event_ids must be unique")
        return self


class VersionHistoryLedger(ImmutableContractModel):
    object_key: MemoryObjectKeyV3
    entries: tuple[VersionHistoryEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _contiguous(self) -> Self:
        if tuple(entry.version_index for entry in self.entries) != tuple(range(len(self.entries))):
            raise ValueError("version history must start at zero and be contiguous")
        return self

    @property
    def semantic_identity(self) -> tuple[str, str, str, str | None]:
        return _identity(self.object_key)

    @property
    def semantic_hash(self) -> str:
        return hashlib.sha256(_canonical_bytes(_semantic_value(self))).hexdigest()


class DerivationStepV3(ImmutableContractModel):
    step_id: StrictString
    operation: StrictString
    input_step_ids: tuple[StrictString, ...] = ()
    supporting_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    supporting_event_ids: tuple[StrictString, ...] = ()

    @model_validator(mode="after")
    def _unique_support(self) -> Self:
        if len(self.input_step_ids) != len(set(self.input_step_ids)):
            raise ValueError("derivation input step IDs must be unique")
        if len(self.supporting_event_ids) != len(set(self.supporting_event_ids)):
            raise ValueError("derivation supporting event IDs must be unique")
        if len({_identity(key) for key in self.supporting_object_keys}) != len(self.supporting_object_keys):
            raise ValueError("derivation supporting object identities must be unique")
        return self


def _validate_derivation_graph(evidence):
    if len(evidence.supporting_event_ids) != len(set(evidence.supporting_event_ids)):
        raise ValueError("gold supporting event IDs must be unique")
    if len({_identity(key) for key in evidence.supporting_object_keys}) != len(evidence.supporting_object_keys):
        raise ValueError("gold supporting object identities must be unique")
    steps = {step.step_id: step for step in evidence.derivation_steps}
    if len(steps) != len(evidence.derivation_steps):
        raise ValueError("derivation step IDs must be unique")
    if evidence.final_derivation_step_id not in steps:
        raise ValueError("final derivation step is unknown")
    positions = {step.step_id: index for index, step in enumerate(evidence.derivation_steps)}
    for step in evidence.derivation_steps:
        if step.operation == "equals" and len(step.input_step_ids) < 2:
            raise ValueError("equals requires at least two operands")
        if set(step.input_step_ids) - steps.keys():
            raise ValueError("derivation references unknown input step")
    visiting: set[str] = set()
    reached: set[str] = set()
    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("cyclic derivation graph")
        if step_id in reached:
            return
        visiting.add(step_id)
        for parent in steps[step_id].input_step_ids:
            visit(parent)
        visiting.remove(step_id)
        reached.add(step_id)
    visit(evidence.final_derivation_step_id)
    if reached != set(steps):
        raise ValueError("disconnected derivation graph")
    for step in evidence.derivation_steps:
        if any(positions[parent] >= positions[step.step_id] for parent in step.input_step_ids):
            raise ValueError("derivation steps must be in topological order")
    return evidence


class StaleAlternativeEvidenceV3(ImmutableContractModel):
    answer: FrozenJsonValue
    supporting_object_keys: tuple[MemoryObjectKeyV3, ...] = Field(min_length=1)
    supporting_event_ids: tuple[StrictString, ...] = Field(min_length=1)
    derivation_steps: tuple[DerivationStepV3, ...] = Field(min_length=1)
    final_derivation_step_id: StrictString

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        return _validate_derivation_graph(self)


class QueryGoldEvidenceV3(ImmutableContractModel):
    query_id: StrictString
    answer: FrozenJsonValue
    supporting_object_keys: tuple[MemoryObjectKeyV3, ...] = Field(min_length=1)
    supporting_event_ids: tuple[StrictString, ...] = Field(min_length=1)
    derivation_steps: tuple[DerivationStepV3, ...] = Field(min_length=1)
    final_derivation_step_id: StrictString
    stale_alternative: StaleAlternativeEvidenceV3 | None = None

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        return _validate_derivation_graph(self)


class MemUpdateTaskV3(ImmutableContractModel):
    task_id: StrictString
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    task_family: StrictString
    difficulty: Difficulty
    source: SourceRecordV3
    events: tuple[MemoryEventV3, ...]
    target_objects: tuple[MemoryObjectKeyV3, ...] = Field(min_length=1)
    actions: tuple[GoldActionV3, ...] = ()
    queries: tuple[MemoryQueryV3, ...] = Field(min_length=1)
    version_history: tuple[VersionHistoryLedger, ...] = Field(min_length=1)
    gold_evidence: tuple[QueryGoldEvidenceV3, ...] = Field(min_length=1)
    metadata: TaskMetadataV3

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)) or [event.sequence_index for event in self.events] != list(range(len(self.events))):
            raise ValueError("events must have unique IDs and contiguous sequence indices")
        declared = {_identity(key) for key in self.target_objects}
        if len(declared) != len(self.target_objects):
            raise ValueError("target object identities must be unique")
        histories = {_identity(item.object_key): item for item in self.version_history}
        if len(histories) != len(self.version_history):
            raise ValueError("duplicate canonical version ledgers are not allowed")
        if set(histories) != declared:
            raise ValueError("version histories must cover declared targets exactly")
        event_position = {event_id: index for index, event_id in enumerate(event_ids)}
        event_times = {
            event.event_id: event.timestamp
            for event in self.events
            if event.timestamp is not None
        }
        action_by_id = {action.action_id: action for action in self.actions}
        if len(action_by_id) != len(self.actions):
            raise ValueError("action IDs must be unique")
        for event in self.events:
            if len(event.gold_action_ids) != len(set(event.gold_action_ids)):
                raise ValueError("event gold action IDs must be unique")
            for action_id in event.gold_action_ids:
                if action_id not in action_by_id or action_by_id[action_id].event_id != event.event_id:
                    raise ValueError("event references missing or mismatched action")
        for action in self.actions:
            if action.event_id not in event_ids:
                raise ValueError("action references unknown event")
            if action.action_id not in self.events[event_position[action.event_id]].gold_action_ids:
                raise ValueError("action is not owned by its source event")
            if {_identity(key) for key in action.target_object_keys} - declared:
                raise ValueError("action targets undeclared object")
        for ledger in self.version_history:
            for index, entry in enumerate(ledger.entries):
                anchors = {entry.valid_from_event_id, entry.valid_until_event_id} - {None}
                if anchors - set(event_ids) or set(entry.source_event_ids) - set(event_ids):
                    raise ValueError("version history references unknown event anchor")
                if entry.valid_from_event_id is not None and entry.valid_until_event_id is not None and event_position[entry.valid_from_event_id] >= event_position[entry.valid_until_event_id]:
                    raise ValueError("version validity event interval must be ordered")
                source_positions = [event_position[event_id] for event_id in entry.source_event_ids]
                if source_positions != sorted(source_positions):
                    raise ValueError("version source events must be chronological")
                if entry.valid_from_event_id is not None and any(position < event_position[entry.valid_from_event_id] for position in source_positions):
                    raise ValueError("source event precedes version validity")
                if entry.valid_until_event_id is not None and any(position >= event_position[entry.valid_until_event_id] for position in source_positions):
                    raise ValueError("source event falls outside version validity")
                if index + 1 < len(ledger.entries):
                    following = ledger.entries[index + 1]
                    event_boundary = (entry.valid_until_event_id, following.valid_from_event_id)
                    if (event_boundary[0] is None) != (event_boundary[1] is None):
                        raise ValueError("adjacent partial event intervals are not allowed")
                    if event_boundary[0] is not None and event_boundary[0] != event_boundary[1]:
                        raise ValueError("adjacent version validity intervals must be contiguous")
                    logical_boundary = (entry.logical_time, following.logical_time)
                    if (logical_boundary[0] is None) != (logical_boundary[1] is None):
                        raise ValueError("adjacent partial logical-time intervals are not allowed")
                    if logical_boundary[0] is not None and logical_boundary[0] >= logical_boundary[1]:
                        raise ValueError("logical-time anchors must be strictly increasing")
                    if event_boundary == (None, None) and logical_boundary == (None, None):
                        raise ValueError("adjacent versions require event or logical-time continuity")
        horizon_candidates = [event.timestamp for event in self.events if event.timestamp is not None]
        horizon_candidates.extend(
            action.effective_at
            for action in self.actions
            if action.effective_at is not None and not (action.operation == Operation.DELETE and action.scope == ActionScope.TTL)
        )
        horizon_candidates.extend(
            query.selector.logical_time
            for query in self.queries
            if isinstance(query.selector, LogicalTimeAnchorSelector)
        )
        task_horizon = max(horizon_candidates) if horizon_candidates else None
        query_by_id = {query.query_id: query for query in self.queries}
        if len(query_by_id) != len(self.queries):
            raise ValueError("query IDs must be unique")
        evidence_by_id = {item.query_id: item for item in self.gold_evidence}
        if len(evidence_by_id) != len(self.gold_evidence):
            raise ValueError("duplicate query evidence rows are not allowed")
        if set(evidence_by_id) != set(query_by_id):
            raise ValueError("gold evidence must cover queries exactly")
        for query in self.queries:
            targets = {_identity(key) for key in query.target_object_keys}
            if not targets <= declared:
                raise ValueError("query targets undeclared object")
            if any(not histories[target].entries for target in targets):
                raise ValueError("query selectors require nonempty version histories")
            selector = query.selector
            if isinstance(selector, EventAnchorSelector) and selector.event_id not in event_ids:
                raise ValueError("query selector references unknown event anchor")
            if isinstance(selector, LogicalTimeAnchorSelector) and any(
                not any(entry.logical_time is not None and entry.logical_time <= selector.logical_time for entry in histories[target].entries)
                for target in targets
            ):
                raise ValueError("query selector precedes all known logical-time anchors")
            if isinstance(selector, ExactVersionSelector):
                if any(selector.version_index >= len(histories[target].entries) for target in targets):
                    raise ValueError("query selector references unknown version")
            if isinstance(selector, PreviousSelector) and any(len(histories[target].entries) < 2 for target in targets):
                raise ValueError("previous selector requires at least two versions")
            if isinstance(selector, TransitionSelector) and any(selector.to_version_index >= len(histories[target].entries) for target in targets):
                raise ValueError("transition selector references unknown version")
            if isinstance(selector, OrderedHistorySelector):
                for target in targets:
                    size = len(_horizon_active_entries(histories[target].entries, task_horizon))
                    if selector.start_version_index is not None and selector.start_version_index >= size:
                        raise ValueError("history selector references unknown start version")
                    if selector.end_version_index is not None and selector.end_version_index >= size:
                        raise ValueError("history selector references unknown end version")
            evidence = evidence_by_id[query.query_id]
            evidence_objects = {_identity(key) for key in evidence.supporting_object_keys}
            if not evidence_objects <= declared or not targets <= evidence_objects:
                raise ValueError("answer evidence is not coherent with query targets")
            if set(evidence.supporting_event_ids) - set(event_ids):
                raise ValueError("gold evidence references unknown event")
            _validate_answer_schema(evidence.answer, query.answer_schema)
            selected_entries_by_target = {
                target: _selector_entries(query.selector, histories[target], event_position, event_times, task_horizon)
                for target in targets
            }
            selected_entries = [
                entry
                for entries in selected_entries_by_target.values()
                for entry in entries
            ]
            if not selected_entries:
                raise ValueError("selector does not resolve any version history entries")
            required_events = {event_id for entry in selected_entries for event_id in entry.source_event_ids}
            if not required_events <= set(evidence.supporting_event_ids):
                raise ValueError("evidence does not support selector-selected versions")
            step_ids = {step.step_id for step in evidence.derivation_steps}
            for step in evidence.derivation_steps:
                if {_identity(key) for key in step.supporting_object_keys} - evidence_objects:
                    raise ValueError("derivation uses object outside evidence scope")
                if set(step.supporting_event_ids) - set(evidence.supporting_event_ids):
                    raise ValueError("derivation uses event outside evidence scope")
            _validate_derivation_read_bindings(
                evidence,
                histories,
                include_implicit=query.query_type in {
                    QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
                    QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
                },
            )
            alternative = evidence.stale_alternative
            if alternative is not None:
                if query.query_type not in {QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY}:
                    raise ValueError("stale alternative is only valid for G synthesis queries")
                alternative_objects = {_identity(key) for key in alternative.supporting_object_keys}
                if not targets <= alternative_objects or not alternative_objects <= declared:
                    raise ValueError("stale alternative objects are not coherent with query targets")
                if set(alternative.supporting_event_ids) - set(event_ids):
                    raise ValueError("stale alternative references unknown event")
                _validate_answer_schema(alternative.answer, query.answer_schema)
                for step in alternative.derivation_steps:
                    if {_identity(key) for key in step.supporting_object_keys} - alternative_objects:
                        raise ValueError("stale derivation uses object outside alternative scope")
                    if set(step.supporting_event_ids) - set(alternative.supporting_event_ids):
                        raise ValueError("stale derivation uses event outside alternative scope")
                _validate_derivation_read_bindings(alternative, histories, include_implicit=True)
            if query.query_type == QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP:
                minimum_hops = query.synthesis.minimum_hops
                if _derivation_depth(evidence) < minimum_hops:
                    raise ValueError("G derivation does not satisfy minimum_hops")
                if alternative is not None and _derivation_depth(alternative) < minimum_hops:
                    raise ValueError("stale alternative does not satisfy minimum_hops")
            if query.query_type == QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY:
                minimum_objects = query.synthesis.minimum_objects
                read_count, read_objects = _derivation_read_support(evidence, targets)
                if len(targets) < minimum_objects or read_count < minimum_objects or len(read_objects) < minimum_objects:
                    raise ValueError("G derivation does not satisfy minimum_objects")
                _validate_consistency_read_eligibility(
                    evidence,
                    targets,
                    histories,
                    selected_entries_by_target,
                )
                if alternative is not None:
                    stale_read_count, stale_read_objects = _derivation_read_support(alternative, targets)
                    if stale_read_count < minimum_objects or len(stale_read_objects) < minimum_objects:
                        raise ValueError("stale alternative does not satisfy minimum_objects")
                    _validate_consistency_read_eligibility(
                        alternative,
                        targets,
                        histories,
                        require_exact_event_coverage=True,
                    )
        semantic_queries = [
            _canonical_bytes(_query_semantic_projection(query, event_position))
            for query in self.queries
        ]
        if len(semantic_queries) != len(set(semantic_queries)):
            raise ValueError("duplicate semantic query projections are not allowed")
        return self

    @property
    def semantic_identity(self) -> Mapping[str, JsonValue]:
        return _semantic_task_projection(self)

    @property
    def semantic_hash(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.semantic_identity)).hexdigest()


def _identity(key: MemoryObjectKeyV3) -> tuple[str, str, str, str | None]:
    return key.namespace, key.entity, key.attribute, key.subkey


def _resolve_derivation_read_versions(step, ledgers, version_rows):
    supporting_events = set(step.supporting_event_ids)
    if step.supporting_object_keys:
        resolved = []
        for key in step.supporting_object_keys:
            ledger = ledgers.get(_identity(key))
            if ledger is None:
                raise ValueError("derivation support object is missing")
            candidates = tuple(
                version
                for version in version_rows(ledger)
                if set(version.source_event_ids) & supporting_events
            )
            if len(candidates) != 1:
                raise ValueError("derivation read support is missing or ambiguous")
            resolved.append(candidates[0])
        return tuple(resolved)
    candidates = tuple(
        version
        for ledger in ledgers.values()
        for version in version_rows(ledger)
        if set(version.source_event_ids) & supporting_events
    )
    if len(candidates) != 1:
        raise ValueError("derivation event support is missing or ambiguous")
    return candidates


def _validate_derivation_read_bindings(evidence, histories, include_implicit=False) -> None:
    for step in evidence.derivation_steps:
        reads_support = (
            step.operation in _DERIVATION_READ_OPERATIONS
            or include_implicit and _derivation_step_reads_support(step)
        )
        if reads_support:
            _resolve_derivation_read_versions(step, histories, lambda ledger: ledger.entries)


def _validate_consistency_read_eligibility(
    evidence,
    targets,
    ledgers,
    selected_by_target=None,
    require_exact_event_coverage=False,
    version_rows=None,
) -> None:
    if version_rows is None:
        version_rows = lambda ledger: ledger.entries
    consumed_events = set()
    for step in evidence.derivation_steps:
        if not _derivation_step_reads_support(step) or len(step.supporting_object_keys) != 1:
            continue
        identity = _identity(step.supporting_object_keys[0])
        if identity not in targets:
            continue
        version = _resolve_derivation_read_versions(
            step,
            ledgers,
            version_rows,
        )[0]
        if selected_by_target is not None and all(
            version.version_index != selected.version_index
            for selected in selected_by_target[identity]
        ):
            raise ValueError("derivation read provenance is not eligible for the query selector")
        consumed_events.update(step.supporting_event_ids)
    if require_exact_event_coverage and consumed_events != set(evidence.supporting_event_ids):
        raise ValueError("derivation read provenance is not eligible for authenticated evidence support")


def _require_unique_objects(keys: tuple[MemoryObjectKeyV3, ...], label: str) -> None:
    if len({_identity(key) for key in keys}) != len(keys):
        raise ValueError(f"{label} must contain unique canonical identities")


def _semantic_anchor_projection(anchor: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    surface_only = {"normalized_text_sha256", "raw_text", "normalized_text", "text"}
    return {
        key: _semantic_value(value)
        for key, value in sorted(anchor.items())
        if key not in surface_only
    }


def _query_semantic_projection(query: MemoryQueryV3, event_index: Mapping[str, int]) -> Mapping[str, JsonValue]:
    selector = _semantic_value(query.selector)
    if isinstance(query.selector, EventAnchorSelector):
        try:
            selector["event_id"] = event_index[query.selector.event_id]
        except KeyError as exc:
            raise ValueError(
                f"query selector references missing event anchor {query.selector.event_id!r}"
            ) from exc
    if isinstance(query.selector, MultiObjectCurrentSelector):
        selector["object_keys"] = sorted(
            (_semantic_value(key) for key in query.selector.object_keys),
            key=_canonical_bytes,
        )
    return {
        "query_type": query.query_type.value,
        "selector": selector,
        "targets": sorted((_semantic_value(key) for key in query.target_object_keys), key=_canonical_bytes),
        "answer_schema": query.answer_schema.value,
        "evaluation_mode": query.evaluation_mode.value,
        "synthesis": _semantic_value(query.synthesis),
    }


def _derivation_semantic_projection(
    gold: QueryGoldEvidenceV3,
    event_index: Mapping[str, int],
) -> Mapping[str, JsonValue]:
    steps = {step.step_id: step for step in gold.derivation_steps}
    canonical_numbers: dict[str, int] = {}
    records: list[Mapping[str, JsonValue] | None] = []
    visiting: set[str] = set()

    def discover(step_id: str) -> int:
        if step_id in visiting:
            raise ValueError("cyclic derivation graph cannot be projected")
        if step_id in canonical_numbers:
            return canonical_numbers[step_id]
        try:
            step = steps[step_id]
        except KeyError as exc:
            raise ValueError(f"derivation references missing step {step_id!r}") from exc
        number = len(records)
        canonical_numbers[step_id] = number
        records.append(None)
        visiting.add(step_id)
        input_numbers = tuple(discover(parent) for parent in step.input_step_ids)
        visiting.remove(step_id)
        records[number] = {
            "operation": step.operation,
            "inputs": input_numbers,
            "supporting_objects": sorted(
                (_semantic_value(key) for key in step.supporting_object_keys),
                key=_canonical_bytes,
            ),
            "supporting_event_indices": sorted(
                event_index[event_id] for event_id in step.supporting_event_ids
            ),
        }
        return number

    root = discover(gold.final_derivation_step_id)
    if any(record is None for record in records):
        raise ValueError("derivation canonicalization left unresolved nodes")
    return {"root": root, "nodes": records}


def _semantic_task_projection(task: MemUpdateTaskV3) -> Mapping[str, JsonValue]:
    event_index = {event.event_id: index for index, event in enumerate(task.events)}
    action_index = {action.action_id: index for index, action in enumerate(task.actions)}

    def event_ref(event_id: str | None):
        return None if event_id is None else event_index[event_id]

    events = []
    for event in task.events:
        events.append({
            "role": event.role.value,
            "timestamp": event.timestamp,
            "source_anchor": _semantic_anchor_projection(event.source_anchor),
            "gold_action_indices": sorted(action_index[action_id] for action_id in event.gold_action_ids),
        })
    actions = [{
        "event_index": event_ref(action.event_id),
        "operation": action.operation.value,
        "scope": None if action.scope is None else action.scope.value,
        "targets": sorted((_semantic_value(key) for key in action.target_object_keys), key=_canonical_bytes),
        "value": _semantic_value(action.value),
        "effective_at": action.effective_at,
        "expected_effect": _semantic_value(action.expected_effect),
    } for action in task.actions]
    query_projection_by_id = {
        query.query_id: _query_semantic_projection(query, event_index)
        for query in task.queries
    }
    histories = []
    for ledger in task.version_history:
        histories.append({
            "object_key": _semantic_value(ledger.object_key),
            "entries": [{
                "version_index": entry.version_index,
                "status": entry.status.value,
                "value": _semantic_value(entry.value),
                "valid_from_event_index": event_ref(entry.valid_from_event_id),
                "valid_until_event_index": event_ref(entry.valid_until_event_id),
                "logical_time": entry.logical_time,
                "source_event_indices": [event_ref(event_id) for event_id in entry.source_event_ids],
            } for entry in ledger.entries],
        })
    histories.sort(key=lambda item: _canonical_bytes(item["object_key"]))
    evidence_by_query = {}
    for gold in task.gold_evidence:
        alternative = gold.stale_alternative
        evidence_by_query[gold.query_id] = {
            "answer": _semantic_value(gold.answer),
            "supporting_objects": sorted((_semantic_value(key) for key in gold.supporting_object_keys), key=_canonical_bytes),
            "supporting_event_indices": sorted(event_ref(event_id) for event_id in gold.supporting_event_ids),
            "derivation_graph": _derivation_semantic_projection(gold, event_index),
            "stale_alternative": None if alternative is None else {
                "answer": _semantic_value(alternative.answer),
                "supporting_objects": sorted((_semantic_value(key) for key in alternative.supporting_object_keys), key=_canonical_bytes),
                "supporting_event_indices": sorted(event_ref(event_id) for event_id in alternative.supporting_event_ids),
                "derivation_graph": _derivation_semantic_projection(alternative, event_index),
            },
        }
    bundles = [
        {
            "query": query_projection_by_id[query.query_id],
            "evidence": evidence_by_query[query.query_id],
        }
        for query in task.queries
    ]
    bundles.sort(key=_canonical_bytes)
    queries = [bundle["query"] for bundle in bundles]
    evidence = [
        {"query_index": index, **bundle["evidence"]}
        for index, bundle in enumerate(bundles)
    ]
    return {
        "schema_version": task.schema_version,
        "task_family": task.task_family,
        "source": {
            "source_type": task.source.source_type.value,
            "normalization_version": task.source.normalization_version,
        },
        "events": events,
        "target_objects": sorted((_semantic_value(key) for key in task.target_objects), key=_canonical_bytes),
        "actions": actions,
        "queries": queries,
        "version_history": histories,
        "gold_evidence": evidence,
    }


def _horizon_active_entries(entries, horizon: str | None):
    return entries if horizon is None else tuple(entry for entry in entries if entry.logical_time is None or entry.logical_time <= horizon)


def _selector_entries(
    selector: SelectorV3,
    ledger: VersionHistoryLedger,
    event_position: Mapping[str, int],
    event_times: Mapping[str, str],
    horizon: str | None = None,
) -> tuple[VersionHistoryEntry, ...]:
    entries = ledger.entries
    active_entries = _horizon_active_entries(entries, horizon)
    if isinstance(selector, (CurrentSelector, MultiObjectCurrentSelector)):
        return active_entries[-1:]
    if isinstance(selector, PreviousSelector):
        return active_entries[-2:-1]
    if isinstance(selector, ExactVersionSelector):
        return (entries[selector.version_index],)
    if isinstance(selector, TransitionSelector):
        return (entries[selector.from_version_index], entries[selector.to_version_index])
    if isinstance(selector, OrderedHistorySelector):
        start = selector.start_version_index or 0
        end = selector.end_version_index if selector.end_version_index is not None else len(active_entries) - 1
        return active_entries[start : end + 1]
    if isinstance(selector, LogicalTimeAnchorSelector):
        eligible = tuple(entry for entry in entries if entry.logical_time is not None and entry.logical_time <= selector.logical_time)
        if not eligible:
            return ()
        latest = max(entry.logical_time for entry in eligible)
        return tuple(entry for entry in eligible if entry.logical_time == latest)
    if isinstance(selector, EventAnchorSelector):
        anchor = event_position[selector.event_id]
        matched = tuple(
            entry
            for entry in entries
            if entry.valid_from_event_id is not None
            and event_position[entry.valid_from_event_id] <= anchor
            and (
                entry.valid_until_event_id is None
                or anchor < event_position[entry.valid_until_event_id]
            )
        )
        event_time = event_times.get(selector.event_id)
        scheduled = tuple(
            entry
            for entry in entries
            if entry.valid_from_event_id is None
            and entry.logical_time is not None
            and event_time is not None
            and entry.logical_time <= event_time
            and all(event_position[event_id] <= anchor for event_id in entry.source_event_ids)
        )
        eligible = (*matched, *scheduled)
        return () if not eligible else (max(eligible, key=lambda entry: entry.version_index),)
    raise TypeError("unknown selector")


def _validate_answer_schema(answer, schema: AnswerSchema) -> None:
    valid = {
        AnswerSchema.STRING: type(answer) is str,
        AnswerSchema.NUMBER: type(answer) in {int, float},
        AnswerSchema.BOOLEAN: type(answer) is bool,
        AnswerSchema.LIST: isinstance(answer, tuple),
        AnswerSchema.OBJECT: isinstance(answer, Mapping),
    }[schema]
    if not valid:
        raise ValueError(f"gold answer does not match answer_schema={schema.value}")


def _derivation_depth(evidence: QueryGoldEvidenceV3) -> int:
    depths: dict[str, int] = {}
    for step in evidence.derivation_steps:
        depths[step.step_id] = 1 + max((depths[parent] for parent in step.input_step_ids), default=0)
    return depths[evidence.final_derivation_step_id]


def _derivation_read_support(
    evidence,
    allowed_objects: set[tuple[str, str, str, str | None]],
) -> tuple[int, set[tuple[str, str, str, str | None]]]:
    read_units = set()
    for step in evidence.derivation_steps:
        if not _derivation_step_reads_support(step):
            continue
        identities = {_identity(key) for key in step.supporting_object_keys}
        if len(identities) == 1:
            identity = next(iter(identities))
            if identity in allowed_objects:
                read_units.add(identity)
    return len(read_units), read_units


def _semantic_value(value):
    if hasattr(value, "namespace") and hasattr(value, "entity") and hasattr(value, "attribute") and hasattr(value, "subkey") and hasattr(value, "object_type"):
        namespace, entity, attribute, subkey = object_identity(value)
        return {"namespace": namespace, "entity": entity, "attribute": attribute, "subkey": subkey}
    if isinstance(value, BaseModel):
        return {name: _semantic_value(getattr(value, name)) for name in value.__class__.model_fields}
    if isinstance(value, Mapping):
        return {key: _semantic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


VersionIndexSelector = ExactVersionSelector
ExactVersionIndexSelector = ExactVersionSelector
LogicalTimeSelector = LogicalTimeAnchorSelector
HistorySelector = OrderedHistorySelector
MultiObjectCurrentStateSelector = MultiObjectCurrentSelector
GoldEvidenceV3 = QueryGoldEvidenceV3
VersionLedgerEntryV3 = VersionHistoryEntry
VersionLedgerV3 = VersionHistoryLedger


__all__ = [
    "CurrentSelector", "DerivationStepV3", "EventAnchorSelector", "ExactVersionIndexSelector", "ExactVersionSelector",
    "GeneratorProvenanceV3", "GoldActionV3", "GoldEvidenceV3", "HistorySelector",
    "LedgerEntryStatus", "LegacyProvenanceV3", "LogicalTimeAnchorSelector", "LogicalTimeSelector", "MemUpdateTaskV3", "MemoryEventV3", "MemoryQueryV3",
    "MultiObjectCurrentConsistencySynthesis", "MultiObjectCurrentSelector", "MultiObjectCurrentStateSelector", "OrderedHistorySelector",
    "PreviousSelector", "QueryGoldEvidenceV3", "SelectorV3", "SourceRecordV3", "SplitKeyV3", "StaleAlternativeEvidenceV3", "SynthesisSpecV3", "TaskMetadataV3",
    "TransitionSelector", "UpdateSensitiveMultiHopSynthesis", "VersionHistoryEntry", "VersionHistoryLedger",
    "VersionIndexSelector", "VersionLedgerEntryV3", "VersionLedgerV3",
]
