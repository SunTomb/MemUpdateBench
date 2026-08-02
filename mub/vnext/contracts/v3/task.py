from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, MemoryObjectKey, SourceRecord
from mub.vnext.contracts.enums import AnswerSchema, Difficulty, EvaluationMode
from mub.vnext.contracts.task import GoldAction, MemoryEvent, TaskMetadata
from mub.vnext.contracts.v3.enums import LedgerEntryStatus, QueryTypeV3, SynthesisKindV3
from mub.vnext.contracts.v3.version import SCHEMA_VERSION_V3

StrictString = Annotated[str, Field(strict=True, min_length=1)]
StrictIndex = Annotated[int, Field(strict=True, ge=0)]


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
    object_keys: tuple[MemoryObjectKey, ...] = Field(min_length=2)

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
    target_object_keys: tuple[MemoryObjectKey, ...] = Field(min_length=1)
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
    value: JsonValue | None = None
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
    object_key: MemoryObjectKey
    entries: tuple[VersionHistoryEntry, ...]

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
        payload = {"identity": self.semantic_identity, "entries": self.model_dump(mode="json")["entries"]}
        return hashlib.sha256(_canonical_bytes(_without_object_type(payload))).hexdigest()


class DerivationStepV3(ImmutableContractModel):
    step_id: StrictString
    operation: StrictString
    input_step_ids: tuple[StrictString, ...] = ()
    supporting_object_keys: tuple[MemoryObjectKey, ...] = ()
    supporting_event_ids: tuple[StrictString, ...] = ()


class QueryGoldEvidenceV3(ImmutableContractModel):
    query_id: StrictString
    answer: JsonValue
    supporting_object_keys: tuple[MemoryObjectKey, ...] = Field(min_length=1)
    supporting_event_ids: tuple[StrictString, ...] = Field(min_length=1)
    derivation_steps: tuple[DerivationStepV3, ...] = Field(min_length=1)
    final_derivation_step_id: StrictString

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        steps = {step.step_id: step for step in self.derivation_steps}
        if len(steps) != len(self.derivation_steps):
            raise ValueError("derivation step IDs must be unique")
        if self.final_derivation_step_id not in steps:
            raise ValueError("final derivation step is unknown")
        for step in self.derivation_steps:
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
        visit(self.final_derivation_step_id)
        if reached != set(steps):
            raise ValueError("disconnected derivation graph")
        return self


class MemUpdateTaskV3(ImmutableContractModel):
    task_id: StrictString
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    task_family: StrictString
    difficulty: Difficulty
    source: SourceRecord
    events: tuple[MemoryEvent, ...]
    target_objects: tuple[MemoryObjectKey, ...] = Field(min_length=1)
    actions: tuple[GoldAction, ...] = ()
    queries: tuple[MemoryQueryV3, ...] = Field(min_length=1)
    version_history: tuple[VersionHistoryLedger, ...] = Field(min_length=1)
    gold_evidence: tuple[QueryGoldEvidenceV3, ...] = Field(min_length=1)
    metadata: TaskMetadata

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)) or [event.sequence_index for event in self.events] != list(range(len(self.events))):
            raise ValueError("events must have unique IDs and contiguous sequence indices")
        declared = {_identity(key) for key in self.target_objects}
        if len(declared) != len(self.target_objects):
            raise ValueError("target object identities must be unique")
        histories = {_identity(item.object_key): item for item in self.version_history}
        if set(histories) != declared:
            raise ValueError("version histories must cover declared targets exactly")
        for action in self.actions:
            if action.event_id not in event_ids:
                raise ValueError("action references unknown event")
            if {_identity(key) for key in action.target_object_keys} - declared:
                raise ValueError("action targets undeclared object")
        event_position = {event_id: index for index, event_id in enumerate(event_ids)}
        for ledger in self.version_history:
            for index, entry in enumerate(ledger.entries):
                anchors = {entry.valid_from_event_id, entry.valid_until_event_id} - {None}
                if anchors - set(event_ids) or set(entry.source_event_ids) - set(event_ids):
                    raise ValueError("version history references unknown event anchor")
                if entry.valid_from_event_id is not None and entry.valid_until_event_id is not None and event_position[entry.valid_from_event_id] >= event_position[entry.valid_until_event_id]:
                    raise ValueError("version validity event interval must be ordered")
                if index + 1 < len(ledger.entries):
                    following = ledger.entries[index + 1]
                    if entry.valid_until_event_id is not None and following.valid_from_event_id is not None and entry.valid_until_event_id != following.valid_from_event_id:
                        raise ValueError("adjacent version validity intervals must be contiguous")
        query_by_id = {query.query_id: query for query in self.queries}
        if len(query_by_id) != len(self.queries):
            raise ValueError("query IDs must be unique")
        evidence_by_id = {item.query_id: item for item in self.gold_evidence}
        if set(evidence_by_id) != set(query_by_id):
            raise ValueError("gold evidence must cover queries exactly")
        for query in self.queries:
            targets = {_identity(key) for key in query.target_object_keys}
            if not targets <= declared:
                raise ValueError("query targets undeclared object")
            selector = query.selector
            if isinstance(selector, EventAnchorSelector):
                if selector.event_id not in event_ids or any(
                    selector.event_id not in {
                        anchor
                        for entry in histories[target].entries
                        for anchor in (entry.valid_from_event_id, entry.valid_until_event_id, *entry.source_event_ids)
                        if anchor is not None
                    }
                    for target in targets
                ):
                    raise ValueError("query selector references unknown event anchor")
            if isinstance(selector, LogicalTimeAnchorSelector) and any(
                selector.logical_time not in {entry.logical_time for entry in histories[target].entries}
                for target in targets
            ):
                raise ValueError("query selector references unknown logical-time anchor")
            if isinstance(selector, ExactVersionSelector):
                if any(selector.version_index >= len(histories[target].entries) for target in targets):
                    raise ValueError("query selector references unknown version")
            if isinstance(selector, PreviousSelector) and any(len(histories[target].entries) < 2 for target in targets):
                raise ValueError("previous selector requires at least two versions")
            if isinstance(selector, TransitionSelector) and any(selector.to_version_index >= len(histories[target].entries) for target in targets):
                raise ValueError("transition selector references unknown version")
            if isinstance(selector, OrderedHistorySelector):
                for target in targets:
                    size = len(histories[target].entries)
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
            step_ids = {step.step_id for step in evidence.derivation_steps}
            for step in evidence.derivation_steps:
                if {_identity(key) for key in step.supporting_object_keys} - evidence_objects:
                    raise ValueError("derivation uses object outside evidence scope")
                if set(step.supporting_event_ids) - set(evidence.supporting_event_ids):
                    raise ValueError("derivation uses event outside evidence scope")
            if query.query_type in {QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY}:
                if len(step_ids) < 2:
                    raise ValueError("G derivations require multiple connected steps")
        return self

    @property
    def semantic_identity(self) -> Mapping[str, JsonValue]:
        payload = self.model_dump(mode="json", exclude={"task_id", "metadata"})
        return _without_object_type(payload)

    @property
    def semantic_hash(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.semantic_identity)).hexdigest()


def _identity(key: MemoryObjectKey) -> tuple[str, str, str, str | None]:
    return key.namespace, key.entity, key.attribute, key.subkey


def _require_unique_objects(keys: tuple[MemoryObjectKey, ...], label: str) -> None:
    if len({_identity(key) for key in keys}) != len(keys):
        raise ValueError(f"{label} must contain unique canonical identities")


def _without_object_type(value):
    if isinstance(value, dict):
        return {key: _without_object_type(item) for key, item in value.items() if key != "object_type"}
    if isinstance(value, (list, tuple)):
        return [_without_object_type(item) for item in value]
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
    "GoldEvidenceV3", "HistorySelector",
    "LedgerEntryStatus", "LogicalTimeAnchorSelector", "LogicalTimeSelector", "MemUpdateTaskV3", "MemoryQueryV3",
    "MultiObjectCurrentConsistencySynthesis", "MultiObjectCurrentSelector", "MultiObjectCurrentStateSelector", "OrderedHistorySelector",
    "PreviousSelector", "QueryGoldEvidenceV3", "SelectorV3", "SynthesisSpecV3",
    "TransitionSelector", "UpdateSensitiveMultiHopSynthesis", "VersionHistoryEntry", "VersionHistoryLedger",
    "VersionIndexSelector", "VersionLedgerEntryV3", "VersionLedgerV3",
]
