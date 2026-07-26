from __future__ import annotations

from typing import Any

from typing_extensions import Self

from pydantic import Field, JsonValue, field_validator, model_validator

from mub.vnext.contracts.common import ContractModel, MemoryObjectKey, SourceRecord
from mub.vnext.contracts.enums import (
    ActionScope,
    AnswerSchema,
    Difficulty,
    EvaluationMode,
    EventRole,
    Operation,
    QueryType,
    Split,
)
from mub.vnext.version import SCHEMA_VERSION


class SplitKey(ContractModel):
    semantic_core_id: str
    source_group_id: str
    trajectory_id: str
    paraphrase_group_id: str | None = None
    source_document_id: str | None = None
    version_group_id: str | None = None
    split_exception_id: str | None = None
    split_policy_version: str


class LegacyProvenance(ContractModel):
    legacy_family_id: str
    legacy_phase: str
    legacy_dataset_id: str
    legacy_split_id: str
    legacy_metric_namespace: str
    legacy_run_condition_id: str | None = None
    checkpoint_family: str | None = None
    training_seed: int | None = None
    answer_mode: str | None = None
    memory_trajectory_id: str | None = None
    source_artifact_path: str
    source_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    known_caveats: list[str] = Field(default_factory=list)


class TaskMetadata(ContractModel):
    split: Split
    split_key: SplitKey
    profile_name: Difficulty
    resolved_profile: dict[str, JsonValue] = Field(default_factory=dict)
    generation_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_version: str
    tags: list[str] = Field(default_factory=list)
    legacy_provenance: LegacyProvenance | None = None
    extra: dict[str, JsonValue] = Field(default_factory=dict)


class GoldAction(ContractModel):
    action_id: str
    event_id: str
    operation: Operation
    scope: ActionScope
    target_object_keys: list[MemoryObjectKey] = Field(default_factory=list)
    value: JsonValue | None = None
    effective_at: str | None = None
    expected_effect: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_operation_payload(self) -> Self:
        if self.operation == Operation.NOOP:
            if self.target_object_keys or self.value is not None:
                raise ValueError("NOOP cannot target objects or carry value")
        elif self.operation in {Operation.ADD, Operation.UPDATE}:
            if not self.target_object_keys:
                raise ValueError("ADD/UPDATE actions need target_object_keys")
        elif self.operation == Operation.DELETE:
            if not self.target_object_keys:
                raise ValueError("DELETE actions must enumerate targets")
        return self


class MemoryEvent(ContractModel):
    event_id: str
    sequence_index: int = Field(ge=0)
    timestamp: str | None = None
    raw_text: str
    normalized_text: str
    speaker: str | None = None
    gold_action_ids: list[str] = Field(default_factory=list)
    role: EventRole
    source_anchor: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class MemoryQuery(ContractModel):
    query_id: str
    query_type: QueryType
    text: str
    target_object_keys: list[MemoryObjectKey] = Field(default_factory=list)
    answer_schema: AnswerSchema
    evaluation_mode: EvaluationMode
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class GoldRecord(ContractModel):
    actions: list[GoldAction] = Field(default_factory=list)
    action_sequence: list[str] = Field(default_factory=list)
    final_state: dict[str, JsonValue] = Field(default_factory=dict)
    version_history: dict[str, list[JsonValue]] = Field(default_factory=dict)
    expected_present_objects: list[MemoryObjectKey] = Field(default_factory=list)
    expected_absent_objects: list[MemoryObjectKey] = Field(default_factory=list)
    gold_source_event_ids: list[str] = Field(default_factory=list)
    gold_answers: dict[str, JsonValue] = Field(default_factory=dict)
    acceptable_answers: dict[str, JsonValue] = Field(default_factory=dict)


class MemUpdateTask(ContractModel):
    task_id: str
    schema_version: str = SCHEMA_VERSION
    task_family: str
    difficulty: Difficulty
    source: SourceRecord
    events: list[MemoryEvent] = Field(default_factory=list)
    target_objects: list[MemoryObjectKey] = Field(default_factory=list)
    queries: list[MemoryQuery] = Field(default_factory=list)
    gold: GoldRecord
    metadata: TaskMetadata

    @field_validator("task_family")
    @classmethod
    def _reject_blank_task_family(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_family must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        _reject_blank_id(self.task_id, "task_id")

        event_ids = [event.event_id for event in self.events]
        _reject_blank_ids(event_ids, "event IDs")
        _reject_duplicates(event_ids, "event IDs")

        sequence_indices = [event.sequence_index for event in self.events]
        if sequence_indices != list(range(len(self.events))):
            raise ValueError("event sequence_index values must be ordered and contiguous")

        action_ids = [action.action_id for action in self.gold.actions]
        _reject_blank_ids(action_ids, "action IDs")
        _reject_duplicates(action_ids, "action IDs")
        _reject_blank_ids([action.event_id for action in self.gold.actions], "action event IDs")

        query_ids = [query.query_id for query in self.queries]
        _reject_blank_ids(query_ids, "query IDs")
        _reject_duplicates(query_ids, "query IDs")

        _reject_blank_ids(self.gold.action_sequence, "action_sequence IDs")
        if sorted(self.gold.action_sequence) != sorted(action_ids) or len(self.gold.action_sequence) != len(action_ids):
            raise ValueError("action_sequence must contain every action exactly once")

        declared_objects = {_object_identity(key) for key in self.target_objects}

        event_by_id = {event.event_id: event for event in self.events}
        action_by_id = {action.action_id: action for action in self.gold.actions}

        for event in self.events:
            _reject_blank_ids(event.gold_action_ids, f"gold_action_ids for event {event.event_id}")
            for action_id in event.gold_action_ids:
                action = action_by_id.get(action_id)
                if action is None:
                    raise ValueError(f"missing gold action {action_id} referenced by event {event.event_id}")
                if action.event_id != event.event_id:
                    raise ValueError(
                        f"action/event mismatch: event {event.event_id} references action {action_id} "
                        f"from event {action.event_id}"
                    )

        for action in self.gold.actions:
            event = event_by_id.get(action.event_id)
            if event is None:
                raise ValueError(f"action {action.action_id} references missing event {action.event_id}")
            if action.action_id not in event.gold_action_ids:
                raise ValueError(
                    f"action/event mismatch: action {action.action_id} is not listed by event {event.event_id}"
                )
            for key in action.target_object_keys:
                if _object_identity(key) not in declared_objects:
                    raise ValueError(f"action {action.action_id} targets undeclared object {key.canonical_id}")

        for query in self.queries:
            for key in query.target_object_keys:
                if _object_identity(key) not in declared_objects:
                    raise ValueError(f"query {query.query_id} targets undeclared object {key.canonical_id}")

        for key in self.gold.expected_present_objects:
            if _object_identity(key) not in declared_objects:
                raise ValueError(f"expected_present_objects contains undeclared object {key.canonical_id}")

        for key in self.gold.expected_absent_objects:
            if _object_identity(key) not in declared_objects:
                raise ValueError(f"expected_absent_objects contains undeclared object {key.canonical_id}")

        _reject_blank_ids(self.gold.gold_source_event_ids, "gold_source_event_ids")
        for event_id in self.gold.gold_source_event_ids:
            if event_id not in event_by_id:
                raise ValueError(f"gold_source_event_ids references missing event {event_id}")

        gold_answer_ids = set(self.gold.gold_answers)
        acceptable_answer_ids = set(self.gold.acceptable_answers)
        query_id_set = set(query_ids)

        if gold_answer_ids != query_id_set:
            missing = query_id_set - gold_answer_ids
            if missing:
                raise ValueError("gold_answers must contain every query ID")
            raise ValueError("gold_answers contains unknown query ID")

        if acceptable_answer_ids != query_id_set:
            missing = query_id_set - acceptable_answer_ids
            if missing:
                raise ValueError("acceptable_answers must contain every query ID")
            raise ValueError("acceptable_answers contains unknown query ID")

        return self


def _object_identity(key: MemoryObjectKey) -> tuple[str, str, str, str | None]:
    return (key.namespace, key.entity, key.attribute, key.subkey)


def _reject_blank_id(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")


def _reject_blank_ids(values: list[str], label: str) -> None:
    for value in values:
        _reject_blank_id(value, label)


def _reject_duplicates(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} are not allowed")


__all__ = [
    "GoldAction",
    "GoldRecord",
    "LegacyProvenance",
    "MemUpdateTask",
    "MemoryEvent",
    "MemoryQuery",
    "SplitKey",
    "TaskMetadata",
]
