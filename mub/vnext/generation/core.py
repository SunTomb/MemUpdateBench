from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ContractModel,
    MemoryObjectKey,
    StrictNonnegativeInt,
    freeze_json,
    freeze_mapping,
    thaw_json,
)
from mub.vnext.contracts.enums import Difficulty, EventRole, Operation, TaskFamily
from mub.vnext.generation.config import PilotConfig
from mub.vnext.io import sha256_model


_CORE_ID_PATTERN = r"^core_[0-9a-f]{16}$"
_TRAJECTORY_ID_PATTERN = r"^trajectory_[0-9a-f]{16}$"


class _FrozenMemoryObjectKey(MemoryObjectKey):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_json_value(value: Any, field_name: str) -> Any:
    value_type = type(value)
    if value is None or value_type in {str, int, bool}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{field_name} must contain only string JSON object keys")
            _validate_json_value(item, field_name)
        return value
    if value_type is list:
        for item in value:
            _validate_json_value(item, field_name)
        return value
    raise ValueError(f"{field_name} must contain only JSON values")


def _clone_object_keys(value: Any) -> Any:
    if type(value) is not list:
        return value
    cloned = []
    for key in value:
        payload = key.model_dump(mode="python") if isinstance(key, MemoryObjectKey) else key
        cloned.append(_FrozenMemoryObjectKey.model_validate(payload))
    return cloned


def _reject_duplicate_keys(
    keys: list[MemoryObjectKey],
    field_name: str,
) -> tuple[MemoryObjectKey, ...]:
    seen: list[MemoryObjectKey] = []
    for key in keys:
        if key in seen:
            raise ValueError(f"duplicate exact identities in {field_name} are not allowed")
        seen.append(key)
    return tuple(keys)


def _serialize_keys(
    keys: tuple[MemoryObjectKey, ...],
    info: SerializationInfo,
) -> list[dict[str, Any]]:
    return [key.model_dump(mode=info.mode) for key in keys]


class _FrozenCoreModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    _copy_sequence_fields: ClassVar[frozenset[str]] = frozenset()

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        normalized_update = dict(update)
        for field_name in self._copy_sequence_fields:
            value = normalized_update.get(field_name)
            if isinstance(value, tuple):
                normalized_update[field_name] = list(value)
        data = self.model_dump(mode="python")
        data.update(normalized_update)
        return type(self).model_validate(data)


class GenerationContext(_FrozenCoreModel):
    config: PilotConfig
    code_revision: str = Field(min_length=1, strict=True)
    compiler_version: str = Field(default="1.0.0", strict=True)
    generator_name: str = Field(
        default="memupdatebench_vnext_pilot",
        strict=True,
    )

    @field_validator("config", mode="before")
    @classmethod
    def _clone_config(cls, value: Any) -> Any:
        if isinstance(value, PilotConfig):
            return value.model_dump(mode="python")
        return value

    @field_validator("code_revision")
    @classmethod
    def _reject_blank_code_revision(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("code_revision must not be blank")
        return value

    @property
    def config_sha256(self) -> str:
        return sha256_model(self.config)

    @property
    def seed(self) -> int:
        return self.config.seed

    @property
    def release_id(self) -> str:
        return self.config.release_id

    @property
    def schema_version(self) -> str:
        return self.config.schema_version

    @property
    def profile_version(self) -> str:
        return self.config.profile_version


class CoreEvent(_FrozenCoreModel):
    _copy_sequence_fields = frozenset({"object_keys"})

    operation: Operation
    object_keys: list[MemoryObjectKey]
    value: JsonValue | None
    role: EventRole
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("object_keys", mode="before")
    @classmethod
    def _copy_object_keys(cls, value: Any) -> Any:
        return _clone_object_keys(value)

    @field_validator("object_keys")
    @classmethod
    def _freeze_object_keys(
        cls,
        value: list[MemoryObjectKey],
    ) -> tuple[MemoryObjectKey, ...]:
        return _reject_duplicate_keys(value, "object_keys")

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: Any) -> Any:
        return _validate_json_value(value, "value")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        return _validate_json_value(value, "metadata")

    @field_validator("value")
    @classmethod
    def _freeze_value(cls, value: JsonValue | None) -> JsonValue | None:
        return freeze_json(value)

    @field_validator("metadata")
    @classmethod
    def _freeze_metadata(cls, value: dict[str, JsonValue]) -> Mapping[str, JsonValue]:
        return freeze_mapping({key: freeze_json(item) for key, item in value.items()})

    @model_validator(mode="after")
    def _validate_operation_payload(self) -> Self:
        if self.operation == Operation.NOOP:
            if self.object_keys or self.value is not None:
                raise ValueError("NOOP requires no targets and a null value")
        elif self.operation in {Operation.ADD, Operation.UPDATE}:
            if not self.object_keys or self.value is None:
                raise ValueError(
                    f"{self.operation.value} requires targets and a non-null value"
                )
        elif self.operation == Operation.DELETE:
            if not self.object_keys or self.value is not None:
                raise ValueError("DELETE requires targets and a null value")
        return self

    @field_serializer("object_keys", when_used="always")
    def _dump_object_keys(
        self,
        value: tuple[MemoryObjectKey, ...],
        info: SerializationInfo,
    ) -> list[dict[str, Any]]:
        return _serialize_keys(value, info)

    @field_serializer("value", "metadata", when_used="always")
    def _dump_json(self, value: Any) -> JsonValue:
        return thaw_json(value)


class SemanticCore(_FrozenCoreModel):
    _copy_sequence_fields = frozenset({"events", "query_targets"})

    core_id: str = Field(pattern=_CORE_ID_PATTERN, strict=True)
    task_family: TaskFamily
    difficulty: Difficulty
    core_index: StrictNonnegativeInt
    trajectory_id: str = Field(pattern=_TRAJECTORY_ID_PATTERN, strict=True)
    events: list[CoreEvent] = Field(min_length=1)
    query_targets: list[MemoryObjectKey] = Field(min_length=1)
    expected_answer: JsonValue | None
    profile: dict[str, JsonValue]
    stratification: dict[str, str | int | float | bool]

    @field_validator("events", mode="before")
    @classmethod
    def _copy_events(cls, value: Any) -> Any:
        if type(value) is not list:
            return value
        return [
            CoreEvent.model_validate(event.model_dump(mode="python"))
            if isinstance(event, CoreEvent)
            else event
            for event in value
        ]

    @field_validator("events")
    @classmethod
    def _freeze_events(cls, value: list[CoreEvent]) -> tuple[CoreEvent, ...]:
        return tuple(value)

    @field_validator("query_targets", mode="before")
    @classmethod
    def _copy_query_targets(cls, value: Any) -> Any:
        return _clone_object_keys(value)

    @field_validator("query_targets")
    @classmethod
    def _freeze_query_targets(
        cls,
        value: list[MemoryObjectKey],
    ) -> tuple[MemoryObjectKey, ...]:
        return _reject_duplicate_keys(value, "query_targets")

    @field_validator("expected_answer", mode="before")
    @classmethod
    def _validate_expected_answer(cls, value: Any) -> Any:
        return _validate_json_value(value, "expected_answer")

    @field_validator("profile", mode="before")
    @classmethod
    def _validate_profile(cls, value: Any) -> Any:
        return _validate_json_value(value, "profile")

    @field_validator("stratification", mode="before")
    @classmethod
    def _validate_stratification(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("stratification keys must be strings")
            if not key.strip():
                raise ValueError("stratification keys must not be blank")
            if type(item) not in {str, int, float, bool}:
                raise ValueError("stratification values must be JSON scalars")
            if type(item) is float and not math.isfinite(item):
                raise ValueError("stratification values must be finite")
        return value

    @field_validator("expected_answer")
    @classmethod
    def _freeze_expected_answer(cls, value: JsonValue | None) -> JsonValue | None:
        return freeze_json(value)

    @field_validator("profile")
    @classmethod
    def _freeze_profile(cls, value: dict[str, JsonValue]) -> Mapping[str, JsonValue]:
        return freeze_mapping({key: freeze_json(item) for key, item in value.items()})

    @field_validator("stratification")
    @classmethod
    def _freeze_stratification(
        cls,
        value: dict[str, str | int | float | bool],
    ) -> Mapping[str, str | int | float | bool]:
        return freeze_mapping(value)

    @field_serializer("events", when_used="always")
    def _dump_events(
        self,
        value: tuple[CoreEvent, ...],
        info: SerializationInfo,
    ) -> list[dict[str, Any]]:
        return [event.model_dump(mode=info.mode) for event in value]

    @field_serializer("query_targets", when_used="always")
    def _dump_query_targets(
        self,
        value: tuple[MemoryObjectKey, ...],
        info: SerializationInfo,
    ) -> list[dict[str, Any]]:
        return _serialize_keys(value, info)

    @field_serializer(
        "expected_answer",
        "profile",
        "stratification",
        when_used="always",
    )
    def _dump_json(self, value: Any) -> JsonValue:
        return thaw_json(value)


__all__ = ["CoreEvent", "GenerationContext", "SemanticCore"]
