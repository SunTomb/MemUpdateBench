from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator, Field, JsonValue, PlainSerializer, field_validator

from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel, MemoryObjectKey, StrictNonnegativeInt, freeze_json, freeze_mapping, thaw_json
from mub.vnext.contracts.enums import ActionScope, Operation


def _validate_identifier(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("identifiers must be exact built-in strings")
    if not value.strip():
        raise ValueError("identifiers must not be blank")
    return value


def _validate_finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("v3 JSON cannot contain non-finite floats")
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite_json(item)
    return value


def _validate_finite_float(value: Any) -> float:
    if type(value) is not float:
        raise ValueError("scores must be exact built-in floats")
    if not math.isfinite(value):
        raise ValueError("scores must be finite")
    return value


StrictIdentifier = Annotated[str, BeforeValidator(_validate_identifier), Field(strict=True, min_length=1)]
StrictFiniteFloat = Annotated[float, BeforeValidator(_validate_finite_float), Field(strict=True, allow_inf_nan=False)]


def _key_input(value: Any) -> Any:
    if isinstance(value, MemoryObjectKey):
        return value.model_dump(mode="python")
    return value


class FrozenMemoryObjectKey(ImmutableContractModel):
    object_type: str = Field(strict=True, min_length=1)
    namespace: str = Field(default="default", strict=True, min_length=1)
    entity: str = Field(strict=True, min_length=1)
    attribute: str = Field(strict=True, min_length=1)
    subkey: str | None = Field(default=None, strict=True)

    @field_validator("object_type", "namespace", "entity", "attribute", "subkey", mode="before")
    @classmethod
    def _normalize_parts(cls, value: Any, info) -> Any:
        if value is None and info.field_name == "subkey":
            return None
        if type(value) is not str:
            raise ValueError("object key parts must be exact built-in strings")
        value = value.strip()
        if info.field_name == "subkey" and not value:
            return None
        return value

    @property
    def canonical_id(self) -> str:
        def escape(value: str) -> str:
            return value.replace("%", "%25").replace("|", "%7C")
        return "|".join(escape(value) for value in (self.namespace, self.entity, self.attribute, self.subkey or ""))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (FrozenMemoryObjectKey, MemoryObjectKey)):
            return NotImplemented
        return object_identity(self) == object_identity(other)

    def __hash__(self) -> int:
        return hash(object_identity(self))


MemoryObjectKeyV3 = Annotated[FrozenMemoryObjectKey, BeforeValidator(_key_input)]
FrozenJsonValue = Annotated[
    JsonValue,
    BeforeValidator(_validate_finite_json),
    AfterValidator(freeze_json),
    PlainSerializer(thaw_json, return_type=JsonValue, when_used="always"),
]
FrozenJsonObjectV3 = Annotated[
    Mapping[str, JsonValue],
    BeforeValidator(_validate_finite_json),
    AfterValidator(freeze_json),
    PlainSerializer(thaw_json, return_type=dict[str, JsonValue], when_used="always"),
]
FrozenUsageMap = Annotated[
    Mapping[str, StrictNonnegativeInt],
    AfterValidator(freeze_mapping),
    PlainSerializer(thaw_json, return_type=dict[str, StrictNonnegativeInt], when_used="always"),
]


def object_identity(key: FrozenMemoryObjectKey | MemoryObjectKey) -> tuple[str, str, str, str | None]:
    return key.namespace, key.entity, key.attribute, key.subkey


def validate_action_coherence(
    *,
    operation: Operation | None,
    scope: ActionScope | None,
    targets: tuple[FrozenMemoryObjectKey, ...],
    value: JsonValue | None,
    executed: bool = False,
) -> None:
    identities = [object_identity(key) for key in targets]
    if len(identities) != len(set(identities)):
        raise ValueError("target object identities must be unique")
    if operation is None:
        if executed:
            raise ValueError("executed actions require an operation")
        if scope is not None or targets or value is not None:
            raise ValueError("actions without an operation cannot carry scope, targets, or value")
        return
    if operation == Operation.NOOP:
        if scope is not None or targets or value is not None:
            raise ValueError("NOOP cannot carry scope, targets, or value")
        return
    if not targets or scope is None:
        raise ValueError("mutating actions require scope and targets")
    if operation in {Operation.ADD, Operation.UPDATE}:
        if scope != ActionScope.OBJECT or len(targets) != 1 or value is None:
            raise ValueError("ADD/UPDATE require one object-scoped target and a value")
        return
    if value is not None:
        raise ValueError("DELETE cannot carry a value")
    if scope in {ActionScope.OBJECT, ActionScope.TTL} and len(targets) != 1:
        raise ValueError("object/TTL scope requires exactly one target")
    if scope == ActionScope.ATTRIBUTE:
        bases = {(key.namespace, key.entity, key.attribute) for key in targets}
        if len(bases) != 1:
            raise ValueError("attribute scope targets must share namespace/entity/attribute")
    elif scope == ActionScope.ENTITY:
        bases = {(key.namespace, key.entity) for key in targets}
        if len(bases) != 1:
            raise ValueError("entity scope targets must share namespace/entity")
    elif scope == ActionScope.NAMESPACE:
        if len({key.namespace for key in targets}) != 1:
            raise ValueError("namespace scope targets must share namespace")


__all__ = ["FrozenJsonObjectV3", "FrozenJsonValue", "FrozenMemoryObjectKey", "FrozenUsageMap", "MemoryObjectKeyV3", "StrictFiniteFloat", "StrictIdentifier", "object_identity", "validate_action_coherence"]
