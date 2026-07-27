from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from mub.vnext.contracts.enums import SourceType, SupportReason

SHA256_PATTERN = r"^[0-9a-f]{64}$"
StrictBool = Annotated[bool, Field(strict=True)]
StrictNonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
StrictNumericScore = Annotated[float, Field(strict=True, allow_inf_nan=False)]
StrictNonnegativeFloat = Annotated[
    float, Field(ge=0, strict=True, allow_inf_nan=False)
]


class FrozenDict(Mapping):
    __slots__ = ("__data",)

    def __init__(self, values=()):
        if hasattr(self, "_FrozenDict__data"):
            raise TypeError("frozen mapping cannot be reinitialized")
        object.__setattr__(
            self,
            "_FrozenDict__data",
            MappingProxyType(dict(values)),
        )

    def __setattr__(self, name, value):
        raise AttributeError("frozen mapping attributes cannot be changed")

    def __getitem__(self, key):
        return self.__data[key]

    def __iter__(self):
        return iter(self.__data)

    def __len__(self):
        return len(self.__data)

    def __repr__(self):
        return repr(dict(self.__data))

    def __eq__(self, other):
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def freeze_mapping(value: Mapping) -> FrozenDict:
    return FrozenDict(value)


def freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


FrozenJsonObject = Annotated[
    Mapping[str, JsonValue],
    PlainSerializer(
        thaw_json,
        return_type=dict[str, JsonValue],
        when_used="always",
    ),
]
FrozenStringMap = Annotated[
    Mapping[str, str],
    PlainSerializer(
        thaw_json,
        return_type=dict[str, str],
        when_used="always",
    ),
]
FrozenNonnegativeIntMap = Annotated[
    Mapping[str, StrictNonnegativeInt],
    PlainSerializer(
        thaw_json,
        return_type=dict[str, StrictNonnegativeInt],
        when_used="always",
    ),
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ImmutableContractModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def validated_replace(self, **changes) -> Self:
        data = self.model_dump(mode="python")
        data.update(changes)
        return type(self).model_validate(data)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is not None:
            return self.validated_replace(**update)
        return super().model_copy(deep=deep)


class ArtifactRef(ImmutableContractModel):
    path: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    media_type: str
    record_count: StrictNonnegativeInt | None = None


class SourceAnchor(ContractModel):
    document_id: str
    section_id: str
    paragraph: int | None = Field(default=None, ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    normalized_text_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_character_span(self):
        has_start = self.start_char is not None
        has_end = self.end_char is not None
        if has_start != has_end:
            raise ValueError("start_char and end_char must be supplied together")
        if has_start and self.start_char > self.end_char:
            raise ValueError("start_char must be less than or equal to end_char")
        return self


class GeneratorProvenance(ContractModel):
    generator_name: str = Field(strict=True)
    seed: int = Field(strict=True)
    config_sha256: str = Field(pattern=SHA256_PATTERN, strict=True)
    code_revision: str = Field(strict=True)
    compiler_version: str = Field(strict=True)


class SourceRecord(ContractModel):
    source_id: str
    source_type: SourceType
    source_uri: str | None
    license_or_privacy: str
    raw_hash: str | None = Field(pattern=SHA256_PATTERN)
    normalized_hash: str = Field(pattern=SHA256_PATTERN)
    normalization_version: str
    provenance: dict[str, JsonValue]
    generator: GeneratorProvenance | None = None

    @model_validator(mode="after")
    def _require_synthetic_generator(self):
        if self.source_type == SourceType.SYNTHETIC and self.generator is None:
            raise ValueError("synthetic sources require generator provenance")
        return self


class MemoryObjectKey(ContractModel):
    object_type: str
    namespace: str = "default"
    entity: str
    attribute: str
    subkey: str | None = None

    @field_validator("object_type", "namespace", "entity", "attribute", "subkey", mode="before")
    @classmethod
    def _strip_string_parts(cls, value: Any, info) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if info.field_name == "subkey" and not stripped:
                return None
            return stripped
        return value

    @field_validator("object_type", "namespace", "entity", "attribute")
    @classmethod
    def _reject_blank_identity_parts(cls, value: str) -> str:
        if not value:
            raise ValueError("identity parts must not be blank")
        return value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryObjectKey):
            return NotImplemented
        return (
            self.namespace,
            self.entity,
            self.attribute,
            self.subkey,
        ) == (
            other.namespace,
            other.entity,
            other.attribute,
            other.subkey,
        )

    @staticmethod
    def _escape_identity_part(value: str) -> str:
        return value.replace("%", "%25").replace("|", "%7C")

    @property
    def canonical_id(self) -> str:
        return "|".join(
            [
                self._escape_identity_part(self.namespace),
                self._escape_identity_part(self.entity),
                self._escape_identity_part(self.attribute),
                self._escape_identity_part(self.subkey or ""),
            ]
        )


class MetricFieldSupport(ImmutableContractModel):
    reason: SupportReason
    null_policy: str
    detail: str | None = None


class RawExtension(ContractModel):
    namespace: str
    payload: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ArtifactRef",
    "ContractModel",
    "FrozenDict",
    "FrozenJsonObject",
    "FrozenNonnegativeIntMap",
    "FrozenStringMap",
    "GeneratorProvenance",
    "ImmutableContractModel",
    "MemoryObjectKey",
    "MetricFieldSupport",
    "RawExtension",
    "SourceAnchor",
    "SourceRecord",
    "StrictBool",
    "StrictNonnegativeFloat",
    "StrictNonnegativeInt",
    "StrictNumericScore",
    "freeze_json",
    "freeze_mapping",
    "thaw_json",
]
