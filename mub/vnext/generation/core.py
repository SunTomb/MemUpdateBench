from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, NoReturn

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
    SourceAnchor,
    StrictNonnegativeInt,
    freeze_json,
    freeze_mapping,
    thaw_json,
)
from mub.vnext.contracts.enums import (
    Difficulty,
    EventRole,
    Operation,
    QueryType,
    ReferenceResolutionStatus,
    TaskFamily,
)
from mub.vnext.contracts.task import (
    CanonicalAnswer,
    ReferenceCandidate,
    SurfaceReference,
)
from mub.vnext.generation.config import (
    DifficultyDensities,
    DifficultyNonnegativeCounts,
    DifficultyPositiveCounts,
    EntityAttributeGroundingConfig,
    InterleavedMultiSlotUpdateConfig,
    MechanismCondition,
    MechanismSliceConfig,
    NoopWriteDisciplineConfig,
    OutputConfig,
    PilotConfig,
    PilotFamiliesConfig,
    RepeatedSameSlotUpdateConfig,
    SplitConfig,
)
from mub.vnext.generation.core_config import (
    CoreConfig,
    CoreEntityAttributeGroundingConfig,
    CoreFamilyASchedule,
    CoreFamilyBSchedule,
    CoreFamilyCSchedule,
    CoreFamilyDSchedule,
    CoreFamiliesConfig,
    CoreSplitCoreCounts,
    CoreInterleavedMultiSlotUpdateConfig,
    CoreNoopWriteDisciplineConfig,
    CoreRepeatedSameSlotUpdateConfig,
    CoreSurfaceDeclaration,
)
from mub.vnext.io import sha256_model
from mub.vnext.version import COMPILER_VERSION


_CORE_ID_PATTERN = r"^core_[0-9a-f]{16}$"
_TRAJECTORY_ID_PATTERN = r"^trajectory_[0-9a-f]{16}$"


class _FrozenMemoryObjectKey(MemoryObjectKey):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _FrozenList(list):
    def _reject_mutation(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("frozen config lists cannot be mutated")

    __delitem__ = __setitem__ = __iadd__ = __imul__ = clear = extend = insert = pop = remove = reverse = sort = _reject_mutation
    append = _reject_mutation


class _FrozenConfigMixin:
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @model_validator(mode="after")
    def _freeze_lists(self):
        for field_name, value in self.__dict__.items():
            if type(value) is list:
                object.__setattr__(self, field_name, _FrozenList(value))
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            if deep:
                return type(self).model_validate(self.model_dump(mode="python"))
            return super().model_copy(deep=False)
        data = self.model_dump(mode="python")
        data.update(dict(update))
        return type(self).model_validate(data)


class _FrozenDifficultyNonnegativeCounts(
    _FrozenConfigMixin,
    DifficultyNonnegativeCounts,
):
    pass


class _FrozenDifficultyPositiveCounts(_FrozenConfigMixin, DifficultyPositiveCounts):
    pass


class _FrozenDifficultyDensities(_FrozenConfigMixin, DifficultyDensities):
    pass


class _FrozenRepeatedSameSlotUpdateConfig(
    _FrozenConfigMixin,
    RepeatedSameSlotUpdateConfig,
):
    same_name_distractors: _FrozenDifficultyNonnegativeCounts
    same_entity_other_attribute: _FrozenDifficultyNonnegativeCounts
    noop_near_miss: _FrozenDifficultyNonnegativeCounts


class _FrozenInterleavedMultiSlotUpdateConfig(
    _FrozenConfigMixin,
    InterleavedMultiSlotUpdateConfig,
):
    active_object_counts: _FrozenDifficultyPositiveCounts
    cross_slot_distractor_density: _FrozenDifficultyDensities


class _FrozenEntityAttributeGroundingConfig(
    _FrozenConfigMixin,
    EntityAttributeGroundingConfig,
):
    pass


class _FrozenNoopWriteDisciplineConfig(
    _FrozenConfigMixin,
    NoopWriteDisciplineConfig,
):
    pass


class _FrozenSplitConfig(_FrozenConfigMixin, SplitConfig):
    pass


class _FrozenPilotFamiliesConfig(_FrozenConfigMixin, PilotFamiliesConfig):
    repeated_same_slot_update: _FrozenRepeatedSameSlotUpdateConfig
    interleaved_multi_slot_update: _FrozenInterleavedMultiSlotUpdateConfig
    entity_attribute_grounding: _FrozenEntityAttributeGroundingConfig
    noop_write_discipline: _FrozenNoopWriteDisciplineConfig


class _FrozenMechanismCondition(_FrozenConfigMixin, MechanismCondition):
    pass


class _FrozenMechanismSliceConfig(_FrozenConfigMixin, MechanismSliceConfig):
    conditions: list[_FrozenMechanismCondition]


class _FrozenOutputConfig(_FrozenConfigMixin, OutputConfig):
    pass


class _FrozenPilotConfig(_FrozenConfigMixin, PilotConfig):
    splits: _FrozenSplitConfig
    families: _FrozenPilotFamiliesConfig
    mechanism_slice: _FrozenMechanismSliceConfig
    output: _FrozenOutputConfig


_FrozenPilotConfig.model_rebuild()


class _FrozenCoreSplitCoreCounts(_FrozenConfigMixin, CoreSplitCoreCounts):
    pass


class _FrozenCoreFamilyASchedule(_FrozenConfigMixin, CoreFamilyASchedule):
    split_core_counts: _FrozenCoreSplitCoreCounts


class _FrozenCoreFamilyBSchedule(_FrozenConfigMixin, CoreFamilyBSchedule):
    split_core_counts: _FrozenCoreSplitCoreCounts


class _FrozenCoreFamilyCSchedule(_FrozenConfigMixin, CoreFamilyCSchedule):
    split_core_counts: _FrozenCoreSplitCoreCounts


class _FrozenCoreFamilyDSchedule(_FrozenConfigMixin, CoreFamilyDSchedule):
    split_core_counts: _FrozenCoreSplitCoreCounts


class _FrozenCoreRepeatedSameSlotUpdateConfig(
    _FrozenConfigMixin,
    CoreRepeatedSameSlotUpdateConfig,
):
    schedule: _FrozenCoreFamilyASchedule


class _FrozenCoreInterleavedMultiSlotUpdateConfig(
    _FrozenConfigMixin,
    CoreInterleavedMultiSlotUpdateConfig,
):
    schedule: _FrozenCoreFamilyBSchedule


class _FrozenCoreEntityAttributeGroundingConfig(
    _FrozenConfigMixin,
    CoreEntityAttributeGroundingConfig,
):
    schedule: _FrozenCoreFamilyCSchedule


class _FrozenCoreNoopWriteDisciplineConfig(
    _FrozenConfigMixin,
    CoreNoopWriteDisciplineConfig,
):
    schedule: _FrozenCoreFamilyDSchedule


class _FrozenCoreFamiliesConfig(_FrozenConfigMixin, CoreFamiliesConfig):
    repeated_same_slot_update: _FrozenCoreRepeatedSameSlotUpdateConfig
    interleaved_multi_slot_update: _FrozenCoreInterleavedMultiSlotUpdateConfig
    entity_attribute_grounding: _FrozenCoreEntityAttributeGroundingConfig
    noop_write_discipline: _FrozenCoreNoopWriteDisciplineConfig


class _FrozenCoreSurfaceDeclaration(_FrozenConfigMixin, CoreSurfaceDeclaration):
    pass


class _FrozenCoreConfig(_FrozenConfigMixin, CoreConfig):
    surfaces: list[_FrozenCoreSurfaceDeclaration]
    splits: _FrozenSplitConfig
    families: _FrozenCoreFamiliesConfig
    mechanism_slice: _FrozenMechanismSliceConfig
    output: _FrozenOutputConfig


_FrozenCoreConfig.model_rebuild()


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


def _clone_object_keys(value: Any, field_name: str) -> list[_FrozenMemoryObjectKey]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be an exact list")
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

    def copy(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("copy() is disabled; use validated model_copy() instead")

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            if deep:
                return type(self).model_validate(self.model_dump(mode="python"))
            return super().model_copy(deep=False)
        normalized_update = dict(update)
        for field_name in self._copy_sequence_fields:
            value = normalized_update.get(field_name)
            if isinstance(value, tuple):
                normalized_update[field_name] = list(value)
        data = self.model_dump(mode="python")
        data.update(normalized_update)
        return type(self).model_validate(data)


class _FrozenSourceAnchor(_FrozenCoreModel, SourceAnchor):
    pass


class _FrozenReferenceCandidate(_FrozenCoreModel, ReferenceCandidate):
    _copy_sequence_fields = frozenset({"source_anchors"})

    object_key: _FrozenMemoryObjectKey
    source_anchors: list[_FrozenSourceAnchor] = Field(default_factory=list)

    @field_validator("object_key", mode="before")
    @classmethod
    def _copy_object_key(cls, value: Any) -> _FrozenMemoryObjectKey:
        payload = (
            value.model_dump(mode="python")
            if isinstance(value, MemoryObjectKey)
            else value
        )
        return _FrozenMemoryObjectKey.model_validate(payload)

    @field_validator("source_anchors", mode="before")
    @classmethod
    def _copy_source_anchors(cls, value: Any) -> list[_FrozenSourceAnchor]:
        if type(value) is not list:
            raise ValueError("source_anchors must be an exact list")
        return [
            _FrozenSourceAnchor.model_validate(
                anchor.model_dump(mode="python")
                if isinstance(anchor, SourceAnchor)
                else anchor
            )
            for anchor in value
        ]

    @field_validator("source_anchors")
    @classmethod
    def _freeze_source_anchors(
        cls,
        value: list[_FrozenSourceAnchor],
    ) -> tuple[_FrozenSourceAnchor, ...]:
        return tuple(value)

    @field_serializer("source_anchors", when_used="always")
    def _dump_source_anchors(
        self,
        value: tuple[_FrozenSourceAnchor, ...],
        info: SerializationInfo,
    ) -> list[dict[str, Any]]:
        return [anchor.model_dump(mode=info.mode) for anchor in value]


class _FrozenSurfaceReference(_FrozenCoreModel, SurfaceReference):
    _copy_sequence_fields = frozenset({"candidate_ids"})

    @field_validator("candidate_ids", mode="before")
    @classmethod
    def _copy_candidate_ids(cls, value: Any) -> Any:
        if type(value) is not list:
            raise ValueError("candidate_ids must be an exact list")
        return list(value)

    @field_validator("candidate_ids")
    @classmethod
    def _freeze_candidate_ids(cls, value: list[str]) -> tuple[str, ...]:
        return tuple(value)

    @field_serializer("candidate_ids", when_used="always")
    def _dump_candidate_ids(self, value: tuple[str, ...]) -> list[str]:
        return list(value)


class _FrozenCanonicalAnswer(_FrozenCoreModel, CanonicalAnswer):
    _copy_sequence_fields = frozenset({"selected_candidate_ids"})

    @field_validator("selected_candidate_ids", mode="before")
    @classmethod
    def _copy_selected_candidate_ids(cls, value: Any) -> Any:
        if type(value) is not list:
            raise ValueError("selected_candidate_ids must be an exact list")
        return list(value)

    @field_validator("selected_candidate_ids")
    @classmethod
    def _freeze_selected_candidate_ids(cls, value: list[str]) -> tuple[str, ...]:
        return tuple(value)

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: Any) -> Any:
        return _validate_json_value(value, "canonical_answer.value")

    @field_validator("value")
    @classmethod
    def _freeze_value(cls, value: JsonValue | None) -> JsonValue | None:
        return freeze_json(value)

    @field_serializer("selected_candidate_ids", when_used="always")
    def _dump_selected_candidate_ids(self, value: tuple[str, ...]) -> list[str]:
        return list(value)

    @field_serializer("value", when_used="always")
    def _dump_value(self, value: Any) -> JsonValue:
        return thaw_json(value)


class GenerationContext(_FrozenCoreModel):
    config: PilotConfig | CoreConfig
    code_revision: str = Field(min_length=1, strict=True)
    compiler_version: str = Field(default=COMPILER_VERSION, strict=True)
    generator_name: str = Field(
        default="memupdatebench_vnext_pilot",
        strict=True,
    )

    @field_validator("config", mode="before")
    @classmethod
    def _clone_config(cls, value: Any) -> Any:
        if isinstance(value, CoreConfig):
            return _FrozenCoreConfig.model_validate(value.model_dump(mode="python"))
        if isinstance(value, PilotConfig):
            return _FrozenPilotConfig.model_validate(value.model_dump(mode="python"))
        if isinstance(value, Mapping) and "surface_catalog_version" in value:
            return _FrozenCoreConfig.model_validate(value)
        return _FrozenPilotConfig.model_validate(value)

    @field_validator("code_revision", "compiler_version", "generator_name")
    @classmethod
    def _reject_blank_provenance(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
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
        return _clone_object_keys(value, "object_keys")

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
        surface_statement = self.metadata.get("surface_statement")
        if "surface_statement" in self.metadata:
            if type(surface_statement) is not str or not surface_statement.strip():
                raise ValueError("surface_statement must be an exact nonblank string")
            if self.operation != Operation.NOOP:
                raise ValueError("surface_statement is valid only for NOOP events")

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
    _copy_sequence_fields = frozenset(
        {
            "events",
            "query_targets",
            "reference_candidates",
            "surface_references",
        }
    )

    core_id: str = Field(pattern=_CORE_ID_PATTERN, strict=True)
    task_family: TaskFamily
    difficulty: Difficulty
    core_index: StrictNonnegativeInt
    trajectory_id: str = Field(pattern=_TRAJECTORY_ID_PATTERN, strict=True)
    events: list[CoreEvent] = Field(min_length=1)
    query_targets: list[MemoryObjectKey] = Field(min_length=1)
    query_type: QueryType = QueryType.CURRENT_STATE
    reference_candidates: list[ReferenceCandidate] = Field(default_factory=list)
    surface_references: list[SurfaceReference] = Field(default_factory=list)
    canonical_answer: CanonicalAnswer | None = None
    expected_answer: JsonValue | None
    profile: dict[str, JsonValue]
    stratification: dict[str, str | int | float | bool]

    @field_validator("events", mode="before")
    @classmethod
    def _copy_events(cls, value: Any) -> Any:
        if type(value) is not list:
            raise ValueError("events must be an exact list")
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
        return _clone_object_keys(value, "query_targets")

    @field_validator("query_targets")
    @classmethod
    def _freeze_query_targets(
        cls,
        value: list[MemoryObjectKey],
    ) -> tuple[MemoryObjectKey, ...]:
        return _reject_duplicate_keys(value, "query_targets")

    @field_validator("reference_candidates", mode="before")
    @classmethod
    def _copy_reference_candidates(cls, value: Any) -> Any:
        if type(value) is not list:
            raise ValueError("reference_candidates must be an exact list")
        return [
            _FrozenReferenceCandidate.model_validate(
                candidate.model_dump(mode="python")
                if isinstance(candidate, ReferenceCandidate)
                else candidate
            )
            for candidate in value
        ]

    @field_validator("reference_candidates")
    @classmethod
    def _freeze_reference_candidates(
        cls,
        value: list[_FrozenReferenceCandidate],
    ) -> tuple[_FrozenReferenceCandidate, ...]:
        return tuple(value)

    @field_validator("surface_references", mode="before")
    @classmethod
    def _copy_surface_references(cls, value: Any) -> Any:
        if type(value) is not list:
            raise ValueError("surface_references must be an exact list")
        return [
            _FrozenSurfaceReference.model_validate(
                reference.model_dump(mode="python")
                if isinstance(reference, SurfaceReference)
                else reference
            )
            for reference in value
        ]

    @field_validator("surface_references")
    @classmethod
    def _freeze_surface_references(
        cls,
        value: list[_FrozenSurfaceReference],
    ) -> tuple[_FrozenSurfaceReference, ...]:
        return tuple(value)

    @field_validator("canonical_answer", mode="before")
    @classmethod
    def _copy_canonical_answer(cls, value: Any) -> Any:
        if value is None:
            return None
        payload = (
            value.model_dump(mode="python")
            if isinstance(value, CanonicalAnswer)
            else value
        )
        return _FrozenCanonicalAnswer.model_validate(payload)

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

    @model_validator(mode="after")
    def _validate_reference_semantics(self) -> Self:
        has_reference_payload = bool(
            self.reference_candidates
            or self.surface_references
            or self.canonical_answer is not None
        )
        if self.query_type is not QueryType.UNRESOLVED_REFERENCE:
            if has_reference_payload:
                raise ValueError(
                    "reference candidates, surface references, and canonical_answer "
                    "are valid only for unresolved-reference queries"
                )
            return self

        if self.expected_answer is not None:
            raise ValueError(
                "unresolved-reference expected_answer must be null; use canonical_answer"
            )
        if not self.reference_candidates:
            raise ValueError("unresolved-reference queries require reference_candidates")
        if not self.surface_references:
            raise ValueError("unresolved-reference queries require surface_references")
        if self.canonical_answer is None:
            raise ValueError("unresolved-reference queries require canonical_answer")

        candidate_ids = [
            candidate.candidate_id for candidate in self.reference_candidates
        ]
        if any(not candidate_id.strip() for candidate_id in candidate_ids):
            raise ValueError("reference candidate IDs must not be blank")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("duplicate reference candidate IDs are not allowed")
        candidate_identities = [
            (
                candidate.object_key.namespace,
                candidate.object_key.entity,
                candidate.object_key.attribute,
                candidate.object_key.subkey,
            )
            for candidate in self.reference_candidates
        ]
        if len(candidate_identities) != len(set(candidate_identities)):
            raise ValueError(
                "duplicate reference candidate identities are not allowed"
            )

        reference_ids = [
            reference.reference_id for reference in self.surface_references
        ]
        if any(not reference_id.strip() for reference_id in reference_ids):
            raise ValueError("surface reference IDs must not be blank")
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("duplicate surface reference IDs are not allowed")

        known_candidate_ids = set(candidate_ids)
        referenced_candidate_ids: set[str] = set()
        for reference in self.surface_references:
            if not reference.surface_text.strip():
                raise ValueError("surface reference text must not be blank")
            if not reference.normalized_text.strip():
                raise ValueError("normalized surface reference text must not be blank")
            if any(not candidate_id.strip() for candidate_id in reference.candidate_ids):
                raise ValueError("surface reference candidate IDs must not be blank")
            if len(reference.candidate_ids) != len(set(reference.candidate_ids)):
                raise ValueError(
                    "duplicate surface reference candidate IDs are not allowed"
                )
            unknown = set(reference.candidate_ids) - known_candidate_ids
            if unknown:
                raise ValueError(
                    "surface references link unknown candidate IDs: "
                    f"{sorted(unknown)}"
                )
            referenced_candidate_ids.update(reference.candidate_ids)

        canonical = self.canonical_answer
        selected_ids = set(canonical.selected_candidate_ids)
        if selected_ids - known_candidate_ids:
            raise ValueError("canonical_answer selects unknown candidate IDs")
        if selected_ids - referenced_candidate_ids:
            raise ValueError("canonical_answer selects unreferenced candidate IDs")
        if canonical.resolution_status is ReferenceResolutionStatus.UNIQUE:
            if len(referenced_candidate_ids) != 1:
                raise ValueError("UNIQUE resolution must reference exactly one candidate")
        elif canonical.resolution_status is ReferenceResolutionStatus.AMBIGUOUS:
            if len(referenced_candidate_ids) < 2:
                raise ValueError("AMBIGUOUS resolution must reference multiple candidates")
        elif referenced_candidate_ids:
            raise ValueError("NO_MATCH resolution cannot reference candidates")
        return self

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
        "reference_candidates",
        "surface_references",
        when_used="always",
    )
    def _dump_reference_records(
        self,
        value: tuple[ReferenceCandidate | SurfaceReference, ...],
        info: SerializationInfo,
    ) -> list[dict[str, Any]]:
        return [record.model_dump(mode=info.mode) for record in value]

    @field_serializer("canonical_answer", when_used="always")
    def _dump_canonical_answer(
        self,
        value: CanonicalAnswer | None,
        info: SerializationInfo,
    ) -> dict[str, Any] | None:
        return None if value is None else value.model_dump(mode=info.mode)

    @field_serializer(
        "expected_answer",
        "profile",
        "stratification",
        when_used="always",
    )
    def _dump_json(self, value: Any) -> JsonValue:
        return thaw_json(value)


__all__ = ["CoreEvent", "GenerationContext", "SemanticCore"]
