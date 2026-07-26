from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Self

from pydantic import JsonValue, field_validator, model_validator

from mub.vnext.contracts.common import (
    FrozenDict,
    FrozenJsonObject,
    ImmutableContractModel,
    freeze_json,
    thaw_json,
)
from mub.vnext.contracts.enums import Difficulty
from mub.vnext.version import PROFILE_VERSION

CANONICAL_CONTROL_KEYS = (
    "update_depth",
    "active_object_count",
    "entity_ambiguity",
    "attribute_ambiguity",
    "noop_density",
    "cross_slot_interleaving",
    "stale_count",
    "context_length",
    "context_order",
    "version_metadata",
    "query_type",
    "source_naturalness",
)

FAMILY_SPECIFIC_PARAMETER_KEYS = (
    "alias_namespace_condition",
    "write_trap_type",
    "duplicate_current_condition",
    "deletion_scope",
    "relearning_condition",
    "requested_version_distance",
    "reasoning_depth",
    "source_type",
    "provenance_class",
)

REGISTERED_PROFILE_PARAMETER_KEYS = frozenset(
    (*CANONICAL_CONTROL_KEYS, *FAMILY_SPECIFIC_PARAMETER_KEYS)
)
PROTECTED_PROFILE_LABELS = frozenset(
    {"task_family", "difficulty", "profile_name", "profile_version"}
)
DERIVED_PROFILE_KEYS = frozenset({"update_depth_bucket"})


def _canonicalize_json(value: Any, path: str = "parameters") -> Any:
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(type(key) is not str for key in keys):
            raise ValueError(f"{path} canonical JSON mapping keys must be exact strings")
        return {
            key: _canonicalize_json(value[key], f"{path}.{key}")
            for key in sorted(keys)
        }
    if type(value) in (list, tuple):
        return [
            _canonicalize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} canonical JSON float must be finite")
        return value
    raise TypeError(
        f"{path} must contain only canonical JSON values; got {type(value).__name__}"
    )


# Phase 0 defaults are deliberately explicit and environment-independent. Repeated
# same-slot depth follows the established 1/4/16 progression and adds 32 for the
# compositional challenge tier.
_CANONICAL_DEFAULT_DATA: dict[str, dict[str, JsonValue]] = {
    "easy": {
        "update_depth": 1,
        "active_object_count": 1,
        "entity_ambiguity": "none",
        "attribute_ambiguity": "none",
        "noop_density": 0.0,
        "cross_slot_interleaving": 0.0,
        "stale_count": 0,
        "context_length": 4,
        "context_order": "chronological",
        "version_metadata": "latest_outdated",
        "query_type": "current_state",
        "source_naturalness": "synthetic_direct",
    },
    "medium": {
        "update_depth": 4,
        "active_object_count": 4,
        "entity_ambiguity": "moderate",
        "attribute_ambiguity": "moderate",
        "noop_density": 0.1,
        "cross_slot_interleaving": 0.5,
        "stale_count": 3,
        "context_length": 16,
        "context_order": "retrieval_score",
        "version_metadata": "event_index",
        "query_type": "current_state",
        "source_naturalness": "mixed_template",
    },
    "hard": {
        "update_depth": 16,
        "active_object_count": 8,
        "entity_ambiguity": "high",
        "attribute_ambiguity": "high",
        "noop_density": 0.25,
        "cross_slot_interleaving": 0.75,
        "stale_count": 15,
        "context_length": 64,
        "context_order": "reverse_chronological",
        "version_metadata": "none",
        "query_type": "current_state",
        "source_naturalness": "semi_natural",
    },
    "challenge": {
        "update_depth": 32,
        "active_object_count": 16,
        "entity_ambiguity": "compositional",
        "attribute_ambiguity": "compositional",
        "noop_density": 0.4,
        "cross_slot_interleaving": 1.0,
        "stale_count": 31,
        "context_length": 128,
        "context_order": "adversarial",
        "version_metadata": "none",
        "query_type": "mixed",
        "source_naturalness": "natural",
    },
}
CANONICAL_PROFILE_DEFAULTS = FrozenDict(
    {
        name: freeze_json(_canonicalize_json(parameters))
        for name, parameters in sorted(_CANONICAL_DEFAULT_DATA.items())
    }
)


class ProfileSpec(ImmutableContractModel):
    name: str
    version: str
    task_family: str
    difficulty: Difficulty
    parameters: FrozenJsonObject
    allowed_overrides: tuple[str, ...]

    @field_validator("name", "version", "task_family")
    @classmethod
    def _validate_nonblank_text(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must be nonblank")
        return value.strip()

    @field_validator("parameters", mode="before")
    @classmethod
    def _canonicalize_parameters(cls, value):
        return _canonicalize_json(value)

    @field_validator("parameters")
    @classmethod
    def _freeze_parameters(cls, value: Mapping[str, JsonValue]):
        return freeze_json(value)

    @field_validator("allowed_overrides", mode="before")
    @classmethod
    def _validate_allowed_overrides_source(cls, value: Any) -> tuple[Any, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("allowed_overrides must be an ordered list or tuple")
        if any(
            type(item) is not str or not item or item != item.strip()
            for item in value
        ):
            raise TypeError(
                "allowed_overrides entries must be nonblank canonical exact strings"
            )
        return tuple(value)

    @field_validator("allowed_overrides")
    @classmethod
    def _validate_allowed_overrides(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(type(item) is not str or not item.strip() for item in value):
            raise ValueError("allowed_overrides entries must be nonblank exact strings")
        normalized = value
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_overrides must be unique")
        return normalized

    @model_validator(mode="after")
    def _validate_profile_contract(self) -> Self:
        if self.name != self.difficulty.value:
            raise ValueError("profile name must equal difficulty.value")
        unknown = set(self.parameters) - REGISTERED_PROFILE_PARAMETER_KEYS
        if unknown:
            raise ValueError(f"unknown parameter keys: {sorted(unknown)}")
        if "update_depth" not in self.parameters:
            raise ValueError("update_depth is required")
        for key, value in self.parameters.items():
            _validate_parameter_value(key, value)
        if not set(self.allowed_overrides).issubset(self.parameters):
            raise ValueError("allowed_overrides must be a subset of parameter keys")
        forbidden = set(self.allowed_overrides) & (PROTECTED_PROFILE_LABELS | DERIVED_PROFILE_KEYS)
        if forbidden:
            raise ValueError(f"allowed_overrides contains protected or derived keys: {sorted(forbidden)}")
        return self


def build_generic_profile(difficulty: Difficulty, task_family: str) -> ProfileSpec:
    if not isinstance(difficulty, Difficulty):
        raise TypeError("difficulty must be a Difficulty")
    parameters = CANONICAL_PROFILE_DEFAULTS[difficulty.value]
    return ProfileSpec(
        name=difficulty.value,
        version=PROFILE_VERSION,
        task_family=task_family,
        difficulty=difficulty,
        parameters=parameters,
        allowed_overrides=tuple(_CANONICAL_DEFAULT_DATA[difficulty.value]),
    )


def easy_profile(task_family: str) -> ProfileSpec:
    return build_generic_profile(Difficulty.EASY, task_family)


def medium_profile(task_family: str) -> ProfileSpec:
    return build_generic_profile(Difficulty.MEDIUM, task_family)


def hard_profile(task_family: str) -> ProfileSpec:
    return build_generic_profile(Difficulty.HARD, task_family)


def challenge_profile(task_family: str) -> ProfileSpec:
    return build_generic_profile(Difficulty.CHALLENGE, task_family)


def resolve_profile(
    profile: ProfileSpec,
    overrides: Mapping[str, JsonValue] | None = None,
) -> FrozenDict:
    if not isinstance(profile, ProfileSpec):
        raise TypeError("profile must be a ProfileSpec")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")

    canonical_overrides = _canonicalize_json(overrides, "overrides")
    copied_overrides = {
        key: thaw_json(value) for key, value in canonical_overrides.items()
    }
    override_keys = set(copied_overrides)
    protected = override_keys & PROTECTED_PROFILE_LABELS
    if protected:
        raise ValueError(f"cannot override protected profile label: {sorted(protected)}")
    derived = override_keys & DERIVED_PROFILE_KEYS
    if derived:
        raise ValueError(f"cannot override derived profile label: {sorted(derived)}")
    unknown_registered = override_keys - REGISTERED_PROFILE_PARAMETER_KEYS
    if unknown_registered:
        raise ValueError(f"unknown override keys: {sorted(unknown_registered)}")

    validated_profile = ProfileSpec.model_validate(
        profile.model_dump(mode="python", warnings=False)
    )
    parameters = {key: thaw_json(value) for key, value in validated_profile.parameters.items()}
    unknown_parameters = set(parameters) - REGISTERED_PROFILE_PARAMETER_KEYS
    if unknown_parameters:
        raise ValueError(f"unknown parameter keys: {sorted(unknown_parameters)}")
    missing_from_profile = override_keys - set(parameters)
    if missing_from_profile:
        raise ValueError(f"unknown override keys for profile: {sorted(missing_from_profile)}")
    disallowed = override_keys - set(validated_profile.allowed_overrides)
    if disallowed:
        raise ValueError(f"override keys are not allowed: {sorted(disallowed)}")

    resolved: dict[str, JsonValue] = {}
    for key in sorted(parameters):
        default = parameters[key]
        value = copied_overrides.get(key, default)
        _validate_type_compatible(value, default, key)
        _validate_parameter_value(key, value)
        resolved[key] = value

    update_depth = resolved.get("update_depth")
    if not _strict_positive_int(update_depth):
        raise ValueError("update_depth must be a strict positive integer")

    resolved.update(
        {
            "task_family": validated_profile.task_family,
            "difficulty": validated_profile.difficulty.value,
            "profile_name": validated_profile.name,
            "profile_version": validated_profile.version,
            "update_depth_bucket": _update_depth_bucket(update_depth),
        }
    )
    return freeze_json(resolved)


def _validate_type_compatible(value: Any, default: Any, path: str) -> None:
    if isinstance(default, bool):
        compatible = isinstance(value, bool)
    elif isinstance(default, int):
        compatible = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        compatible = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif isinstance(default, str):
        compatible = isinstance(value, str)
    elif default is None:
        compatible = value is None
    elif isinstance(default, Mapping):
        compatible = isinstance(value, Mapping) and set(value) == set(default)
        if compatible:
            for key in sorted(default):
                _validate_type_compatible(value[key], default[key], f"{path}.{key}")
    elif isinstance(default, Sequence) and not isinstance(default, (str, bytes, bytearray)):
        compatible = isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        if compatible and default:
            for index, item in enumerate(value):
                _validate_type_compatible(item, default[min(index, len(default) - 1)], f"{path}[{index}]")
    else:
        compatible = type(value) is type(default)
    if not compatible:
        raise TypeError(f"{path} override must remain type-compatible with its default")


def _validate_parameter_value(key: str, value: Any) -> None:
    _reject_nonfinite_or_blank(value, key)
    if key in {"noop_density", "cross_slot_interleaving"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if key in {"update_depth", "active_object_count", "context_length", "reasoning_depth"}:
        if not _strict_positive_int(value):
            raise ValueError(f"{key} must be a strict positive integer")
    if key in {"stale_count", "requested_version_distance"}:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a strict nonnegative integer")


def _reject_nonfinite_or_blank(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{path} must be nonblank")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_or_blank(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_nonfinite_or_blank(item, f"{path}[{index}]")


def _strict_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _update_depth_bucket(update_depth: int) -> str:
    if update_depth == 1:
        return "1"
    if update_depth <= 3:
        return "2-3"
    if update_depth <= 7:
        return "4-7"
    if update_depth <= 15:
        return "8-15"
    return "16+"


__all__ = [
    "CANONICAL_CONTROL_KEYS",
    "CANONICAL_PROFILE_DEFAULTS",
    "FAMILY_SPECIFIC_PARAMETER_KEYS",
    "ProfileSpec",
    "REGISTERED_PROFILE_PARAMETER_KEYS",
    "build_generic_profile",
    "challenge_profile",
    "easy_profile",
    "hard_profile",
    "medium_profile",
    "resolve_profile",
]
