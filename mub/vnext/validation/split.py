from __future__ import annotations

import hashlib
import math
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import JsonValue, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ArtifactRef,
    FrozenDict,
    FrozenJsonObject,
    ImmutableContractModel,
    freeze_json,
    thaw_json,
)
from mub.vnext.contracts.enums import Difficulty, QueryType, SourceType, Split, TaskFamily
from mub.vnext.contracts.manifest import TaskManifest
from mub.vnext.contracts.task import MemUpdateTask, SplitKey
from mub.vnext.io.canonical import canonical_json_bytes, sha256_model
from mub.vnext.profiles import REGISTERED_PROFILE_PARAMETER_KEYS
from mub.vnext.validation.issues import ValidationIssue, ValidationReport, build_report

_GROUP_FIELDS = (
    "semantic_core_id",
    "trajectory_id",
    "paraphrase_group_id",
    "source_group_id",
    "source_document_id",
    "version_group_id",
)
_REQUIRED_GROUP_FIELDS = frozenset({"semantic_core_id", "trajectory_id", "source_group_id"})
_STANDARD_SPLITS = (Split.TRAIN, Split.DEV, Split.TEST)
_ALL_SPLITS = (Split.TRAIN, Split.DEV, Split.TEST, Split.EVALUATION_ONLY)
_BUCKETS = frozenset({"1", "2-3", "4-7", "8-15", "16+"})

FAMILY_STRATIFICATION_AXES = FrozenDict(
    {
        TaskFamily.REPEATED_SAME_SLOT.value: (
            "update_depth_bucket",
            "active_object_count",
            "cross_slot_interleaving",
        ),
        TaskFamily.INTERLEAVED_MULTI_SLOT.value: (
            "update_depth_bucket",
            "active_object_count",
            "cross_slot_interleaving",
        ),
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: (
            "entity_ambiguity",
            "attribute_ambiguity",
            "alias_namespace_condition",
        ),
        TaskFamily.NOOP_WRITE_DISCIPLINE.value: (
            "write_trap_type",
            "noop_density",
            "duplicate_current_condition",
        ),
        TaskFamily.DELETION_FORGETTING.value: (
            "deletion_scope",
            "relearning_condition",
        ),
        TaskFamily.CURRENT_HISTORICAL_QUERY.value: (
            "query_type",
            "requested_version_distance",
        ),
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value: (
            "reasoning_depth",
            "active_object_count",
        ),
        TaskFamily.REALISTIC_SOURCE_UPDATE.value: (
            "source_type",
            "provenance_class",
        ),
    }
)

_SLICE_DIRECT_FILTERS = frozenset(
    {"task_family", "difficulty", "split", "source_type", "query_type", "tags"}
)
_SLICE_PROFILE_KEYS = frozenset(
    {*REGISTERED_PROFILE_PARAMETER_KEYS, "update_depth_bucket"}
)
_POSITIVE_PROFILE_KEYS = frozenset(
    {"update_depth", "active_object_count", "context_length", "reasoning_depth"}
)
_NONNEGATIVE_PROFILE_KEYS = frozenset({"stale_count", "requested_version_distance"})
_DENSITY_PROFILE_KEYS = frozenset({"noop_density", "cross_slot_interleaving"})


class _CanonicalPayload(ImmutableContractModel):
    payload: Any


class SplitException(ImmutableContractModel):
    split_exception_id: str
    version: str
    rationale: str
    allowed_group_ids: tuple[str, ...]
    reviewer: str

    @field_validator("split_exception_id", "version", "rationale", "reviewer")
    @classmethod
    def _validate_nonblank_text(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must be nonblank")
        return value.strip()

    @field_validator("allowed_group_ids")
    @classmethod
    def _validate_allowed_group_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("allowed_group_ids must be nonempty")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("allowed_group_ids must contain nonblank strings")
        normalized = tuple(value.strip() for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_group_ids must be unique")
        return normalized


class SliceDefinition(ImmutableContractModel):
    name: str
    filters: FrozenJsonObject
    task_ids: tuple[str, ...]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be nonblank")
        return value.strip()

    @field_validator("filters")
    @classmethod
    def _freeze_filters(cls, value: Mapping[str, JsonValue]):
        return freeze_json(value)

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("task_ids must contain nonblank strings")
        if len(values) != len(set(values)):
            raise ValueError("task_ids must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("task_ids must be sorted")
        return values

    @model_validator(mode="after")
    def _validate_filter_keys(self) -> Self:
        for key in self.filters:
            if key in _SLICE_DIRECT_FILTERS:
                continue
            if key.startswith("resolved_profile.") and key.removeprefix("resolved_profile.") in _SLICE_PROFILE_KEYS:
                continue
            raise ValueError(f"unsupported slice filter: {key}")
        return self


def validate_splits(
    tasks,
    declared_exceptions=(),
    task_manifest: TaskManifest | None = None,
    slice_definitions=(),
) -> ValidationReport:
    input_issues: list[ValidationIssue] = []
    hash_issues: list[ValidationIssue] = []
    group_issues: list[ValidationIssue] = []
    exception_issues: list[ValidationIssue] = []
    manifest_issues: list[ValidationIssue] = []
    strata_issues: list[ValidationIssue] = []
    slice_issues: list[ValidationIssue] = []

    try:
        copied_tasks = tuple(tasks)
    except Exception as exc:
        _issue(input_issues, "malformed_tasks_iterable", f"tasks could not be consumed: {type(exc).__name__}: {exc}", "tasks")
        copied_tasks = ()
    try:
        copied_exceptions = tuple(declared_exceptions)
    except Exception as exc:
        _issue(exception_issues, "malformed_exceptions_iterable", f"declared exceptions could not be consumed: {type(exc).__name__}: {exc}", "declared_exceptions")
        copied_exceptions = ()
    try:
        copied_slices = tuple(slice_definitions)
    except Exception as exc:
        _issue(slice_issues, "malformed_slices_iterable", f"slice definitions could not be consumed: {type(exc).__name__}: {exc}", "slice_definitions")
        copied_slices = ()

    try:
        task_records = _inspect_tasks(copied_tasks, input_issues, hash_issues)
    except Exception as exc:
        _issue(input_issues, "internal_input_validation_error", f"unexpected input validation failure: {type(exc).__name__}: {exc}", "tasks")
        task_records = []
    try:
        valid_exceptions, declared_exception_ids = _inspect_exceptions(
            copied_exceptions, exception_issues
        )
    except Exception as exc:
        _issue(exception_issues, "internal_exception_validation_error", f"unexpected exception validation failure: {type(exc).__name__}: {exc}", "declared_exceptions")
        valid_exceptions, declared_exception_ids = {}, set()
    try:
        _validate_groups(
            task_records,
            valid_exceptions,
            declared_exception_ids,
            group_issues,
        )
    except Exception as exc:
        _issue(group_issues, "internal_group_validation_error", f"unexpected group validation failure: {type(exc).__name__}: {exc}", "groups")
    try:
        normalized_manifest = _normalize_task_manifest(task_manifest, manifest_issues)
        _validate_manifest(task_records, normalized_manifest, manifest_issues)
    except Exception as exc:
        _issue(manifest_issues, "internal_manifest_validation_error", f"unexpected manifest validation failure: {type(exc).__name__}: {exc}", "task_manifest")
        normalized_manifest = None
    try:
        _validate_strata(
            task_records,
            normalized_manifest,
            strata_issues,
            raw_manifest=task_manifest,
        )
    except Exception as exc:
        _issue(strata_issues, "internal_strata_validation_error", f"unexpected strata validation failure: {type(exc).__name__}: {exc}", "strata")
    try:
        _validate_slices(task_records, copied_slices, slice_issues)
    except Exception as exc:
        _issue(slice_issues, "internal_slice_validation_error", f"unexpected slice validation failure: {type(exc).__name__}: {exc}", "slice_definitions")
    slice_issues.sort(
        key=lambda issue: (issue.path, issue.code, issue.severity, issue.message)
    )

    return build_report(
        (
            *input_issues,
            *hash_issues,
            *group_issues,
            *exception_issues,
            *manifest_issues,
            *strata_issues,
            *slice_issues,
        )
    )


def _issue(issues: list[ValidationIssue], code: str, message: str, path: str) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path, severity="error"))


def _nonblank(value: Any) -> bool:
    return _plain_canonical_text(value, require_canonical=False) is not None


def _plain_canonical_text(
    value: Any, *, require_canonical: bool = True
) -> str | None:
    if type(value) is not str:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if require_canonical and value != stripped:
        return None
    return value


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _strict_json_copy(
    value: Any, path: str = "$", active: set[int] | None = None
) -> Any:
    if active is None:
        active = set()
    if isinstance(value, Enum):
        return _strict_json_copy(value.value, path, active)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"cyclic mapping at {path}")
        active.add(identity)
        try:
            keys = tuple(value.keys())
            if any(type(key) is not str for key in keys):
                raise TypeError(f"JSON object keys must be exact strings at {path}")
            return {
                key: _strict_json_copy(value[key], f"{path}.{key}", active)
                for key in sorted(keys)
            }
        finally:
            active.remove(identity)
    if type(value) in (list, tuple):
        identity = id(value)
        if identity in active:
            raise ValueError(f"cyclic sequence at {path}")
        active.add(identity)
        try:
            return [
                _strict_json_copy(item, f"{path}[{index}]", active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise TypeError(f"non-JSON value {type(value).__name__} at {path}")


def _task_sort_key(task: Any) -> tuple[str, str, str]:
    task_id = getattr(task, "task_id", None)
    metadata = getattr(task, "metadata", None)
    split = _enum_value(getattr(metadata, "split", None))
    try:
        digest = _artifact_hash(task)
    except Exception:
        digest = type(task).__name__
    safe_task_id = _plain_canonical_text(task_id, require_canonical=False) or ""
    safe_split = _plain_canonical_text(split, require_canonical=False) or ""
    return safe_task_id, safe_split, digest


def _inspect_tasks(
    tasks: tuple[Any, ...],
    input_issues: list[ValidationIssue],
    hash_issues: list[ValidationIssue],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in sorted(tasks, key=_task_sort_key):
        task_id = getattr(task, "task_id", None)
        path_id = task_id if _nonblank(task_id) else "<blank>"
        path = f"tasks.{path_id}"
        if not isinstance(task, MemUpdateTask):
            _issue(input_issues, "malformed_task", "task must be a MemUpdateTask", path)
            records.append({"task": task, "task_id": task_id, "task_family": None, "path": path, "split": None, "split_key": None})
            continue
        if not _nonblank(task_id):
            _issue(input_issues, "blank_task_id", "task_id must be nonblank", f"{path}.task_id")
        task_family = getattr(task, "task_family", None)
        canonical_task_family = _plain_canonical_text(task_family)
        if canonical_task_family is None:
            _issue(input_issues, "invalid_task_family", "task_family must be canonical nonblank text", f"{path}.task_family")

        metadata = getattr(task, "metadata", None)
        split = getattr(metadata, "split", None)
        if not isinstance(split, Split):
            _issue(input_issues, "invalid_split", "metadata.split must be a Split", f"{path}.metadata.split")
            split = None
        raw_split_key = getattr(metadata, "split_key", None)
        split_key = None
        has_raw_exception_reference = False
        if not isinstance(raw_split_key, SplitKey):
            _issue(input_issues, "malformed_split_key", "metadata.split_key must be a SplitKey contract", f"{path}.metadata.split_key")
        else:
            try:
                raw_values = {
                    field: getattr(raw_split_key, field, None)
                    for field in SplitKey.model_fields
                }
                has_raw_exception_reference = (
                    raw_values["split_exception_id"] is not None
                )
                required_text_fields = (
                    "semantic_core_id",
                    "source_group_id",
                    "trajectory_id",
                    "split_policy_version",
                )
                optional_text_fields = (
                    "paraphrase_group_id",
                    "source_document_id",
                    "version_group_id",
                    "split_exception_id",
                )
                canonical_values: dict[str, str | None] = {}
                invalid_fields: list[str] = []
                for field in required_text_fields:
                    canonical = _plain_canonical_text(raw_values[field])
                    if canonical is None:
                        invalid_fields.append(field)
                    else:
                        canonical_values[field] = canonical
                for field in optional_text_fields:
                    raw_value = raw_values[field]
                    if raw_value is None:
                        canonical_values[field] = None
                        continue
                    canonical = _plain_canonical_text(raw_value)
                    if canonical is None:
                        invalid_fields.append(field)
                    else:
                        canonical_values[field] = canonical
                for field in sorted(invalid_fields):
                    _issue(
                        input_issues,
                        "malformed_split_key_field",
                        f"{field} must be canonical nonblank text without surrounding whitespace",
                        f"{path}.metadata.split_key.{field}",
                    )
                if invalid_fields:
                    raise ValueError("SplitKey text fields are not canonical")
                split_key = SplitKey.model_validate(canonical_values)
                required_valid = all(
                    _nonblank(getattr(split_key, field, None))
                    for field in (*_REQUIRED_GROUP_FIELDS, "split_policy_version")
                )
                optional_valid = all(
                    getattr(split_key, field, None) is None
                    or _nonblank(getattr(split_key, field, None))
                    for field in (
                        "paraphrase_group_id",
                        "source_document_id",
                        "version_group_id",
                        "split_exception_id",
                    )
                )
                if not required_valid or not optional_valid:
                    raise ValueError("SplitKey text fields violate the nonblank/null contract")
            except Exception as exc:
                _issue(input_issues, "malformed_split_key", f"metadata.split_key fails SplitKey validation: {type(exc).__name__}: {exc}", f"{path}.metadata.split_key")
                split_key = None
        if split_key is not None:
            for field in _GROUP_FIELDS:
                value = getattr(split_key, field, None)
                if value is None and field not in _REQUIRED_GROUP_FIELDS:
                    continue
                if not _nonblank(value):
                    _issue(input_issues, "malformed_group_id", f"{field} must be nonblank when present", f"{path}.metadata.split_key.{field}")
            if not _nonblank(getattr(split_key, "split_policy_version", None)):
                _issue(input_issues, "invalid_split_policy_version", "split_policy_version must be nonblank", f"{path}.metadata.split_key.split_policy_version")
            exception_id = getattr(split_key, "split_exception_id", None)
            if exception_id is not None and not _nonblank(exception_id):
                _issue(input_issues, "invalid_split_exception_id", "split_exception_id must be null or nonblank", f"{path}.metadata.split_key.split_exception_id")

        artifact_hash = None
        exact_hash = None
        try:
            artifact_hash = _artifact_hash(task)
        except Exception as exc:
            _issue(hash_issues, "task_artifact_hash_error", f"could not compute sha256_model: {type(exc).__name__}: {exc}", f"{path}.artifact_hash")
        try:
            exact_hash = _split_invariant_task_hash(task)
        except Exception as exc:
            _issue(hash_issues, "noncanonical_task_content_hash", f"split-invariant content is not canonical-safe: {type(exc).__name__}: {exc}", f"{path}.content_hash")
        records.append(
            {
                "task": task,
                "task_id": task_id,
                "task_family": canonical_task_family,
                "path": path,
                "split": split,
                "split_key": split_key,
                "has_raw_exception_reference": has_raw_exception_reference,
                "artifact_hash": artifact_hash,
                "exact_hash": exact_hash,
            }
        )

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if _nonblank(record["task_id"]):
            by_id[record["task_id"]].append(record)
    for task_id in sorted(by_id):
        same_id = by_id[task_id]
        if len(same_id) > 1:
            for duplicate_index in range(1, len(same_id)):
                _issue(input_issues, "duplicate_task_id", f"duplicate task_id: {task_id}", f"tasks.{task_id}[{duplicate_index}].task_id")
            splits = {_enum_value(record["split"]) for record in same_id if record["split"] is not None}
            if len(splits) > 1:
                _issue(input_issues, "conflicting_task_split", f"task_id {task_id} has conflicting splits: {sorted(splits)}", f"tasks.{task_id}.metadata.split")

    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("exact_hash") is not None and record.get("split") is not None:
            exact_groups[record["exact_hash"]].append(record)
    for digest in sorted(exact_groups):
        same_content = exact_groups[digest]
        splits = {_enum_value(record["split"]) for record in same_content}
        if len(splits) > 1:
            ids = sorted(str(record["task_id"]) for record in same_content)
            _issue(hash_issues, "exact_task_content_leakage", f"split-invariant exact task content crosses splits for task IDs {ids}", f"hashes.{digest}")
    return records


def _artifact_hash(task: MemUpdateTask) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return sha256_model(task)


def _split_invariant_task_hash(task: MemUpdateTask) -> str:
    """Hash surface-exact task content without task ID or split assignment metadata.

    The projection retains source text, events, queries, gold content, profile, tags,
    compiler metadata, and unrelated ``metadata.extra`` fields. It removes only
    ``task_id``, ``metadata.split``, the complete ``metadata.split_key``, and
    ``metadata.extra.evaluation_slice`` so every split/named-slice assignment is
    invariant. Semantic equivalence across non-exact surfaces remains governed by
    ``semantic_core_id`` rather than this hash.
    """
    payload = task.model_dump(
        mode="python",
        exclude_none=False,
        exclude_computed_fields=True,
        warnings=False,
    )
    payload.pop("task_id", None)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("split", None)
        metadata.pop("split_key", None)
        extra = metadata.get("extra")
        if isinstance(extra, dict):
            extra.pop("evaluation_slice", None)
    canonical_payload = _strict_json_copy(payload)
    wrapper = _CanonicalPayload.model_construct(payload=canonical_payload)
    return hashlib.sha256(canonical_json_bytes(wrapper)).hexdigest()


def _inspect_exceptions(
    exceptions: tuple[Any, ...], issues: list[ValidationIssue]
) -> tuple[dict[str, SplitException], set[str]]:
    valid: dict[str, SplitException] = {}
    declared_ids: set[str] = set()
    by_id: dict[str, list[SplitException]] = defaultdict(list)
    malformed_without_id: list[Any] = []
    for exception in exceptions:
        exception_id = getattr(exception, "split_exception_id", None)
        if isinstance(exception, SplitException) and _nonblank(exception_id):
            by_id[exception_id].append(exception)
        else:
            malformed_without_id.append(exception)

    for index, exception in enumerate(sorted(malformed_without_id, key=repr)):
        path = f"declared_exceptions.{index}"
        if not isinstance(exception, SplitException):
            _issue(issues, "invalid_split_exception", "declared exception must be a SplitException", path)
        else:
            _issue(issues, "invalid_split_exception", "split_exception_id must be a nonblank string", path)

    for exception_id in sorted(by_id):
        declarations = by_id[exception_id]
        path = f"declared_exceptions.{exception_id}"
        declared_ids.add(exception_id)
        if len(declarations) > 1:
            _issue(
                issues,
                "duplicate_split_exception_id",
                f"split_exception_id {exception_id} has {len(declarations)} ambiguous declarations",
                path,
            )
            continue
        exception = declarations[0]
        runtime_normalized = (
            isinstance(exception.split_exception_id, str)
            and isinstance(exception.version, str)
            and isinstance(exception.rationale, str)
            and type(exception.allowed_group_ids) is tuple
            and isinstance(exception.reviewer, str)
        )
        try:
            normalized = SplitException.model_validate(
                {
                    field: getattr(exception, field, None)
                    for field in SplitException.model_fields
                }
            )
        except Exception as exc:
            _issue(issues, "invalid_split_exception", f"exception fails SplitException validation: {type(exc).__name__}: {exc}", path)
            continue
        if not runtime_normalized:
            _issue(issues, "invalid_split_exception", "constructed exception fields must already use immutable normalized runtime types", path)
        valid[exception_id] = normalized
    return valid, declared_ids


def _evaluation_slice(record: dict[str, Any]) -> str | None:
    metadata = getattr(record["task"], "metadata", None)
    extra = getattr(metadata, "extra", None)
    if isinstance(extra, Mapping):
        value = extra.get("evaluation_slice")
        return value.strip() if _nonblank(value) else None
    return None


def _validate_groups(
    records: list[dict[str, Any]],
    valid_exceptions: Mapping[str, SplitException],
    declared_exception_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    for record in records:
        if (
            record.get("split") == Split.TRAIN
            and record.get("has_raw_exception_reference", False)
        ):
            _issue(
                issues,
                "training_split_exception_reference",
                "training tasks cannot reference split exceptions",
                f"{record['path']}.metadata.split_key.split_exception_id",
            )

    undeclared_references: dict[str, list[str]] = defaultdict(list)
    for record in records:
        split_key = record.get("split_key")
        exception_id = getattr(split_key, "split_exception_id", None) if split_key is not None else None
        if _nonblank(exception_id) and exception_id not in declared_exception_ids:
            undeclared_references[exception_id].append(str(record.get("task_id")))
    for exception_id in sorted(undeclared_references):
        task_ids = sorted(undeclared_references[exception_id])
        _issue(issues, "undeclared_split_exception", f"split_exception_id {exception_id} is not declared for task IDs {task_ids}", f"exceptions.references.{exception_id}")

    for field in _GROUP_FIELDS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            split_key = record.get("split_key")
            split = record.get("split")
            value = getattr(split_key, field, None) if split_key is not None else None
            if split is not None and _nonblank(value):
                groups[value].append(record)
        for group_id in sorted(groups):
            members = groups[group_id]
            partitions = set()
            for member in members:
                split = member["split"]
                evaluation_slice = _evaluation_slice(member) if split == Split.EVALUATION_ONLY else None
                partitions.add((split.value, evaluation_slice))
            if len(partitions) <= 1:
                continue
            path = f"groups.{field}.{group_id}"
            task_ids = sorted(str(member["task_id"]) for member in members)
            if any(member["split"] == Split.TRAIN for member in members):
                _issue(issues, "group_leakage_training", f"group crosses training and another partition for task IDs {task_ids}", path)
                continue
            exception_ids = [
                getattr(member["split_key"], "split_exception_id", None)
                for member in members
            ]
            nonnull = {value for value in exception_ids if _nonblank(value)}
            if any(not _nonblank(value) for value in exception_ids):
                _issue(issues, "group_leakage_missing_exception", f"nontraining group overlap lacks one shared nonnull exception for task IDs {task_ids}", path)
                continue
            if len(nonnull) != 1:
                _issue(issues, "group_leakage_mismatched_exception", f"nontraining group overlap uses mismatched exceptions: {sorted(nonnull)}", path)
                continue
            exception_id = next(iter(nonnull))
            if exception_id not in declared_exception_ids:
                continue
            exception = valid_exceptions.get(exception_id)
            if exception is None:
                _issue(issues, "invalid_split_exception_usage", f"split_exception_id {exception_id} is malformed", path)
                continue
            qualified = f"{field}:{group_id}"
            if group_id not in exception.allowed_group_ids and qualified not in exception.allowed_group_ids:
                _issue(issues, "disallowed_split_exception_group", f"split_exception_id {exception_id} does not allow {qualified}", path)


def _normalize_task_manifest(
    manifest: Any, issues: list[ValidationIssue]
) -> TaskManifest | None:
    if not isinstance(manifest, TaskManifest):
        _issue(
            issues,
            "malformed_task_manifest",
            "task_manifest must be a TaskManifest contract",
            "task_manifest",
        )
        return None

    payload = {
        field: getattr(manifest, field, None) for field in TaskManifest.model_fields
    }
    map_fields = (
        "compiler_versions",
        "split_counts",
        "family_difficulty_counts",
        "semantic_core_counts",
    )
    for field in map_fields:
        if not isinstance(payload[field], FrozenDict):
            _issue(
                issues,
                "malformed_task_manifest",
                f"{field} must already use immutable normalized mapping type",
                f"task_manifest.{field}",
            )
    if not isinstance(payload["leakage_check_summary"], FrozenDict) or not _is_frozen_json(
        payload["leakage_check_summary"]
    ):
        _issue(
            issues,
            "malformed_task_manifest",
            "leakage_check_summary must already be recursively immutable",
            "task_manifest.leakage_check_summary",
        )

    artifact_fields = (
        "source_manifest_paths_and_hashes",
        "generation_configs_and_hashes",
        "task_file_paths_and_hashes",
        "human_audit_artifacts",
    )
    artifact_error = False
    for field in artifact_fields:
        raw_artifacts = payload[field]
        if type(raw_artifacts) is not tuple:
            _issue(
                issues,
                "malformed_task_manifest",
                f"{field} must already be an immutable tuple",
                f"task_manifest.{field}",
            )
        if not _sequence(raw_artifacts):
            artifact_error = True
            continue
        normalized_artifacts: list[ArtifactRef] = []
        for index, artifact in enumerate(raw_artifacts):
            path = f"task_manifest.{field}[{index}]"
            if not isinstance(artifact, ArtifactRef):
                _issue(
                    issues,
                    "malformed_manifest_artifact",
                    "manifest artifact must be an ArtifactRef contract",
                    path,
                )
            try:
                source = (
                    {
                        nested_field: getattr(artifact, nested_field, None)
                        for nested_field in ArtifactRef.model_fields
                    }
                    if isinstance(artifact, ArtifactRef)
                    else artifact
                )
                normalized_artifacts.append(ArtifactRef.model_validate(source))
            except Exception as exc:
                _issue(
                    issues,
                    "malformed_manifest_artifact",
                    f"artifact fails ArtifactRef validation: {type(exc).__name__}: {exc}",
                    path,
                )
                artifact_error = True
        payload[field] = tuple(normalized_artifacts)

    if artifact_error:
        _issue(
            issues,
            "malformed_task_manifest",
            "task_manifest contains invalid nested artifacts",
            "task_manifest",
        )
        return None
    validation_payload = dict(payload)
    for field in map_fields:
        if isinstance(validation_payload[field], Mapping):
            validation_payload[field] = dict(validation_payload[field])
    if isinstance(validation_payload["leakage_check_summary"], Mapping):
        validation_payload["leakage_check_summary"] = thaw_json(
            validation_payload["leakage_check_summary"]
        )
    try:
        return TaskManifest.model_validate(validation_payload)
    except Exception as exc:
        _issue(
            issues,
            "malformed_task_manifest",
            f"task_manifest fails TaskManifest validation: {type(exc).__name__}: {exc}",
            "task_manifest",
        )
        return None


def _validate_manifest(
    records: list[dict[str, Any]],
    manifest: TaskManifest | None,
    issues: list[ValidationIssue],
) -> None:
    if manifest is None:
        return
    if not isinstance(manifest, TaskManifest):
        _issue(issues, "malformed_task_manifest", "task_manifest must be a TaskManifest", "task_manifest")
        return

    split_counts = _manifest_map(manifest, "split_counts", issues)
    family_counts = _manifest_map(manifest, "family_difficulty_counts", issues)
    semantic_counts = _manifest_map(manifest, "semantic_core_counts", issues)
    actual_split_counts = {
        split.value: sum(record.get("split") == split for record in records)
        for split in _ALL_SPLITS
    }
    actual_family_counts: dict[str, int] = {}
    for record in records:
        task = record["task"]
        family = record.get("task_family")
        difficulty = getattr(task, "difficulty", None)
        if _nonblank(family) and isinstance(difficulty, Difficulty):
            key = f"{family}|{difficulty.value}"
            actual_family_counts[key] = actual_family_counts.get(key, 0) + 1
    actual_semantic_counts = {}
    for split in _ALL_SPLITS:
        actual_semantic_counts[split.value] = len(
            {
                getattr(record.get("split_key"), "semantic_core_id", None)
                for record in records
                if record.get("split") == split
                and _nonblank(getattr(record.get("split_key"), "semantic_core_id", None))
            }
        )

    if split_counts is not None and dict(split_counts) != actual_split_counts:
        _issue(issues, "manifest_split_counts_mismatch", f"split_counts must equal {actual_split_counts}", "task_manifest.split_counts")
    if family_counts is not None and dict(family_counts) != actual_family_counts:
        _issue(issues, "manifest_family_difficulty_counts_mismatch", f"family_difficulty_counts must equal {actual_family_counts}", "task_manifest.family_difficulty_counts")
    if semantic_counts is not None and dict(semantic_counts) != actual_semantic_counts:
        _issue(issues, "manifest_semantic_core_counts_mismatch", f"semantic_core_counts must equal {actual_semantic_counts}", "task_manifest.semantic_core_counts")

    if split_counts is not None and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in split_counts.values()
    ):
        unique_task_ids = {record["task_id"] for record in records if _nonblank(record["task_id"])}
        if sum(split_counts.values()) != len(unique_task_ids):
            _issue(issues, "manifest_unique_task_count_mismatch", f"manifest split-count total {sum(split_counts.values())} does not equal {len(unique_task_ids)} unique task IDs", "task_manifest.split_counts")

    manifest_schema = getattr(manifest, "task_schema_version", None)
    manifest_policy = getattr(manifest, "split_policy_version", None)
    if not _nonblank(manifest_schema) or not _nonblank(manifest_policy):
        _issue(issues, "malformed_manifest", "task_schema_version and split_policy_version must be nonblank", "task_manifest")
    for record in records:
        task = record["task"]
        if getattr(task, "schema_version", None) != manifest_schema:
            _issue(issues, "task_schema_version_mismatch", f"task schema_version does not equal manifest task_schema_version {manifest_schema}", f"{record['path']}.schema_version")
        task_policy = getattr(record.get("split_key"), "split_policy_version", None)
        if task_policy != manifest_policy:
            _issue(issues, "task_split_policy_version_mismatch", f"task split_policy_version does not equal manifest split_policy_version {manifest_policy}", f"{record['path']}.metadata.split_key.split_policy_version")

    artifacts = getattr(manifest, "task_file_paths_and_hashes", None)
    record_count_sum = 0
    if not _sequence(artifacts):
        _issue(issues, "malformed_manifest", "task_file_paths_and_hashes must be a sequence", "task_manifest.task_file_paths_and_hashes")
    else:
        for index, artifact in enumerate(artifacts):
            count = getattr(artifact, "record_count", None)
            if count is None:
                _issue(issues, "missing_task_file_record_count", "every task artifact needs record_count", f"task_manifest.task_file_paths_and_hashes[{index}].record_count")
            elif not isinstance(count, int) or isinstance(count, bool) or count < 0:
                _issue(issues, "malformed_task_file_record_count", "record_count must be a strict nonnegative integer", f"task_manifest.task_file_paths_and_hashes[{index}].record_count")
            else:
                record_count_sum += count
        if record_count_sum != len(records):
            _issue(issues, "task_file_record_count_mismatch", f"sum of non-null task artifact record_count values {record_count_sum} does not equal {len(records)} tasks", "task_manifest.task_file_paths_and_hashes")

    summary = getattr(manifest, "leakage_check_summary", None)
    if not isinstance(summary, Mapping):
        _issue(issues, "malformed_manifest", "leakage_check_summary must be a mapping", "task_manifest.leakage_check_summary")
        return
    if "task_hashes" not in summary:
        _issue(issues, "missing_manifest_task_hashes", "leakage_check_summary.task_hashes is required", "task_manifest.leakage_check_summary.task_hashes")
        return
    task_hashes = summary.get("task_hashes")
    if not isinstance(task_hashes, Mapping):
        _issue(issues, "malformed_manifest_task_hashes", "task_hashes must be a mapping", "task_manifest.leakage_check_summary.task_hashes")
        return
    actual_hashes = {
        record["task_id"]: record["artifact_hash"]
        for record in records
        if _nonblank(record["task_id"]) and record.get("artifact_hash") is not None
    }
    if set(task_hashes) != set(actual_hashes):
        _issue(issues, "manifest_task_hash_set_mismatch", f"task_hashes keys must equal exact task set {sorted(actual_hashes)}", "task_manifest.leakage_check_summary.task_hashes")
    for task_id in sorted(set(task_hashes) & set(actual_hashes)):
        declared_hash = task_hashes[task_id]
        if not isinstance(declared_hash, str) or len(declared_hash) != 64 or any(char not in "0123456789abcdef" for char in declared_hash):
            _issue(issues, "malformed_manifest_task_hash", f"task hash for {task_id} must be lowercase SHA-256", f"task_manifest.leakage_check_summary.task_hashes.{task_id}")
        if declared_hash != actual_hashes[task_id]:
            _issue(issues, "manifest_task_hash_mismatch", f"task hash for {task_id} does not equal sha256_model(task)", f"task_manifest.leakage_check_summary.task_hashes.{task_id}")


def _manifest_map(manifest: TaskManifest, field: str, issues: list[ValidationIssue]) -> Mapping[str, Any] | None:
    value = getattr(manifest, field, None)
    if not isinstance(value, Mapping):
        _issue(issues, "malformed_manifest", f"{field} must be a mapping", f"task_manifest.{field}")
        return None
    return value


def _derive_bucket(update_depth: Any) -> str | None:
    if not isinstance(update_depth, int) or isinstance(update_depth, bool) or update_depth <= 0:
        return None
    if update_depth == 1:
        return "1"
    if update_depth <= 3:
        return "2-3"
    if update_depth <= 7:
        return "4-7"
    if update_depth <= 15:
        return "8-15"
    return "16+"


def _validate_strata(
    records: list[dict[str, Any]],
    manifest: TaskManifest | None,
    issues: list[ValidationIssue],
    *,
    raw_manifest: Any = None,
) -> None:
    stratum_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for record in records:
        task = record["task"]
        metadata = getattr(task, "metadata", None)
        family = record.get("task_family")
        if family is None:
            continue
        profile = getattr(metadata, "resolved_profile", None)
        if not isinstance(profile, Mapping):
            _issue(issues, "malformed_resolved_profile", "resolved_profile must be a mapping", f"{record['path']}.metadata.resolved_profile")
            continue
        difficulty = getattr(task, "difficulty", None)
        metadata_profile_name = getattr(metadata, "profile_name", None)
        expected_difficulty = difficulty.value if isinstance(difficulty, Difficulty) else None
        if metadata_profile_name != difficulty:
            _issue(issues, "profile_difficulty_mismatch", "metadata.profile_name must equal task difficulty", f"{record['path']}.metadata.profile_name")
        resolved_difficulty = _plain_canonical_text(profile.get("difficulty"))
        resolved_profile_name = _plain_canonical_text(profile.get("profile_name"))
        resolved_family = _plain_canonical_text(profile.get("task_family"))
        if resolved_difficulty != expected_difficulty or resolved_profile_name != expected_difficulty:
            _issue(issues, "profile_difficulty_mismatch", "resolved profile difficulty/profile_name must equal task difficulty", f"{record['path']}.metadata.resolved_profile.difficulty")
        if resolved_family != family:
            _issue(issues, "profile_task_family_mismatch", "resolved profile task_family must equal task_family", f"{record['path']}.metadata.resolved_profile.task_family")
        if not _nonblank(profile.get("profile_version")):
            _issue(issues, "missing_profile_version", "resolved profile requires profile_version", f"{record['path']}.metadata.resolved_profile.profile_version")

        derived = _derive_bucket(profile.get("update_depth"))
        raw_bucket_value = profile.get("update_depth_bucket")
        raw_bucket = _plain_canonical_text(raw_bucket_value)
        raw_bucket_valid = raw_bucket is not None and raw_bucket in _BUCKETS
        bucket = raw_bucket if raw_bucket_valid else derived
        if bucket is None:
            _issue(issues, "missing_update_depth_bucket", "minimum stratum requires a valid update_depth_bucket or strict positive update_depth", f"{record['path']}.metadata.resolved_profile.update_depth_bucket")
        if raw_bucket_value is not None and not raw_bucket_valid:
            _issue(issues, "invalid_update_depth_bucket", f"update_depth_bucket must be one of {sorted(_BUCKETS)}", f"{record['path']}.metadata.resolved_profile.update_depth_bucket")
        elif raw_bucket_value is not None and derived is not None and raw_bucket != derived:
            _issue(issues, "update_depth_bucket_mismatch", f"update_depth_bucket {raw_bucket} does not match update_depth-derived bucket {derived}", f"{record['path']}.metadata.resolved_profile.update_depth_bucket")

        family_axes = FAMILY_STRATIFICATION_AXES.get(family, ()) if isinstance(family, str) else ()
        for axis in family_axes:
            if axis == "update_depth_bucket":
                axis_present = raw_bucket_value is not None or bucket is not None
                axis_usable = (raw_bucket_value is None and bucket is not None) or raw_bucket_valid
            else:
                axis_present = axis in profile
                axis_usable = axis_present and _valid_registered_profile_value(axis, profile.get(axis))
            if not axis_present:
                _issue(issues, "missing_stratification_axis", f"known family {family} requires stratification axis {axis}", f"{record['path']}.metadata.resolved_profile.{axis}")
            elif not axis_usable:
                _issue(issues, "invalid_stratification_axis", f"stratification axis {axis} has an unusable value", f"{record['path']}.metadata.resolved_profile.{axis}")
        split = record.get("split")
        if _nonblank(family) and expected_difficulty is not None and bucket is not None and split is not None:
            stratum_counts[(family, expected_difficulty, bucket, split.value)] += 1

    if not isinstance(manifest, TaskManifest):
        return
    summary = getattr(manifest, "leakage_check_summary", None)
    if not isinstance(summary, Mapping):
        return
    raw_summary = (
        getattr(raw_manifest, "leakage_check_summary", None)
        if isinstance(raw_manifest, TaskManifest)
        else None
    )
    ledger_summary = raw_summary if isinstance(raw_summary, Mapping) else summary
    required = ledger_summary.get("required_minimum_strata")
    deviations = ledger_summary.get("small_cell_deviations")
    declared_required_records = _parse_required_strata(required, issues)
    derived_required_records = tuple(
        sorted({(family, difficulty, bucket) for family, difficulty, bucket, _ in stratum_counts})
    )
    if set(declared_required_records) != set(derived_required_records):
        _issue(
            issues,
            "required_minimum_strata_mismatch",
            f"required_minimum_strata must equal derived canonical task strata {list(derived_required_records)}",
            "task_manifest.leakage_check_summary.required_minimum_strata",
        )
    split_counts = getattr(manifest, "split_counts", {})
    if not isinstance(split_counts, Mapping):
        return
    valid_deviations = _parse_small_cell_deviations(
        deviations,
        derived_required_records,
        stratum_counts,
        split_counts,
        issues,
    )
    for family, difficulty, bucket in derived_required_records:
        for split in _STANDARD_SPLITS:
            declared_count = split_counts.get(split.value, 0)
            if not isinstance(declared_count, int) or isinstance(declared_count, bool) or declared_count <= 0:
                continue
            key = (family, difficulty, bucket, split.value)
            if stratum_counts.get(key, 0) == 0 and key not in valid_deviations:
                _issue(issues, "missing_required_minimum_stratum", f"required minimum stratum {(family, difficulty, bucket)} is absent from {split.value}", f"task_manifest.leakage_check_summary.required_minimum_strata.{family}|{difficulty}|{bucket}.{split.value}")


def _parse_required_strata(value: Any, issues: list[ValidationIssue]) -> tuple[tuple[str, str, str], ...]:
    if not _sequence(value):
        _issue(issues, "malformed_required_minimum_strata", "required_minimum_strata must be a list of records", "task_manifest.leakage_check_summary.required_minimum_strata")
        return ()
    parsed: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    expected_fields = {
        "task_family",
        "difficulty",
        "update_depth_bucket",
    }
    for index, record in enumerate(value):
        path = f"task_manifest.leakage_check_summary.required_minimum_strata[{index}]"
        if not isinstance(record, Mapping):
            _issue(issues, "invalid_required_minimum_stratum", "required stratum must be a mapping", path)
            continue
        keys = tuple(record.keys())
        if any(type(key) is not str for key in keys) or set(keys) != expected_fields:
            _issue(
                issues,
                "invalid_required_minimum_stratum",
                "required stratum must use exactly task_family, difficulty, and update_depth_bucket with exact string keys",
                path,
            )
            continue
        family = _plain_canonical_text(record["task_family"])
        difficulty = _plain_canonical_text(record["difficulty"])
        bucket = _plain_canonical_text(record["update_depth_bucket"])
        if (
            family is None
            or difficulty is None
            or difficulty not in {item.value for item in Difficulty}
            or bucket is None
            or bucket not in _BUCKETS
        ):
            _issue(issues, "invalid_required_minimum_stratum", "required stratum needs nonblank string task_family, valid string difficulty, and valid string update_depth_bucket", path)
            continue
        key = (family, difficulty, bucket)
        if key in seen:
            _issue(issues, "duplicate_required_minimum_stratum", f"duplicate required stratum {key}", path)
            continue
        seen.add(key)
        parsed.append(key)
    return tuple(parsed)


def _parse_small_cell_deviations(
    value: Any,
    required: tuple[tuple[str, str, str], ...],
    counts: Mapping[tuple[str, str, str, str], int],
    split_counts: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> set[tuple[str, str, str, str]]:
    if not _sequence(value):
        _issue(issues, "malformed_small_cell_deviations", "small_cell_deviations must be a list of records", "task_manifest.leakage_check_summary.small_cell_deviations")
        return set()
    valid: set[tuple[str, str, str, str]] = set()
    required_set = set(required)
    expected_fields = {
        "task_family",
        "difficulty",
        "update_depth_bucket",
        "split",
        "observed_count",
        "rationale",
    }
    for index, record in enumerate(value):
        path = f"task_manifest.leakage_check_summary.small_cell_deviations[{index}]"
        if not isinstance(record, Mapping):
            _issue(issues, "invalid_small_cell_deviation", "small-cell deviation must be a mapping", path)
            continue
        keys = tuple(record.keys())
        if any(type(key) is not str for key in keys) or set(keys) != expected_fields:
            _issue(issues, "invalid_small_cell_deviation", "deviation must use exact fields and strict scalar types for a standard split", path)
            continue
        family = _plain_canonical_text(record["task_family"])
        difficulty = _plain_canonical_text(record["difficulty"])
        bucket = _plain_canonical_text(record["update_depth_bucket"])
        split = _plain_canonical_text(record["split"])
        observed = record["observed_count"]
        rationale = record["rationale"]
        scalar_types_valid = (
            family is not None
            and difficulty is not None
            and difficulty in {item.value for item in Difficulty}
            and bucket is not None
            and bucket in _BUCKETS
            and split is not None
            and split in {item.value for item in _STANDARD_SPLITS}
            and isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed >= 0
            and _nonblank(rationale)
        )
        if not scalar_types_valid:
            _issue(issues, "invalid_small_cell_deviation", "deviation must use exact fields and strict scalar types for a standard split", path)
            continue
        key = (family, difficulty, bucket, split)
        declared_split_count = split_counts.get(split)
        valid_shape = (
            (family, difficulty, bucket) in required_set
            and isinstance(declared_split_count, int)
            and not isinstance(declared_split_count, bool)
            and declared_split_count > 0
            and observed == counts.get(key, 0)
            and observed == 0
        )
        if not valid_shape or key in valid:
            _issue(issues, "invalid_small_cell_deviation", "deviation must identify a derived missing cell in a positive-count standard split with consistent observed_count=0", path)
            continue
        valid.add(key)
    return valid


def _valid_registered_profile_value(key: str, value: Any) -> bool:
    if key == "update_depth_bucket":
        plain = _plain_canonical_text(value)
        return plain is not None and plain in _BUCKETS
    if key == "duplicate_current_condition":
        return type(value) is bool
    if key in _POSITIVE_PROFILE_KEYS:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if key in _NONNEGATIVE_PROFILE_KEYS:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if key in _DENSITY_PROFILE_KEYS:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        )
    if key == "query_type":
        plain = _plain_canonical_text(value)
        return plain is not None and plain in {item.value for item in QueryType}
    if key == "source_type":
        plain = _plain_canonical_text(value)
        return plain is not None and plain in {item.value for item in SourceType}
    return _plain_canonical_text(value) is not None


def _is_frozen_json(value: Any) -> bool:
    if isinstance(value, Mapping):
        return isinstance(value, FrozenDict) and all(
            _is_frozen_json(item) for item in value.values()
        )
    if isinstance(value, tuple):
        return all(_is_frozen_json(item) for item in value)
    if isinstance(value, list):
        return False
    return True


def _slice_value_category(value: Any) -> bytes:
    if value is None:
        return b"none"
    if type(value) is str:
        return b"text"
    if isinstance(value, str):
        return b"text-subclass"
    if type(value) is bytes:
        return b"bytes"
    if type(value) is bytearray:
        return b"bytearray"
    if type(value) is tuple:
        return b"tuple"
    if type(value) is list:
        return b"list"
    if isinstance(value, Mapping):
        return b"mapping"
    return b"other"


def _slice_name_sort_component(raw_name: Any) -> bytes:
    try:
        if type(raw_name) is not str:
            return b"1:" + _slice_value_category(raw_name)
        encoded = raw_name.encode("utf-8")
        if raw_name and raw_name == raw_name.strip():
            return b"0:" + encoded
        return b"1:invalid-text:" + encoded
    except Exception:
        return b"2:unrepresentable-text"


def _slice_filter_sort_component(raw_filters: Any) -> bytes:
    try:
        payload = _strict_json_copy(thaw_json(raw_filters))
        canonical = canonical_json_bytes(
            _CanonicalPayload.model_construct(payload=payload)
        )
        return b"0:" + canonical
    except Exception:
        return b"1:" + _slice_value_category(raw_filters)


def _slice_task_ids_sort_component(raw_task_ids: Any) -> bytes:
    try:
        if type(raw_task_ids) not in (list, tuple):
            return b"2:" + _slice_value_category(raw_task_ids)
        shape = []
        for item in raw_task_ids:
            if type(item) is str:
                shape.append(
                    {
                        "kind": "canonical-text"
                        if item and item == item.strip()
                        else "invalid-text",
                        "value": item,
                    }
                )
            else:
                shape.append(
                    {
                        "kind": "invalid-type",
                        "category": _slice_value_category(item).decode("ascii"),
                    }
                )
        canonical = canonical_json_bytes(
            _CanonicalPayload.model_construct(payload=shape)
        )
        marker = b"0:tuple:" if type(raw_task_ids) is tuple else b"1:list:"
        return marker + canonical
    except Exception:
        return b"3:unrepresentable:" + _slice_value_category(raw_task_ids)


def _slice_sort_key(definition: Any) -> tuple[bytes, bytes, bytes]:
    try:
        if not isinstance(definition, SliceDefinition):
            return b"1:non-slice", b"1:other", b"2:other"
        fields = object.__getattribute__(definition, "__dict__")
        raw_name = fields.get("name")
        raw_filters = fields.get("filters")
        raw_task_ids = fields.get("task_ids")
    except Exception:
        return b"1:unreadable", b"1:other", b"2:other"
    try:
        name_component = _slice_name_sort_component(raw_name)
    except Exception:
        name_component = b"2:unrepresentable-name"
    try:
        filter_component = _slice_filter_sort_component(raw_filters)
    except Exception:
        filter_component = b"2:unrepresentable-filter"
    try:
        task_ids_component = _slice_task_ids_sort_component(raw_task_ids)
    except Exception:
        task_ids_component = b"3:unrepresentable-task-ids"
    try:
        return name_component, filter_component, task_ids_component
    except Exception:
        return (
            b"2:unrepresentable-name",
            b"2:unrepresentable-filter",
            b"3:unrepresentable-task-ids",
        )


def _validate_slices(
    records: list[dict[str, Any]],
    definitions: tuple[Any, ...],
    issues: list[ValidationIssue],
) -> None:
    tasks_by_id: dict[str, MemUpdateTask] = {}
    for record in records:
        if _nonblank(record["task_id"]) and isinstance(record["task"], MemUpdateTask):
            tasks_by_id.setdefault(record["task_id"], record["task"])
    ordered_tasks = tuple(tasks_by_id[task_id] for task_id in sorted(tasks_by_id))
    seen_names: set[str] = set()
    for index, definition in enumerate(
        sorted(definitions, key=_slice_sort_key)
    ):
        if not isinstance(definition, SliceDefinition):
            path = f"slice_definitions.{index}"
            _issue(issues, "malformed_slice_definition", "slice must be a SliceDefinition", path)
            continue
        try:
            fields = object.__getattribute__(definition, "__dict__")
            raw_name = fields.get("name")
            raw_filters = fields.get("filters")
            raw_task_ids = fields.get("task_ids")
        except Exception:
            _issue(
                issues,
                "malformed_slice_definition",
                "slice fields could not be safely extracted",
                f"slice_definitions.{index}",
            )
            continue
        path = f"slice_definitions.{raw_name if _nonblank(raw_name) else index}"
        try:
            if type(raw_name) is not str or not raw_name or raw_name != raw_name.strip():
                raise TypeError("slice name must be canonical exact text")
            raw_name.encode("utf-8")
            if type(raw_task_ids) is not tuple or any(
                type(task_id) is not str
                or not task_id
                or task_id != task_id.strip()
                for task_id in raw_task_ids
            ):
                raise TypeError("slice task IDs must be canonical exact strings in a tuple")
            for task_id in raw_task_ids:
                task_id.encode("utf-8")
            duplicate_ids = len(raw_task_ids) != len(set(raw_task_ids))
            unsorted_ids = raw_task_ids != tuple(sorted(raw_task_ids))
            if duplicate_ids:
                _issue(issues, "duplicate_slice_task_id", "slice task IDs must be unique", f"{path}.task_ids")
            if unsorted_ids:
                _issue(issues, "unsorted_slice_task_ids", "slice task IDs must be sorted", f"{path}.task_ids")
            if duplicate_ids or unsorted_ids:
                raise ValueError("slice task IDs violate uniqueness or ordering")
            if not isinstance(raw_filters, FrozenDict) or not _is_frozen_json(
                raw_filters
            ):
                raise TypeError("slice filters must be recursively immutable")
            normalized = SliceDefinition.model_validate(
                {
                    "name": raw_name,
                    "filters": thaw_json(raw_filters),
                    "task_ids": list(raw_task_ids),
                }
            )
        except Exception:
            _issue(issues, "malformed_slice_definition", "slice fails SliceDefinition validation", path)
            continue
        name = normalized.name
        filters = normalized.filters
        task_ids = normalized.task_ids
        if name in seen_names:
            _issue(issues, "duplicate_slice_name", f"duplicate slice name: {name}", f"{path}.name")
        seen_names.add(name)
        unsupported = [key for key in filters if not _supported_slice_filter(key)]
        if unsupported:
            for key in sorted(unsupported):
                _issue(issues, "unsupported_slice_filter", f"unsupported slice filter: {key}", f"{path}.filters.{key}")
            continue
        malformed_values = [
            key for key in filters if not _valid_slice_filter_value(key, filters[key])
        ]
        if malformed_values:
            for key in sorted(malformed_values):
                _issue(issues, "malformed_slice_filter", f"slice filter {key} has an unusable value", f"{path}.filters.{key}")
            continue
        try:
            expected = tuple(
                task.task_id
                for task in ordered_tasks
                if _task_matches_filters(task, filters)
            )
        except Exception as exc:
            _issue(issues, "malformed_slice_filter", f"slice filter could not be evaluated: {type(exc).__name__}: {exc}", f"{path}.filters")
            continue
        if tuple(task_ids) != expected:
            _issue(issues, "slice_task_ids_mismatch", f"declared task_ids {list(task_ids)} do not equal deterministic filter output {list(expected)}", f"{path}.task_ids")


def _supported_slice_filter(key: Any) -> bool:
    if type(key) is not str:
        return False
    if key in _SLICE_DIRECT_FILTERS:
        return True
    return (
        key.startswith("resolved_profile.")
        and key.removeprefix("resolved_profile.") in _SLICE_PROFILE_KEYS
    )


def _valid_slice_filter_value(key: str, value: Any) -> bool:
    if key == "task_family":
        return _plain_canonical_text(value) is not None
    if key == "difficulty":
        plain = _plain_canonical_text(value)
        return plain is not None and plain in {item.value for item in Difficulty}
    if key == "split":
        plain = _plain_canonical_text(value)
        return plain is not None and plain in {item.value for item in Split}
    if key == "source_type":
        plain = _plain_canonical_text(value)
        return plain is not None and plain in {item.value for item in SourceType}
    if key == "query_type":
        plain = _plain_canonical_text(value)
        return plain is not None and plain in {item.value for item in QueryType}
    if key == "tags":
        if isinstance(value, str):
            return _plain_canonical_text(value) is not None
        return (
            _sequence(value)
            and bool(value)
            and all(_plain_canonical_text(tag) is not None for tag in value)
        )
    if key.startswith("resolved_profile."):
        return _valid_registered_profile_value(
            key.removeprefix("resolved_profile."), value
        )
    return False


def _task_matches_filters(task: MemUpdateTask, filters: Mapping[str, Any]) -> bool:
    for key in sorted(filters):
        expected = filters[key]
        if key == "task_family":
            actual = task.task_family
        elif key == "difficulty":
            actual = _enum_value(task.difficulty)
        elif key == "split":
            actual = _enum_value(task.metadata.split)
        elif key == "source_type":
            actual = _enum_value(task.source.source_type)
        elif key == "query_type":
            expected_value = _enum_value(expected)
            if not any(_enum_value(query.query_type) == expected_value for query in task.queries):
                return False
            continue
        elif key == "tags":
            required_tags = (expected,) if isinstance(expected, str) else tuple(expected) if _sequence(expected) else None
            if required_tags is None or any(not isinstance(tag, str) for tag in required_tags):
                raise ValueError("tags filter must be a string or list of strings")
            if not set(required_tags).issubset(set(task.metadata.tags)):
                return False
            continue
        elif key.startswith("resolved_profile."):
            profile_key = key.removeprefix("resolved_profile.")
            actual = task.metadata.resolved_profile.get(profile_key)
        else:
            return False
        if _enum_value(actual) != _enum_value(expected):
            return False
    return True


__all__ = [
    "FAMILY_STRATIFICATION_AXES",
    "SliceDefinition",
    "SplitException",
    "validate_splits",
]
