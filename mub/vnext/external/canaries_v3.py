from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Literal
import uuid

from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts.common import (
    ArtifactRef,
    FrozenDict,
    FrozenNonnegativeIntMap,
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictNonnegativeInt,
    freeze_mapping,
)
from mub.vnext.contracts.enums import Operation, Split
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.io import canonical_json_bytes, sha256_model


CANARY_SCHEMA_VERSION = "memupdatebench.external.canary.v1"
CORE_TASK_RELEASE_MANIFEST_HASH = (
    "f953283a10dd45d3f9d1de066570a9c09b9d132ed458f8dea3c948641b89e99d"
)
_FAMILY_LETTERS = {
    "repeated_same_slot_update": "A",
    "interleaved_multi_slot_update": "B",
    "entity_attribute_grounding": "C",
    "noop_write_discipline": "D",
    "deletion_forgetting": "E",
    "current_historical_query": "F",
    "long_horizon_memory_synthesis": "G",
}
_CANARY_FAMILY_QUOTAS = {
    "A": 8,
    "B": 8,
    "C": 8,
    "D": 8,
    "E": 12,
    "F": 12,
    "G": 8,
}
CANARY_SELECTION_POLICY_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "family_letters": _FAMILY_LETTERS,
            "family_quotas": _CANARY_FAMILY_QUOTAS,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
CANARY_SELECTION_VERSION = (
    "core_task10_phase2_canary_selection_v1:"
    f"{CANARY_SELECTION_POLICY_SHA256}"
)
FAMILY_LETTERS = MappingProxyType(_FAMILY_LETTERS)
CANARY_FAMILY_QUOTAS = MappingProxyType(_CANARY_FAMILY_QUOTAS)
_IMMUTABLE_CORE_ROOT = (
    Path(__file__).resolve().parents[3] / "data" / "vnext" / "core" / "v3"
)
_CANARY_IDS = ("canary_a", "canary_b")
_INDEPENDENCE_FIELDS = (
    "task_id",
    "semantic_core_id",
    "trajectory_id",
    "version_group_id",
    "source_group_id",
    "source_document_id",
    "paraphrase_group_id",
)
_WITHIN_CANARY_UNIQUE_FIELDS = (
    "task_id",
    "semantic_core_id",
    "source_group_id",
    "source_document_id",
    "paraphrase_group_id",
)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _strict_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonblank exact built-in string")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_payload_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class CoreReleaseArtifactRefV1(ImmutableContractModel):
    path: str = Field(strict=True, min_length=1)
    sha256: str = Field(strict=True, pattern=SHA256_PATTERN)
    media_type: str = Field(strict=True, min_length=1)
    size_bytes: StrictNonnegativeInt

    @field_validator("path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if (
            path.is_absolute()
            or path.drive
            or path.root
            or ".." in path.parts
            or path.as_posix() != value
        ):
            raise ValueError("release artifact path must be a normalized relative path")
        return value


class CoreTaskReleaseManifestV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core.task_release.v1"]
    artifact_refs: tuple[CoreReleaseArtifactRefV1, ...]
    candidate_generation_revision: str = Field(strict=True, pattern=r"^[0-9a-f]{40}$")
    candidate_receipt_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    candidate_root_digest: str = Field(strict=True, pattern=SHA256_PATTERN)
    gate_attestation_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    hard_suite_core_count: StrictNonnegativeInt
    hard_suite_task_count: StrictNonnegativeInt
    release_manifest_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    release_root_digest: str = Field(strict=True, pattern=SHA256_PATTERN)
    release_stage: Literal["task_release"]
    release_status: Literal["FINAL_APPROVED"]
    review_context_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    selection_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    semantic_core_count: StrictNonnegativeInt
    source_task_manifest_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    split_task_counts: FrozenNonnegativeIntMap
    structural_report_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    task_count: StrictNonnegativeInt

    @field_validator("split_task_counts")
    @classmethod
    def _freeze_split_counts(cls, value):
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _unique_artifact_paths(self):
        paths = tuple(ref.path for ref in self.artifact_refs)
        if len(paths) != len(set(paths)):
            raise ValueError("release artifact paths must be unique")
        if set(self.split_task_counts) != {"train", "dev", "test"}:
            raise ValueError(
                "release split counts must name train, dev, and test exactly"
            )
        return self


class SelectedCanaryTaskV1(ImmutableContractModel):
    task_id: StrictIdentifier
    task_record_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    family_letter: Literal["A", "B", "C", "D", "E", "F", "G"]
    semantic_core_id: StrictIdentifier
    trajectory_id: StrictIdentifier
    version_group_id: StrictIdentifier
    source_group_id: StrictIdentifier
    source_document_id: StrictIdentifier
    paraphrase_group_id: StrictIdentifier


class CanaryManifestV1(ImmutableContractModel):
    schema_version: Literal[CANARY_SCHEMA_VERSION] = CANARY_SCHEMA_VERSION
    selection_version: str = Field(strict=True, min_length=1)
    canary_id: Literal["canary_a", "canary_b"]
    source_release_manifest_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    source_task_manifest_ref: ArtifactRef
    source_tasks_ref: ArtifactRef
    tasks_ref: ArtifactRef
    selected_tasks: tuple[SelectedCanaryTaskV1, ...]
    family_task_counts: FrozenNonnegativeIntMap

    @field_validator("family_task_counts")
    @classmethod
    def _freeze_family_counts(cls, value):
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _coherent(self):
        _validate_manifest_shape(self)
        return self


class CanarySetManifestV1(ImmutableContractModel):
    schema_version: Literal[CANARY_SCHEMA_VERSION] = CANARY_SCHEMA_VERSION
    selection_version: str = Field(strict=True, min_length=1)
    source_release_manifest_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    canary_manifest_refs: tuple[ArtifactRef, ArtifactRef]
    canary_ids: tuple[Literal["canary_a", "canary_b"], Literal["canary_a", "canary_b"]]
    independence_fields: tuple[str, ...]

    @model_validator(mode="after")
    def _coherent(self):
        if self.selection_version != CANARY_SELECTION_VERSION:
            raise ValueError("unsupported canary set selection version")
        if self.canary_ids != _CANARY_IDS:
            raise ValueError("canary set manifest must order canary_a before canary_b")
        if self.independence_fields != _INDEPENDENCE_FIELDS:
            raise ValueError(
                "canary set manifest independence fields are not canonical"
            )
        if any(
            type(ref) is not ArtifactRef
            for ref in self.canary_manifest_refs
        ):
            raise ValueError(
                "canary manifest references must be exact ArtifactRef instances"
            )
        for ref in self.canary_manifest_refs:
            _revalidate_exact(ArtifactRef, ref, "canary manifest reference")
        return self


@dataclass(frozen=True)
class AuthenticatedCoreReleaseV1:
    root: Path
    release_manifest: CoreTaskReleaseManifestV1
    release_manifest_hash: str
    task_manifest: TaskManifestV3
    task_manifest_ref: ArtifactRef
    tasks_ref: ArtifactRef
    raw_record_by_task_id: Mapping[str, bytes]
    task_by_id: Mapping[str, MemUpdateTaskV3]
    dev_tasks: tuple[MemUpdateTaskV3, ...]


@dataclass(frozen=True)
class CanaryBundleV1:
    manifest: CanaryManifestV1
    tasks: tuple[MemUpdateTaskV3, ...]
    records: tuple[bytes, ...]
    manifest_bytes: bytes


@dataclass(frozen=True)
class CanarySetBundleV1:
    canaries: tuple[CanaryBundleV1, CanaryBundleV1]
    set_manifest: CanarySetManifestV1
    set_manifest_bytes: bytes


def _revalidate_exact(model_type, value, label: str) -> None:
    if type(value) is not model_type:
        raise ValueError(f"{label} must have an exact trusted model type")
    try:
        rebuilt = model_type.model_validate(value.model_dump(mode="python"))
    except Exception as exc:
        raise ValueError(f"{label} fails trust-boundary validation") from exc
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(value):
        raise ValueError(f"{label} serialization is not stable")


def _validate_manifest_shape(manifest: CanaryManifestV1) -> None:
    if manifest.selection_version != CANARY_SELECTION_VERSION:
        raise ValueError("unsupported canary selection version")
    if (
        type(manifest.selected_tasks) is not tuple
        or type(manifest.family_task_counts) is not FrozenDict
    ):
        raise ValueError("canary manifest has untrusted container types")
    artifact_refs = (
        manifest.source_task_manifest_ref,
        manifest.source_tasks_ref,
        manifest.tasks_ref,
    )
    if any(type(value) is not ArtifactRef for value in artifact_refs):
        raise ValueError(
            "canary artifact references must be exact ArtifactRef instances"
        )
    if any(
        type(item) is not SelectedCanaryTaskV1
        for item in manifest.selected_tasks
    ):
        raise ValueError(
            "selected canary records must be exact SelectedCanaryTaskV1 instances"
        )
    for value in artifact_refs:
        _revalidate_exact(ArtifactRef, value, "canary artifact reference")
    for item in manifest.selected_tasks:
        _revalidate_exact(SelectedCanaryTaskV1, item, "selected canary record")
    if len(manifest.selected_tasks) != 64:
        raise ValueError("canary manifest must select exactly 64 tasks")
    task_ids = tuple(item.task_id for item in manifest.selected_tasks)
    if task_ids != tuple(sorted(task_ids)) or len(task_ids) != len(set(task_ids)):
        raise ValueError("selected task IDs must be unique and sorted")
    family_counts = Counter(
        item.family_letter for item in manifest.selected_tasks
    )
    if family_counts != CANARY_FAMILY_QUOTAS:
        raise ValueError("canary family quotas are invalid")
    if dict(manifest.family_task_counts) != CANARY_FAMILY_QUOTAS:
        raise ValueError("canary manifest family counts are invalid")
    if (
        manifest.tasks_ref.path != "tasks.jsonl"
        or manifest.tasks_ref.media_type != "application/x-ndjson"
        or manifest.tasks_ref.record_count != 64
    ):
        raise ValueError("derived tasks reference is invalid")
    if (
        manifest.source_task_manifest_ref.path
        != "candidate/task_manifest.json"
        or manifest.source_tasks_ref.path != "candidate/tasks.jsonl"
    ):
        raise ValueError("canary source artifact paths are invalid")
    if (
        manifest.source_task_manifest_ref.media_type != "application/json"
        or manifest.source_tasks_ref.media_type
        != "application/x-ndjson"
    ):
        raise ValueError("canary source artifact media types are invalid")
    for field in _WITHIN_CANARY_UNIQUE_FIELDS[1:]:
        values = tuple(getattr(item, field) for item in manifest.selected_tasks)
        if len(values) != len(set(values)):
            raise ValueError(f"canary {field} values must be unique")


def _validate_canary_manifest_shape(manifest: CanaryManifestV1) -> CanaryManifestV1:
    if type(manifest) is not CanaryManifestV1:
        raise ValueError("canary manifest must be an exact CanaryManifestV1 instance")
    _validate_manifest_shape(manifest)
    try:
        rebuilt = CanaryManifestV1.model_validate(manifest.model_dump(mode="python"))
    except Exception as exc:
        raise ValueError("canary manifest fails trust-boundary validation") from exc
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(manifest):
        raise ValueError("canary manifest serialization is not stable")
    return rebuilt


def _release_artifact_ref(
    ref: CoreReleaseArtifactRefV1,
    *,
    record_count: int | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        path=ref.path,
        sha256=ref.sha256,
        media_type=ref.media_type,
        record_count=record_count,
    )


def _is_reparse_point(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _regular_file(path: Path) -> None:
    if not path.is_file() or _is_reparse_point(path):
        raise ValueError(f"release artifact is not a regular file: {path}")
    if path.lstat().st_nlink != 1:
        raise ValueError(f"release artifact must be a single-link file: {path}")


def _validate_release_tree(
    root: Path,
    release: CoreTaskReleaseManifestV1,
    release_payload: dict[str, Any],
) -> None:
    expected_files = {
        "task_release_manifest.json",
        *(ref.path for ref in release.artifact_refs),
    }
    expected_directories: set[str] = set()
    for relative_path in expected_files:
        parent = Path(relative_path).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        if _is_reparse_point(path):
            raise ValueError("Core release tree contains a reparse point")
        relative_path = path.relative_to(root).as_posix()
        if path.is_file():
            observed_files.add(relative_path)
        elif path.is_dir():
            observed_directories.add(relative_path)
        else:
            raise ValueError("Core release tree contains a non-file entry")
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise ValueError("Core release tree does not match its artifact manifest")
    observed_root_digest = _sha256(
        _canonical_payload_bytes(release_payload["artifact_refs"])
    )
    if observed_root_digest != release.release_root_digest:
        raise ValueError("Core release root digest is invalid")


def _release_member(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    _regular_file(path)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("release artifact escapes the release root") from exc
    return path


def _read_canonical_model(path: Path, model_type):
    _regular_file(path)
    raw = path.read_bytes()
    try:
        model = model_type.model_validate_json(raw)
    except Exception as exc:
        raise ValueError(f"invalid {path.name}") from exc
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"{path.name} is not canonical JSON")
    return model, raw


def authenticate_core_release(release_root: str | Path) -> AuthenticatedCoreReleaseV1:
    root = Path(release_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Core release root does not exist")
    release_path = root / "task_release_manifest.json"
    _regular_file(release_path)
    release_raw = release_path.read_bytes()
    try:
        release_payload = json.loads(release_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("task release manifest is not JSON") from exc
    if (
        type(release_payload) is not dict
        or _canonical_payload_bytes(release_payload) != release_raw
    ):
        raise ValueError("task release manifest is not canonical JSON")
    try:
        release = CoreTaskReleaseManifestV1.model_validate(release_payload)
    except Exception as exc:
        raise ValueError("task release manifest contract is invalid") from exc
    self_payload = dict(release_payload)
    declared_hash = self_payload.pop("release_manifest_hash", None)
    observed_hash = _sha256(_canonical_payload_bytes(self_payload))
    if (
        declared_hash != observed_hash
        or release.release_manifest_hash != observed_hash
    ):
        raise ValueError("task release manifest self-hash is invalid")
    if observed_hash != CORE_TASK_RELEASE_MANIFEST_HASH:
        raise ValueError(
            "task release manifest is not the immutable approved Core release"
        )
    _validate_release_tree(root, release, release_payload)
    artifacts = {ref.path: ref for ref in release.artifact_refs}
    if len(artifacts) != len(release.artifact_refs):
        raise ValueError("task release manifest contains duplicate artifact references")
    for ref in release.artifact_refs:
        path = _release_member(root, ref.path)
        if ref.path == "candidate/tasks.jsonl":
            continue
        raw = path.read_bytes()
        if len(raw) != ref.size_bytes or _sha256(raw) != ref.sha256:
            raise ValueError(f"task release artifact binding is invalid: {ref.path}")
    try:
        task_manifest_release_ref = artifacts["candidate/task_manifest.json"]
        tasks_release_ref = artifacts["candidate/tasks.jsonl"]
    except KeyError as exc:
        raise ValueError("task release lacks required candidate artifacts") from exc
    task_manifest, task_manifest_raw = _read_canonical_model(
        _release_member(root, task_manifest_release_ref.path), TaskManifestV3
    )
    if _sha256(task_manifest_raw) != release.source_task_manifest_hash:
        raise ValueError("release source task-manifest hash is invalid")
    if task_manifest_release_ref.sha256 != release.source_task_manifest_hash:
        raise ValueError("release task-manifest artifact binding is invalid")
    source_task_manifest_ref = _release_artifact_ref(
        task_manifest_release_ref,
        record_count=1,
    )
    source_tasks_ref = _release_artifact_ref(
        tasks_release_ref,
        record_count=release.task_count,
    )
    if tuple(task_manifest.task_file_paths_and_hashes) != (
        ArtifactRef(
            path="tasks.jsonl",
            sha256=source_tasks_ref.sha256,
            media_type="application/x-ndjson",
            record_count=release.task_count,
        ),
    ):
        raise ValueError("candidate task manifest does not authenticate tasks.jsonl")
    if len(task_manifest.task_record_hashes) != release.task_count:
        raise ValueError("candidate task manifest record coverage is invalid")
    raw_by_id: dict[str, bytes] = {}
    task_by_id: dict[str, MemUpdateTaskV3] = {}
    seen_task_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    tasks_path = _release_member(root, tasks_release_ref.path)
    tasks_hasher = hashlib.sha256()
    tasks_size = 0
    previous_task_id: str | None = None
    with tasks_path.open("rb") as task_file:
        for line_number, raw_line in enumerate(task_file, start=1):
            tasks_hasher.update(raw_line)
            tasks_size += len(raw_line)
            if not raw_line.endswith(b"\n") or raw_line == b"\n":
                raise ValueError(
                    f"candidate tasks line {line_number} is not canonical"
                )
            raw = raw_line[:-1]
            try:
                task = MemUpdateTaskV3.model_validate_json(raw)
            except Exception as exc:
                raise ValueError(
                    f"candidate task line {line_number} is invalid"
                ) from exc
            if canonical_json_bytes(task) != raw:
                raise ValueError(
                    f"candidate task line {line_number} is not canonical"
                )
            if task.task_id in seen_task_ids:
                raise ValueError("candidate tasks contain duplicate IDs")
            if previous_task_id is not None and task.task_id <= previous_task_id:
                raise ValueError("candidate tasks must be sorted by task ID")
            previous_task_id = task.task_id
            seen_task_ids.add(task.task_id)
            if (
                task_manifest.task_record_hashes.get(task.task_id)
                != sha256_model(task)
            ):
                raise ValueError(
                    f"candidate task record hash is invalid: {task.task_id}"
                )
            split_counts[task.metadata.split.value] += 1
            if task.metadata.split is Split.DEV:
                raw_by_id[task.task_id] = raw_line
                task_by_id[task.task_id] = task
    if (
        tasks_size != tasks_release_ref.size_bytes
        or tasks_hasher.hexdigest() != tasks_release_ref.sha256
    ):
        raise ValueError(
            "task release artifact binding is invalid: "
            "candidate/tasks.jsonl"
        )
    if (
        len(seen_task_ids) != release.task_count
        or seen_task_ids != set(task_manifest.task_record_hashes)
    ):
        raise ValueError("candidate task count or record coverage is invalid")
    if {
        split: split_counts[split] for split in ("train", "dev", "test")
    } != dict(release.split_task_counts):
        raise ValueError("candidate split counts are invalid")
    dev_tasks = tuple(task_by_id.values())
    if len(dev_tasks) != release.split_task_counts["dev"]:
        raise ValueError("candidate dev split count is invalid")
    return AuthenticatedCoreReleaseV1(
        root=root,
        release_manifest=release,
        release_manifest_hash=observed_hash,
        task_manifest=task_manifest,
        task_manifest_ref=source_task_manifest_ref,
        tasks_ref=source_tasks_ref,
        raw_record_by_task_id=MappingProxyType(raw_by_id),
        task_by_id=MappingProxyType(task_by_id),
        dev_tasks=dev_tasks,
    )


def coverage_tokens(task: MemUpdateTaskV3) -> frozenset[str]:
    tokens: set[str] = set()
    for action in task.actions:
        if action.operation in {Operation.ADD, Operation.UPDATE, Operation.NOOP}:
            tokens.add(f"operation:{action.operation.value}")
        if (
            task.task_family == "deletion_forgetting"
            and action.operation is Operation.DELETE
            and action.scope is not None
        ):
            tokens.add(f"E:delete:{action.scope.value}")
    if any(len(query.target_object_keys) > 1 for query in task.queries):
        tokens.add("multi_object")
    if task.task_family == "current_historical_query":
        if any(query.selector.kind == "current" for query in task.queries):
            tokens.add("F:current")
        if any(query.selector.kind != "current" for query in task.queries):
            tokens.add("F:historical")
    if task.task_family == "long_horizon_memory_synthesis":
        if task.queries:
            tokens.add("G:query")
        if any(query.synthesis is not None for query in task.queries):
            tokens.add("G:synthesis")
    return frozenset(tokens)


def _required_coverage(tasks: tuple[MemUpdateTaskV3, ...]) -> tuple[str, ...]:
    source_tokens = set().union(*(coverage_tokens(task) for task in tasks))
    tracked_tokens = {
        "operation:ADD",
        "operation:UPDATE",
        "operation:NOOP",
        "F:current",
        "F:historical",
        "multi_object",
        "G:synthesis",
        "G:query",
    }
    required = {
        token
        for token in source_tokens
        if token in tracked_tokens or token.startswith("E:delete:")
    }
    return tuple(sorted(required))


def _selection_rank(
    release: AuthenticatedCoreReleaseV1,
    canary_id: str,
    task: MemUpdateTaskV3,
) -> str:
    split_key = task.metadata.split_key
    surface = task.metadata.extra.get("surface_variant")
    if type(surface) is not int:
        raise ValueError(f"task {task.task_id} has no strict surface identity")
    components = (
        CANARY_SELECTION_VERSION,
        canary_id,
        release.task_manifest_ref.sha256,
        FAMILY_LETTERS.get(task.task_family, ""),
        split_key.semantic_core_id,
        str(surface),
        task.task_id,
    )
    return _sha256("\x1f".join(components).encode("utf-8"))


def _selected_item(
    release: AuthenticatedCoreReleaseV1,
    task: MemUpdateTaskV3,
) -> SelectedCanaryTaskV1:
    key = task.metadata.split_key
    values = {
        "task_id": task.task_id,
        "task_record_hash": release.task_manifest.task_record_hashes[task.task_id],
        "family_letter": FAMILY_LETTERS[task.task_family],
        "semantic_core_id": key.semantic_core_id,
        "trajectory_id": key.trajectory_id,
        "version_group_id": key.version_group_id,
        "source_group_id": key.source_group_id,
        "source_document_id": key.source_document_id,
        "paraphrase_group_id": key.paraphrase_group_id,
    }
    for field in _INDEPENDENCE_FIELDS:
        _strict_string(values[field], field)
    return SelectedCanaryTaskV1(**values)


def _independence_values(task: MemUpdateTaskV3) -> dict[str, str]:
    key = task.metadata.split_key
    return {
        "task_id": task.task_id,
        "semantic_core_id": key.semantic_core_id,
        "trajectory_id": key.trajectory_id,
        "version_group_id": key.version_group_id,
        "source_group_id": key.source_group_id,
        "source_document_id": key.source_document_id,
        "paraphrase_group_id": key.paraphrase_group_id,
    }


def _can_add(
    task: MemUpdateTaskV3,
    local_reserved: dict[str, set[str]],
    cross_canary_reserved: dict[str, set[str]],
    family_counts: Counter[str],
) -> bool:
    family = FAMILY_LETTERS[task.task_family]
    if family_counts[family] >= CANARY_FAMILY_QUOTAS[family]:
        return False
    candidate = _independence_values(task)
    if any(type(value) is not str or not value.strip() for value in candidate.values()):
        return False
    if any(
        candidate[field] in cross_canary_reserved[field]
        for field in _INDEPENDENCE_FIELDS
    ):
        return False
    return all(
        candidate[field] not in local_reserved[field]
        for field in _WITHIN_CANARY_UNIQUE_FIELDS
    )


def _reserve(
    task: MemUpdateTaskV3,
    selected: list[MemUpdateTaskV3],
    local_reserved: dict[str, set[str]],
    cross_canary_reserved: dict[str, set[str]],
    family_counts: Counter[str],
) -> None:
    if not _can_add(
        task,
        local_reserved,
        cross_canary_reserved,
        family_counts,
    ):
        raise ValueError(
            "canary selection conflicts with an independence or quota "
            "constraint"
        )
    selected.append(task)
    family_counts[FAMILY_LETTERS[task.task_family]] += 1
    candidate = _independence_values(task)
    for field in _WITHIN_CANARY_UNIQUE_FIELDS:
        local_reserved[field].add(candidate[field])


def _family_f_group_partition(
    release: AuthenticatedCoreReleaseV1,
) -> dict[str, frozenset[tuple[str, str]]]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for task in release.dev_tasks:
        if task.task_family != "current_historical_query":
            continue
        key = task.metadata.split_key
        groups[(key.trajectory_id, key.version_group_id)].add(
            key.semantic_core_id
        )
    if not groups or len(groups) % len(_CANARY_IDS):
        raise ValueError("Family F groups cannot be evenly partitioned")
    ranked = tuple(
        sorted(
            groups,
            key=lambda group: (
                _sha256(
                    "\x1f".join(
                        (
                            CANARY_SELECTION_VERSION,
                            "family_f_group_partition",
                            release.task_manifest_ref.sha256,
                            group[0],
                            group[1],
                        )
                    ).encode("utf-8")
                ),
                group,
            ),
        )
    )
    group_count = len(ranked) // len(_CANARY_IDS)
    partition = {
        canary_id: frozenset(
            ranked[index * group_count : (index + 1) * group_count]
        )
        for index, canary_id in enumerate(_CANARY_IDS)
    }
    for canary_id, assigned in partition.items():
        capacity = sum(len(groups[group]) for group in assigned)
        if capacity < CANARY_FAMILY_QUOTAS["F"]:
            raise ValueError(
                f"insufficient Family F group capacity for {canary_id}"
            )
    return partition


def _choose_canary(
    release: AuthenticatedCoreReleaseV1,
    canary_id: str,
    cross_canary_reserved: dict[str, set[str]],
    family_f_groups: frozenset[tuple[str, str]],
) -> tuple[MemUpdateTaskV3, ...]:
    def allowed(task: MemUpdateTaskV3) -> bool:
        if task.task_family != "current_historical_query":
            return True
        key = task.metadata.split_key
        return (key.trajectory_id, key.version_group_id) in family_f_groups

    candidates = tuple(
        sorted(
            (task for task in release.dev_tasks if allowed(task)),
            key=lambda task: (
                _selection_rank(release, canary_id, task),
                task.task_id,
            ),
        )
    )
    selected: list[MemUpdateTaskV3] = []
    local_reserved = {
        field: set() for field in _WITHIN_CANARY_UNIQUE_FIELDS
    }
    counts: Counter[str] = Counter()
    selected_tokens: set[str] = set()
    required = _required_coverage(release.dev_tasks)
    required_set = set(required)
    for requirement in required:
        if requirement in selected_tokens:
            continue
        eligible = [
            task
            for task in candidates
            if requirement in coverage_tokens(task)
            and _can_add(
                task,
                local_reserved,
                cross_canary_reserved,
                counts,
            )
        ]
        if not eligible:
            raise ValueError(
                f"insufficient dev capacity for {canary_id} "
                f"coverage requirement {requirement}"
            )
        best_score = max(
            len((coverage_tokens(task) & required_set) - selected_tokens)
            for task in eligible
        )
        chosen = next(
            task
            for task in eligible
            if len((coverage_tokens(task) & required_set) - selected_tokens)
            == best_score
        )
        _reserve(
            chosen,
            selected,
            local_reserved,
            cross_canary_reserved,
            counts,
        )
        selected_tokens.update(coverage_tokens(chosen))
    for family, quota in CANARY_FAMILY_QUOTAS.items():
        while counts[family] < quota:
            chosen = next(
                (
                    task
                    for task in candidates
                    if FAMILY_LETTERS[task.task_family] == family
                    and _can_add(
                        task,
                        local_reserved,
                        cross_canary_reserved,
                        counts,
                    )
                ),
                None,
            )
            if chosen is None:
                break
            _reserve(
                chosen,
                selected,
                local_reserved,
                cross_canary_reserved,
                counts,
            )
            selected_tokens.update(coverage_tokens(chosen))
        if counts[family] != quota:
            raise ValueError(
                f"insufficient dev capacity for {canary_id} family {family}"
            )
    if len(selected) != 64 or required_set - selected_tokens:
        raise ValueError(f"canary {canary_id} coverage selection is incomplete")
    return tuple(sorted(selected, key=lambda task: task.task_id))


def _make_bundle(
    release: AuthenticatedCoreReleaseV1,
    canary_id: str,
    tasks: tuple[MemUpdateTaskV3, ...],
) -> CanaryBundleV1:
    records = tuple(release.raw_record_by_task_id[task.task_id] for task in tasks)
    items = tuple(_selected_item(release, task) for task in tasks)
    manifest = CanaryManifestV1(
        selection_version=CANARY_SELECTION_VERSION,
        canary_id=canary_id,
        source_release_manifest_hash=release.release_manifest_hash,
        source_task_manifest_ref=release.task_manifest_ref,
        source_tasks_ref=release.tasks_ref,
        tasks_ref=ArtifactRef(
            path="tasks.jsonl",
            sha256=_sha256(b"".join(records)),
            media_type="application/x-ndjson",
            record_count=len(records),
        ),
        selected_tasks=items,
        family_task_counts=CANARY_FAMILY_QUOTAS,
    )
    manifest = _validate_canary_manifest_shape(manifest)
    return CanaryBundleV1(
        manifest=manifest,
        tasks=tasks,
        records=records,
        manifest_bytes=canonical_json_bytes(manifest),
    )


def _build_canary_set_authenticated(
    release: AuthenticatedCoreReleaseV1,
) -> CanarySetBundleV1:
    cross_canary_reserved = {field: set() for field in _INDEPENDENCE_FIELDS}
    family_f_partition = _family_f_group_partition(release)
    built: list[CanaryBundleV1] = []
    for canary_id in _CANARY_IDS:
        tasks = _choose_canary(
            release,
            canary_id,
            cross_canary_reserved,
            family_f_partition[canary_id],
        )
        bundle = _make_bundle(release, canary_id, tasks)
        built.append(bundle)
        for task in tasks:
            candidate = _independence_values(task)
            for field in _INDEPENDENCE_FIELDS:
                cross_canary_reserved[field].add(candidate[field])
    bundles = tuple(built)
    _validate_pair_independence(bundles)
    refs = tuple(
        ArtifactRef(
            path=(
                f"{bundle.manifest.canary_id}/canary_manifest.json"
            ),
            sha256=_sha256(bundle.manifest_bytes),
            media_type="application/json",
            record_count=1,
        )
        for bundle in bundles
    )
    set_manifest = CanarySetManifestV1(
        selection_version=CANARY_SELECTION_VERSION,
        source_release_manifest_hash=release.release_manifest_hash,
        canary_manifest_refs=refs,
        canary_ids=_CANARY_IDS,
        independence_fields=_INDEPENDENCE_FIELDS,
    )
    return CanarySetBundleV1(
        canaries=bundles,
        set_manifest=set_manifest,
        set_manifest_bytes=canonical_json_bytes(set_manifest),
    )


def _fresh_authenticated_release(
    release: AuthenticatedCoreReleaseV1,
) -> AuthenticatedCoreReleaseV1:
    if type(release) is not AuthenticatedCoreReleaseV1:
        raise ValueError("canary operation requires an authenticated Core release")
    return authenticate_core_release(release.root)


def build_canary_set(
    release: AuthenticatedCoreReleaseV1,
) -> CanarySetBundleV1:
    fresh_release = _fresh_authenticated_release(release)
    return _build_canary_set_authenticated(fresh_release)


def _validate_pair_independence(
    bundles: tuple[CanaryBundleV1, CanaryBundleV1],
) -> None:
    for field in _INDEPENDENCE_FIELDS:
        left = {getattr(item, field) for item in bundles[0].manifest.selected_tasks}
        right = {getattr(item, field) for item in bundles[1].manifest.selected_tasks}
        if left & right:
            raise ValueError(f"canary independence violation for {field}")


def _validate_canary_bundle_authenticated(
    bundle: CanaryBundleV1,
    release: AuthenticatedCoreReleaseV1,
) -> CanaryBundleV1:
    if (
        type(bundle) is not CanaryBundleV1
        or type(release) is not AuthenticatedCoreReleaseV1
    ):
        raise ValueError(
            "canary bundle validation requires exact trusted types"
        )
    manifest = _validate_canary_manifest_shape(bundle.manifest)
    if (
        type(bundle.tasks) is not tuple
        or type(bundle.records) is not tuple
        or len(bundle.tasks) != 64
        or len(bundle.records) != 64
    ):
        raise ValueError("canary bundle task and record counts are invalid")
    if canonical_json_bytes(manifest) != bundle.manifest_bytes:
        raise ValueError("canary manifest bytes are invalid")
    if (
        manifest.source_release_manifest_hash
        != release.release_manifest_hash
        or manifest.source_task_manifest_ref != release.task_manifest_ref
        or manifest.source_tasks_ref != release.tasks_ref
    ):
        raise ValueError("canary source release bindings are invalid")
    if _sha256(b"".join(bundle.records)) != manifest.tasks_ref.sha256:
        raise ValueError("canary derived task artifact hash is invalid")
    selected_ids = tuple(item.task_id for item in manifest.selected_tasks)
    if selected_ids != tuple(task.task_id for task in bundle.tasks):
        raise ValueError("canary selected task order is invalid")
    for item, task, record in zip(
        manifest.selected_tasks,
        bundle.tasks,
        bundle.records,
        strict=True,
    ):
        if type(task) is not MemUpdateTaskV3 or type(record) is not bytes:
            raise ValueError("canary bundle contains untrusted nested types")
        if (
            task.metadata.split is not Split.DEV
            or release.task_by_id.get(task.task_id) != task
        ):
            raise ValueError("canary task is not an authenticated dev record")
        if record != release.raw_record_by_task_id[task.task_id]:
            raise ValueError("canary record does not preserve exact source bytes")
        if item != _selected_item(release, task):
            raise ValueError("canary selected task binding is invalid")
    required = set(_required_coverage(release.dev_tasks))
    if required - set().union(*(coverage_tokens(task) for task in bundle.tasks)):
        raise ValueError("canary coverage requirements are not met")
    return bundle


def _validate_canary_set_authenticated(
    bundle: CanarySetBundleV1,
    release: AuthenticatedCoreReleaseV1,
) -> CanarySetBundleV1:
    if (
        type(bundle) is not CanarySetBundleV1
        or type(bundle.canaries) is not tuple
        or len(bundle.canaries) != 2
    ):
        raise ValueError("canary set must contain exactly two exact bundles")
    if any(type(canary) is not CanaryBundleV1 for canary in bundle.canaries):
        raise ValueError("canary set contains an untrusted bundle")
    if type(bundle.set_manifest) is not CanarySetManifestV1:
        raise ValueError(
            "canary set manifest must be an exact "
            "CanarySetManifestV1 instance"
        )
    manifest_refs = bundle.set_manifest.canary_manifest_refs
    if type(manifest_refs) is not tuple or any(
        type(ref) is not ArtifactRef for ref in manifest_refs
    ):
        raise ValueError("canary set manifest contains untrusted artifact references")
    try:
        set_manifest = CanarySetManifestV1.model_validate(
            bundle.set_manifest.model_dump(mode="python")
        )
    except Exception as exc:
        raise ValueError("canary set manifest fails trust-boundary validation") from exc
    if canonical_json_bytes(set_manifest) != bundle.set_manifest_bytes:
        raise ValueError("canary set manifest bytes are invalid")
    expected_refs = tuple(
        ArtifactRef(
            path=f"{canary.manifest.canary_id}/canary_manifest.json",
            sha256=_sha256(canary.manifest_bytes),
            media_type="application/json",
            record_count=1,
        )
        for canary in bundle.canaries
    )
    if set_manifest.canary_manifest_refs != expected_refs:
        raise ValueError("canary set manifest references are invalid")
    source_hashes = {
        canary.manifest.source_release_manifest_hash
        for canary in bundle.canaries
    }
    if source_hashes != {set_manifest.source_release_manifest_hash}:
        raise ValueError("canary set source release bindings differ")
    _validate_pair_independence(bundle.canaries)
    for canary in bundle.canaries:
        _validate_canary_bundle_authenticated(canary, release)
    return bundle


def _require_canonical_canary_set(
    bundle: CanarySetBundleV1,
    expected: CanarySetBundleV1,
) -> None:
    if bundle.set_manifest_bytes != expected.set_manifest_bytes:
        raise ValueError("bundle is not the canonical canary selection")
    for observed, canonical in zip(
        bundle.canaries,
        expected.canaries,
        strict=True,
    ):
        if (
            observed.manifest_bytes != canonical.manifest_bytes
            or observed.records != canonical.records
            or tuple(task.task_id for task in observed.tasks)
            != tuple(task.task_id for task in canonical.tasks)
        ):
            raise ValueError("bundle is not the canonical canary selection")


def validate_canary_bundle(
    bundle: CanaryBundleV1,
    release: AuthenticatedCoreReleaseV1,
) -> CanaryBundleV1:
    fresh_release = _fresh_authenticated_release(release)
    _validate_canary_bundle_authenticated(bundle, fresh_release)
    expected = _build_canary_set_authenticated(fresh_release)
    canonical = next(
        (
            canary
            for canary in expected.canaries
            if canary.manifest.canary_id == bundle.manifest.canary_id
        ),
        None,
    )
    if canonical is None or (
        bundle.manifest_bytes != canonical.manifest_bytes
        or bundle.records != canonical.records
        or tuple(task.task_id for task in bundle.tasks)
        != tuple(task.task_id for task in canonical.tasks)
    ):
        raise ValueError("bundle is not the canonical canary selection")
    return bundle


def validate_canary_set(
    bundle: CanarySetBundleV1,
    release: AuthenticatedCoreReleaseV1,
) -> CanarySetBundleV1:
    fresh_release = _fresh_authenticated_release(release)
    _validate_canary_set_authenticated(bundle, fresh_release)
    expected = _build_canary_set_authenticated(fresh_release)
    _require_canonical_canary_set(bundle, expected)
    return bundle


def _assert_no_reparse_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and _is_reparse_point(current):
            raise ValueError("derived canary output path contains a reparse point")


def _ensure_output_root_is_safe(output_root: Path, release_root: Path) -> None:
    _assert_no_reparse_components(output_root)
    resolved = output_root.resolve(strict=False)
    protected_roots = [release_root.resolve(strict=True)]
    if _IMMUTABLE_CORE_ROOT.exists():
        protected_roots.append(_IMMUTABLE_CORE_ROOT.resolve(strict=True))
    for source in protected_roots:
        try:
            resolved.relative_to(source)
        except ValueError:
            continue
        raise ValueError(
            "derived canary output root must be outside "
            "data/vnext/core/v3"
        )


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _require_stable_parent(
    parent: Path,
    expected_identity: tuple[int, int],
    output: Path,
    release_root: Path,
) -> None:
    _assert_no_reparse_components(parent)
    if _directory_identity(parent) != expected_identity:
        raise ValueError("derived canary output parent changed during publication")
    _ensure_output_root_is_safe(output, release_root)


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return

    import ctypes

    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is unavailable",
                destination,
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is unavailable",
                destination,
            )
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(source),
            os.fsencode(destination),
            4,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace rename is unsupported on this platform",
            destination,
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination,
    )


def publish_canary_set(
    bundle: CanarySetBundleV1,
    output_root: str | Path,
    *,
    release: AuthenticatedCoreReleaseV1,
) -> Path:
    if type(bundle) is not CanarySetBundleV1:
        raise ValueError("publication requires an exact CanarySetBundleV1")
    if type(release) is not AuthenticatedCoreReleaseV1:
        raise ValueError("publication requires an authenticated Core release")
    output = Path(output_root).absolute()
    source_root = Path(release.root).resolve(strict=True)
    _ensure_output_root_is_safe(output, source_root)
    if os.path.lexists(output):
        raise FileExistsError(f"canary output root already exists: {output}")
    parent = output.parent
    if not parent.is_dir():
        raise ValueError("canary output parent directory does not exist")
    _assert_no_reparse_components(parent)
    parent_identity = _directory_identity(parent)

    fresh_release = _fresh_authenticated_release(release)
    source_root = fresh_release.root
    _require_stable_parent(
        parent,
        parent_identity,
        output,
        source_root,
    )
    _validate_canary_set_authenticated(bundle, fresh_release)
    canonical_bundle = _build_canary_set_authenticated(fresh_release)
    _require_canonical_canary_set(bundle, canonical_bundle)
    bundle = canonical_bundle

    stage = parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if os.path.lexists(stage):
        raise FileExistsError("canary output staging path already exists")
    try:
        _require_stable_parent(
            parent,
            parent_identity,
            output,
            source_root,
        )
        stage.mkdir()
        if (
            _is_reparse_point(stage)
            or stage.resolve(strict=True).parent != parent.resolve(strict=True)
        ):
            raise ValueError("canary staging directory is not anchored to its parent")
        for canary in bundle.canaries:
            _require_stable_parent(
                parent,
                parent_identity,
                output,
                source_root,
            )
            directory = stage / canary.manifest.canary_id
            directory.mkdir()
            _write_fsynced(
                directory / "tasks.jsonl",
                b"".join(canary.records),
            )
            _write_fsynced(
                directory / "canary_manifest.json",
                canary.manifest_bytes,
            )
            _fsync_directory(directory)
        _write_fsynced(
            stage / "canary_set_manifest.json",
            bundle.set_manifest_bytes,
        )
        _fsync_directory(stage)
        _require_stable_parent(
            parent,
            parent_identity,
            output,
            source_root,
        )
        if os.path.lexists(output):
            raise FileExistsError(
                f"canary output root already exists: {output}"
            )
        _rename_no_replace(stage, output)
        _fsync_directory(parent)
    except BaseException:
        if os.path.lexists(stage) and not _is_reparse_point(stage):
            for path in sorted(stage.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            stage.rmdir()
        raise
    return output


__all__ = [
    "AuthenticatedCoreReleaseV1",
    "CANARY_FAMILY_QUOTAS",
    "CANARY_SCHEMA_VERSION",
    "CANARY_SELECTION_POLICY_SHA256",
    "CANARY_SELECTION_VERSION",
    "CanaryBundleV1",
    "CanaryManifestV1",
    "CanarySetBundleV1",
    "CanarySetManifestV1",
    "CoreTaskReleaseManifestV1",
    "FAMILY_LETTERS",
    "SelectedCanaryTaskV1",
    "authenticate_core_release",
    "build_canary_set",
    "coverage_tokens",
    "publish_canary_set",
    "validate_canary_bundle",
    "validate_canary_set",
]
