from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Iterator
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import threading
import weakref
from typing import Any, Literal

from pydantic import BaseModel

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.runtime import TaskRunRecordV3
from mub.vnext.contracts.v3.score import ScoreRecordV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.preparation.task12 import (
    Task12DryRunPlanV1,
    Task12InterventionCellV1,
    Task12PreparationManifestV1,
    _read_artifact,
)
from mub.vnext.runtime.run_v3 import ExternalRunConfigV1
from mub.vnext.runtime.task12_bundle_v3 import (
    _read_core_tasks,
    _validate_existing_bundle_root,
    _validate_task12_run_bundle_v3,
    load_task12_frozen_trajectories_v3,
    validate_task12_manifest_plan_v3,
)
from mub.vnext.runtime.task12_execution_v3 import (
    Task12ExecutionAuthorizationV1,
    Task12RuntimeCodeBindingV1,
    load_finalized_task12_run_v3,
    parse_task12_control_json_bytes_v3,
    read_task12_regular_file_v3,
    verify_task12_score_artifact_v3,
)
from mub.vnext.runtime.task12_matrix_v3 import (
    Task12MatrixBundleManifestV1,
    Task12MatrixRunSummaryV1,
)
from mub.vnext.statistics.contracts_v3 import (
    SHA256,
    TASK13_SEMANTIC_CORE_COUNT,
    TASK13_TASK_COUNT,
    TASK13_TASKS_PER_CORE,
    Task13AnswerModelSlot,
    Task13RetrievalK,
    Task13RunSourceV1,
)


_SHA256_HEX = frozenset("0123456789abcdef")
_TASK13_LOADER_TOKEN = object()
_TASK13_MATRIX_REGISTRY_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class Task13LoaderRootCapabilityV1:
    name: str
    path: Path
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Task13LoaderFileCapabilityV1:
    name: str
    path: Path
    identity: tuple[int, int]
    sha256: str


@dataclass(frozen=True, slots=True)
class Task13LoaderCapabilityV1:
    matrix_digest: str
    roots: Mapping[str, Task13LoaderRootCapabilityV1]
    controls: Mapping[str, Task13LoaderFileCapabilityV1]

    def __post_init__(self) -> None:
        expected_roots = {"repository", "core", "evidence", "matrix"}
        expected_controls = {
            "preparation_manifest",
            "plan",
            "matrix_manifest",
            "matrix_summary",
            "integrity_audit",
        }
        if set(self.roots) != expected_roots:
            raise ValueError("Task 13 loader capability must register all four source roots")
        if set(self.controls) != expected_controls:
            raise ValueError("Task 13 loader capability must register all five control files")
        for digest, label in (
            (self.matrix_digest, "matrix digest"),
            *( (entry.sha256, f"{entry.name} hash") for entry in self.controls.values() ),
        ):
            if type(digest) is not str or len(digest) != 64 or any(
                char not in _SHA256_HEX for char in digest
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        for entry in (*self.roots.values(), *self.controls.values()):
            if (
                type(entry.identity) is not tuple
                or len(entry.identity) != 2
                or any(type(value) is not int or value < 0 for value in entry.identity)
            ):
                raise ValueError(f"{entry.name} identity is invalid")
        object.__setattr__(self, "roots", MappingProxyType(dict(self.roots)))
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))

    @property
    def digest(self) -> str:
        return self.matrix_digest

    @property
    def repository_root(self) -> Path:
        return self.roots["repository"].path

    @property
    def core_root(self) -> Path:
        return self.roots["core"].path

    @property
    def evidence_root(self) -> Path:
        return self.roots["evidence"].path

    @property
    def matrix_root(self) -> Path:
        return self.roots["matrix"].path

    @property
    def control_paths(self) -> Mapping[str, Path]:
        return MappingProxyType({name: entry.path for name, entry in self.controls.items()})


@dataclass(frozen=True)
class _Task13MatrixRegistryEntry:
    reference: weakref.ReferenceType[Task13AuthenticatedMatrixV1]
    capability: Task13LoaderCapabilityV1
    nonce: object
    lock: threading.RLock


_TASK13_MATRIX_REGISTRY: dict[int, _Task13MatrixRegistryEntry] = {}


class Task13IntegrityCountsV1(ImmutableContractModel):
    run_count: Literal[18]
    total_task_rows: Literal[1440]
    total_score_rows: Literal[1440]
    failed: Literal[0]
    partial: Literal[0]
    semantic_multiset_mismatches: Literal[0]


class Task13IntegrityAuditV1(ImmutableContractModel):
    status: Literal["verified"]
    runtime_code_binding: Task12RuntimeCodeBindingV1
    matrix_bundle_manifest_sha256: SHA256
    matrix_summary_sha256: SHA256
    counts: Task13IntegrityCountsV1


class _Task12MatrixIntegrityAuditLegacyV1(BaseModel):
    schema_version: Literal["memupdatebench.core-task12-matrix-integrity-audit.v1"]
    status: Literal["verified"]
    runtime_revision: str
    runtime_tree_sha256: str
    preparation_manifest_sha256: str
    plan_fingerprint_sha256: str
    matrix_bundle_manifest_sha256: str
    matrix_run_summary_sha256: str
    run_count: Literal[18]
    total_task_rows: Literal[1440]
    total_score_rows: Literal[1440]
    failed_or_partial_rows: Literal[0]
    retrieval_multiset_mismatches: Literal[0]
    retrieval_incomplete_groups: int
    retrieval_multiset_group_count: int
    snapshot_path_hits: int
    runs: tuple[dict[str, Any], ...]


class _Task13ObservationEvidencePayloadV1(ImmutableContractModel):
    cell_id: str
    slot: Task13AnswerModelSlot
    k: Task13RetrievalK
    context_order: Literal["chronological", "reverse_chronological"]
    context_annotation: Literal["none", "latest_outdated_label"]
    semantic_core_id: str
    task: MemUpdateTaskV3
    run: TaskRunRecordV3
    score: ScoreRecordV3
    source: Task13RunSourceV1


class _Task13ObservationRootPayloadV1(ImmutableContractModel):
    evidence_sha256: tuple[str, ...]


@dataclass(frozen=True)
class Task13AuthenticatedObservationV1:
    cell_id: str
    slot: Task13AnswerModelSlot
    k: Task13RetrievalK
    context_order: Literal["chronological", "reverse_chronological"]
    context_annotation: Literal["none", "latest_outdated_label"]
    semantic_core_id: str
    task: MemUpdateTaskV3
    run: TaskRunRecordV3
    score: ScoreRecordV3
    source: Task13RunSourceV1
    evidence_sha256: str = ""


def _task13_observation_evidence_sha256(
    observation: Task13AuthenticatedObservationV1,
) -> str:
    return sha256_model(
        _Task13ObservationEvidencePayloadV1(
            cell_id=observation.cell_id,
            slot=observation.slot,
            k=observation.k,
            context_order=observation.context_order,
            context_annotation=observation.context_annotation,
            semantic_core_id=observation.semantic_core_id,
            task=observation.task,
            run=observation.run,
            score=observation.score,
            source=observation.source,
        )
    )


def _task13_observation_membership_root_sha256(
    observations: tuple[Task13AuthenticatedObservationV1, ...],
) -> str:
    return sha256_model(
        _Task13ObservationRootPayloadV1(
            evidence_sha256=tuple(
                observation.evidence_sha256 for observation in observations
            )
        )
    )


@dataclass(frozen=True)
class Task13AuthenticatedRunV1:
    cell: Task12InterventionCellV1
    run_configuration: ExternalRunConfigV1
    authorization: Task12ExecutionAuthorizationV1
    source: Task13RunSourceV1
    observations: tuple[Task13AuthenticatedObservationV1, ...]
    observation_membership_root_sha256: str = ""

    @property
    def run_config(self) -> ExternalRunConfigV1:
        return self.run_configuration

    @property
    def auth(self) -> Task12ExecutionAuthorizationV1:
        return self.authorization


@dataclass(frozen=True)
class Task13AuthenticatedMatrixV1:
    manifest: Task12PreparationManifestV1
    plan: Task12DryRunPlanV1
    matrix_manifest: Task12MatrixBundleManifestV1
    summary: Task12MatrixRunSummaryV1
    integrity_audit: Task13IntegrityAuditV1
    runtime: Task12RuntimeCodeBindingV1
    runs: tuple[Task13AuthenticatedRunV1, ...]
    canonical_core_ids: tuple[str, ...]
    input_hashes: Mapping[str, str]
    _loader_token: InitVar[object | None] = None
    observation_membership_roots: Mapping[str, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _loader_seal_valid: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _loader_token: object | None) -> None:
        loader_seal_valid = _loader_token is _TASK13_LOADER_TOKEN
        roots = (
            {
                run.source.run_id: run.observation_membership_root_sha256
                for run in self.runs
            }
            if loader_seal_valid
            else {}
        )
        object.__setattr__(self, "observation_membership_roots", MappingProxyType(roots))
        object.__setattr__(self, "_loader_seal_valid", loader_seal_valid)

    @property
    def matrix_run_summary(self) -> Task12MatrixRunSummaryV1:
        return self.summary



class _Task13DigestPayloadV1(ImmutableContractModel):
    payload: Any


def _task13_matrix_digest(matrix: Task13AuthenticatedMatrixV1) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return normalize(value.model_dump(mode="json", exclude_none=False, exclude_computed_fields=True))
        if is_dataclass(value) and not isinstance(value, type):
            return {item.name: normalize(getattr(value, item.name)) for item in fields(value)}
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, tuple | list):
            return [normalize(item) for item in value]
        if type(value) in {str, int, float, bool} or value is None:
            return value
        raise TypeError(f"Task 13 matrix digest cannot serialize {type(value).__name__}")

    payload = _Task13DigestPayloadV1(payload=normalize(matrix))
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _register_loader_task13_matrix_v1(
    matrix: Task13AuthenticatedMatrixV1,
    capability: Task13LoaderCapabilityV1,
) -> None:
    identifier = id(matrix)

    def cleanup(reference: weakref.ReferenceType[Task13AuthenticatedMatrixV1]) -> None:
        with _TASK13_MATRIX_REGISTRY_LOCK:
            entry = _TASK13_MATRIX_REGISTRY.get(identifier)
            if entry is not None and entry.reference is reference:
                _TASK13_MATRIX_REGISTRY.pop(identifier, None)

    reference = weakref.ref(matrix, cleanup)
    entry = _Task13MatrixRegistryEntry(
        reference,
        capability,
        object(),
        threading.RLock(),
    )
    with _TASK13_MATRIX_REGISTRY_LOCK:
        _TASK13_MATRIX_REGISTRY[identifier] = entry


def _loader_registered_task13_matrix_entry(
    matrix: Task13AuthenticatedMatrixV1,
) -> _Task13MatrixRegistryEntry:
    if not isinstance(matrix, Task13AuthenticatedMatrixV1):
        raise ValueError("Task 13 matrix is not loader-registered")
    with _TASK13_MATRIX_REGISTRY_LOCK:
        entry = _TASK13_MATRIX_REGISTRY.get(id(matrix))
    if entry is None or entry.reference() is not matrix:
        raise ValueError("Task 13 matrix is not loader-registered")
    if _task13_matrix_digest(matrix) != entry.capability.matrix_digest:
        raise ValueError("Task 13 loader-registered matrix content changed")
    return entry


def require_loader_registered_task13_matrix_v1(
    matrix: Task13AuthenticatedMatrixV1,
) -> Task13LoaderCapabilityV1:
    return _loader_registered_task13_matrix_entry(matrix).capability


@contextmanager
def loader_registered_task13_matrix_lease_v1(
    matrix: Task13AuthenticatedMatrixV1,
) -> Iterator[Task13LoaderCapabilityV1]:
    entry = _loader_registered_task13_matrix_entry(matrix)
    with entry.lock:
        capability = require_loader_registered_task13_matrix_v1(matrix)
        yield capability
        if require_loader_registered_task13_matrix_v1(matrix).matrix_digest != capability.matrix_digest:
            raise ValueError("Task 13 loader-registered matrix content changed")


def validate_task13_authenticated_matrix_v1(
    matrix: Task13AuthenticatedMatrixV1,
) -> Task13AuthenticatedMatrixV1:
    require_loader_registered_task13_matrix_v1(matrix)
    from mub.vnext.statistics.cases_v3 import _validate_authenticated_matrix

    return _validate_authenticated_matrix(matrix)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in _SHA256_HEX for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _loader_file_capability(name: str, path: str | Path) -> Task13LoaderFileCapabilityV1:
    checked = Path(path).resolve(strict=True)
    result = checked.stat(follow_symlinks=False)
    return Task13LoaderFileCapabilityV1(
        name=name,
        path=checked,
        identity=(result.st_dev, result.st_ino),
        sha256=hashlib.sha256(checked.read_bytes()).hexdigest(),
    )


def _identity(path: Path) -> tuple[int, int]:
    result = path.stat(follow_symlinks=False)
    return result.st_dev, result.st_ino


def _read_expected(path: str | Path, expected_sha256: str, *, label: str) -> bytes:
    expected = _require_sha256(expected_sha256, f"expected {label} hash")
    raw = read_task12_regular_file_v3(path)
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return raw


def _load_hashed_control(
    path: str | Path,
    model_type,
    expected_sha256: str,
    *,
    label: str,
    allow_trailing_lf: bool = False,
):
    raw = _read_expected(path, expected_sha256, label=label)
    model = parse_task12_control_json_bytes_v3(
        raw,
        model_type,
        source=path,
        allow_trailing_lf=allow_trailing_lf,
    )
    return model, raw


def _load_integrity_audit(
    path: str | Path,
    expected_sha256: str,
    *,
    matrix_manifest_sha256: str,
    summary_sha256: str,
) -> Task13IntegrityAuditV1:
    raw = _read_expected(path, expected_sha256, label="Task 12 integrity audit")
    try:
        audit = Task13IntegrityAuditV1.model_validate_json(raw)
        if canonical_json_bytes(audit) != raw:
            raise ValueError("integrity audit is not canonical JSON")
    except ValueError:
        try:
            legacy = _Task12MatrixIntegrityAuditLegacyV1.model_validate_json(raw)
            if canonical_json_bytes(legacy) != raw or len(legacy.runs) != 18:
                raise ValueError("legacy integrity audit is not canonical or complete")
            if any(
                run.get("task_rows") != 80 or run.get("score_rows") != 80
                or type(run.get("run_manifest_sha256")) is not str
                or type(run.get("score_artifact_sha256")) is not str
                for run in legacy.runs
            ):
                raise ValueError("legacy integrity audit run rows are incomplete")
            runtime = Task12RuntimeCodeBindingV1(
                code_revision=legacy.runtime_revision,
                code_tree_sha256=legacy.runtime_tree_sha256,
            )
            audit = Task13IntegrityAuditV1(
                status="verified",
                runtime_code_binding=runtime,
                matrix_bundle_manifest_sha256=legacy.matrix_bundle_manifest_sha256,
                matrix_summary_sha256=legacy.matrix_run_summary_sha256,
                counts=Task13IntegrityCountsV1(
                    run_count=18,
                    total_task_rows=1440,
                    total_score_rows=1440,
                    failed=0,
                    partial=0,
                    semantic_multiset_mismatches=0,
                ),
            )
        except Exception as exc:
            raise ValueError("integrity audit is invalid") from exc
    if audit.matrix_bundle_manifest_sha256 != matrix_manifest_sha256:
        raise ValueError("integrity audit matrix hash mismatch")
    if audit.matrix_summary_sha256 != summary_sha256:
        raise ValueError("integrity audit summary hash mismatch")
    return audit


def _source_task_inputs(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    core_root: Path,
) -> tuple[TaskManifestV3, dict[str, Any]]:
    manifest_raw = _read_artifact(root=core_root, location=manifest.task_manifest)
    if hashlib.sha256(manifest_raw).hexdigest() != plan.core_task_manifest_sha256:
        raise ValueError("Core task manifest differs from the admitted plan")
    source_manifest = TaskManifestV3.model_validate_json(manifest_raw)
    if canonical_json_bytes(source_manifest) != manifest_raw:
        raise ValueError("Core task manifest is not canonical JSON")
    tasks_raw = _read_artifact(root=core_root, location=manifest.tasks)
    if hashlib.sha256(tasks_raw).hexdigest() != plan.core_tasks_sha256:
        raise ValueError("Core tasks differ from the admitted plan")
    tasks_by_id = _read_core_tasks(tasks_raw)
    if dict(source_manifest.task_record_hashes) != {
        task_id: sha256_model(task) for task_id, task in tasks_by_id.items()
    }:
        raise ValueError("Core task manifest record hashes do not authenticate tasks")
    return source_manifest, tasks_by_id


def _matrix_path(root: Path, leaf: str, name: str) -> Path:
    path = root / leaf
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Task 12 matrix bundle is not a real directory: {leaf}")
    return path / name


def _exact_matrix_pairs(
    plan: Task12DryRunPlanV1,
    matrix_manifest: Task12MatrixBundleManifestV1,
    summary: Task12MatrixRunSummaryV1,
) -> tuple[tuple[str, str], ...]:
    expected = tuple((run.cell_id, run.answer_model_slot) for run in plan.admitted_answer_runs)
    observed_manifest = tuple((run.cell_id, run.answer_model_slot) for run in matrix_manifest.run_bundles)
    observed_summary = tuple((run.cell_id, run.answer_model_slot) for run in summary.completed_runs)
    if observed_manifest != expected:
        raise ValueError("Task 12 matrix manifest order differs from admitted answer runs")
    if observed_summary != expected:
        raise ValueError("Task 12 matrix summary order differs from admitted answer runs")
    return expected


def _validate_canonical_core_ids(core_ids: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(core_ids)
    if len(core_ids) != TASK13_TASK_COUNT:
        raise ValueError(
            f"each Task 12 run must contain exactly {TASK13_TASK_COUNT} task observations"
        )
    if len(counts) != TASK13_SEMANTIC_CORE_COUNT:
        raise ValueError(
            f"each Task 12 run must contain exactly {TASK13_SEMANTIC_CORE_COUNT} unique semantic cores"
        )
    if any(count != TASK13_TASKS_PER_CORE for count in counts.values()):
        raise ValueError(
            "each semantic core must have exactly four task IDs"
        )
    if core_ids != tuple(sorted(core_ids, key=lambda value: value.encode("utf-8"))):
        raise ValueError("semantic core IDs must be in canonical UTF-8 order")
    return tuple(sorted(counts, key=lambda value: value.encode("utf-8")))


def _validate_canonical_task_identity(
    identity: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if len(identity) != TASK13_TASK_COUNT:
        raise ValueError(
            f"each Task 12 run must contain exactly {TASK13_TASK_COUNT} task identities"
        )
    task_ids = tuple(task_id for task_id, _ in identity)
    if len(set(task_ids)) != TASK13_TASK_COUNT:
        raise ValueError("each Task 12 run must contain unique task IDs")
    expected = tuple(
        sorted(
            identity,
            key=lambda pair: (pair[1].encode("utf-8"), pair[0].encode("utf-8")),
        )
    )
    if identity != expected:
        raise ValueError("Task 12 task identities must be in canonical UTF-8 order")
    _validate_canonical_core_ids(tuple(core_id for _, core_id in identity))
    return identity


def _validate_task_identity_mapping(
    observed: tuple[tuple[str, str], ...],
    expected: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if len(observed) != TASK13_TASK_COUNT:
        raise ValueError(
            f"each Task 12 run must contain exactly {TASK13_TASK_COUNT} task identities"
        )
    task_ids = tuple(task_id for task_id, _ in observed)
    if len(set(task_ids)) != TASK13_TASK_COUNT:
        raise ValueError("each Task 12 run must contain unique task IDs")
    expected_sorted = tuple(
        sorted(expected, key=lambda pair: (pair[1].encode("utf-8"), pair[0].encode("utf-8")))
    )
    observed_sorted = tuple(
        sorted(observed, key=lambda pair: (pair[1].encode("utf-8"), pair[0].encode("utf-8")))
    )
    observed_counts = Counter(core_id for _, core_id in observed_sorted)
    if len(observed_counts) != TASK13_SEMANTIC_CORE_COUNT or any(
        count != TASK13_TASKS_PER_CORE for count in observed_counts.values()
    ):
        raise ValueError("each semantic core must have exactly four task IDs")
    if observed_sorted != expected_sorted:
        raise ValueError("Task 12 task IDs or semantic-core assignments differ")
    _validate_canonical_core_ids(tuple(core_id for _, core_id in observed_sorted))
    return observed_sorted


def _canonical_task_identity(
    tasks: tuple[MemUpdateTaskV3, ...],
) -> tuple[tuple[str, str], ...]:
    ordered = sorted(
        tasks,
        key=lambda task: (
            task.metadata.split_key.semantic_core_id.encode("utf-8"),
            task.task_id.encode("utf-8"),
        ),
    )
    return tuple(
        (task.task_id, task.metadata.split_key.semantic_core_id)
        for task in ordered
    )


def load_task13_authenticated_matrix_v1(
    *,
    preparation_manifest_path: str | Path,
    plan_path: str | Path,
    core_root: str | Path,
    evidence_root: str | Path,
    matrix_root: str | Path,
    matrix_manifest_path: str | Path,
    matrix_summary_path: str | Path,
    integrity_audit_path: str | Path,
    repository_root: str | Path,
    expected_preparation_manifest_sha256: str,
    expected_plan_sha256: str,
    expected_matrix_manifest_sha256: str,
    expected_matrix_summary_sha256: str,
    expected_integrity_audit_sha256: str,
) -> Task13AuthenticatedMatrixV1:
    manifest, _ = _load_hashed_control(
        preparation_manifest_path,
        Task12PreparationManifestV1,
        expected_preparation_manifest_sha256,
        label="Task 12 preparation manifest",
    )
    plan, _ = _load_hashed_control(
        plan_path,
        Task12DryRunPlanV1,
        expected_plan_sha256,
        label="Task 12 dry-run plan",
        allow_trailing_lf=True,
    )
    validate_task12_manifest_plan_v3(manifest, plan)

    core = Path(core_root).resolve(strict=True)
    evidence = Path(evidence_root).resolve(strict=True)
    repository = Path(repository_root).resolve(strict=True)
    matrix_bundle_root = _validate_existing_bundle_root(
        bundle_root=matrix_root,
        core_root=core,
        evidence_root=evidence,
        repository_root=repository,
    )
    source_manifest, tasks_by_id = _source_task_inputs(
        manifest=manifest,
        plan=plan,
        core_root=core,
    )
    matrix_manifest, matrix_manifest_raw = _load_hashed_control(
        matrix_manifest_path,
        Task12MatrixBundleManifestV1,
        expected_matrix_manifest_sha256,
        label="Task 12 matrix bundle manifest",
    )
    summary, summary_raw = _load_hashed_control(
        matrix_summary_path,
        Task12MatrixRunSummaryV1,
        expected_matrix_summary_sha256,
        label="Task 12 matrix summary",
    )
    if matrix_manifest.preparation_manifest_sha256 != sha256_model(manifest):
        raise ValueError("Task 12 matrix manifest is not bound to preparation manifest")
    if matrix_manifest.plan_fingerprint_sha256 != plan.plan_fingerprint_sha256:
        raise ValueError("Task 12 matrix manifest is not bound to dry-run plan")
    if summary.preparation_manifest_sha256 != sha256_model(manifest):
        raise ValueError("Task 12 matrix summary is not bound to preparation manifest")
    if summary.plan_fingerprint_sha256 != plan.plan_fingerprint_sha256:
        raise ValueError("Task 12 matrix summary is not bound to dry-run plan")
    matrix_manifest_sha256 = hashlib.sha256(matrix_manifest_raw).hexdigest()
    summary_sha256 = hashlib.sha256(summary_raw).hexdigest()
    if summary.matrix_bundle_manifest_sha256 != matrix_manifest_sha256:
        raise ValueError("Task 12 matrix summary manifest hash mismatch")
    _exact_matrix_pairs(plan, matrix_manifest, summary)

    audit = _load_integrity_audit(
        integrity_audit_path,
        expected_integrity_audit_sha256,
        matrix_manifest_sha256=matrix_manifest_sha256,
        summary_sha256=summary_sha256,
    )
    runtime = audit.runtime_code_binding
    frozen_tasks = tuple(
        tasks_by_id[task_id] for task_id in manifest.semantic_matrix.task_scope.task_ids
    )
    if tuple(task.task_id for task in frozen_tasks) != manifest.semantic_matrix.task_scope.task_ids:
        raise ValueError("Task 12 semantic matrix scope references unknown Core tasks")
    canonical_task_identity = _validate_canonical_task_identity(
        _canonical_task_identity(frozen_tasks)
    )
    _validate_task_identity_mapping(
        tuple(
            (task.task_id, task.metadata.split_key.semantic_core_id)
            for task in frozen_tasks
        ),
        canonical_task_identity,
    )
    canonical_core_ids = _validate_canonical_core_ids(
        tuple(core_id for _, core_id in canonical_task_identity)
    )
    canonical_task_ids = tuple(task_id for task_id, _ in canonical_task_identity)
    frozen_trajectories = load_task12_frozen_trajectories_v3(
        manifest=manifest,
        evidence_root=evidence,
        tasks=frozen_tasks,
    )
    summary_by_pair = {
        (run.cell_id, run.answer_model_slot): run for run in summary.completed_runs
    }
    cell_by_id = {cell.cell_id: cell for cell in manifest.semantic_matrix.intervention_cells}
    runs: list[Task13AuthenticatedRunV1] = []
    shared_task_identity: tuple[tuple[str, str], ...] | None = None
    for ref in matrix_manifest.run_bundles:
        pair = (ref.cell_id, ref.answer_model_slot)
        cell = cell_by_id.get(ref.cell_id)
        if cell is None:
            raise ValueError(f"matrix run references unknown cell: {ref.cell_id}")
        bundle_root = _matrix_path(matrix_bundle_root, ref.bundle_leaf, "authorization.json").parent
        authorization_raw = read_task12_regular_file_v3(bundle_root / "authorization.json")
        if hashlib.sha256(authorization_raw).hexdigest() != ref.authorization_sha256:
            raise ValueError("Task 12 authorization hash differs from matrix manifest")
        authorization = Task12ExecutionAuthorizationV1.model_validate_json(authorization_raw)
        if canonical_json_bytes(authorization) != authorization_raw:
            raise ValueError("Task 12 authorization is not canonical JSON")
        if (authorization.cell_id, authorization.answer_model_slot) != pair:
            raise ValueError("Task 12 authorization pair differs from matrix manifest")
        if authorization.runtime_code_binding != runtime:
            raise ValueError("Task 12 authorization runtime differs from integrity audit")
        validated = _validate_task12_run_bundle_v3(
            manifest=manifest,
            plan=plan,
            core_root=core,
            evidence_root=evidence,
            repository_root=repository,
            runtime_code_binding=authorization.runtime_code_binding,
            bundle_root=bundle_root,
            source_manifest=source_manifest,
            source_tasks_by_id=tasks_by_id,
            frozen_trajectories=frozen_trajectories,
        )
        if (
            validated.authorization.cell_id,
            validated.authorization.answer_model_slot,
        ) != pair:
            raise ValueError("validated Task 12 bundle pair differs from matrix manifest")
        if (
            validated.authorization.task_manifest_sha256 != ref.task_manifest_sha256
            or validated.authorization.task_view_sha256 != ref.task_view_sha256
            or validated.authorization.run_config_sha256 != ref.run_config_sha256
            or validated.authorization.output_leaf != ref.output_leaf
        ):
            raise ValueError("Task 12 authorization does not match matrix bundle reference")
        run_manifest, rows = load_finalized_task12_run_v3(
            validated.execution_output_root,
            validated.run_configuration,
        )
        run_manifest_sha256 = hashlib.sha256(canonical_json_bytes(run_manifest)).hexdigest()
        scores, receipt = verify_task12_score_artifact_v3(
            validated.execution_output_root / "scores",
            expected_task_ids=validated.run_configuration.expected_task_ids,
            run_manifest_sha256=run_manifest_sha256,
            task_manifest_sha256=validated.authorization.task_manifest_sha256,
        )
        summary_ref = summary_by_pair[pair]
        score_hash = receipt.get("score_artifact_sha256")
        if (
            summary_ref.bundle_leaf != ref.bundle_leaf
            or summary_ref.output_leaf != ref.output_leaf
            or summary_ref.task_count != 80
            or summary_ref.score_count != 80
            or run_manifest_sha256 != summary_ref.run_manifest_sha256
            or score_hash != summary_ref.score_artifact_sha256
        ):
            raise ValueError("Task 12 run or score hash differs from matrix summary")
        if len(rows) != 80 or len(scores) != 80:
            raise ValueError("Task 12 matrix runs must contain exactly 80 rows and scores")
        if any(row.completion_status.value in {"failed", "partial"} for row in rows):
            raise ValueError("Task 12 matrix run contains failed or partial rows")
        if any(score.completion_status.value in {"failed", "partial"} for score in scores):
            raise ValueError("Task 12 matrix score contains failed or partial rows")
        source = Task13RunSourceV1(
            run_id=validated.run_configuration.run_id,
            answer_model_slot=validated.run_configuration.answer_model_slot,
            k=cell.retrieval.configuration.retrieval_k,
            run_manifest_sha256=run_manifest_sha256,
            score_artifact_sha256=score_hash,
        )
        tasks = validated.tasks
        observed_task_identity = _validate_task_identity_mapping(
            tuple(
                (task.task_id, task.metadata.split_key.semantic_core_id)
                for task in tasks
            ),
            canonical_task_identity,
        )
        if tuple(task_id for task_id, _ in observed_task_identity) != canonical_task_ids:
            raise ValueError("Task 12 run task identity differs from canonical task identity")
        if tuple(task.task_id for task in tasks) != validated.run_configuration.expected_task_ids:
            raise ValueError("Task 12 task order differs from run configuration")
        if tuple(row.task_id for row in rows) != tuple(task.task_id for task in tasks):
            raise ValueError("Task 12 run row order differs from task order")
        if tuple(score.task_id for score in scores) != tuple(task.task_id for task in tasks):
            raise ValueError("Task 12 score row order differs from task order")
        if any(
            row.run_id != validated.run_configuration.run_id
            or score.run_id != validated.run_configuration.run_id
            for row, score in zip(rows, scores)
        ):
            raise ValueError("Task 12 run or score row has a foreign run ID")
        score_by_id = dict(zip((score.task_id for score in scores), scores))
        run_by_id = dict(zip((row.task_id for row in rows), rows))
        ordered_tasks = tuple(
            sorted(
                tasks,
                key=lambda task: (
                    task.metadata.split_key.semantic_core_id.encode("utf-8"),
                    task.task_id.encode("utf-8"),
                ),
            )
        )
        observations = tuple(
            Task13AuthenticatedObservationV1(
                cell_id=cell.cell_id,
                slot=validated.run_configuration.answer_model_slot,
                k=cell.retrieval.configuration.retrieval_k,
                context_order=cell.context_intervention.context_order,
                context_annotation=cell.context_intervention.context_annotation,
                semantic_core_id=task.metadata.split_key.semantic_core_id,
                task=task,
                run=run_by_id[task.task_id],
                score=score_by_id[task.task_id],
                source=source,
            )
            for task in ordered_tasks
        )
        observations = tuple(
            replace(
                observation,
                evidence_sha256=_task13_observation_evidence_sha256(observation),
            )
            for observation in observations
        )
        observation_identity = tuple(
            (obs.task.task_id, obs.semantic_core_id) for obs in observations
        )
        if observation_identity != canonical_task_identity:
            raise ValueError("Task 12 observations are not in canonical task order")
        if tuple(obs.semantic_core_id for obs in observations) != tuple(
            core_id
            for _, core_id in canonical_task_identity
        ):
            raise ValueError("Task 12 observations have a forged task-to-core assignment")
        _validate_canonical_core_ids(
            tuple(obs.semantic_core_id for obs in observations)
        )
        if shared_task_identity is None:
            shared_task_identity = observation_identity
        elif observation_identity != shared_task_identity:
            raise ValueError("Task 12 matrix runs do not share one canonical task identity sequence")
        runs.append(
            Task13AuthenticatedRunV1(
                cell=cell,
                run_configuration=validated.run_configuration,
                authorization=validated.authorization,
                source=source,
                observations=observations,
                observation_membership_root_sha256=_task13_observation_membership_root_sha256(
                    observations
                ),
            )
        )
    if len(runs) != 18 or shared_task_identity is None:
        raise ValueError("Task 13 input must contain exactly 18 authenticated runs")
    if summary.total_task_rows != 1440 or summary.total_score_rows != 1440:
        raise ValueError("Task 12 matrix totals must be 1440 task and score rows")
    input_hashes = MappingProxyType(
        {
            "task12_preparation_manifest": _require_sha256(
                expected_preparation_manifest_sha256,
                "preparation manifest",
            ),
            "task12_plan": _require_sha256(expected_plan_sha256, "plan"),
            "task12_matrix_manifest": matrix_manifest_sha256,
            "task12_matrix_summary": summary_sha256,
            "task12_integrity_audit": _require_sha256(expected_integrity_audit_sha256, "integrity audit"),
            "core_task_manifest": _require_sha256(plan.core_task_manifest_sha256, "Core task manifest"),
            "core_tasks": _require_sha256(plan.core_tasks_sha256, "Core tasks"),
        }
    )
    matrix = Task13AuthenticatedMatrixV1(
        manifest=manifest,
        plan=plan,
        matrix_manifest=matrix_manifest,
        summary=summary,
        integrity_audit=audit,
        runtime=runtime,
        runs=tuple(runs),
        canonical_core_ids=canonical_core_ids,
        input_hashes=input_hashes,
        _loader_token=_TASK13_LOADER_TOKEN,
    )
    roots = {
        "repository": Task13LoaderRootCapabilityV1(
            "repository", repository, _identity(repository)
        ),
        "core": Task13LoaderRootCapabilityV1("core", core, _identity(core)),
        "evidence": Task13LoaderRootCapabilityV1(
            "evidence", evidence, _identity(evidence)
        ),
        "matrix": Task13LoaderRootCapabilityV1(
            "matrix", matrix_bundle_root, _identity(matrix_bundle_root)
        ),
    }
    controls = {
        "preparation_manifest": _loader_file_capability(
            "preparation_manifest", preparation_manifest_path
        ),
        "plan": _loader_file_capability("plan", plan_path),
        "matrix_manifest": _loader_file_capability(
            "matrix_manifest", matrix_manifest_path
        ),
        "matrix_summary": _loader_file_capability("matrix_summary", matrix_summary_path),
        "integrity_audit": _loader_file_capability(
            "integrity_audit", integrity_audit_path
        ),
    }
    capability = Task13LoaderCapabilityV1(_task13_matrix_digest(matrix), roots, controls)
    _register_loader_task13_matrix_v1(matrix, capability)
    return matrix


__all__ = [
    "Task13AuthenticatedMatrixV1",
    "Task13AuthenticatedObservationV1",
    "Task13AuthenticatedRunV1",
    "Task13IntegrityAuditV1",
    "Task13IntegrityCountsV1",
    "Task13LoaderCapabilityV1",
    "Task13LoaderFileCapabilityV1",
    "Task13LoaderRootCapabilityV1",
    "load_task13_authenticated_matrix_v1",
    "loader_registered_task13_matrix_lease_v1",
    "require_loader_registered_task13_matrix_v1",
    "validate_task13_authenticated_matrix_v1",
]
