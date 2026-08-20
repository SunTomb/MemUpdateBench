from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import threading
from types import MappingProxyType
import uuid
import weakref

from pydantic import BaseModel

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.statistics.bootstrap_v3 import (
    BootstrapIndicesV1,
    FROZEN_BOOTSTRAP_INDEX_SHA256,
    build_bootstrap_indices_v1,
)
from mub.vnext.statistics.cases_v3 import (
    Task13CasesResultV1,
    build_task13_cases_v1,
    verify_task13_cases_v1,
)
from mub.vnext.statistics.contracts_v3 import (
    TASK13_ARTIFACT_PATHS,
    TASK13_METRIC_PATHS,
    Task13ArtifactBindingV1,
    Task13ArtifactIndexV1,
    Task13BootstrapConfigV1,
    Task13CaseIndexV1,
    Task13CaseRecordV1,
    Task13CellStatisticV1,
    Task13ClaimLedgerRecordV1,
    Task13PairedContrastV1,
    Task13RunSourceV1,
    Task13StatisticsReceiptV1,
)
from mub.vnext.statistics.input_v3 import (
    Task13AuthenticatedMatrixV1,
    Task13LoaderCapabilityV1,
    require_loader_registered_task13_matrix_v1,
    validate_task13_authenticated_matrix_v1,
)
from mub.vnext.statistics.ledger_v3 import (
    build_task13_case_index_v1,
    build_task13_claim_ledger_v1,
    build_task13_statistics_receipt_v1,
    canonical_jsonl_bytes_v1,
    verify_task13_claim_ledger_v1,
)
from mub.vnext.statistics.statistics_v3 import (
    Task13StatisticsResultV1,
    compute_task13_statistics_v1,
)


TASK13_FINAL_INDEX_PATH = "task13_artifact_index.json"
TASK13_PUBLICATION_PATHS = (*TASK13_ARTIFACT_PATHS, TASK13_FINAL_INDEX_PATH)
_STAGE_PREFIX = ".mub-task13-stage-"


@dataclass(frozen=True, slots=True)
class _Task13PublicationRegistryEntry:
    publication: weakref.ReferenceType[Task13PublicationV1]
    matrix: weakref.ReferenceType[Task13AuthenticatedMatrixV1]
    matrix_digest: str
    publication_digest: str


_PUBLICATION_REGISTRY_LOCK = threading.RLock()
_PUBLICATION_REGISTRY: dict[int, _Task13PublicationRegistryEntry] = {}


@dataclass(frozen=True, slots=True)
class Task13RuntimeBindingV1:
    runtime_revision: str
    runtime_tree_sha256: str

    def __post_init__(self) -> None:
        if not _sha256_text(self.runtime_revision):
            raise ValueError("Task 13 runtime revision must be a full SHA-256-style revision")
        if not _sha256_text(self.runtime_tree_sha256):
            raise ValueError("Task 13 runtime tree must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class Task13ArtifactRefsV1:
    bootstrap_indices: ArtifactRef
    cell_statistics: ArtifactRef
    paired_contrasts: ArtifactRef
    statistics_receipt: ArtifactRef
    cases: ArtifactRef
    case_index: ArtifactRef
    claim_ledger: ArtifactRef
    task13_artifact_index: ArtifactRef

    def ordered(self) -> tuple[ArtifactRef, ...]:
        return (
            self.bootstrap_indices, self.cell_statistics, self.paired_contrasts,
            self.statistics_receipt, self.cases, self.case_index,
            self.claim_ledger, self.task13_artifact_index,
        )


@dataclass(frozen=True)
class Task13PublicationV1:
    bootstrap: BootstrapIndicesV1
    statistics: Task13StatisticsResultV1
    cases_result: Task13CasesResultV1
    receipt: Task13StatisticsReceiptV1
    case_index: Task13CaseIndexV1
    claims: tuple[Task13ClaimLedgerRecordV1, ...]
    artifact_index: Task13ArtifactIndexV1
    artifact_refs: Task13ArtifactRefsV1
    artifact_bytes: Mapping[str, bytes]
    matrix_identity: int = 0
    publication_seal: object | None = None

    def __post_init__(self) -> None:
        if tuple(self.artifact_bytes) != TASK13_PUBLICATION_PATHS:
            raise ValueError("Task 13 publication bytes must use the frozen eight-file order")
        if tuple(ref.path for ref in self.artifact_refs.ordered()) != TASK13_PUBLICATION_PATHS:
            raise ValueError("Task 13 publication refs must use the frozen eight-file order")
        for ref in self.artifact_refs.ordered():
            content = self.artifact_bytes[ref.path]
            if hashlib.sha256(content).hexdigest() != ref.sha256:
                raise ValueError(f"Task 13 artifact ref disagrees with final bytes: {ref.path}")


def _task13_publication_digest(publication: Task13PublicationV1) -> str:
    digest = hashlib.sha256()
    for path in TASK13_PUBLICATION_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(publication.artifact_bytes[path])
        digest.update(b"\0")
    return digest.hexdigest()


def _register_builder_task13_publication_v1(
    publication: Task13PublicationV1,
    matrix: Task13AuthenticatedMatrixV1,
    matrix_digest: str,
) -> None:
    identifier = id(publication)

    def cleanup(reference: weakref.ReferenceType[Task13PublicationV1]) -> None:
        with _PUBLICATION_REGISTRY_LOCK:
            entry = _PUBLICATION_REGISTRY.get(identifier)
            if entry is not None and entry.publication is reference:
                _PUBLICATION_REGISTRY.pop(identifier, None)

    reference = weakref.ref(publication, cleanup)
    entry = _Task13PublicationRegistryEntry(reference, weakref.ref(matrix), matrix_digest, _task13_publication_digest(publication))
    with _PUBLICATION_REGISTRY_LOCK:
        _PUBLICATION_REGISTRY[identifier] = entry


def require_builder_registered_task13_publication_v1(
    publication: Task13PublicationV1,
    matrix: Task13AuthenticatedMatrixV1,
    matrix_digest: str,
) -> None:
    with _PUBLICATION_REGISTRY_LOCK:
        entry = _PUBLICATION_REGISTRY.get(id(publication))
    if (
        entry is None
        or entry.publication() is not publication
        or entry.matrix() is not matrix
        or entry.matrix_digest != matrix_digest
        or _task13_publication_digest(publication) != entry.publication_digest
    ):
        raise ValueError("Task 13 publication is not builder-registered")


@dataclass(frozen=True, slots=True)
class Task13PublicationResultV1:
    output_root: Path
    artifact_refs: Task13ArtifactRefsV1
    artifact_index: Task13ArtifactIndexV1

    @property
    def artifact_index_sha256(self) -> str:
        return self.artifact_refs.task13_artifact_index.sha256


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: Path
    identity: tuple[int, int]
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _RootMemberSnapshot:
    root: Path
    relative_path: str
    kind: str
    identity: tuple[int, int]
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Task13SourceSnapshotV1:
    files: tuple[_SourceSnapshot, ...]
    roots: tuple[_DirectoryIdentity, ...]
    root_members: tuple[_RootMemberSnapshot, ...]
    shallow_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.files or not self.roots or not self.root_members:
            raise ValueError("Task 13 source snapshot requires source files, roots, and membership")
        root_keys = {_snapshot_path_key(root.path) for root in self.roots}
        shallow_keys = tuple(_snapshot_path_key(root) for root in self.shallow_roots)
        if len(set(shallow_keys)) != len(shallow_keys) or not set(shallow_keys) <= root_keys:
            raise ValueError("Task 13 shallow roots must be unique registered roots")


@dataclass(frozen=True, slots=True)
class _Task13SourceSnapshotRegistryEntry:
    snapshot: weakref.ReferenceType[Task13SourceSnapshotV1]
    digest: str


_SOURCE_SNAPSHOT_REGISTRY_LOCK = threading.RLock()
_SOURCE_SNAPSHOT_REGISTRY: dict[int, _Task13SourceSnapshotRegistryEntry] = {}


def _task13_source_snapshot_digest_v3(snapshot: Task13SourceSnapshotV1) -> str:
    digest = hashlib.sha256()

    def add(*values: object) -> None:
        for value in values:
            digest.update(os.fsencode(value) if isinstance(value, Path) else str(value).encode("utf-8"))
            digest.update(b"\0")

    for source in snapshot.files:
        add("file", source.path, *source.identity, source.size, source.sha256)
    for root in snapshot.roots:
        add("root", root.path, *root.identity)
    for member in snapshot.root_members:
        add(
            "member", member.root, member.relative_path, member.kind,
            *member.identity, member.size, member.sha256,
        )
    for root in snapshot.shallow_roots:
        add("shallow", root)
    return digest.hexdigest()


def _register_task13_source_snapshot_v3(snapshot: Task13SourceSnapshotV1) -> None:
    identifier = id(snapshot)

    def cleanup(reference: weakref.ReferenceType[Task13SourceSnapshotV1]) -> None:
        with _SOURCE_SNAPSHOT_REGISTRY_LOCK:
            entry = _SOURCE_SNAPSHOT_REGISTRY.get(identifier)
            if entry is not None and entry.snapshot is reference:
                _SOURCE_SNAPSHOT_REGISTRY.pop(identifier, None)

    reference = weakref.ref(snapshot, cleanup)
    entry = _Task13SourceSnapshotRegistryEntry(
        reference, _task13_source_snapshot_digest_v3(snapshot)
    )
    with _SOURCE_SNAPSHOT_REGISTRY_LOCK:
        _SOURCE_SNAPSHOT_REGISTRY[identifier] = entry


def require_capture_registered_task13_source_snapshot_v3(
    snapshot: Task13SourceSnapshotV1,
) -> None:
    with _SOURCE_SNAPSHOT_REGISTRY_LOCK:
        entry = _SOURCE_SNAPSHOT_REGISTRY.get(id(snapshot))
    if (
        entry is None
        or entry.snapshot() is not snapshot
        or entry.digest != _task13_source_snapshot_digest_v3(snapshot)
    ):
        raise ValueError("Task 13 source snapshot is not capture-registered")


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    path: Path
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _OwnedStagingMember:
    name: str
    identity: tuple[int, int]
    size: int
    sha256: str


def _sha256_text(value: object) -> bool:
    return type(value) is str and len(value) in {40, 64} and all(char in "0123456789abcdef" for char in value)


def _artifact(path: str, content: bytes, media_type: str, record_count: int) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        media_type=media_type,
        record_count=record_count,
    )


def _canonical_statistics_v3(statistics: Task13StatisticsResultV1) -> Task13StatisticsResultV1:
    metric_order = {metric: index for index, metric in enumerate(TASK13_METRIC_PATHS)}
    cells = tuple(sorted(
        statistics.cell_statistics,
        key=lambda record: (
            record.answer_model_slot.encode("utf-8"), record.k,
            record.cell_id.encode("utf-8"), metric_order[record.metric_path],
        ),
    ))
    contrasts = tuple(sorted(
        statistics.paired_contrasts,
        key=lambda record: (
            record.answer_model_slot.encode("utf-8"), record.k,
            record.left_cell_id.encode("utf-8"), record.right_cell_id.encode("utf-8"),
            record.contrast_id.encode("utf-8"), metric_order[record.metric_path],
        ),
    ))
    return Task13StatisticsResultV1(cells, contrasts)


def _run_sources(statistics: Task13StatisticsResultV1) -> tuple[Task13RunSourceV1, ...]:
    seen: dict[str, Task13RunSourceV1] = {}
    for record in statistics.cell_statistics:
        source = Task13RunSourceV1(
            run_id=record.run_id,
            answer_model_slot=record.answer_model_slot,
            k=record.k,
            run_manifest_sha256=record.run_manifest_sha256,
            score_artifact_sha256=record.score_artifact_sha256,
        )
        previous = seen.setdefault(source.run_id, source)
        if previous != source:
            raise ValueError("Task 13 cell statistics disagree about a run source")
    return tuple(sorted(seen.values(), key=lambda source: source.run_id.encode("utf-8")))


def build_task13_publication_v3(
    *,
    matrix: Task13AuthenticatedMatrixV1,
    bootstrap_config: Task13BootstrapConfigV1,
    statistics_config_sha256: str,
    runtime: Task13RuntimeBindingV1,
    source_hashes: Mapping[str, str],
) -> Task13PublicationV1:
    """Build a complete immutable Task 13 publication without filesystem output."""
    try:
        matrix_capability = require_loader_registered_task13_matrix_v1(matrix)
        matrix_digest = matrix_capability.matrix_digest
        validate_task13_authenticated_matrix_v1(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError("Task 13 matrix is not loader-registered") from exc
    if not isinstance(bootstrap_config, Task13BootstrapConfigV1):
        raise TypeError("bootstrap_config must be Task13BootstrapConfigV1")
    if statistics_config_sha256 != sha256_model(bootstrap_config):
        raise ValueError("statistics config hash does not match the typed configuration")
    if not isinstance(runtime, Task13RuntimeBindingV1):
        raise TypeError("runtime must be Task13RuntimeBindingV1")
    required_hashes = {
        "preparation_manifest", "plan", "matrix_manifest", "matrix_summary",
        "integrity_audit", "core_tasks", "core_task_manifest",
    }
    authenticated_hashes = {
        "preparation_manifest": matrix.input_hashes["task12_preparation_manifest"],
        "plan": matrix.input_hashes["task12_plan"],
        "matrix_manifest": matrix.input_hashes["task12_matrix_manifest"],
        "matrix_summary": matrix.input_hashes["task12_matrix_summary"],
        "integrity_audit": matrix.input_hashes["task12_integrity_audit"],
        "core_tasks": matrix.input_hashes["core_tasks"],
        "core_task_manifest": matrix.input_hashes["core_task_manifest"],
    }
    if (
        set(source_hashes) != required_hashes
        or any(not _sha256_text(value) for value in source_hashes.values())
        or dict(source_hashes) != authenticated_hashes
    ):
        raise ValueError("Task 13 source hashes must exactly equal authenticated matrix provenance")

    bootstrap = build_bootstrap_indices_v1(matrix.canonical_core_ids, bootstrap_config)
    statistics = _canonical_statistics_v3(
        compute_task13_statistics_v1(matrix, bootstrap, bootstrap_config)
    )
    cases_result = build_task13_cases_v1(matrix)
    verify_task13_cases_v1(cases_result.cases, matrix)

    bootstrap_bytes = bootstrap.raw
    cell_bytes = canonical_jsonl_bytes_v1(statistics.cell_statistics)
    contrast_bytes = canonical_jsonl_bytes_v1(statistics.paired_contrasts)
    cases_bytes = canonical_jsonl_bytes_v1(cases_result.cases)
    cell_ref = _artifact("cell_statistics.jsonl", cell_bytes, "application/x-ndjson", len(statistics.cell_statistics))
    contrast_ref = _artifact("paired_contrasts.jsonl", contrast_bytes, "application/x-ndjson", len(statistics.paired_contrasts))
    cases_ref = _artifact("cases.jsonl", cases_bytes, "application/x-ndjson", len(cases_result.cases))

    receipt = build_task13_statistics_receipt_v1(
        statistics,
        task12_hashes=source_hashes,
        statistics_config_sha256=statistics_config_sha256,
        task13_runtime_revision=runtime.runtime_revision,
        task13_runtime_tree_sha256=runtime.runtime_tree_sha256,
        core_ids_sha256=bootstrap.core_ids_sha256,
        bootstrap_indices_sha256=bootstrap.sha256,
        cell_statistics_artifact=cell_ref,
        paired_contrasts_artifact=contrast_ref,
    )
    case_index = build_task13_case_index_v1(
        cases_result.cases,
        cases_result.coverage,
        _run_sources(statistics),
        cases_ref,
    )
    receipt_bytes = canonical_json_bytes(receipt)
    case_index_bytes = canonical_json_bytes(case_index)
    ledger = build_task13_claim_ledger_v1(
        statistics,
        receipt=receipt,
        case_index=case_index,
        expected_statistics_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        expected_case_index_sha256=hashlib.sha256(case_index_bytes).hexdigest(),
    )
    verify_task13_claim_ledger_v1(
        ledger.claims, statistics, receipt=receipt, case_index=case_index,
    )
    ledger_bytes = canonical_jsonl_bytes_v1(ledger.claims)

    payloads: dict[str, bytes] = {
        "bootstrap_indices.bin": bootstrap_bytes,
        "cell_statistics.jsonl": cell_bytes,
        "paired_contrasts.jsonl": contrast_bytes,
        "statistics_receipt.json": receipt_bytes,
        "cases.jsonl": cases_bytes,
        "case_index.json": case_index_bytes,
        "claim_ledger.jsonl": ledger_bytes,
    }
    first_seven_refs = (
        _artifact("bootstrap_indices.bin", bootstrap_bytes, "application/octet-stream", len(bootstrap.rows)),
        cell_ref, contrast_ref,
        _artifact("statistics_receipt.json", receipt_bytes, "application/json", 1),
        cases_ref,
        _artifact("case_index.json", case_index_bytes, "application/json", 1),
        _artifact("claim_ledger.jsonl", ledger_bytes, "application/x-ndjson", len(ledger.claims)),
    )
    artifact_index = Task13ArtifactIndexV1(
        artifacts=tuple(
            Task13ArtifactBindingV1(artifact_id=ref.path, role=ref.path, artifact=ref)
            for ref in first_seven_refs
        )
    )
    index_bytes = canonical_json_bytes(artifact_index)
    payloads[TASK13_FINAL_INDEX_PATH] = index_bytes
    refs = Task13ArtifactRefsV1(
        *first_seven_refs,
        _artifact(TASK13_FINAL_INDEX_PATH, index_bytes, "application/json", 1),
    )
    if require_loader_registered_task13_matrix_v1(matrix).matrix_digest != matrix_digest:
        raise ValueError("Task 13 loader-registered matrix content changed")
    publication = Task13PublicationV1(
        bootstrap=bootstrap,
        statistics=statistics,
        cases_result=cases_result,
        receipt=receipt,
        case_index=case_index,
        claims=ledger.claims,
        artifact_index=artifact_index,
        artifact_refs=refs,
        matrix_identity=id(matrix),
        artifact_bytes=MappingProxyType(payloads),
    )
    _register_builder_task13_publication_v1(publication, matrix, matrix_digest)
    return publication


def _lstat(path: Path) -> os.stat_result:
    return path.stat(follow_symlinks=False)


def _identity(path: Path) -> tuple[int, int]:
    result = _lstat(path)
    return result.st_dev, result.st_ino


def _present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_reparse(path: Path) -> bool:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(result.st_mode) or bool(getattr(result, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _regular_single_link(path: Path) -> bool:
    result = _lstat(path)
    return not _is_reparse(path) and stat.S_ISREG(result.st_mode) and getattr(result, "st_nlink", 1) == 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_no_reparse(path: Path, *, require_exists: bool) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected

    # Inspect the caller's lexical path before canonicalization.  Resolving first
    # would erase a symlink or junction component that must be rejected.
    lexical_anchor = Path(selected.anchor)
    if not _present(lexical_anchor) or not stat.S_ISDIR(_lstat(lexical_anchor).st_mode):
        raise NotADirectoryError(f"path has no directory anchor: {selected}")
    current = lexical_anchor
    for part in selected.parts[1:]:
        if part in {"", "."}:
            continue
        current = current / part
        if _present(current) and _is_reparse(current):
            raise ValueError(f"path contains a reparse-point component: {current}")
        if not _present(current):
            break

    value = selected.resolve(strict=False)
    if os.name == "nt":
        existing = value
        missing: list[str] = []
        while not _present(existing) and existing != existing.parent:
            missing.append(existing.name)
            existing = existing.parent
        get_long = ctypes.windll.kernel32.GetLongPathNameW
        length = get_long(str(existing), None, 0)
        if length:
            buffer = ctypes.create_unicode_buffer(length)
            if get_long(str(existing), buffer, length):
                value = Path(buffer.value, *reversed(missing))

    anchor = Path(value.anchor)
    if not _present(anchor) or not stat.S_ISDIR(_lstat(anchor).st_mode):
        raise NotADirectoryError(f"path has no directory anchor: {value}")
    current = anchor
    existing_parent = anchor
    for part in value.parts[1:]:
        if part in {"", "."}:
            continue
        current = current / part
        if not _present(current):
            break
        if _is_reparse(current):
            raise ValueError(f"path contains a reparse-point component: {current}")
        existing_parent = current
    if _present(value) and _is_reparse(value):
        raise ValueError(f"path contains a reparse-point component: {value}")
    if require_exists and not _present(value):
        raise FileNotFoundError(value)
    if not _present(value) and not stat.S_ISDIR(_lstat(existing_parent).st_mode):
        raise NotADirectoryError(f"path parent is not a directory: {existing_parent}")
    return value


def _contains(parent: Path, child: Path) -> bool:
    normalized_parent = os.path.normcase(os.path.abspath(str(parent)))
    normalized_child = os.path.normcase(os.path.abspath(str(child)))
    try:
        return os.path.commonpath((normalized_parent, normalized_child)) == normalized_parent
    except ValueError:
        return False


def _assert_nonoverlap(final_root: Path, protected: Sequence[Path]) -> None:
    final_root = _absolute_no_reparse(final_root, require_exists=False)
    for protected_path in protected:
        checked = _absolute_no_reparse(protected_path, require_exists=True)
        if _contains(checked, final_root) or _contains(final_root, checked):
            raise ValueError(f"Task 13 output root overlaps protected source: {checked}")
        try:
            if os.path.samefile(final_root, checked):
                raise ValueError(f"Task 13 output root aliases protected source: {checked}")
        except FileNotFoundError:
            pass


def _snapshot_source_files(source_paths: Sequence[Path], source_roots: Sequence[Path]) -> tuple[_SourceSnapshot, ...]:
    candidates: dict[Path, None] = {}
    for root in source_roots:
        checked = _absolute_no_reparse(root, require_exists=True)
        if not stat.S_ISDIR(_lstat(checked).st_mode):
            raise NotADirectoryError(f"source root is not a directory: {checked}")
        for current, directories, files in os.walk(checked, followlinks=False):
            current_path = Path(current)
            if _is_reparse(current_path):
                raise ValueError("source root contains a reparse-point directory")
            for name in (*directories, *files):
                member = current_path / name
                if _is_reparse(member):
                    raise ValueError("source root contains a reparse-point member")
            for name in files:
                member = current_path / name
                if not _regular_single_link(member):
                    raise ValueError(f"source input must be a regular single-link file: {member}")
                candidates[member] = None
    for source in source_paths:
        checked = _absolute_no_reparse(source, require_exists=True)
        if not _regular_single_link(checked):
            raise ValueError(f"source input must be a regular single-link file: {checked}")
        candidates[checked] = None
    return tuple(
        _SourceSnapshot(path, _identity(path), _lstat(path).st_size, _sha256_file(path))
        for path in sorted(candidates, key=lambda item: os.fsencode(str(item)))
    )


def _snapshot_root_members_v3(root: Path) -> tuple[_RootMemberSnapshot, ...]:
    members: list[_RootMemberSnapshot] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        if _is_reparse(current_path) or not stat.S_ISDIR(_lstat(current_path).st_mode):
            raise ValueError("source root contains an unsafe directory")
        relative = current_path.relative_to(root).as_posix() or "."
        members.append(_RootMemberSnapshot(root, relative, "directory", _identity(current_path)))
        directories.sort(key=lambda name: os.fsencode(name))
        files.sort(key=lambda name: os.fsencode(name))
        for name in directories:
            member = current_path / name
            if _is_reparse(member) or not stat.S_ISDIR(_lstat(member).st_mode):
                raise ValueError("source root contains a reparse-point or unknown directory")
        for name in files:
            member = current_path / name
            if not _regular_single_link(member):
                raise ValueError("source root contains a non-regular or linked file")
            result = _lstat(member)
            members.append(
                _RootMemberSnapshot(
                    root,
                    member.relative_to(root).as_posix(),
                    "file",
                    _identity(member),
                    result.st_size,
                    _sha256_file(member),
                )
            )
    return tuple(sorted(members, key=lambda item: (os.fsencode(str(item.root)), item.relative_path.encode("utf-8"), item.kind)))


def capture_task13_source_snapshot_v3(
    source_paths: Sequence[Path],
    source_roots: Sequence[Path],
    *,
    shallow_roots: Sequence[Path] = (),
) -> Task13SourceSnapshotV1:
    roots = tuple(
        _DirectoryIdentity(
            _absolute_no_reparse(root, require_exists=True),
            _identity(_absolute_no_reparse(root, require_exists=True)),
        )
        for root in source_roots
    )
    if any(not stat.S_ISDIR(_lstat(root.path).st_mode) for root in roots):
        raise NotADirectoryError("Task 13 source roots must be directories")
    normalized_shallow_roots = tuple(
        _absolute_no_reparse(root, require_exists=True) for root in shallow_roots
    )
    shallow_keys = {_snapshot_path_key(root) for root in normalized_shallow_roots}
    root_keys = {_snapshot_path_key(root.path) for root in roots}
    if not shallow_keys <= root_keys:
        raise ValueError("Task 13 shallow source roots must be registered source roots")
    recursive_roots = tuple(
        root.path
        for root in roots
        if os.path.normcase(os.path.abspath(os.fspath(root.path))) not in shallow_keys
    )
    root_members = tuple(
        member
        for root in roots
        for member in (
            (_RootMemberSnapshot(root.path, ".", "shallow_directory", root.identity),)
            if os.path.normcase(os.path.abspath(os.fspath(root.path))) in shallow_keys
            else _snapshot_root_members_v3(root.path)
        )
    )
    snapshot = Task13SourceSnapshotV1(
        files=_snapshot_source_files(source_paths, recursive_roots),
        roots=roots,
        root_members=root_members,
        shallow_roots=normalized_shallow_roots,
    )
    _register_task13_source_snapshot_v3(snapshot)
    return snapshot


def source_snapshot_sha256_v3(snapshot: Task13SourceSnapshotV1, path: Path) -> str:
    if not isinstance(snapshot, Task13SourceSnapshotV1):
        raise TypeError("snapshot must be Task13SourceSnapshotV1")
    checked = _absolute_no_reparse(path, require_exists=True)
    for member in snapshot.files:
        if member.path == checked:
            return member.sha256
    raise ValueError(f"source path is not a member of the initial Task 13 snapshot: {checked}")


def _revalidate_source_snapshot(snapshot: Task13SourceSnapshotV1) -> None:
    for root in snapshot.roots:
        if (
            not _present(root.path)
            or _is_reparse(root.path)
            or not stat.S_ISDIR(_lstat(root.path).st_mode)
            or _identity(root.path) != root.identity
        ):
            raise RuntimeError(f"Task 13 source root changed during publication: {root.path}")
    observed_members: list[_RootMemberSnapshot] = []
    shallow_keys = {_snapshot_path_key(path) for path in snapshot.shallow_roots}
    for root in snapshot.roots:
        expected_members = tuple(
            member for member in snapshot.root_members if member.root == root.path
        )
        if _snapshot_path_key(root.path) in shallow_keys:
            marker = (_RootMemberSnapshot(root.path, ".", "shallow_directory", root.identity),)
            if expected_members != marker:
                raise RuntimeError("Task 13 shallow source root snapshot is malformed")
            observed_members.extend(marker)
        else:
            observed_members.extend(_snapshot_root_members_v3(root.path))
    if tuple(observed_members) != snapshot.root_members:
        raise RuntimeError("Task 13 source root membership changed during publication")
    _revalidate_sources(snapshot.files)


def _revalidate_sources(snapshots: Sequence[_SourceSnapshot]) -> None:
    for snapshot in snapshots:
        if (
            not _regular_single_link(snapshot.path)
            or _identity(snapshot.path) != snapshot.identity
            or _lstat(snapshot.path).st_size != snapshot.size
            or _sha256_file(snapshot.path) != snapshot.sha256
        ):
            raise RuntimeError(f"Task 13 source changed during publication: {snapshot.path}")


def _snapshot_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _validate_source_snapshot_against_loader_capability_v3(
    snapshot: Task13SourceSnapshotV1,
    capability: Task13LoaderCapabilityV1,
    repository_root: Path,
) -> None:
    if not isinstance(capability, Task13LoaderCapabilityV1):
        raise ValueError("Task 13 loader capability metadata is invalid")
    observed_roots: dict[str, _DirectoryIdentity] = {}
    for root in snapshot.roots:
        key = _snapshot_path_key(root.path)
        if key in observed_roots:
            raise ValueError("Task 13 source snapshot contains duplicate roots")
        observed_roots[key] = root
    expected_roots = {
        _snapshot_path_key(entry.path): entry
        for entry in capability.roots.values()
    }
    if set(observed_roots) != set(expected_roots):
        raise ValueError("Task 13 source snapshot does not contain the exact loader source roots")
    for key, expected in expected_roots.items():
        observed = observed_roots[key]
        if observed.identity != expected.identity:
            raise ValueError(f"Task 13 source root identity differs: {expected.name}")

    observed_files: dict[str, _SourceSnapshot] = {}
    for source in snapshot.files:
        key = _snapshot_path_key(source.path)
        if key in observed_files:
            raise ValueError("Task 13 source snapshot contains duplicate files")
        observed_files[key] = source
    for expected in capability.controls.values():
        observed = observed_files.get(_snapshot_path_key(expected.path))
        if observed is None:
            raise ValueError(f"Task 13 source snapshot omits registered control: {expected.name}")
        if observed.path != expected.path or observed.identity != expected.identity or observed.sha256 != expected.sha256:
            raise ValueError(f"Task 13 source snapshot differs for registered control: {expected.name}")

    repository_capability = capability.roots["repository"]
    expected_shallow = (_snapshot_path_key(repository_capability.path),)
    observed_shallow = tuple(_snapshot_path_key(path) for path in snapshot.shallow_roots)
    if observed_shallow != expected_shallow:
        raise ValueError("Task 13 source snapshot must shallow-snapshot only the loader repository root")
    checked_repository = _absolute_no_reparse(repository_root, require_exists=True)
    if (
        checked_repository != repository_capability.path
        or _identity(checked_repository) != repository_capability.identity
    ):
        raise ValueError("Task 13 repository root differs from loader capability")


@contextmanager
def _task13_parent_lock_v3(parent: Path) -> Iterator[None]:
    identity = _identity(parent)
    canonical_parent = os.path.normcase(str(parent.resolve(strict=True)))
    key = hashlib.sha256(f"{canonical_parent}\0{identity[0]}:{identity[1]}".encode("utf-8")).hexdigest()
    lock = parent / f".mub-task13-publish-{key}.lock"
    if _is_reparse(lock):
        raise ValueError("Task 13 parent lock may not be a reparse point")
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        descriptor = os.open(lock, flags)
    try:
        with os.fdopen(descriptor, "r+b", closefd=False) as handle:
            lock_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or getattr(lock_stat, "st_nlink", 1) != 1
                or _identity(lock) != (lock_stat.st_dev, lock_stat.st_ino)
            ):
                raise ValueError("Task 13 parent lock identity is unsafe")
            if os.name == "nt":
                import msvcrt
                if os.fstat(descriptor).st_size == 0:
                    handle.write(b"\0"); handle.flush(); os.fsync(handle.fileno())
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    if _identity(parent) != identity: raise RuntimeError("Task 13 parent identity changed")
                    yield
                finally:
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    if _identity(parent) != identity: raise RuntimeError("Task 13 parent identity changed")
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _create_staging_root_v3(parent: Path) -> tuple[Path, _DirectoryIdentity]:
    for _ in range(32):
        staging = parent / f"{_STAGE_PREFIX}{uuid.uuid4().hex}"
        try:
            staging.mkdir(mode=0o700)
        except FileExistsError:
            continue
        if _is_reparse(staging) or not stat.S_ISDIR(_lstat(staging).st_mode):
            raise RuntimeError("Task 13 staging directory has unsafe type")
        if _identity(staging)[0] != _identity(parent)[0]:
            raise OSError("Task 13 staging directory is on another device")
        return staging, _DirectoryIdentity(staging, _identity(staging))
    raise RuntimeError("could not create an exclusive Task 13 staging directory")


def _json_model_bytes(path: Path, model_type: type[BaseModel]) -> BaseModel:
    raw = path.read_bytes()
    try:
        model = model_type.model_validate_json(raw)
    except Exception as exc:
        raise ValueError(f"Task 13 artifact has invalid typed JSON: {path.name}") from exc
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"Task 13 artifact is not canonical JSON: {path.name}")
    return model


def _jsonl_models(path: Path, model_type: type[BaseModel]) -> tuple[BaseModel, ...]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"Task 13 JSONL canonical form must be nonempty and LF-terminated: {path.name}")
    rows: list[BaseModel] = []
    for line in raw[:-1].split(b"\n"):
        if not line:
            raise ValueError(f"Task 13 JSONL contains an empty row: {path.name}")
        try:
            row = model_type.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"Task 13 JSONL contains invalid typed row: {path.name}") from exc
        if canonical_json_bytes(row) != line:
            raise ValueError(f"Task 13 JSONL has noncanonical row: {path.name}")
        rows.append(row)
    if canonical_jsonl_bytes_v1(rows) != raw:
        raise ValueError(f"Task 13 JSONL does not round-trip canonically: {path.name}")
    return tuple(rows)


def _validate_frozen_bootstrap_bytes_v3(raw: bytes) -> None:
    if len(raw) != 200_000:
        raise ValueError("Task 13 frozen bootstrap must be exactly 200000 bytes")
    if hashlib.sha256(raw).hexdigest() != FROZEN_BOOTSTRAP_INDEX_SHA256:
        raise ValueError("Task 13 frozen bootstrap hash does not match the frozen bootstrap")


def verify_task13_artifact_root_v3(root: Path) -> Task13PublicationResultV1:
    """Validate only internal Task 13 artifact self-consistency.

    Matrix-aware staging validation is the authenticated release-validation
    boundary; this standalone verifier intentionally has no source-matrix context.
    """
    root = _absolute_no_reparse(root, require_exists=True)
    if _is_reparse(root) or not stat.S_ISDIR(_lstat(root).st_mode):
        raise ValueError("Task 13 artifact root must be a regular directory")
    entries = tuple(root.iterdir())
    if tuple(sorted(entry.name for entry in entries)) != tuple(sorted(TASK13_PUBLICATION_PATHS)):
        raise ValueError("Task 13 artifact root must contain exactly eight public artifacts")
    for entry in entries:
        if entry.name.endswith((".tmp", ".bak", ".journal")) or _is_reparse(entry) or not _regular_single_link(entry):
            raise ValueError("Task 13 artifact root contains an unsafe or temporary artifact")

    bootstrap_path = root / "bootstrap_indices.bin"
    _validate_frozen_bootstrap_bytes_v3(bootstrap_path.read_bytes())
    cells = tuple(_jsonl_models(root / "cell_statistics.jsonl", Task13CellStatisticV1))
    contrasts = tuple(_jsonl_models(root / "paired_contrasts.jsonl", Task13PairedContrastV1))
    receipt = _json_model_bytes(root / "statistics_receipt.json", Task13StatisticsReceiptV1)
    cases = tuple(_jsonl_models(root / "cases.jsonl", Task13CaseRecordV1))
    case_index = _json_model_bytes(root / "case_index.json", Task13CaseIndexV1)
    claims = tuple(_jsonl_models(root / "claim_ledger.jsonl", Task13ClaimLedgerRecordV1))
    artifact_index = _json_model_bytes(root / TASK13_FINAL_INDEX_PATH, Task13ArtifactIndexV1)

    content_by_path = {path.name: path.read_bytes() for path in entries}
    expected_refs = tuple(
        _artifact(path, content_by_path[path], "application/octet-stream" if path == "bootstrap_indices.bin" else "application/x-ndjson" if path.endswith(".jsonl") else "application/json", 10_000 if path == "bootstrap_indices.bin" else len(cells) if path == "cell_statistics.jsonl" else len(contrasts) if path == "paired_contrasts.jsonl" else len(cases) if path == "cases.jsonl" else len(claims) if path == "claim_ledger.jsonl" else 1)
        for path in TASK13_PUBLICATION_PATHS
    )
    first_seven = expected_refs[:-1]
    if tuple(binding.artifact for binding in artifact_index.artifacts) != first_seven:
        raise ValueError("Task 13 final index does not bind the exact first seven artifact bytes")
    if receipt.cell_statistics_artifact != first_seven[1] or receipt.paired_contrasts_artifact != first_seven[2]:
        raise ValueError("Task 13 receipt artifact bindings do not close")
    if case_index.cases_artifact != first_seven[4]:
        raise ValueError("Task 13 case index cases binding does not close")
    if receipt.cell_statistic_count != len(cells) or receipt.paired_contrast_count != len(contrasts):
        raise ValueError("Task 13 receipt counts do not close")
    if receipt.bootstrap_indices_sha256 != first_seven[0].sha256:
        raise ValueError("Task 13 receipt bootstrap binding does not close")
    if any(row.bootstrap_indices_sha256 != first_seven[0].sha256 for row in (*cells, *contrasts)):
        raise ValueError("Task 13 statistic rows do not bind bootstrap bytes")
    if hashlib.sha256(canonical_json_bytes(receipt)).hexdigest() != hashlib.sha256(content_by_path["statistics_receipt.json"]).hexdigest():
        raise ValueError("Task 13 receipt hash does not match final bytes")
    if hashlib.sha256(canonical_json_bytes(case_index)).hexdigest() != hashlib.sha256(content_by_path["case_index.json"]).hexdigest():
        raise ValueError("Task 13 case index hash does not match final bytes")
    verify_task13_claim_ledger_v1(claims, cells, contrasts, receipt=receipt, case_index=case_index)
    return Task13PublicationResultV1(
        output_root=root,
        artifact_refs=Task13ArtifactRefsV1(*expected_refs),
        artifact_index=artifact_index,
    )


def validate_task13_staging_root_v3(
    staging_root: Path,
    expected: Task13PublicationV1,
    matrix: Task13AuthenticatedMatrixV1,
) -> None:
    result = verify_task13_artifact_root_v3(staging_root)
    if result.artifact_refs != expected.artifact_refs:
        raise ValueError("Task 13 staging artifacts differ from the computed publication")
    cases = tuple(_jsonl_models(staging_root / "cases.jsonl", Task13CaseRecordV1))
    case_index = _json_model_bytes(staging_root / "case_index.json", Task13CaseIndexV1)
    verify_task13_cases_v1(cases, matrix)
    rebuilt_index = build_task13_case_index_v1(
        cases,
        case_index.coverage,
        case_index.run_sources,
        case_index.cases_artifact,
        source_bindings=case_index.source_bindings,
    )
    if canonical_json_bytes(rebuilt_index) != (staging_root / "case_index.json").read_bytes():
        raise ValueError("Task 13 case index does not equal the authenticated cases closure")


def _renameat2_noreplace_v3(staging: Path, final_root: Path) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            renameat2.restype = ctypes.c_int
            result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(final_root), 1)
        else:
            syscall = getattr(libc, "syscall", None)
            if syscall is None:
                raise RuntimeError("no safe no-replace directory commit primitive is available")
            number = {"x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276}.get(__import__("platform").machine().lower())
            if number is None:
                raise RuntimeError("renameat2 syscall number is unavailable for this architecture")
            syscall.restype = ctypes.c_long
            result = syscall(number, -100, os.fsencode(staging), -100, os.fsencode(final_root), 1)
    except (AttributeError, OSError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError("no safe no-replace directory commit primitive is available") from exc
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("Task 13 final root appeared during no-replace commit")
        raise OSError(error, "renameat2 RENAME_NOREPLACE directory commit failed")


def _directory_commit_noreplace_v3(staging: Path, final_root: Path) -> None:
    if os.name == "nt":
        move_file_ex = ctypes.windll.kernel32.MoveFileExW
        move_file_ex.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move_file_ex.restype = ctypes.c_int
        if not move_file_ex(str(staging), str(final_root), 0x00000008):  # MOVEFILE_WRITE_THROUGH; no REPLACE_EXISTING
            error = ctypes.windll.kernel32.GetLastError()
            if error in {80, 183}:
                raise FileExistsError("Task 13 final root appeared during no-replace commit")
            raise OSError(error, "MoveFileExW no-replace directory commit failed")
        return
    _renameat2_noreplace_v3(staging, final_root)


def _fsync_parent_directory_v3(parent: Path) -> None:
    if os.name == "nt":
        return
    unsupported = {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _commit_staged_task13_root_v3(
    *,
    staging: Path,
    final_root: Path,
    parent_identity: _DirectoryIdentity,
    source_snapshot: Task13SourceSnapshotV1,
    ownership: tuple[_OwnedStagingMember, ...] = (),
    repository_root: Path | None = None,
    expected_runtime: Task13RuntimeBindingV1 | None = None,
) -> None:
    if not ownership and _present(staging) and tuple(staging.iterdir()):
        raise RuntimeError("Task 13 staging ownership is required before commit")
    if ownership:
        _validate_staging_ownership_v3(staging, ownership)
    if _identity(parent_identity.path) != parent_identity.identity or _present(final_root):
        raise FileExistsError("Task 13 final root exists or parent changed before commit")
    staging_identity = _identity(staging)
    _revalidate_source_snapshot(source_snapshot)
    if (repository_root is None) != (expected_runtime is None):
        raise ValueError("Task 13 commit runtime binding arguments must be supplied together")
    if repository_root is not None and expected_runtime is not None:
        _revalidate_clean_task13_runtime_v3(repository_root, expected_runtime)
    if ownership:
        _validate_staging_ownership_v3(staging, ownership)
    _directory_commit_noreplace_v3(staging, final_root)
    try:
        final_stat = _lstat(final_root)
        if (
            _is_reparse(final_root)
            or not stat.S_ISDIR(final_stat.st_mode)
            or (final_stat.st_dev, final_stat.st_ino) != staging_identity
        ):
            raise RuntimeError("Task 13 committed-path-substitution detected after no-replace commit")
    except FileNotFoundError as exc:
        raise RuntimeError("Task 13 committed-path-substitution detected after no-replace commit") from exc
    try:
        _fsync_parent_directory_v3(parent_identity.path)
    except OSError as exc:
        raise RuntimeError(
            "Task 13 final root committed but parent-directory durability failed"
        ) from exc


def _capture_staging_ownership_v3(
    staging: Path, expected_names: Sequence[str]
) -> tuple[_OwnedStagingMember, ...]:
    expected = tuple(sorted(expected_names))
    observed = tuple(sorted(member.name for member in staging.iterdir()))
    if observed != expected:
        raise RuntimeError("owned staging members are not the expected publication set")
    ownership: list[_OwnedStagingMember] = []
    for name in expected:
        member = staging / name
        if not _regular_single_link(member):
            raise RuntimeError("owned staging member has unsafe type or link count")
        result = _lstat(member)
        ownership.append(_OwnedStagingMember(name, _identity(member), result.st_size, _sha256_file(member)))
    return tuple(ownership)


def _owned_staging_member_matches_v3(staging: Path, member: _OwnedStagingMember) -> bool:
    path = staging / member.name
    return (
        _regular_single_link(path)
        and _identity(path) == member.identity
        and _lstat(path).st_size == member.size
        and _sha256_file(path) == member.sha256
    )


def _validate_staging_ownership_v3(
    staging: Path, ownership: tuple[_OwnedStagingMember, ...]
) -> None:
    expected = tuple(member.name for member in ownership)
    observed = tuple(sorted(member.name for member in staging.iterdir()))
    if observed != tuple(sorted(expected)):
        raise RuntimeError("Task 13 staging ownership set changed before commit")
    for member in ownership:
        if not _owned_staging_member_matches_v3(staging, member):
            raise RuntimeError("Task 13 staging ownership/hash changed before commit")


def _unlink_owned_member_v3(staging: Path, member: _OwnedStagingMember) -> None:
    path = staging / member.name
    if os.name != "nt":
        raise RuntimeError("identity-safe staging unlink is unavailable; preserving staging directory")
    if not _owned_staging_member_matches_v3(staging, member):
        raise RuntimeError("Task 13 owned staging member changed; preserving staging directory")
    kernel = ctypes.windll.kernel32
    kernel.CreateFileW.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(
        str(path), 0x00010000 | 0x0080, 0x00000007, None, 3, 0x00200000, None
    )
    if handle in {None, ctypes.c_void_p(-1).value}:
        raise RuntimeError("identity-safe staging handle open failed; preserving staging directory")
    try:
        class _FileInfo(ctypes.Structure):
            _fields_ = [("dwFileAttributes", ctypes.c_uint32), ("ftCreationTime", ctypes.c_ulonglong), ("ftLastAccessTime", ctypes.c_ulonglong), ("ftLastWriteTime", ctypes.c_ulonglong), ("dwVolumeSerialNumber", ctypes.c_uint32), ("nFileSizeHigh", ctypes.c_uint32), ("nFileSizeLow", ctypes.c_uint32), ("nNumberOfLinks", ctypes.c_uint32), ("nFileIndexHigh", ctypes.c_uint32), ("nFileIndexLow", ctypes.c_uint32)]
        info = _FileInfo()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise RuntimeError("identity-safe staging identity read failed; preserving staging directory")
        current_identity = (int(info.dwVolumeSerialNumber), (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow))
        if current_identity != member.identity:
            raise RuntimeError("Task 13 owned staging member identity changed; preserving staging directory")
        class _Disposition(ctypes.Structure):
            _fields_ = [("DeleteFile", ctypes.c_uint8)]
        disposition = _Disposition(1)
        if not kernel.SetFileInformationByHandle(handle, 4, ctypes.byref(disposition), ctypes.sizeof(disposition)):
            raise RuntimeError("identity-safe staging unlink failed; preserving staging directory")
    finally:
        kernel.CloseHandle(handle)


def _safe_cleanup_staging_v3(
    staging: Path,
    identity: _DirectoryIdentity,
    parent_identity: _DirectoryIdentity,
    ownership: tuple[_OwnedStagingMember, ...] | None,
) -> None:
    if not _present(staging):
        return
    if (
        _identity(parent_identity.path) != parent_identity.identity
        or _is_reparse(staging)
        or _identity(staging) != identity.identity
    ):
        raise RuntimeError("Task 13 staging identity changed; preserving staging directory")
    names = tuple(sorted(member.name for member in staging.iterdir()))
    if ownership is None:
        if names:
            raise RuntimeError("Task 13 unowned staging contents preserved after failure")
        staging.rmdir()
        return
    expected = tuple(member.name for member in ownership)
    if names != expected:
        raise RuntimeError("Task 13 owned staging contents changed; preserving staging directory")
    if any(not _owned_staging_member_matches_v3(staging, member) for member in ownership):
        raise RuntimeError("Task 13 owned staging member changed; preserving staging directory")
    for member in ownership:
        path = staging / member.name
        if not _owned_staging_member_matches_v3(staging, member):
            raise RuntimeError("Task 13 owned staging member changed during cleanup; preserving staging directory")
        _unlink_owned_member_v3(staging, member)
    staging.rmdir()


def _publication_runtime_binding_v3(
    publication: Task13PublicationV1,
) -> Task13RuntimeBindingV1:
    raw = publication.artifact_bytes["statistics_receipt.json"]
    try:
        receipt = Task13StatisticsReceiptV1.model_validate_json(raw)
    except Exception as exc:
        raise ValueError("Task 13 publication receipt bytes are invalid") from exc
    if canonical_json_bytes(receipt) != raw:
        raise ValueError("Task 13 publication receipt bytes are not canonical")
    return Task13RuntimeBindingV1(
        receipt.task13_runtime_revision,
        receipt.task13_runtime_tree_sha256,
    )


def publish_task13_artifacts_v3(
    publication: Task13PublicationV1,
    *,
    matrix: Task13AuthenticatedMatrixV1,
    output_root: Path,
    source_snapshot: Task13SourceSnapshotV1,
    repository_root: Path,
) -> Task13PublicationResultV1:
    """Publish a validated Task 13 result via owned sibling staging and no-replace commit."""
    try:
        matrix_capability = require_loader_registered_task13_matrix_v1(matrix)
        matrix_digest = matrix_capability.matrix_digest
        validate_task13_authenticated_matrix_v1(matrix)
        require_builder_registered_task13_publication_v1(publication, matrix, matrix_digest)
    except (TypeError, ValueError) as exc:
        raise ValueError("Task 13 matrix or publication capability is invalid") from exc
    if not isinstance(source_snapshot, Task13SourceSnapshotV1):
        raise TypeError("source_snapshot must be Task13SourceSnapshotV1")
    require_capture_registered_task13_source_snapshot_v3(source_snapshot)
    _validate_source_snapshot_against_loader_capability_v3(
        source_snapshot, matrix_capability, repository_root
    )
    output_root = _absolute_no_reparse(output_root, require_exists=False)
    if _present(output_root):
        raise FileExistsError("Task 13 output root must not already exist")
    parent = _absolute_no_reparse(output_root.parent, require_exists=True)
    if output_root.name in {"", ".", ".."}:
        raise ValueError("Task 13 output root must be a named child directory")
    protected = (
        *(entry.path for entry in matrix_capability.roots.values()),
        *(entry.path for entry in matrix_capability.controls.values()),
        *(snapshot.path for snapshot in source_snapshot.files),
    )
    _assert_nonoverlap(output_root, protected)
    _revalidate_source_snapshot(source_snapshot)
    payloads = {Path(name): content for name, content in publication.artifact_bytes.items()}

    with _task13_parent_lock_v3(parent):
        if _present(output_root):
            raise FileExistsError("Task 13 output root appeared while waiting for publication lock")
        parent_identity = _DirectoryIdentity(parent, _identity(parent))
        staging, staging_identity = _create_staging_root_v3(parent)
        destinations = {staging / name: content for name, content in publication.artifact_bytes.items()}
        validators = {
            staging / name: (lambda path, name=name: _validate_staged_member_v3(path, name, publication))
            for name in publication.artifact_bytes
        }
        committed = False
        preserve_staging = False
        ownership: tuple[_OwnedStagingMember, ...] | None = None
        try:
            publish_files_atomically(
                destinations,
                overwrite=False,
                source_paths=tuple(snapshot.path for snapshot in source_snapshot.files),
                validators=validators,
            )
            validate_task13_staging_root_v3(staging, publication, matrix)
            ownership = _capture_staging_ownership_v3(staging, TASK13_PUBLICATION_PATHS)
            _validate_staging_ownership_v3(staging, ownership)
            if require_loader_registered_task13_matrix_v1(matrix).matrix_digest != matrix_digest:
                raise ValueError("Task 13 loader-registered matrix content changed")
            require_builder_registered_task13_publication_v1(publication, matrix, matrix_digest)
            _commit_staged_task13_root_v3(
                staging=staging, final_root=output_root,
                parent_identity=parent_identity, source_snapshot=source_snapshot,
                ownership=ownership,
                repository_root=repository_root,
                expected_runtime=_publication_runtime_binding_v3(publication),
            )
            committed = True
        except OSError as exc:
            if exc.errno == errno.EXDEV or (os.name == "nt" and exc.errno == 17):
                # A cross-device failure cannot have installed the final root; retain owned staging.
                preserve_staging = True
                raise OSError("Task 13 no-replace commit crossed devices; owned staging retained") from exc
            raise
        finally:
            if not committed and not preserve_staging:
                if ownership is not None and os.name != "nt":
                    # POSIX has no identity-bound unlink primitive here; retain the
                    # authenticated stage without replacing the publication error.
                    preserve_staging = True
                elif ownership is None and _present(staging) and tuple(staging.iterdir()):
                    # Never destroy a populated staging directory whose ownership
                    # was not authenticated; preserve it and retain the original
                    # publication error rather than masking it during cleanup.
                    preserve_staging = True
                else:
                    _safe_cleanup_staging_v3(staging, staging_identity, parent_identity, ownership)
    return Task13PublicationResultV1(
        output_root=output_root,
        artifact_refs=publication.artifact_refs,
        artifact_index=publication.artifact_index,
    )


def _validate_staged_member_v3(path: Path, name: str, publication: Task13PublicationV1) -> None:
    if not _regular_single_link(path):
        raise ValueError("Task 13 staged artifact is not a regular single-link file")
    content = path.read_bytes()
    if content != publication.artifact_bytes[name]:
        raise ValueError("Task 13 staged artifact bytes changed")
    ref = next(ref for ref in publication.artifact_refs.ordered() if ref.path == name)
    if hashlib.sha256(content).hexdigest() != ref.sha256:
        raise ValueError("Task 13 staged artifact hash changed")


def _validate_task13_runtime_index_flags_v3(root: Path) -> None:
    process = subprocess.run(
        ("git", "-C", str(root), "ls-files", "-v", "-z"),
        check=False,
        capture_output=True,
    )
    records = tuple(record for record in process.stdout.split(b"\0") if record)
    if process.returncode != 0 or any(not record.startswith(b"H ") for record in records):
        raise RuntimeError("Task 13 runtime binding rejects nonstandard Git index flags")


def _revalidate_clean_task13_runtime_v3(
    repository_root: Path,
    expected: Task13RuntimeBindingV1,
) -> None:
    if not isinstance(expected, Task13RuntimeBindingV1):
        raise TypeError("expected runtime must be Task13RuntimeBindingV1")
    root = _absolute_no_reparse(repository_root, require_exists=True)
    observed = current_clean_task13_runtime_v3(root)
    _validate_task13_runtime_index_flags_v3(root)
    if observed != expected:
        raise RuntimeError("Task 13 repository runtime changed before commit")
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--ignored=matching", "--ignore-submodules=none"),
        check=False,
        capture_output=True,
        text=True,
    )
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        status.returncode != 0
        or status.stdout
        or revision.returncode != 0
        or revision.stdout.strip() != expected.runtime_revision
    ):
        raise RuntimeError("Task 13 repository runtime changed before commit")


def current_clean_task13_runtime_v3(repository_root: Path) -> Task13RuntimeBindingV1:
    root = _absolute_no_reparse(repository_root, require_exists=True)
    def git(*args: str) -> str:
        process = subprocess.run(("git", "-C", str(root), *args), check=False, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError("could not calculate Task 13 runtime binding from repository")
        return process.stdout.strip()
    top_level = _absolute_no_reparse(Path(git("rev-parse", "--show-toplevel")), require_exists=True)
    if top_level != root or _identity(top_level) != _identity(root):
        raise RuntimeError("Task 13 runtime binding requires the Git worktree root")
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--ignored=matching", "--ignore-submodules=none"),
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout:
        raise RuntimeError("Task 13 runtime binding requires a clean repository")
    _validate_task13_runtime_index_flags_v3(root)
    revision = git("rev-parse", "HEAD")
    tree_process = subprocess.run(
        ("git", "-C", str(root), "ls-tree", "-r", "-z", revision),
        check=False,
        capture_output=True,
    )
    if tree_process.returncode != 0:
        raise RuntimeError("could not calculate Task 13 runtime tree binding")
    tree = hashlib.sha256(tree_process.stdout).hexdigest()
    return Task13RuntimeBindingV1(revision, tree)


__all__ = [
    "TASK13_FINAL_INDEX_PATH", "TASK13_PUBLICATION_PATHS", "Task13ArtifactRefsV1",
    "Task13PublicationResultV1", "Task13PublicationV1", "Task13RuntimeBindingV1",
    "Task13SourceSnapshotV1", "build_task13_publication_v3",
    "capture_task13_source_snapshot_v3", "current_clean_task13_runtime_v3",
    "require_capture_registered_task13_source_snapshot_v3",
    "publish_task13_artifacts_v3", "validate_task13_staging_root_v3",
    "verify_task13_artifact_root_v3",
]
