from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Annotated, Literal
import uuid

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ArtifactRef,
    FrozenStringMap,
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
    freeze_mapping,
)
from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.common import (
    FrozenJsonObjectV3,
    StrictIdentifier,
)
from mub.vnext.contracts.v3.manifest import RunManifestV3
from mub.vnext.contracts.v3.runtime import TaskRunRecordV3
from mub.vnext.contracts.v3.score import ScorerConfigV3
from mub.vnext.external.artifacts import (
    RawPayloadLicenseStatus,
    assert_no_reparse_components,
)
from mub.vnext.external.registry import validate_artifact_provenance
from mub.vnext.external.security import require_redistributable_payload
from mub.vnext.io import canonical_json_bytes, sha256_model


StrictSha256 = Annotated[str, Field(strict=True, pattern=SHA256_PATTERN)]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_IMMUTABLE_CORE_ROOT = _PROJECT_ROOT / "data" / "vnext" / "core" / "v3"
_WRITER_CONSTRUCTION_TOKEN = object()
_ARTIFACT_FIELDS = (
    "source_task_manifest_ref",
    "task_view_ref",
    "adapter_configuration_ref",
    "capability_verification_ref",
    "model_provenance_ref",
    "package_provenance_ref",
    "environment_lock_ref",
)


class ExternalRunConfigV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.run_config.v1"] = (
        "memupdatebench.external.run_config.v1"
    )
    run_id: StrictIdentifier
    code_revision: str = Field(strict=True, pattern=r"^[0-9a-f]{40}$")
    dirty_state: StrictBool
    source_task_manifest_ref: ArtifactRef
    task_view_ref: ArtifactRef
    adapter_configuration_ref: ArtifactRef
    capability_verification_ref: ArtifactRef
    model_provenance_ref: ArtifactRef
    package_provenance_ref: ArtifactRef
    environment_lock_ref: ArtifactRef
    adapter_info: AdapterInfoV3
    adapter_capabilities: AdapterCapabilitiesV3
    retrieval_policy: StrictIdentifier
    answer_mode: StrictIdentifier
    runtime_configuration_hash: StrictSha256
    evaluation_configuration_hash: StrictSha256
    model_name: StrictIdentifier | None = None
    provider: StrictIdentifier | None = None
    model_revision: StrictIdentifier | None = None
    prompt_config: FrozenJsonObjectV3 = Field(default_factory=dict)
    decoding_config: FrozenJsonObjectV3 = Field(default_factory=dict)
    seed_information: FrozenJsonObjectV3 = Field(default_factory=dict)
    environment_summary: FrozenJsonObjectV3 = Field(default_factory=dict)
    package_summary: FrozenJsonObjectV3 = Field(default_factory=dict)
    action_parser_version: StrictIdentifier
    answer_parser_version: StrictIdentifier
    memory_entry_extractor_version: StrictIdentifier
    object_value_extractor_config_hash: StrictSha256
    redaction_policy_version: StrictIdentifier
    normalized_license_status: RawPayloadLicenseStatus
    scorer_config: ScorerConfigV3 = Field(default_factory=ScorerConfigV3)
    repetition_index: int = Field(strict=True, ge=0)
    repetition_count: int = Field(strict=True, gt=0)
    expected_task_ids: tuple[StrictIdentifier, ...] = Field(min_length=1)
    task_record_hashes: FrozenStringMap

    @field_validator("task_record_hashes")
    @classmethod
    def _freeze_task_hashes(cls, value):
        copied = dict(value)
        if any(
            type(key) is not str
            or type(item) is not str
            or not key.strip()
            or len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for key, item in copied.items()
        ):
            raise ValueError("task record hashes must be task-id SHA-256 pairs")
        return freeze_mapping(copied)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.expected_task_ids) != len(set(self.expected_task_ids)):
            raise ValueError("expected task IDs must be unique")
        if set(self.task_record_hashes) != set(self.expected_task_ids):
            raise ValueError("task record hashes must cover expected task IDs")
        if self.repetition_index >= self.repetition_count:
            raise ValueError("repetition index must be below repetition count")
        if (
            type(self.normalized_license_status)
            is not RawPayloadLicenseStatus
            or self.normalized_license_status
            is not RawPayloadLicenseStatus.REDISTRIBUTABLE
        ):
            raise ValueError(
                "normalized license must be explicitly redistributable"
            )
        for field_name in _ARTIFACT_FIELDS:
            value = getattr(self, field_name)
            if type(value) is not ArtifactRef:
                raise ValueError(
                    f"{field_name} must be an exact ArtifactRef"
                )
            validated = validate_artifact_provenance(value)
            object.__setattr__(self, field_name, validated)
        if type(self.adapter_info) is not AdapterInfoV3:
            raise ValueError("adapter info must be an exact AdapterInfoV3")
        if type(self.adapter_capabilities) is not AdapterCapabilitiesV3:
            raise ValueError(
                "adapter capabilities must be exact AdapterCapabilitiesV3"
            )
        if type(self.scorer_config) is not ScorerConfigV3:
            raise ValueError("scorer config must be an exact ScorerConfigV3")
        if (
            self.adapter_configuration_ref.sha256
            != self.adapter_info.configuration_hash
        ):
            raise ValueError(
                "adapter configuration reference must match adapter info"
            )
        return self


class ExternalRunProgressV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.run_progress.v1"] = (
        "memupdatebench.external.run_progress.v1"
    )
    run_identity: StrictSha256
    expected_task_ids: tuple[StrictIdentifier, ...]
    completed_task_ids: tuple[StrictIdentifier, ...]
    failed_task_ids: tuple[StrictIdentifier, ...]
    partial_task_ids: tuple[StrictIdentifier, ...]
    not_supported_task_ids: tuple[StrictIdentifier, ...]
    run_record_hashes: FrozenStringMap
    finalized: StrictBool

    @field_validator("run_record_hashes")
    @classmethod
    def _freeze_run_hashes(cls, value):
        copied = dict(value)
        if any(
            type(key) is not str
            or type(item) is not str
            or not key.strip()
            or len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for key, item in copied.items()
        ):
            raise ValueError("run record hashes must be task-id SHA-256 pairs")
        return freeze_mapping(copied)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.expected_task_ids) != len(set(self.expected_task_ids)):
            raise ValueError("progress expected task IDs must be unique")
        status_groups = (
            self.completed_task_ids,
            self.failed_task_ids,
            self.partial_task_ids,
            self.not_supported_task_ids,
        )
        observed_ids = tuple(
            task_id for group in status_groups for task_id in group
        )
        if (
            len(observed_ids) != len(set(observed_ids))
            or not set(observed_ids) <= set(self.expected_task_ids)
            or set(self.run_record_hashes) != set(observed_ids)
        ):
            raise ValueError("progress task IDs and row hashes are inconsistent")
        order = {
            task_id: index
            for index, task_id in enumerate(self.expected_task_ids)
        }
        if any(
            group != tuple(sorted(group, key=order.__getitem__))
            for group in status_groups
        ):
            raise ValueError("progress task IDs must preserve expected order")
        if self.finalized and set(observed_ids) != set(
            self.expected_task_ids
        ):
            raise ValueError("finalized progress requires complete coverage")
        return self


class ExternalRunIdentityV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.run_identity.v1"] = (
        "memupdatebench.external.run_identity.v1"
    )
    run_identity: StrictSha256
    configuration: ExternalRunConfigV1


class ExternalRunWriterV1:
    def __init__(
        self,
        output_root: Path,
        configuration: ExternalRunConfigV1,
        run_identity: str,
        rows: tuple[TaskRunRecordV3, ...],
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _WRITER_CONSTRUCTION_TOKEN:
            raise ValueError(
                "external run writers require create or resume"
            )
        self.output_root = output_root
        self.configuration = configuration
        self.run_identity = run_identity
        self._rows = list(rows)
        self._directory_identity = _directory_identity(output_root)

    @property
    def rows(self) -> tuple[TaskRunRecordV3, ...]:
        return tuple(self._rows)

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        configuration: ExternalRunConfigV1,
    ) -> ExternalRunWriterV1:
        configuration = _revalidate_configuration(configuration)
        output = _prepare_new_output(output_root)
        run_identity = compute_external_run_identity(configuration)
        identity = ExternalRunIdentityV1(
            run_identity=run_identity,
            configuration=configuration,
        )
        try:
            _write_new_file(
                output / "run_identity.json",
                canonical_json_bytes(identity),
            )
            _write_new_file(output / "task_runs.jsonl", b"")
            writer = cls(
                output,
                configuration,
                run_identity,
                (),
                _construction_token=_WRITER_CONSTRUCTION_TOKEN,
            )
            writer._write_progress(finalized=False)
            _fsync_directory(output)
            return writer
        except BaseException:
            _remove_run_directory(output)
            raise

    @classmethod
    def resume(
        cls,
        output_root: str | Path,
        configuration: ExternalRunConfigV1,
    ) -> ExternalRunWriterV1:
        configuration = _revalidate_configuration(configuration)
        output = _prepare_existing_output(output_root)
        if os.path.lexists(output / "run_manifest.json"):
            raise FileExistsError("external run is already finalized")
        _validate_incomplete_run_tree(output)
        run_identity = compute_external_run_identity(configuration)
        expected_identity = ExternalRunIdentityV1(
            run_identity=run_identity,
            configuration=configuration,
        )
        identity_path = output / "run_identity.json"
        if _read_exact_bytes(identity_path) != canonical_json_bytes(
            expected_identity
        ):
            raise ValueError("external run identity does not match resume config")
        rows = _read_task_rows(
            output / "task_runs.jsonl",
            configuration,
        )
        writer = cls(
            output,
            configuration,
            run_identity,
            rows,
            _construction_token=_WRITER_CONSTRUCTION_TOKEN,
        )
        writer._validate_or_repair_progress()
        return writer

    def append(self, row: TaskRunRecordV3) -> None:
        self._require_directory_identity()
        if os.path.lexists(self.output_root / "run_manifest.json"):
            raise FileExistsError("external run is already finalized")
        _validate_incomplete_run_tree(self.output_root)
        expected_index = len(self._rows)
        if expected_index >= len(self.configuration.expected_task_ids):
            raise ValueError("no next expected task remains")
        expected_task_id = self.configuration.expected_task_ids[
            expected_index
        ]
        row = _validate_public_row(row, self.configuration)
        if row.task_id != expected_task_id:
            raise ValueError(
                "row does not match the next expected task: "
                f"{expected_task_id}"
            )
        task_runs_path = self.output_root / "task_runs.jsonl"
        with task_runs_path.open("ab") as handle:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._rows.append(row)
        self._write_progress(finalized=False)

    def finalize(self) -> RunManifestV3:
        self._require_directory_identity()
        manifest_path = self.output_root / "run_manifest.json"
        if os.path.lexists(manifest_path):
            raise FileExistsError("external run is already finalized")
        _validate_incomplete_run_tree(self.output_root)
        expected_identity = ExternalRunIdentityV1(
            run_identity=self.run_identity,
            configuration=self.configuration,
        )
        if _read_exact_bytes(
            self.output_root / "run_identity.json"
        ) != canonical_json_bytes(expected_identity):
            raise ValueError(
                "run identity artifact does not match current configuration"
            )
        expected_ids = self.configuration.expected_task_ids
        if tuple(row.task_id for row in self._rows) != expected_ids:
            raise ValueError(
                "cannot finalize without complete ordered coverage"
            )
        if any(
            row.completion_status
            in {CompletionStatus.FAILED, CompletionStatus.PARTIAL}
            for row in self._rows
        ):
            raise ValueError("cannot finalize rows with FAILED or PARTIAL status")
        validated_rows = _read_task_rows(
            self.output_root / "task_runs.jsonl",
            self.configuration,
        )
        if validated_rows != tuple(self._rows):
            raise ValueError("task-runs file differs from in-memory rows")
        manifest = _build_manifest(
            self.output_root,
            self.configuration,
            self.run_identity,
            validated_rows,
        )
        self._write_progress(finalized=True)
        try:
            _write_new_atomic(
                manifest_path,
                canonical_json_bytes(manifest),
            )
        except BaseException:
            self._write_progress(finalized=False)
            raise
        return manifest

    def _progress(
        self,
        *,
        finalized: bool,
        rows: tuple[TaskRunRecordV3, ...] | None = None,
    ) -> ExternalRunProgressV1:
        selected_rows = tuple(self._rows) if rows is None else rows
        status_ids = {status: [] for status in CompletionStatus}
        for row in selected_rows:
            status_ids[row.completion_status].append(row.task_id)
        return ExternalRunProgressV1(
            run_identity=self.run_identity,
            expected_task_ids=self.configuration.expected_task_ids,
            completed_task_ids=tuple(
                status_ids[CompletionStatus.COMPLETED]
            ),
            failed_task_ids=tuple(status_ids[CompletionStatus.FAILED]),
            partial_task_ids=tuple(status_ids[CompletionStatus.PARTIAL]),
            not_supported_task_ids=tuple(
                status_ids[CompletionStatus.NOT_SUPPORTED]
            ),
            run_record_hashes={
                row.task_id: sha256_model(row) for row in selected_rows
            },
            finalized=finalized,
        )

    def _write_progress(self, *, finalized: bool) -> None:
        self._require_directory_identity()
        _write_atomic_replace(
            self.output_root / "progress.json",
            canonical_json_bytes(self._progress(finalized=finalized)),
        )

    def _validate_or_repair_progress(self) -> None:
        progress_path = self.output_root / "progress.json"
        if progress_path.exists():
            try:
                raw = _read_exact_bytes(progress_path)
                observed = ExternalRunProgressV1.model_validate_json(raw)
                if canonical_json_bytes(observed) != raw:
                    raise ValueError
                if observed.run_identity != self.run_identity:
                    raise ValueError("progress run identity is invalid")
                observed_count = len(observed.run_record_hashes)
                if observed_count > len(self._rows):
                    raise ValueError("progress row coverage is invalid")
                expected_prefix = self._progress(
                    finalized=False,
                    rows=tuple(self._rows[:observed_count]),
                )
                valid_precommit = False
                if observed.finalized:
                    valid_precommit = (
                        observed_count
                        == len(self.configuration.expected_task_ids)
                        and observed
                        == self._progress(
                            finalized=True,
                            rows=tuple(self._rows[:observed_count]),
                        )
                    )
                if observed != expected_prefix and not valid_precommit:
                    raise ValueError("progress does not match a valid row prefix")
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError("external run progress is invalid") from exc
        self._write_progress(finalized=False)

    def _require_directory_identity(self) -> None:
        assert_no_reparse_components(self.output_root)
        if _directory_identity(self.output_root) != self._directory_identity:
            raise ValueError("external run directory identity changed")


def compute_external_run_identity(
    configuration: ExternalRunConfigV1,
) -> str:
    configuration = _revalidate_configuration(configuration)
    return hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()


def _revalidate_configuration(
    configuration: ExternalRunConfigV1,
) -> ExternalRunConfigV1:
    if type(configuration) is not ExternalRunConfigV1:
        raise ValueError(
            "external run configuration requires an exact trusted type"
        )
    try:
        rebuilt = ExternalRunConfigV1.model_validate(
            configuration.model_dump(mode="python", warnings=False),
            strict=True,
        )
    except Exception as exc:
        raise ValueError(
            "external run configuration fails trust-boundary validation"
        ) from exc
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(configuration):
        raise ValueError("external run configuration serialization is unstable")
    return rebuilt


def _validate_public_row(
    row: TaskRunRecordV3,
    configuration: ExternalRunConfigV1,
) -> TaskRunRecordV3:
    if type(row) is not TaskRunRecordV3:
        raise ValueError("task run row requires an exact TaskRunRecordV3")
    try:
        rebuilt = TaskRunRecordV3.model_validate(
            row.model_dump(mode="python", warnings=False),
            strict=True,
        )
    except Exception as exc:
        raise ValueError(
            "task run row fails trust-boundary validation"
        ) from exc
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(row):
        raise ValueError("task run row serialization is unstable")
    if rebuilt.run_id != configuration.run_id:
        raise ValueError("task run row run_id does not match configuration")
    if rebuilt.adapter_id != configuration.adapter_info.adapter_id:
        raise ValueError("task run row adapter_id does not match configuration")
    provenance = rebuilt.parser_extractor_provenance
    if (
        provenance.action_parser_version
        != configuration.action_parser_version
        or provenance.answer_parser_version
        != configuration.answer_parser_version
        or provenance.memory_entry_extractor_version
        != configuration.memory_entry_extractor_version
        or provenance.object_value_extractor_config_hash
        != configuration.object_value_extractor_config_hash
        or provenance.redaction_policy_version
        != configuration.redaction_policy_version
    ):
        raise ValueError("task run row parser/extractor provenance differs")
    if (
        provenance.raw_provider_artifact_path is not None
        or provenance.raw_adapter_state_path is not None
    ):
        raise ValueError("public task rows cannot contain private raw paths")
    if any(
        snapshot.raw_adapter_state is not None
        for snapshot in rebuilt.memory_snapshots
    ):
        raise ValueError("public task rows cannot contain raw adapter state")
    require_redistributable_payload(
        rebuilt.model_dump(mode="json"),
        license_status=configuration.normalized_license_status,
    )
    return rebuilt


def _read_task_rows(
    path: Path,
    configuration: ExternalRunConfigV1,
) -> tuple[TaskRunRecordV3, ...]:
    if not path.is_file():
        raise ValueError("task-runs file is missing")
    rows: list[TaskRunRecordV3] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith(b"\n") or line == b"\n":
                raise ValueError(
                    f"task-runs line {line_number} is not canonical"
                )
            raw = line[:-1]
            try:
                row = TaskRunRecordV3.model_validate_json(raw)
            except Exception as exc:
                raise ValueError("task-runs file contains an invalid row") from exc
            if canonical_json_bytes(row) != raw:
                raise ValueError("task-runs file contains a noncanonical row")
            row = _validate_public_row(row, configuration)
            expected_index = len(rows)
            if expected_index >= len(configuration.expected_task_ids):
                raise ValueError("task-runs file has unexpected extra rows")
            if row.task_id != configuration.expected_task_ids[expected_index]:
                raise ValueError("task-runs file order or IDs are invalid")
            rows.append(row)
    return tuple(rows)


def _build_manifest(
    output_root: Path,
    configuration: ExternalRunConfigV1,
    run_identity: str,
    rows: tuple[TaskRunRecordV3, ...],
) -> RunManifestV3:
    task_runs_path = output_root / "task_runs.jsonl"
    task_runs_raw = _read_exact_bytes(task_runs_path)
    identity_raw = _read_exact_bytes(output_root / "run_identity.json")
    run_record_hashes = {
        row.task_id: sha256_model(row) for row in rows
    }
    completed = sum(
        row.completion_status is CompletionStatus.COMPLETED for row in rows
    )
    unsupported = sum(
        row.completion_status is CompletionStatus.NOT_SUPPORTED for row in rows
    )
    private_raw_hashes: list[str] = []
    for row in rows:
        provenance = row.parser_extractor_provenance
        for value in (
            provenance.raw_provider_artifact_hash,
            provenance.raw_adapter_state_hash,
        ):
            if value is not None and value not in private_raw_hashes:
                private_raw_hashes.append(value)
    task_runs_ref = ArtifactRef(
        path="task_runs.jsonl",
        sha256=hashlib.sha256(task_runs_raw).hexdigest(),
        media_type="application/x-ndjson",
        record_count=len(rows),
    )
    identity_ref = ArtifactRef(
        path="run_identity.json",
        sha256=hashlib.sha256(identity_raw).hexdigest(),
        media_type="application/json",
        record_count=1,
    )
    return RunManifestV3(
        run_id=configuration.run_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        code_revision=configuration.code_revision,
        dirty_state=configuration.dirty_state,
        task_manifest=configuration.source_task_manifest_ref,
        scorer_config=configuration.scorer_config,
        adapter_info=configuration.adapter_info,
        adapter_capabilities=configuration.adapter_capabilities,
        capability_verification_artifact=(
            configuration.capability_verification_ref
        ),
        model_name=configuration.model_name,
        provider=configuration.provider,
        model_revision=configuration.model_revision,
        prompt_config=configuration.prompt_config,
        decoding_config=configuration.decoding_config,
        seed_information=configuration.seed_information,
        action_parser_version=configuration.action_parser_version,
        answer_parser_version=configuration.answer_parser_version,
        memory_entry_extractor_version=(
            configuration.memory_entry_extractor_version
        ),
        object_value_extractor_config_hash=(
            configuration.object_value_extractor_config_hash
        ),
        redaction_policy_version=configuration.redaction_policy_version,
        environment_summary=configuration.environment_summary,
        package_summary=configuration.package_summary,
        expected_task_count=len(rows),
        completed_task_count=completed,
        failed_task_count=0,
        not_supported_task_count=unsupported,
        raw_provider_response_artifacts=(),
        raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(task_runs_ref, identity_ref),
        score_artifacts=(),
        native_vs_extracted_field_summary={
            "run_identity": run_identity,
            "task_view_hash": configuration.task_view_ref.sha256,
            "runtime_configuration_hash": (
                configuration.runtime_configuration_hash
            ),
            "evaluation_configuration_hash": (
                configuration.evaluation_configuration_hash
            ),
            "model_provenance_hash": (
                configuration.model_provenance_ref.sha256
            ),
            "package_provenance_hash": (
                configuration.package_provenance_ref.sha256
            ),
            "environment_lock_hash": (
                configuration.environment_lock_ref.sha256
            ),
            "normalized_license_status": (
                configuration.normalized_license_status.value
            ),
            "repetition_index": configuration.repetition_index,
            "repetition_count": configuration.repetition_count,
            "private_raw_hashes": list(private_raw_hashes),
        },
        run_record_hashes=run_record_hashes,
    )


def _prepare_new_output(output_root: str | Path) -> Path:
    output = Path(output_root).absolute()
    _validate_output_location(output)
    if os.path.lexists(output):
        raise FileExistsError(f"external run output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError("external run output parent does not exist")
    output.mkdir()
    assert_no_reparse_components(output)
    return output


def _prepare_existing_output(output_root: str | Path) -> Path:
    output = Path(output_root).absolute()
    _validate_output_location(output)
    if not output.is_dir() or output.is_symlink():
        raise ValueError("external run output must be a real directory")
    return output.resolve(strict=True)


def _validate_output_location(output: Path) -> None:
    assert_no_reparse_components(output)
    resolved = output.resolve(strict=False)
    if _IMMUTABLE_CORE_ROOT.exists():
        immutable = _IMMUTABLE_CORE_ROOT.resolve(strict=True)
        if _contains(immutable, resolved) or _contains(resolved, immutable):
            raise ValueError("external run output must be outside immutable Core")


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _file_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _validate_incomplete_run_tree(output: Path) -> None:
    expected_files = {
        "progress.json",
        "run_identity.json",
        "task_runs.jsonl",
    }
    observed_files: set[str] = set()
    for path in output.rglob("*"):
        assert_no_reparse_components(path)
        if path.is_symlink():
            raise ValueError("incomplete run tree contains a reparse point")
        relative_path = path.relative_to(output).as_posix()
        if path.is_dir():
            raise ValueError("incomplete run tree contains an unexpected directory")
        if not path.is_file():
            raise ValueError("incomplete run tree contains a non-file entry")
        if path.lstat().st_nlink != 1:
            raise ValueError(
                f"run artifact must be single-link: {relative_path}"
            )
        observed_files.add(relative_path)
    if observed_files != expected_files:
        raise ValueError("incomplete run tree shape is invalid")


def _read_exact_bytes(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"run artifact is not a regular file: {path.name}")
    if path.lstat().st_nlink != 1:
        raise ValueError(f"run artifact must be single-link: {path.name}")
    return path.read_bytes()


def _write_new_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic_replace(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_new_file(temporary, content)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary) and not temporary.is_symlink():
            temporary.unlink()


def _write_new_atomic(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    installed_identity: tuple[int, int] | None = None
    try:
        _write_new_file(temporary, content)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        except OSError as exc:
            raise OSError(
                "atomic no-replace manifest publication is unavailable"
            ) from exc
        installed_identity = _file_identity(path)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if (
            installed_identity is not None
            and os.path.lexists(path)
            and not path.is_symlink()
            and _file_identity(path) == installed_identity
        ):
            path.unlink()
            try:
                _fsync_directory(path.parent)
            except OSError:
                pass
        raise
    finally:
        if os.path.lexists(temporary) and not temporary.is_symlink():
            temporary.unlink()


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


def _remove_run_directory(output: Path) -> None:
    if not os.path.lexists(output) or output.is_symlink():
        return
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_file() and not path.is_symlink():
            path.unlink()
        elif path.is_dir() and not path.is_symlink():
            path.rmdir()
    output.rmdir()


__all__ = [
    "ExternalRunConfigV1",
    "ExternalRunIdentityV1",
    "ExternalRunProgressV1",
    "ExternalRunWriterV1",
    "compute_external_run_identity",
]
