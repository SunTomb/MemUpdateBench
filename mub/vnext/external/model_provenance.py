from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import re
from typing import Annotated, Literal
import uuid

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ArtifactRef,
    FrozenStringMap,
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
)
from mub.vnext.contracts.v3.common import StrictIdentifier, StrictPositiveInt
from mub.vnext.external.artifacts import (
    PrivateRawArtifactRefV1,
    RawPayloadLicenseStatus,
    assert_no_reparse_components,
)
from mub.vnext.external.registry import validate_artifact_provenance
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.io import canonical_json_bytes

MODEL_PROVENANCE_SCHEMA_VERSION = (
    "memupdatebench.external.model_provenance.v1"
)
EVALUATION_CONFIGURATION_SCHEMA_VERSION = (
    "memupdatebench.external.evaluation_configuration.v1"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_IMMUTABLE_CORE_ROOT = _PROJECT_ROOT / "data" / "vnext" / "core" / "v3"
StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]
StrictRevision = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{40}$", strict=True),
]
CORE_TASK10_SOURCE_TASK_MANIFEST_SHA256 = (
    "38e623e6888c8f692e6aeb4d7f8c593e72c8fab655d52aca96de954339a439d3"
)
CORE_TASK10_CANARY_SET_MANIFEST_SHA256 = (
    "3c822b014af2b1026056f81b9284bbb6a4ed52d9072ac5524c7aa2fb6c8f95a8"
)
_FROZEN_MODEL_LOCK = (
    (
        "instruction",
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        "apache-2.0",
        "Qwen2ForCausalLM",
        "relative-path-sha256-size-canonical-json-v1",
        "d2d9ab0fbeed7ab74ff3dc433209aec9b01952ccc4d88eec16c0d9aaf1fef9c8",
        14,
        15_242_807_270,
        True,
    ),
    (
        "embedding",
        "sentence-transformers/all-MiniLM-L6-v2",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "apache-2.0",
        "BertModel",
        "relative-path-sha256-size-canonical-json-v1",
        "d17624986b02a007e8de99a086d7541ae0119b3f5840890ff196e687b846925b",
        11,
        91_578_367,
        True,
    ),
)
_FROZEN_PACKAGE_ITEMS = (
    ("cuda_runtime", "12.1"),
    ("numpy", "2.2.6"),
    ("sentence_transformers", "5.3.0"),
    ("torch", "2.5.1+cu121"),
    ("transformers", "4.46.3"),
)
CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SHA256 = (
    "c2e3084b0239a62031c02573b4b0b65f0c538feb44474ad5a757f0f82321032e"
)
CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SIZE = 3574
CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SHA256 = (
    "024a24eea20c0188d0a7666a093b2adf59992c7556129965b057d5bffba24655"
)
CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SIZE = 1886
CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SHA256 = (
    "51e44dc22ac808f7df563aed8b5771989334443fec2d277095255a244d1f777c"
)
CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SIZE = 1617
_FROZEN_TREE_EVIDENCE = (
    CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SHA256,
    CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SIZE,
)
_FROZEN_PROBE_EVIDENCE = (
    CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SHA256,
    CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SIZE,
)
_FROZEN_PACKAGE_EVIDENCE = (
    CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SHA256,
    CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SIZE,
)


class ModelRole(str, Enum):
    INSTRUCTION = "instruction"
    EMBEDDING = "embedding"


class ModelSnapshotProvenanceV1(ImmutableContractModel):
    schema_version: Literal[
        "memupdatebench.external.model_provenance.v1"
    ] = MODEL_PROVENANCE_SCHEMA_VERSION
    role: ModelRole
    model_id: StrictIdentifier
    revision: StrictRevision
    source_uri: StrictIdentifier
    license_id: StrictIdentifier
    architecture: StrictIdentifier
    tree_manifest_version: Literal[
        "relative-path-sha256-size-canonical-json-v1"
    ] = "relative-path-sha256-size-canonical-json-v1"
    tree_manifest_sha256: StrictSha256
    file_count: StrictPositiveInt
    size_bytes: StrictPositiveInt
    local_files_only: Literal[True]

    @model_validator(mode="after")
    def _portable_official_snapshot(self) -> Self:
        if not self.source_uri.startswith("https://huggingface.co/"):
            raise ValueError("model source URI must be an official HTTPS URI")
        if self.license_id != "apache-2.0":
            raise ValueError("model snapshot license must be apache-2.0")
        return self


class InstructionModelProbeV1(ImmutableContractModel):
    schema_version: Literal[
        "memupdatebench.external.model_provenance.v1"
    ] = MODEL_PROVENANCE_SCHEMA_VERSION
    model_revision: StrictRevision
    loaded: Literal[True]
    local_files_only: Literal[True]
    prompt_sha256: StrictSha256
    do_sample: Literal[False]
    max_new_tokens: Literal[16]
    deterministic: StrictBool
    response_sha256: StrictSha256

    @model_validator(mode="after")
    def _passed(self) -> Self:
        if self.deterministic is not True:
            raise ValueError("offline instruction probe must be deterministic")
        return self


class EmbeddingModelProbeV1(ImmutableContractModel):
    schema_version: Literal[
        "memupdatebench.external.model_provenance.v1"
    ] = MODEL_PROVENANCE_SCHEMA_VERSION
    model_revision: StrictRevision
    loaded: Literal[True]
    local_files_only: Literal[True]
    probe_inputs_sha256: StrictSha256
    device: Literal["cpu"]
    embedding_shape: tuple[StrictPositiveInt, StrictPositiveInt]
    finite: StrictBool
    nonzero: StrictBool
    repeatable: StrictBool

    @model_validator(mode="after")
    def _passed(self) -> Self:
        if self.embedding_shape != (2, 384) or not all(
            (self.finite, self.nonzero, self.repeatable)
        ):
            raise ValueError("offline embedding probe did not pass")
        return self


class ExternalEvaluationConfigV1(ImmutableContractModel):
    schema_version: Literal[
        "memupdatebench.external.evaluation_configuration.v1"
    ] = EVALUATION_CONFIGURATION_SCHEMA_VERSION
    source_task_manifest_hash: StrictSha256
    canary_set_manifest_hash: StrictSha256
    canary_ids: tuple[Literal["canary_a", "canary_b"], ...] = (
        "canary_a",
        "canary_b",
    )
    namespace_reset_trials: Literal[20] = 20
    determinism_probe_fresh_namespaces: Literal[3] = 3
    deterministic_repetitions: Literal[1] = 1
    nondeterministic_repetitions: Literal[3] = 3
    retrieval_policy: Literal["normal_topk"] = "normal_topk"
    answer_mode: Literal["slot_direct"] = "slot_direct"
    terminal_rows_required: Literal[True] = True
    failed_or_partial_rows_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _canonical_canaries(self) -> Self:
        if self.canary_ids != ("canary_a", "canary_b"):
            raise ValueError("evaluation requires both canonical canaries")
        return self


def _model_lock_tuple(model: ModelSnapshotProvenanceV1) -> tuple[object, ...]:
    return (
        model.role.value,
        model.model_id,
        model.revision,
        model.source_uri,
        model.license_id,
        model.architecture,
        model.tree_manifest_version,
        model.tree_manifest_sha256,
        model.file_count,
        model.size_bytes,
        model.local_files_only,
    )


class ExternalModelProvenanceV1(ImmutableContractModel):
    schema_version: Literal[
        "memupdatebench.external.model_provenance.v1"
    ] = MODEL_PROVENANCE_SCHEMA_VERSION
    status: Literal["verified_offline"] = "verified_offline"
    source_task_manifest_hash: StrictSha256
    source_task_manifest_ref: ArtifactRef
    canary_set_manifest_hash: StrictSha256
    canary_set_manifest_ref: ArtifactRef
    evaluation_configuration_hash: StrictSha256
    evaluation_configuration_ref: ArtifactRef
    models: tuple[ModelSnapshotProvenanceV1, ...]
    instruction_probe: InstructionModelProbeV1
    embedding_probe: EmbeddingModelProbeV1
    snapshot_tree_raw_evidence: PrivateRawArtifactRefV1
    offline_probe_raw_evidence: PrivateRawArtifactRefV1
    package_versions_raw_evidence: PrivateRawArtifactRefV1
    package_versions: FrozenStringMap

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        source_ref = validate_artifact_provenance(
            self.source_task_manifest_ref
        )
        canary_ref = validate_artifact_provenance(
            self.canary_set_manifest_ref
        )
        evaluation_ref = validate_artifact_provenance(
            self.evaluation_configuration_ref
        )
        object.__setattr__(self, "source_task_manifest_ref", source_ref)
        object.__setattr__(self, "canary_set_manifest_ref", canary_ref)
        object.__setattr__(
            self,
            "evaluation_configuration_ref",
            evaluation_ref,
        )
        if source_ref.sha256 != self.source_task_manifest_hash:
            raise ValueError("source task manifest hash does not match its ref")
        if canary_ref.sha256 != self.canary_set_manifest_hash:
            raise ValueError("canary set manifest hash does not match its ref")
        if evaluation_ref.sha256 != self.evaluation_configuration_hash:
            raise ValueError(
                "evaluation configuration hash does not match its ref"
            )
        if (
            source_ref.path != "candidate/task_manifest.json"
            or source_ref.sha256 != CORE_TASK10_SOURCE_TASK_MANIFEST_SHA256
            or source_ref.media_type != "application/json"
            or source_ref.record_count != 1
        ):
            raise ValueError("frozen source task manifest ref does not match")
        if (
            canary_ref.path != "canaries/canary_set_manifest.json"
            or canary_ref.sha256 != CORE_TASK10_CANARY_SET_MANIFEST_SHA256
            or canary_ref.media_type != "application/json"
            or canary_ref.record_count != 1
        ):
            raise ValueError("frozen canary set manifest ref does not match")

        validated_models = tuple(
            _revalidate_nested(model, ModelSnapshotProvenanceV1)
            for model in self.models
        )
        expected_roles = (ModelRole.INSTRUCTION, ModelRole.EMBEDDING)
        if tuple(model.role for model in validated_models) != expected_roles:
            raise ValueError("models must use canonical model order")
        for index, (model, expected) in enumerate(
            zip(validated_models, _FROZEN_MODEL_LOCK)
        ):
            if _model_lock_tuple(model) != expected:
                role = "instruction" if index == 0 else "embedding"
                raise ValueError(f"frozen {role} model lock does not match")
        object.__setattr__(self, "models", validated_models)

        instruction_probe = _revalidate_nested(
            self.instruction_probe,
            InstructionModelProbeV1,
        )
        embedding_probe = _revalidate_nested(
            self.embedding_probe,
            EmbeddingModelProbeV1,
        )
        object.__setattr__(self, "instruction_probe", instruction_probe)
        object.__setattr__(self, "embedding_probe", embedding_probe)
        if instruction_probe.model_revision != validated_models[0].revision:
            raise ValueError("instruction probe revision does not match model")
        if embedding_probe.model_revision != validated_models[1].revision:
            raise ValueError("embedding probe revision does not match model")
        expected_prompt_hash = hashlib.sha256(
            b"Reply with the single word OK."
        ).hexdigest()
        expected_inputs_hash = hashlib.sha256(
            b'["A current memory value.","An outdated memory value."]'
        ).hexdigest()
        if instruction_probe.prompt_sha256 != expected_prompt_hash:
            raise ValueError("instruction probe input does not match")
        if embedding_probe.probe_inputs_sha256 != expected_inputs_hash:
            raise ValueError("embedding probe inputs do not match")

        tree_evidence = _revalidate_nested(
            self.snapshot_tree_raw_evidence,
            PrivateRawArtifactRefV1,
        )
        probe_evidence = _revalidate_nested(
            self.offline_probe_raw_evidence,
            PrivateRawArtifactRefV1,
        )
        package_evidence = _revalidate_nested(
            self.package_versions_raw_evidence,
            PrivateRawArtifactRefV1,
        )
        object.__setattr__(
            self,
            "snapshot_tree_raw_evidence",
            tree_evidence,
        )
        object.__setattr__(
            self,
            "offline_probe_raw_evidence",
            probe_evidence,
        )
        object.__setattr__(
            self,
            "package_versions_raw_evidence",
            package_evidence,
        )
        for evidence, expected, label in (
            (tree_evidence, _FROZEN_TREE_EVIDENCE, "snapshot tree"),
            (probe_evidence, _FROZEN_PROBE_EVIDENCE, "offline probe"),
            (
                package_evidence,
                _FROZEN_PACKAGE_EVIDENCE,
                "package versions",
            ),
        ):
            if (
                (evidence.sha256, evidence.size_bytes) != expected
                or evidence.media_type != "text/plain; charset=utf-8"
                or evidence.storage_class != "private_raw"
                or evidence.license_status is not RawPayloadLicenseStatus.PRIVATE
            ):
                raise ValueError(f"frozen {label} evidence does not match")

        expected_packages = dict(_FROZEN_PACKAGE_ITEMS)
        if dict(self.package_versions) != expected_packages:
            raise ValueError("model probe package versions do not match lock")
        if scan_for_secrets(
            self.model_dump(mode="python", warnings=False)
        ):
            raise ValueError("model provenance failed security scan")
        return self


@dataclass(frozen=True)
class ModelProvenanceBundleV1:
    evaluation_configuration: ExternalEvaluationConfigV1
    model_provenance: ExternalModelProvenanceV1

    @property
    def evaluation_configuration_bytes(self) -> bytes:
        return canonical_json_bytes(self.evaluation_configuration)

    @property
    def model_provenance_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_provenance)

    @property
    def model_provenance_ref(self) -> ArtifactRef:
        content = self.model_provenance_bytes
        return ArtifactRef(
            path="model_provenance.json",
            sha256=hashlib.sha256(content).hexdigest(),
            media_type="application/json",
            record_count=1,
        )


def _revalidate_nested(value: object, expected_type: type):
    if type(value) is not expected_type:
        raise ValueError(
            f"model provenance trust-boundary requires exact "
            f"{expected_type.__name__}"
        )
    try:
        payload = {
            field_name: value.__dict__[field_name]
            for field_name in expected_type.model_fields
        }
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "model provenance trust-boundary stored fields are incomplete"
        ) from exc
    try:
        return expected_type.model_validate(payload, strict=True)
    except Exception:
        raise ValueError(
            f"model provenance trust-boundary rejected "
            f"{expected_type.__name__}"
        ) from None


def validate_model_provenance(
    value: ExternalModelProvenanceV1,
) -> ExternalModelProvenanceV1:
    return _revalidate_nested(value, ExternalModelProvenanceV1)


def build_task10_model_provenance(
    *,
    source_task_manifest_ref: ArtifactRef,
    canary_set_manifest_ref: ArtifactRef,
    snapshot_tree_raw_evidence: PrivateRawArtifactRefV1,
    offline_probe_raw_evidence: PrivateRawArtifactRefV1,
    package_versions_raw_evidence: PrivateRawArtifactRefV1,
) -> ModelProvenanceBundleV1:
    source_ref = validate_artifact_provenance(source_task_manifest_ref)
    canary_ref = validate_artifact_provenance(canary_set_manifest_ref)
    if (
        source_ref.path != "candidate/task_manifest.json"
        or source_ref.sha256 != CORE_TASK10_SOURCE_TASK_MANIFEST_SHA256
        or source_ref.media_type != "application/json"
        or source_ref.record_count != 1
    ):
        raise ValueError("frozen source task manifest ref does not match")
    if (
        canary_ref.path != "canaries/canary_set_manifest.json"
        or canary_ref.sha256 != CORE_TASK10_CANARY_SET_MANIFEST_SHA256
        or canary_ref.media_type != "application/json"
        or canary_ref.record_count != 1
    ):
        raise ValueError("frozen canary set manifest ref does not match")
    tree_evidence = _revalidate_nested(
        snapshot_tree_raw_evidence,
        PrivateRawArtifactRefV1,
    )
    probe_evidence = _revalidate_nested(
        offline_probe_raw_evidence,
        PrivateRawArtifactRefV1,
    )
    package_evidence = _revalidate_nested(
        package_versions_raw_evidence,
        PrivateRawArtifactRefV1,
    )
    if (
        (tree_evidence.sha256, tree_evidence.size_bytes)
        != _FROZEN_TREE_EVIDENCE
        or tree_evidence.media_type != "text/plain; charset=utf-8"
        or tree_evidence.license_status is not RawPayloadLicenseStatus.PRIVATE
    ):
        raise ValueError("frozen snapshot tree evidence does not match")
    if (
        (probe_evidence.sha256, probe_evidence.size_bytes)
        != _FROZEN_PROBE_EVIDENCE
        or probe_evidence.media_type != "text/plain; charset=utf-8"
        or probe_evidence.license_status is not RawPayloadLicenseStatus.PRIVATE
    ):
        raise ValueError("frozen offline probe evidence does not match")
    if (
        (package_evidence.sha256, package_evidence.size_bytes)
        != _FROZEN_PACKAGE_EVIDENCE
        or package_evidence.media_type != "text/plain; charset=utf-8"
        or package_evidence.license_status
        is not RawPayloadLicenseStatus.PRIVATE
    ):
        raise ValueError("frozen package versions evidence does not match")
    evaluation = ExternalEvaluationConfigV1(
        source_task_manifest_hash=source_ref.sha256,
        canary_set_manifest_hash=canary_ref.sha256,
    )
    evaluation_bytes = canonical_json_bytes(evaluation)
    evaluation_ref = ArtifactRef(
        path="evaluation_configuration.json",
        sha256=hashlib.sha256(evaluation_bytes).hexdigest(),
        media_type="application/json",
        record_count=1,
    )
    qwen_revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    minilm_revision = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    models = (
        ModelSnapshotProvenanceV1(
            role=ModelRole.INSTRUCTION,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision=qwen_revision,
            source_uri=(
                "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
            ),
            license_id="apache-2.0",
            architecture="Qwen2ForCausalLM",
            tree_manifest_sha256=(
                "d2d9ab0fbeed7ab74ff3dc433209aec9b01952ccc4d88eec16c0d9aaf1fef9c8"
            ),
            file_count=14,
            size_bytes=15_242_807_270,
            local_files_only=True,
        ),
        ModelSnapshotProvenanceV1(
            role=ModelRole.EMBEDDING,
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            revision=minilm_revision,
            source_uri=(
                "https://huggingface.co/sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            license_id="apache-2.0",
            architecture="BertModel",
            tree_manifest_sha256=(
                "d17624986b02a007e8de99a086d7541ae0119b3f5840890ff196e687b846925b"
            ),
            file_count=11,
            size_bytes=91_578_367,
            local_files_only=True,
        ),
    )
    provenance = ExternalModelProvenanceV1(
        source_task_manifest_hash=source_ref.sha256,
        source_task_manifest_ref=source_ref,
        canary_set_manifest_hash=canary_ref.sha256,
        canary_set_manifest_ref=canary_ref,
        evaluation_configuration_hash=evaluation_ref.sha256,
        evaluation_configuration_ref=evaluation_ref,
        models=models,
        instruction_probe=InstructionModelProbeV1(
            model_revision=qwen_revision,
            loaded=True,
            local_files_only=True,
            prompt_sha256=hashlib.sha256(
                b"Reply with the single word OK."
            ).hexdigest(),
            do_sample=False,
            max_new_tokens=16,
            deterministic=True,
            response_sha256=hashlib.sha256(b"OK").hexdigest(),
        ),
        embedding_probe=EmbeddingModelProbeV1(
            model_revision=minilm_revision,
            loaded=True,
            local_files_only=True,
            probe_inputs_sha256=hashlib.sha256(
                b'["A current memory value.","An outdated memory value."]'
            ).hexdigest(),
            device="cpu",
            embedding_shape=(2, 384),
            finite=True,
            nonzero=True,
            repeatable=True,
        ),
        snapshot_tree_raw_evidence=tree_evidence,
        offline_probe_raw_evidence=probe_evidence,
        package_versions_raw_evidence=package_evidence,
        package_versions=dict(_FROZEN_PACKAGE_ITEMS),
    )
    return ModelProvenanceBundleV1(
        evaluation_configuration=evaluation,
        model_provenance=provenance,
    )


def verify_model_input_artifact(
    path: str | Path,
    expected_sha256: str,
) -> bytes:
    if type(expected_sha256) is not str or not re.fullmatch(
        SHA256_PATTERN,
        expected_sha256,
    ):
        raise ValueError("expected artifact hash must be canonical SHA-256")
    artifact = Path(path).absolute()
    assert_no_reparse_components(artifact)
    if not artifact.is_file() or artifact.is_symlink():
        raise ValueError("model input artifact must be a real file")
    metadata_before = artifact.stat(follow_symlinks=False)
    if metadata_before.st_nlink != 1:
        raise ValueError("model input artifact must be a single-link file")
    content = artifact.read_bytes()
    metadata_after = artifact.stat(follow_symlinks=False)
    before_identity = (metadata_before.st_dev, metadata_before.st_ino)
    after_identity = (metadata_after.st_dev, metadata_after.st_ino)
    if before_identity != after_identity or metadata_after.st_nlink != 1:
        raise ValueError("model input artifact changed during authentication")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("model input artifact hash does not match")
    return content


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _require_stable_parent(
    parent: Path,
    expected_identity: tuple[int, int],
    output: Path,
) -> None:
    assert_no_reparse_components(parent)
    if (
        not parent.is_dir()
        or _directory_identity(parent) != expected_identity
    ):
        raise ValueError("model provenance output parent changed")
    candidate = parent.resolve(strict=True) / output.name
    immutable_root = _IMMUTABLE_CORE_ROOT.resolve(strict=True)
    if _contains(immutable_root, candidate) or _contains(
        candidate,
        immutable_root,
    ):
        raise ValueError(
            "model provenance output must be outside immutable Core"
        )


def _remove_owned_tree(
    root: Path,
    expected_identity: tuple[int, int],
) -> bool:
    if (
        not os.path.lexists(root)
        or root.is_symlink()
        or not root.is_dir()
        or _directory_identity(root) != expected_identity
    ):
        return False
    expected_names = {
        "evaluation_configuration.json",
        "model_provenance.json",
    }
    observed = {path.name: path for path in root.iterdir()}
    if not set(observed).issubset(expected_names):
        return False
    for path in observed.values():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat(follow_symlinks=False).st_nlink != 1
        ):
            return False
    for path in observed.values():
        path.unlink()
    root.rmdir()
    return True


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


def publish_model_provenance(
    bundle: ModelProvenanceBundleV1,
    output_root: str | Path,
) -> Path:
    if type(bundle) is not ModelProvenanceBundleV1:
        raise ValueError("publication requires an exact model provenance bundle")
    evaluation = _revalidate_nested(
        bundle.evaluation_configuration,
        ExternalEvaluationConfigV1,
    )
    provenance = validate_model_provenance(bundle.model_provenance)
    evaluation_bytes = canonical_json_bytes(evaluation)
    if hashlib.sha256(evaluation_bytes).hexdigest() != (
        provenance.evaluation_configuration_hash
    ):
        raise ValueError("model provenance bundle evaluation hash is invalid")
    if (
        evaluation.source_task_manifest_hash
        != provenance.source_task_manifest_hash
    ):
        raise ValueError(
            "model provenance bundle source task context is inconsistent"
        )
    if (
        evaluation.canary_set_manifest_hash
        != provenance.canary_set_manifest_hash
    ):
        raise ValueError(
            "model provenance bundle canary context is inconsistent"
        )
    provenance_bytes = canonical_json_bytes(provenance)

    output = Path(output_root).absolute()
    if os.path.lexists(output):
        raise FileExistsError(
            f"model provenance output root already exists: {output}"
        )
    parent = output.parent
    assert_no_reparse_components(parent)
    if not parent.is_dir():
        raise ValueError("model provenance output parent does not exist")
    parent_identity = _directory_identity(parent)
    _require_stable_parent(parent, parent_identity, output)
    stage = parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if os.path.lexists(stage):
        raise FileExistsError("model provenance staging path already exists")
    stage_identity: tuple[int, int] | None = None
    installed = False
    try:
        _require_stable_parent(parent, parent_identity, output)
        stage.mkdir()
        if (
            stage.is_symlink()
            or stage.resolve(strict=True).parent
            != parent.resolve(strict=True)
        ):
            raise ValueError(
                "model provenance staging directory is not anchored"
            )
        stage_identity = _directory_identity(stage)
        _require_stable_parent(parent, parent_identity, output)
        _write_fsynced(
            stage / "evaluation_configuration.json",
            evaluation_bytes,
        )
        _require_stable_parent(parent, parent_identity, output)
        if _directory_identity(stage) != stage_identity:
            raise ValueError("model provenance staging directory changed")
        _write_fsynced(
            stage / "model_provenance.json",
            provenance_bytes,
        )
        _require_stable_parent(parent, parent_identity, output)
        if _directory_identity(stage) != stage_identity:
            raise ValueError("model provenance staging directory changed")
        _fsync_directory(stage)
        expected = {
            "evaluation_configuration.json": evaluation_bytes,
            "model_provenance.json": provenance_bytes,
        }
        observed = {path.name: path for path in stage.iterdir()}
        if (
            _directory_identity(stage) != stage_identity
            or set(observed) != set(expected)
        ):
            raise ValueError("staged model provenance tree is invalid")
        for name, expected_bytes in expected.items():
            path = observed[name]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat(follow_symlinks=False).st_nlink != 1
                or path.read_bytes() != expected_bytes
            ):
                raise ValueError("staged model provenance artifact is invalid")
        _require_stable_parent(parent, parent_identity, output)
        if _directory_identity(stage) != stage_identity:
            raise ValueError("model provenance staging directory changed")
        if os.path.lexists(output):
            raise FileExistsError(
                f"model provenance output root already exists: {output}"
            )
        from mub.vnext.external.canaries_v3 import _rename_no_replace

        _rename_no_replace(stage, output)
        installed = True
        _fsync_directory(parent)
    except BaseException:
        if stage_identity is not None:
            owned_path = output if installed else stage
            if _remove_owned_tree(owned_path, stage_identity):
                _fsync_directory(parent)
        raise
    return output


__all__ = [
    "CORE_TASK10_CANARY_SET_MANIFEST_SHA256",
    "CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SHA256",
    "CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SIZE",
    "CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SHA256",
    "CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SIZE",
    "CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SHA256",
    "CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SIZE",
    "CORE_TASK10_SOURCE_TASK_MANIFEST_SHA256",
    "EVALUATION_CONFIGURATION_SCHEMA_VERSION",
    "MODEL_PROVENANCE_SCHEMA_VERSION",
    "EmbeddingModelProbeV1",
    "ExternalEvaluationConfigV1",
    "ExternalModelProvenanceV1",
    "InstructionModelProbeV1",
    "ModelProvenanceBundleV1",
    "ModelRole",
    "ModelSnapshotProvenanceV1",
    "build_task10_model_provenance",
    "publish_model_provenance",
    "validate_model_provenance",
    "verify_model_input_artifact",
]
