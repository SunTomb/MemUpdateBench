from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ArtifactRef,
    ImmutableContractModel,
    SHA256_PATTERN,
)
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.registry import validate_artifact_provenance
from mub.vnext.io import canonical_json_bytes

MEM0_PACKAGE_VERSION = "2.0.17"
MEM0_PROVIDER_CONTRACT_VERSION = "memupdatebench.external.mem0.v1"
MEM0_MODEL_PROVENANCE_SHA256 = (
    "8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e"
)
StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]
StrictCommit = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{40}$", strict=True),
]

_MEM0_PACKAGE_LOCK = (
    "mem0ai",
    "2.0.17",
    "v2.0.17",
    "https://github.com/mem0ai/mem0",
    "https://github.com/mem0ai/mem0/releases/tag/v2.0.17",
    "12c47f524935692e27ad48d829f35fa1e4417181",
    "mem0ai-2.0.17-py3-none-any.whl",
    "https://files.pythonhosted.org/packages/f4/bf/"
    "f8167f4e9e5d39c698a82c411700fffcbefeced0f8dec6b0e6a718ed0537/"
    "mem0ai-2.0.17-py3-none-any.whl",
    "1521209f0ab4c77b7e5777aa1b0b5f0104efa06ca5b9eddb804cdd091c17726a",
    343876,
    "Apache-2.0",
    ">=3.10,<4.0",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_IMMUTABLE_CORE_ROOT = _PROJECT_ROOT / "data" / "vnext" / "core" / "v3"


class Mem0PackageProvenanceV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.v1"] = (
        MEM0_PROVIDER_CONTRACT_VERSION
    )
    package_name: Literal["mem0ai"]
    package_version: Literal["2.0.17"]
    release_tag: Literal["v2.0.17"]
    repository_url: StrictIdentifier
    release_url: StrictIdentifier
    release_commit: StrictCommit
    wheel_filename: Literal["mem0ai-2.0.17-py3-none-any.whl"]
    wheel_url: StrictIdentifier
    wheel_sha256: StrictSha256
    wheel_size_bytes: Literal[343876]
    license_id: Literal["Apache-2.0"]
    python_requires: Literal[">=3.10,<4.0"]

    @model_validator(mode="after")
    def _frozen_release(self) -> Self:
        if _package_lock_tuple(self) != _MEM0_PACKAGE_LOCK:
            raise ValueError("frozen mem0 package provenance does not match")
        return self


class Mem0AdapterConfigurationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.v1"] = (
        MEM0_PROVIDER_CONTRACT_VERSION
    )
    run_id: StrictIdentifier
    candidate_id: Literal["mem0_oss"] = "mem0_oss"
    package_provenance: Mem0PackageProvenanceV1
    model_provenance_ref: ArtifactRef
    collection_name: StrictIdentifier
    namespace_filter_field: Literal["user_id"] = "user_id"
    reset_method: Literal["delete_all_user_id"] = "delete_all_user_id"
    retrieval_policy: Literal["normal_topk"] = "normal_topk"
    rerank: Literal[False] = False
    infer_memories: Literal[True] = True
    llm_provider: Literal["mub_local_qwen_v1"] = "mub_local_qwen_v1"
    embedding_provider: Literal["huggingface"] = "huggingface"
    embedding_dims: Literal[384] = 384
    vector_store_provider: Literal["qdrant"] = "qdrant"
    telemetry_enabled: Literal[False] = False

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        package = validate_mem0_package_provenance(self.package_provenance)
        model_ref = validate_artifact_provenance(self.model_provenance_ref)
        object.__setattr__(self, "package_provenance", package)
        object.__setattr__(self, "model_provenance_ref", model_ref)
        if (
            model_ref.path != "model_provenance.json"
            or model_ref.sha256 != MEM0_MODEL_PROVENANCE_SHA256
            or model_ref.media_type != "application/json"
            or model_ref.record_count != 1
        ):
            raise ValueError("frozen model provenance ref does not match")
        expected_collection = _collection_name(
            self.run_id,
            model_ref.sha256,
        )
        if self.collection_name != expected_collection:
            raise ValueError("Mem0 collection name is not run-isolated")
        lowered = self.collection_name.casefold()
        if "default" in lowered or "shared" in lowered:
            raise ValueError("Mem0 collection name must not be default or shared")
        return self


class Mem0WorkerConfigurationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.v1"] = (
        MEM0_PROVIDER_CONTRACT_VERSION
    )
    public_configuration: Mem0AdapterConfigurationV1
    qwen_local_path: StrictIdentifier
    minilm_local_path: StrictIdentifier
    qdrant_path: StrictIdentifier
    history_db_path: StrictIdentifier

    @model_validator(mode="after")
    def _private_paths(self) -> Self:
        public = _revalidate_exact(
            self.public_configuration,
            Mem0AdapterConfigurationV1,
            "Mem0 public configuration",
        )
        object.__setattr__(self, "public_configuration", public)
        paths = {
            name: _real_directory(value, name)
            for name, value in (
                ("qwen_local_path", self.qwen_local_path),
                ("minilm_local_path", self.minilm_local_path),
                ("qdrant_path", self.qdrant_path),
            )
        }
        history_path = Path(self.history_db_path)
        if not history_path.is_absolute():
            raise ValueError("history database path must be absolute")
        assert_no_reparse_components(history_path)
        history_parent = _real_directory(
            str(history_path.parent),
            "history_db_path",
        )
        history_path = history_parent / history_path.name
        if history_path.name != "mem0-history.db":
            raise ValueError("history database filename is not canonical")
        if paths["qwen_local_path"].name != "Qwen2.5-7B-Instruct":
            raise ValueError("Qwen local snapshot path is not frozen")
        if paths["minilm_local_path"].name != (
            "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        ):
            raise ValueError("MiniLM local snapshot path is not frozen")
        storage_roots = (paths["qdrant_path"], history_parent)
        if _overlap(*storage_roots):
            raise ValueError("Qdrant and history storage must not overlap")
        model_roots = (
            paths["qwen_local_path"],
            paths["minilm_local_path"],
        )
        if any(
            _overlap(storage, model)
            for storage in storage_roots
            for model in model_roots
        ):
            raise ValueError("Mem0 storage must not overlap model snapshots")
        immutable = _IMMUTABLE_CORE_ROOT.resolve(strict=True)
        for path in (*storage_roots, history_path):
            if _contains(immutable, path) or _contains(path, immutable):
                raise ValueError("Mem0 storage must be outside immutable Core")
        for name, path in paths.items():
            object.__setattr__(self, name, str(path))
        object.__setattr__(self, "history_db_path", str(history_path))
        return self


def _package_lock_tuple(
    provenance: Mem0PackageProvenanceV1,
) -> tuple[object, ...]:
    return (
        provenance.package_name,
        provenance.package_version,
        provenance.release_tag,
        provenance.repository_url,
        provenance.release_url,
        provenance.release_commit,
        provenance.wheel_filename,
        provenance.wheel_url,
        provenance.wheel_sha256,
        provenance.wheel_size_bytes,
        provenance.license_id,
        provenance.python_requires,
    )


def _revalidate_exact(value: object, expected_type: type, label: str):
    if type(value) is not expected_type:
        raise ValueError(f"{label} trust-boundary requires exact type")
    try:
        payload = {
            field_name: value.__dict__[field_name]
            for field_name in expected_type.model_fields
        }
        return expected_type.model_validate(payload, strict=True)
    except Exception:
        raise ValueError(f"{label} trust-boundary validation failed") from None


def fixed_mem0_package_provenance() -> Mem0PackageProvenanceV1:
    return Mem0PackageProvenanceV1(
        package_name=_MEM0_PACKAGE_LOCK[0],
        package_version=_MEM0_PACKAGE_LOCK[1],
        release_tag=_MEM0_PACKAGE_LOCK[2],
        repository_url=_MEM0_PACKAGE_LOCK[3],
        release_url=_MEM0_PACKAGE_LOCK[4],
        release_commit=_MEM0_PACKAGE_LOCK[5],
        wheel_filename=_MEM0_PACKAGE_LOCK[6],
        wheel_url=_MEM0_PACKAGE_LOCK[7],
        wheel_sha256=_MEM0_PACKAGE_LOCK[8],
        wheel_size_bytes=_MEM0_PACKAGE_LOCK[9],
        license_id=_MEM0_PACKAGE_LOCK[10],
        python_requires=_MEM0_PACKAGE_LOCK[11],
    )


def validate_mem0_package_provenance(
    value: Mem0PackageProvenanceV1,
) -> Mem0PackageProvenanceV1:
    return _revalidate_exact(
        value,
        Mem0PackageProvenanceV1,
        "frozen mem0 package",
    )


def _collection_name(run_id: str, model_provenance_hash: str) -> str:
    digest = hashlib.sha256(
        (
            f"mem0ai-{MEM0_PACKAGE_VERSION}\x1f{run_id}\x1f"
            f"{model_provenance_hash}"
        ).encode("utf-8")
    ).hexdigest()
    return f"mub_mem0_{digest[:32]}"


def build_mem0_adapter_configuration(
    *,
    run_id: str,
    model_provenance_ref: ArtifactRef,
) -> Mem0AdapterConfigurationV1:
    model_ref = validate_artifact_provenance(model_provenance_ref)
    if (
        model_ref.path != "model_provenance.json"
        or model_ref.sha256 != MEM0_MODEL_PROVENANCE_SHA256
        or model_ref.media_type != "application/json"
        or model_ref.record_count != 1
    ):
        raise ValueError("frozen model provenance ref does not match")
    return Mem0AdapterConfigurationV1(
        run_id=run_id,
        package_provenance=fixed_mem0_package_provenance(),
        model_provenance_ref=model_ref,
        collection_name=_collection_name(run_id, model_ref.sha256),
    )


def compute_mem0_configuration_hash(
    configuration: Mem0AdapterConfigurationV1,
) -> str:
    validated = _revalidate_exact(
        configuration,
        Mem0AdapterConfigurationV1,
        "Mem0 adapter configuration",
    )
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _overlap(left: Path, right: Path) -> bool:
    return _contains(left, right) or _contains(right, left)


def _real_directory(value: str, label: str) -> Path:
    if type(value) is not str:
        raise ValueError(f"{label} must be an exact path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    assert_no_reparse_components(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a real directory")
    return resolved


def build_mem0_worker_configuration(
    *,
    public_configuration: Mem0AdapterConfigurationV1,
    qwen_local_path: str | Path,
    minilm_local_path: str | Path,
    qdrant_path: str | Path,
    history_directory: str | Path,
) -> Mem0WorkerConfigurationV1:
    qwen = _real_directory(str(Path(qwen_local_path)), "qwen_local_path")
    minilm = _real_directory(
        str(Path(minilm_local_path)),
        "minilm_local_path",
    )
    qdrant = _real_directory(str(Path(qdrant_path)), "qdrant_path")
    history_root = _real_directory(
        str(Path(history_directory)),
        "history_directory",
    )
    return Mem0WorkerConfigurationV1(
        public_configuration=public_configuration,
        qwen_local_path=str(qwen),
        minilm_local_path=str(minilm),
        qdrant_path=str(qdrant),
        history_db_path=str(history_root / "mem0-history.db"),
    )


def validate_mem0_worker_configuration(
    value: Mem0WorkerConfigurationV1,
) -> Mem0WorkerConfigurationV1:
    return _revalidate_exact(
        value,
        Mem0WorkerConfigurationV1,
        "Mem0 worker configuration",
    )


__all__ = [
    "MEM0_MODEL_PROVENANCE_SHA256",
    "MEM0_PACKAGE_VERSION",
    "MEM0_PROVIDER_CONTRACT_VERSION",
    "Mem0AdapterConfigurationV1",
    "Mem0PackageProvenanceV1",
    "Mem0WorkerConfigurationV1",
    "build_mem0_adapter_configuration",
    "build_mem0_worker_configuration",
    "compute_mem0_configuration_hash",
    "fixed_mem0_package_provenance",
    "validate_mem0_package_provenance",
    "validate_mem0_worker_configuration",
]
