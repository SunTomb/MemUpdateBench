from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.io import canonical_json_bytes


MODEL_PROVENANCE_SHA256 = (
    "8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e"
)


def _model_ref() -> ArtifactRef:
    return ArtifactRef(
        path="model_provenance.json",
        sha256=MODEL_PROVENANCE_SHA256,
        media_type="application/json",
        record_count=1,
    )


def test_mem0_provider_contract_imports_no_optional_sdk() -> None:
    before = set(sys.modules)
    from mub.vnext.external.providers.mem0 import (
        MEM0_PACKAGE_VERSION,
        Mem0AdapterConfigurationV1,
        Mem0PackageProvenanceV1,
        build_mem0_adapter_configuration,
        fixed_mem0_package_provenance,
        validate_mem0_package_provenance,
    )

    imported = set(sys.modules) - before
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imported
        for prefix in (
            "mem0",
            "qdrant_client",
            "sentence_transformers",
            "torch",
            "transformers",
        )
    )
    assert MEM0_PACKAGE_VERSION == "2.0.17"
    assert Mem0AdapterConfigurationV1.__name__ == (
        "Mem0AdapterConfigurationV1"
    )
    assert Mem0PackageProvenanceV1.__name__ == "Mem0PackageProvenanceV1"
    assert build_mem0_adapter_configuration.__name__ == (
        "build_mem0_adapter_configuration"
    )
    assert fixed_mem0_package_provenance.__name__ == (
        "fixed_mem0_package_provenance"
    )
    assert validate_mem0_package_provenance.__name__ == (
        "validate_mem0_package_provenance"
    )


def test_mem0_package_provenance_is_exact_official_release() -> None:
    from mub.vnext.external.providers.mem0 import (
        fixed_mem0_package_provenance,
    )

    provenance = fixed_mem0_package_provenance()
    assert provenance.package_name == "mem0ai"
    assert provenance.package_version == "2.0.17"
    assert provenance.release_tag == "v2.0.17"
    assert provenance.repository_url == "https://github.com/mem0ai/mem0"
    assert provenance.release_url == (
        "https://github.com/mem0ai/mem0/releases/tag/v2.0.17"
    )
    assert provenance.release_commit == (
        "12c47f524935692e27ad48d829f35fa1e4417181"
    )
    assert provenance.wheel_filename == "mem0ai-2.0.17-py3-none-any.whl"
    assert provenance.wheel_sha256 == (
        "1521209f0ab4c77b7e5777aa1b0b5f0104efa06ca5b9eddb804cdd091c17726a"
    )
    assert provenance.wheel_size_bytes == 343876
    assert provenance.license_id == "Apache-2.0"
    assert provenance.python_requires == ">=3.10,<4.0"


def test_mem0_package_provenance_rejects_constructed_drift() -> None:
    from mub.vnext.external.providers.mem0 import (
        Mem0PackageProvenanceV1,
        fixed_mem0_package_provenance,
        validate_mem0_package_provenance,
    )

    valid = fixed_mem0_package_provenance()
    forged = Mem0PackageProvenanceV1.model_construct(
        **{
            **valid.model_dump(mode="python"),
            "wheel_sha256": "f" * 64,
        }
    )
    with pytest.raises(ValueError, match="frozen mem0 package"):
        validate_mem0_package_provenance(forged)


def test_mem0_adapter_configuration_is_deterministic_and_isolated() -> None:
    from mub.vnext.external.providers.mem0 import (
        build_mem0_adapter_configuration,
        compute_mem0_configuration_hash,
    )

    first = build_mem0_adapter_configuration(
        run_id="task10-mem0-preflight",
        model_provenance_ref=_model_ref(),
    )
    second = build_mem0_adapter_configuration(
        run_id="task10-mem0-preflight",
        model_provenance_ref=_model_ref(),
    )
    other = build_mem0_adapter_configuration(
        run_id="task10-mem0-other",
        model_provenance_ref=_model_ref(),
    )

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.collection_name != other.collection_name
    assert first.collection_name.startswith("mub_mem0_")
    assert "default" not in first.collection_name
    assert "shared" not in first.collection_name
    assert first.namespace_filter_field == "user_id"
    assert first.reset_method == "delete_all_user_id"
    assert first.retrieval_policy == "normal_topk"
    assert first.rerank is False
    assert first.infer_memories is True
    assert first.embedding_provider == "huggingface"
    assert first.embedding_dims == 384
    assert first.llm_provider == "mub_local_qwen_v1"
    assert first.vector_store_provider == "qdrant"
    assert first.telemetry_enabled is False
    assert compute_mem0_configuration_hash(first) == hashlib.sha256(
        canonical_json_bytes(first)
    ).hexdigest()


def test_mem0_adapter_configuration_requires_frozen_model_provenance() -> None:
    from mub.vnext.external.providers.mem0 import (
        build_mem0_adapter_configuration,
    )

    with pytest.raises(ValueError, match="frozen model provenance"):
        build_mem0_adapter_configuration(
            run_id="task10-mem0-preflight",
            model_provenance_ref=ArtifactRef(
                path="model_provenance.json",
                sha256="f" * 64,
                media_type="application/json",
                record_count=1,
            ),
        )


def test_mem0_worker_configuration_keeps_private_paths_out_of_public_config(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.providers.mem0 import (
        Mem0WorkerConfigurationV1,
        build_mem0_adapter_configuration,
        build_mem0_worker_configuration,
        validate_mem0_worker_configuration,
    )

    qwen = tmp_path / "Qwen2.5-7B-Instruct"
    minilm = tmp_path / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    qdrant = tmp_path / "qdrant"
    history = tmp_path / "history"
    for path in (qwen, minilm, qdrant, history):
        path.mkdir()
    public = build_mem0_adapter_configuration(
        run_id="task10-mem0-preflight",
        model_provenance_ref=_model_ref(),
    )
    worker = build_mem0_worker_configuration(
        public_configuration=public,
        qwen_local_path=qwen,
        minilm_local_path=minilm,
        qdrant_path=qdrant,
        history_directory=history,
    )

    assert worker.public_configuration == public
    assert worker.qwen_local_path == str(qwen.resolve())
    assert worker.minilm_local_path == str(minilm.resolve())
    assert worker.qdrant_path == str(qdrant.resolve())
    assert worker.history_db_path == str(
        (history / "mem0-history.db").resolve()
    )
    public_bytes = canonical_json_bytes(public)
    assert str(tmp_path).encode() not in public_bytes
    assert validate_mem0_worker_configuration(worker) == worker

    forged = Mem0WorkerConfigurationV1.model_construct(
        **{
            **worker.model_dump(mode="python"),
            "qwen_local_path": str(minilm.resolve()),
        }
    )
    with pytest.raises(ValueError, match="trust-boundary"):
        validate_mem0_worker_configuration(forged)


def test_mem0_worker_configuration_rejects_overlapping_storage(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.providers.mem0 import (
        build_mem0_adapter_configuration,
        build_mem0_worker_configuration,
    )

    qwen = tmp_path / "Qwen2.5-7B-Instruct"
    minilm = tmp_path / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    storage = tmp_path / "storage"
    for path in (qwen, minilm, storage):
        path.mkdir()
    public = build_mem0_adapter_configuration(
        run_id="task10-mem0-preflight",
        model_provenance_ref=_model_ref(),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        build_mem0_worker_configuration(
            public_configuration=public,
            qwen_local_path=qwen,
            minilm_local_path=minilm,
            qdrant_path=storage,
            history_directory=storage,
        )
    with pytest.raises(ValueError, match="model snapshots"):
        build_mem0_worker_configuration(
            public_configuration=public,
            qwen_local_path=qwen,
            minilm_local_path=minilm,
            qdrant_path=qwen,
            history_directory=storage,
        )
    with pytest.raises(ValueError, match="must be absolute"):
        build_mem0_worker_configuration(
            public_configuration=public,
            qwen_local_path="Qwen2.5-7B-Instruct",
            minilm_local_path=minilm,
            qdrant_path=storage,
            history_directory=storage,
        )
