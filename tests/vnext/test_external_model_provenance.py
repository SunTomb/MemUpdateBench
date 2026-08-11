from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.io import canonical_json_bytes


SOURCE_TASK_MANIFEST_SHA256 = (
    "38e623e6888c8f692e6aeb4d7f8c593e72c8fab655d52aca96de954339a439d3"
)
CANARY_SET_MANIFEST_SHA256 = (
    "3c822b014af2b1026056f81b9284bbb6a4ed52d9072ac5524c7aa2fb6c8f95a8"
)


def _ref(path: str, sha256: str) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=sha256,
        media_type="application/json",
        record_count=1,
    )


def _private_ref(sha256: str, size_bytes: int):
    from mub.vnext.external.artifacts import (
        PrivateRawArtifactRefV1,
        RawPayloadLicenseStatus,
    )

    return PrivateRawArtifactRefV1(
        sha256=sha256,
        size_bytes=size_bytes,
        media_type="text/plain; charset=utf-8",
        license_status=RawPayloadLicenseStatus.PRIVATE,
    )


def _bundle():
    from mub.vnext.external.model_provenance import (
        build_task10_model_provenance,
    )

    return build_task10_model_provenance(
        source_task_manifest_ref=_ref(
            "candidate/task_manifest.json",
            SOURCE_TASK_MANIFEST_SHA256,
        ),
        canary_set_manifest_ref=_ref(
            "canaries/canary_set_manifest.json",
            CANARY_SET_MANIFEST_SHA256,
        ),
        snapshot_tree_raw_evidence=_private_ref(
            "c2e3084b0239a62031c02573b4b0b65f0c538feb44474ad5a757f0f82321032e",
            3574,
        ),
        offline_probe_raw_evidence=_private_ref(
            "024a24eea20c0188d0a7666a093b2adf59992c7556129965b057d5bffba24655",
            1886,
        ),
        package_versions_raw_evidence=_private_ref(
            "51e44dc22ac808f7df563aed8b5771989334443fec2d277095255a244d1f777c",
            1617,
        ),
    )


def test_external_facade_exports_model_provenance_contracts() -> None:
    from mub.vnext.external import (
        EmbeddingModelProbeV1,
        ExternalEvaluationConfigV1,
        ExternalModelProvenanceV1,
        InstructionModelProbeV1,
        ModelSnapshotProvenanceV1,
        build_task10_model_provenance,
        publish_model_provenance,
        validate_model_provenance,
    )

    assert EmbeddingModelProbeV1.__name__ == "EmbeddingModelProbeV1"
    assert ExternalEvaluationConfigV1.__name__ == "ExternalEvaluationConfigV1"
    assert ExternalModelProvenanceV1.__name__ == "ExternalModelProvenanceV1"
    assert InstructionModelProbeV1.__name__ == "InstructionModelProbeV1"
    assert ModelSnapshotProvenanceV1.__name__ == "ModelSnapshotProvenanceV1"
    assert build_task10_model_provenance.__name__ == (
        "build_task10_model_provenance"
    )
    assert publish_model_provenance.__name__ == "publish_model_provenance"
    assert validate_model_provenance.__name__ == "validate_model_provenance"


def test_task10_model_provenance_binds_frozen_models_and_evaluation() -> None:
    from mub.vnext.external.model_provenance import ModelRole

    bundle = _bundle()
    evaluation = bundle.evaluation_configuration
    provenance = bundle.model_provenance

    assert evaluation.source_task_manifest_hash == SOURCE_TASK_MANIFEST_SHA256
    assert evaluation.canary_set_manifest_hash == CANARY_SET_MANIFEST_SHA256
    assert evaluation.canary_ids == ("canary_a", "canary_b")
    assert evaluation.namespace_reset_trials == 20
    assert evaluation.determinism_probe_fresh_namespaces == 3
    assert evaluation.deterministic_repetitions == 1
    assert evaluation.nondeterministic_repetitions == 3
    assert evaluation.retrieval_policy == "normal_topk"
    assert evaluation.answer_mode == "slot_direct"

    instruction, embedding = provenance.models
    assert instruction.role is ModelRole.INSTRUCTION
    assert instruction.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert instruction.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert instruction.tree_manifest_sha256 == (
        "d2d9ab0fbeed7ab74ff3dc433209aec9b01952ccc4d88eec16c0d9aaf1fef9c8"
    )
    assert instruction.license_id == "apache-2.0"
    assert instruction.local_files_only is True

    assert embedding.role is ModelRole.EMBEDDING
    assert embedding.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding.revision == "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    assert embedding.tree_manifest_sha256 == (
        "d17624986b02a007e8de99a086d7541ae0119b3f5840890ff196e687b846925b"
    )
    assert embedding.license_id == "apache-2.0"
    assert embedding.local_files_only is True

    assert provenance.instruction_probe.loaded is True
    assert provenance.instruction_probe.deterministic is True
    assert provenance.instruction_probe.prompt_sha256 == hashlib.sha256(
        b"Reply with the single word OK."
    ).hexdigest()
    assert provenance.embedding_probe.embedding_shape == (2, 384)
    assert provenance.embedding_probe.finite is True
    assert provenance.embedding_probe.nonzero is True
    assert provenance.embedding_probe.repeatable is True
    assert provenance.snapshot_tree_raw_evidence.sha256 == (
        "c2e3084b0239a62031c02573b4b0b65f0c538feb44474ad5a757f0f82321032e"
    )
    assert provenance.snapshot_tree_raw_evidence.size_bytes == 3574
    assert provenance.offline_probe_raw_evidence.sha256 == (
        "024a24eea20c0188d0a7666a093b2adf59992c7556129965b057d5bffba24655"
    )
    assert provenance.offline_probe_raw_evidence.size_bytes == 1886
    assert provenance.package_versions_raw_evidence.sha256 == (
        "51e44dc22ac808f7df563aed8b5771989334443fec2d277095255a244d1f777c"
    )
    assert provenance.package_versions_raw_evidence.size_bytes == 1617
    assert dict(provenance.package_versions) == {
        "cuda_runtime": "12.1",
        "numpy": "2.2.6",
        "sentence_transformers": "5.3.0",
        "torch": "2.5.1+cu121",
        "transformers": "4.46.3",
    }

    evaluation_hash = hashlib.sha256(
        canonical_json_bytes(evaluation)
    ).hexdigest()
    assert provenance.evaluation_configuration_hash == evaluation_hash
    assert provenance.evaluation_configuration_ref.sha256 == evaluation_hash


def test_task10_builder_requires_frozen_release_and_canary_anchors() -> None:
    from mub.vnext.external.model_provenance import (
        build_task10_model_provenance,
    )

    evidence = {
        "snapshot_tree_raw_evidence": _private_ref(
            "c2e3084b0239a62031c02573b4b0b65f0c538feb44474ad5a757f0f82321032e",
            3574,
        ),
        "offline_probe_raw_evidence": _private_ref(
            "024a24eea20c0188d0a7666a093b2adf59992c7556129965b057d5bffba24655",
            1886,
        ),
        "package_versions_raw_evidence": _private_ref(
            "51e44dc22ac808f7df563aed8b5771989334443fec2d277095255a244d1f777c",
            1617,
        ),
    }

    with pytest.raises(ValueError, match="frozen source task manifest"):
        build_task10_model_provenance(
            source_task_manifest_ref=_ref(
                "candidate/task_manifest.json",
                "f" * 64,
            ),
            canary_set_manifest_ref=_ref(
                "canaries/canary_set_manifest.json",
                CANARY_SET_MANIFEST_SHA256,
            ),
            **evidence,
        )
    with pytest.raises(ValueError, match="frozen canary set manifest"):
        build_task10_model_provenance(
            source_task_manifest_ref=_ref(
                "candidate/task_manifest.json",
                SOURCE_TASK_MANIFEST_SHA256,
            ),
            canary_set_manifest_ref=_ref(
                "canaries/other.json",
                CANARY_SET_MANIFEST_SHA256,
            ),
            **evidence,
        )
    with pytest.raises(ValueError, match="frozen snapshot tree evidence"):
        build_task10_model_provenance(
            source_task_manifest_ref=_ref(
                "candidate/task_manifest.json",
                SOURCE_TASK_MANIFEST_SHA256,
            ),
            canary_set_manifest_ref=_ref(
                "canaries/canary_set_manifest.json",
                CANARY_SET_MANIFEST_SHA256,
            ),
            snapshot_tree_raw_evidence=_private_ref("f" * 64, 3574),
            offline_probe_raw_evidence=evidence[
                "offline_probe_raw_evidence"
            ],
            package_versions_raw_evidence=evidence[
                "package_versions_raw_evidence"
            ],
        )


def test_model_provenance_rejects_mismatched_or_unverified_evidence() -> None:
    from mub.vnext.external.model_provenance import (
        ExternalModelProvenanceV1,
        ModelSnapshotProvenanceV1,
        validate_model_provenance,
    )

    provenance = _bundle().model_provenance
    with pytest.raises(ValueError, match="evaluation configuration"):
        provenance.validated_replace(evaluation_configuration_hash="f" * 64)
    with pytest.raises(ValueError, match="source task manifest"):
        provenance.validated_replace(source_task_manifest_hash="f" * 64)
    alternate_source = _ref("candidate/task_manifest.json", "f" * 64)
    with pytest.raises(ValueError, match="frozen source task manifest"):
        provenance.validated_replace(
            source_task_manifest_hash="f" * 64,
            source_task_manifest_ref=alternate_source,
        )
    with pytest.raises(ValueError, match="offline instruction probe"):
        provenance.validated_replace(
            instruction_probe=provenance.instruction_probe.validated_replace(
                deterministic=False
            )
        )
    with pytest.raises(ValueError, match="offline embedding probe"):
        provenance.validated_replace(
            embedding_probe=provenance.embedding_probe.validated_replace(
                repeatable=False
            )
        )
    with pytest.raises(ValueError, match="model order"):
        provenance.validated_replace(models=tuple(reversed(provenance.models)))
    secret_model = provenance.models[0].validated_replace(
        model_id="client_secret=hunter2"
    )
    with pytest.raises(ValueError, match="frozen instruction") as exc_info:
        provenance.validated_replace(
            models=(secret_model, provenance.models[1])
        )
    assert "hunter2" not in str(exc_info.value)
    drifted_model = provenance.models[0].validated_replace(
        model_id="Qwen/other-model",
        source_uri="https://huggingface.co/Qwen/other-model",
    )
    with pytest.raises(ValueError, match="frozen instruction model"):
        provenance.validated_replace(
            models=(drifted_model, provenance.models[1])
        )

    forged_model = ModelSnapshotProvenanceV1.model_construct(
        **{
            **provenance.models[0].model_dump(mode="python"),
            "local_files_only": False,
        }
    )
    forged = ExternalModelProvenanceV1.model_construct(
        **{
            **provenance.model_dump(mode="python"),
            "models": (forged_model, provenance.models[1]),
        }
    )
    with pytest.raises(ValueError, match="trust-boundary"):
        validate_model_provenance(forged)


def test_model_provenance_bytes_are_deterministic_and_portable() -> None:
    first = _bundle()
    second = _bundle()

    assert first.evaluation_configuration_bytes == (
        second.evaluation_configuration_bytes
    )
    assert first.model_provenance_bytes == second.model_provenance_bytes
    assert b"/NAS/" not in first.model_provenance_bytes
    assert b"api_key" not in first.model_provenance_bytes
    assert first.model_provenance_ref.path == "model_provenance.json"
    assert first.model_provenance_ref.sha256 == hashlib.sha256(
        first.model_provenance_bytes
    ).hexdigest()


def test_publish_model_provenance_is_exact_and_no_replace(tmp_path: Path) -> None:
    from mub.vnext.external.model_provenance import publish_model_provenance

    bundle = _bundle()
    output = tmp_path / "model-provenance"
    published = publish_model_provenance(bundle, output)

    assert published == output.absolute()
    assert {path.name for path in output.iterdir()} == {
        "evaluation_configuration.json",
        "model_provenance.json",
    }
    assert (output / "evaluation_configuration.json").read_bytes() == (
        bundle.evaluation_configuration_bytes
    )
    assert (output / "model_provenance.json").read_bytes() == (
        bundle.model_provenance_bytes
    )
    with pytest.raises(FileExistsError, match="already exists"):
        publish_model_provenance(bundle, output)


def test_publish_model_provenance_rejects_cross_context_evaluation(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.model_provenance import (
        ModelProvenanceBundleV1,
        publish_model_provenance,
    )

    bundle = _bundle()
    evaluation = bundle.evaluation_configuration.validated_replace(
        source_task_manifest_hash="f" * 64
    )
    evaluation_hash = hashlib.sha256(
        canonical_json_bytes(evaluation)
    ).hexdigest()
    provenance = bundle.model_provenance.validated_replace(
        evaluation_configuration_hash=evaluation_hash,
        evaluation_configuration_ref=_ref(
            "evaluation_configuration.json",
            evaluation_hash,
        ),
    )
    forged = ModelProvenanceBundleV1(
        evaluation_configuration=evaluation,
        model_provenance=provenance,
    )
    output = tmp_path / "cross-context"
    with pytest.raises(ValueError, match="source task context"):
        publish_model_provenance(forged, output)
    assert not output.exists()


def test_publish_model_provenance_rejects_parent_identity_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mub.vnext.external.model_provenance as provenance_module

    output = tmp_path / "model-provenance"
    original_identity = provenance_module._directory_identity
    parent_calls = 0

    def changed_parent_identity(path: Path) -> tuple[int, int]:
        nonlocal parent_calls
        identity = original_identity(path)
        if path == tmp_path:
            parent_calls += 1
            if parent_calls > 1:
                return identity[0], identity[1] + 1
        return identity

    monkeypatch.setattr(
        provenance_module,
        "_directory_identity",
        changed_parent_identity,
    )
    with pytest.raises(ValueError, match="parent changed"):
        provenance_module.publish_model_provenance(_bundle(), output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".model-provenance.staging-*"))


def test_publish_model_provenance_fsync_failure_rolls_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mub.vnext.external.model_provenance as provenance_module

    output = tmp_path / "model-provenance"
    original_fsync = provenance_module._fsync_directory

    def fail_after_publish(path: Path) -> None:
        if output.exists():
            raise OSError("injected provenance fsync failure")
        original_fsync(path)

    monkeypatch.setattr(
        provenance_module,
        "_fsync_directory",
        fail_after_publish,
    )
    with pytest.raises(OSError, match="provenance fsync"):
        provenance_module.publish_model_provenance(_bundle(), output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".model-provenance.staging-*"))


def test_publish_model_provenance_rejects_immutable_core() -> None:
    from mub.vnext.external.model_provenance import publish_model_provenance

    project_root = Path(__file__).resolve().parents[2]
    output = project_root / "data" / "vnext" / "core" / "v3" / "forbidden"
    with pytest.raises(ValueError, match="immutable Core"):
        publish_model_provenance(_bundle(), output)
    assert not output.exists()


def test_prepare_model_provenance_cli_rejects_untrusted_inputs(
    tmp_path: Path,
) -> None:
    from scripts.vnext_prepare_external_model_provenance import main

    canary = tmp_path / "canary-set-manifest.json"
    canary.write_bytes(b'{"source":"canary-set"}')
    output = tmp_path / "published"

    with pytest.raises((ValueError, FileNotFoundError)):
        main(
            [
                "--release-root",
                str(tmp_path / "not-a-release"),
                "--canary-set-manifest",
                str(canary),
                "--snapshot-tree-evidence",
                str(canary),
                "--offline-probe-evidence",
                str(canary),
                "--package-versions-evidence",
                str(canary),
                "--output-root",
                str(output),
            ]
        )
    assert not output.exists()


def test_prepare_model_provenance_cli_reports_unicode_path_as_ascii(
    monkeypatch,
) -> None:
    from scripts.vnext_prepare_external_model_provenance import _report_output

    writes: list[str] = []

    class AsciiStdout:
        def write(self, value: str) -> int:
            value.encode("ascii")
            writes.append(value)
            return len(value)

    monkeypatch.setattr(sys, "stdout", AsciiStdout())
    _report_output(
        Path("模型证据"),
        evaluation_hash="1" * 64,
        provenance_hash="2" * 64,
    )
    assert "\\u6a21" in writes[0]


def test_prepare_model_provenance_script_bootstraps_project_imports(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "vnext_prepare_external_model_provenance.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--release-root" in result.stdout


def test_verify_model_input_artifact_binds_exact_single_link_bytes(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.model_provenance import verify_model_input_artifact

    path = tmp_path / "manifest.json"
    content = b'{"manifest":"test"}'
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()

    assert verify_model_input_artifact(path, expected) == content
    with pytest.raises(ValueError, match="hash"):
        verify_model_input_artifact(path, "0" * 64)

    hardlink = tmp_path / "manifest-hardlink.json"
    try:
        hardlink.hardlink_to(path)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    with pytest.raises(ValueError, match="single-link"):
        verify_model_input_artifact(path, expected)
