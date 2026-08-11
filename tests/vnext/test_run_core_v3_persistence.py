from __future__ import annotations

import hashlib

import pytest

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.runtime import (
    MemorySnapshotV3,
    ParserExtractorProvenanceV3,
    TaskRunRecordV3,
)
from mub.vnext.io import canonical_json_bytes, sha256_model


def _artifact(path: str, digit: str) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=digit * 64,
        media_type="application/json",
        record_count=1,
    )


def _config(**changes):
    from mub.vnext.external.artifacts import RawPayloadLicenseStatus
    from mub.vnext.runtime.run_v3 import ExternalRunConfigV1

    values = {
        "run_id": "external-run-v1",
        "code_revision": "a" * 40,
        "dirty_state": False,
        "source_task_manifest_ref": _artifact(
            "candidate/task_manifest.json",
            "1",
        ),
        "task_view_ref": _artifact(
            "canaries/canary_a/canary_manifest.json",
            "2",
        ),
        "adapter_configuration_ref": _artifact(
            "configuration/adapter.json",
            "3",
        ),
        "capability_verification_ref": _artifact(
            "evidence/capability_verification.json",
            "4",
        ),
        "model_provenance_ref": _artifact(
            "provenance/model.json",
            "5",
        ),
        "package_provenance_ref": _artifact(
            "provenance/package.json",
            "6",
        ),
        "environment_lock_ref": _artifact(
            "provenance/environment.lock.json",
            "7",
        ),
        "adapter_info": AdapterInfoV3(
            adapter_id="mem0_oss",
            adapter_version="adapter-v1",
            system_name="mem0_oss",
            system_version="system-v1",
            sdk_version="sdk-v1",
            configuration_hash="3" * 64,
        ),
        "adapter_capabilities": AdapterCapabilitiesV3(
            supports_isolated_reset=True,
            supports_event_ingest=True,
            supports_add=True,
            supports_update=True,
            exports_entries=True,
            exports_object_keys=True,
            exports_values=True,
        ),
        "retrieval_policy": "normal_topk",
        "answer_mode": "slot_direct",
        "runtime_configuration_hash": "8" * 64,
        "evaluation_configuration_hash": "9" * 64,
        "model_name": "local-model",
        "provider": "offline",
        "model_revision": "model-revision",
        "prompt_config": {},
        "decoding_config": {"temperature": 0},
        "seed_information": {"seed": 20260811},
        "environment_summary": {"isolation": "subprocess"},
        "package_summary": {"lock_hash": "7" * 64},
        "action_parser_version": "visible-action-v1",
        "answer_parser_version": "typed-answer-v1",
        "memory_entry_extractor_version": "provider-entry-v1",
        "object_value_extractor_config_hash": "0" * 64,
        "redaction_policy_version": "external-redaction-v1",
        "normalized_license_status": (
            RawPayloadLicenseStatus.REDISTRIBUTABLE
        ),
        "repetition_index": 0,
        "repetition_count": 1,
        "expected_task_ids": ("task-1", "task-2"),
        "task_record_hashes": {
            "task-1": "b" * 64,
            "task-2": "c" * 64,
        },
    }
    values.update(changes)
    return ExternalRunConfigV1(**values)


def _row(
    task_id: str,
    *,
    status: CompletionStatus = CompletionStatus.COMPLETED,
    raw_hash: str | None = None,
    raw_path: str | None = None,
    raw_state=None,
) -> TaskRunRecordV3:
    snapshots = ()
    if raw_state is not None:
        snapshots = (
            MemorySnapshotV3(
                entries=(),
                state_by_object={},
                store_size=0,
                raw_adapter_state=raw_state,
            ),
        )
    return TaskRunRecordV3(
        task_id=task_id,
        adapter_id="mem0_oss",
        run_id="external-run-v1",
        memory_snapshots=snapshots,
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="visible-action-v1",
            answer_parser_version="typed-answer-v1",
            memory_entry_extractor_version="provider-entry-v1",
            object_value_extractor_config_hash="0" * 64,
            redaction_policy_version="external-redaction-v1",
            raw_provider_artifact_path=raw_path,
            raw_provider_artifact_hash=raw_hash,
        ),
        completion_status=status,
    )


def test_runtime_facade_exports_strict_v3_external_persistence():
    from mub.vnext.runtime import (
        ExternalRunConfigV1,
        ExternalRunProgressV1,
        ExternalRunWriterV1,
        compute_external_run_identity,
    )

    assert ExternalRunConfigV1.__name__ == "ExternalRunConfigV1"
    assert ExternalRunProgressV1.__name__ == "ExternalRunProgressV1"
    assert ExternalRunWriterV1.__name__ == "ExternalRunWriterV1"
    assert compute_external_run_identity.__name__ == (
        "compute_external_run_identity"
    )


def test_external_run_appends_and_fsyncs_one_canonical_row_at_a_time(tmp_path):
    from mub.vnext.runtime.run_v3 import (
        ExternalRunProgressV1,
        ExternalRunWriterV1,
    )

    output = tmp_path / "run"
    writer = ExternalRunWriterV1.create(output, _config())
    row = _row("task-1", raw_hash="d" * 64)
    writer.append(row)

    assert (output / "task_runs.jsonl").read_bytes() == (
        canonical_json_bytes(row) + b"\n"
    )
    progress_raw = (output / "progress.json").read_bytes()
    progress_model = ExternalRunProgressV1.model_validate_json(progress_raw)
    assert progress_raw == canonical_json_bytes(progress_model)
    progress = progress_model.model_dump(mode="json")
    assert progress["completed_task_ids"] == ["task-1"]
    assert progress["run_record_hashes"] == {
        "task-1": sha256_model(row)
    }
    assert not (output / "run_manifest.json").exists()


def test_external_run_requires_exact_order_and_one_terminal_row(tmp_path):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    writer = ExternalRunWriterV1.create(tmp_path / "run", _config())
    with pytest.raises(ValueError, match="next expected task"):
        writer.append(_row("task-2"))
    writer.append(_row("task-1"))
    with pytest.raises(ValueError, match="next expected task"):
        writer.append(_row("task-1"))


def test_external_run_strict_resume_and_atomic_finalize(tmp_path):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    output = tmp_path / "run"
    first = ExternalRunWriterV1.create(output, _config())
    first.append(_row("task-1", raw_hash="d" * 64))

    resumed = ExternalRunWriterV1.resume(output, _config())
    assert type(resumed.rows) is tuple
    assert tuple(row.task_id for row in resumed.rows) == ("task-1",)
    with pytest.raises(AttributeError):
        resumed.rows.append(_row("task-2"))
    resumed.append(_row("task-2", raw_hash="e" * 64))
    manifest = resumed.finalize()

    assert (output / "run_manifest.json").read_bytes() == (
        canonical_json_bytes(manifest)
    )
    assert dict(manifest.run_record_hashes) == {
        "task-1": sha256_model(_row("task-1", raw_hash="d" * 64)),
        "task-2": sha256_model(_row("task-2", raw_hash="e" * 64)),
    }
    assert manifest.raw_provider_response_artifacts == ()
    assert manifest.raw_adapter_state_artifacts == ()
    assert manifest.native_vs_extracted_field_summary[
        "private_raw_hashes"
    ] == ("d" * 64, "e" * 64)
    assert manifest.normalized_runtime_artifacts[0].path == "task_runs.jsonl"
    assert manifest.normalized_runtime_artifacts[0].record_count == 2
    assert manifest.normalized_runtime_artifacts[1].path == "run_identity.json"
    assert manifest.normalized_runtime_artifacts[1].sha256 == hashlib.sha256(
        (output / "run_identity.json").read_bytes()
    ).hexdigest()

    with pytest.raises(FileExistsError, match="already finalized"):
        ExternalRunWriterV1.resume(output, _config())


def test_external_run_resume_rejects_identity_or_row_tampering(tmp_path):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    output = tmp_path / "run"
    writer = ExternalRunWriterV1.create(output, _config())
    writer.append(_row("task-1"))

    changed = _config(runtime_configuration_hash="f" * 64)
    with pytest.raises(ValueError, match="run identity"):
        ExternalRunWriterV1.resume(output, changed)

    task_runs = output / "task_runs.jsonl"
    task_runs.write_bytes(task_runs.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="task-runs"):
        ExternalRunWriterV1.resume(output, _config())


def test_external_run_resume_repairs_only_exact_stale_progress_prefix(tmp_path):
    from mub.vnext.runtime.run_v3 import (
        ExternalRunProgressV1,
        ExternalRunWriterV1,
    )

    output = tmp_path / "run"
    writer = ExternalRunWriterV1.create(output, _config())
    stale_progress = (output / "progress.json").read_bytes()
    writer.append(_row("task-1"))
    (output / "progress.json").write_bytes(stale_progress)

    resumed = ExternalRunWriterV1.resume(output, _config())
    repaired = ExternalRunProgressV1.model_validate_json(
        (output / "progress.json").read_bytes()
    )
    assert repaired.completed_task_ids == ("task-1",)

    with pytest.raises(ValueError, match="progress task IDs"):
        repaired.validated_replace(
            completed_task_ids=("task-2",),
        )
    tampered = ExternalRunProgressV1.model_construct(
        **{
            **repaired.model_dump(mode="python", warnings=False),
            "completed_task_ids": ("task-2",),
        }
    )
    (output / "progress.json").write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="progress"):
        ExternalRunWriterV1.resume(output, _config())


def test_external_run_resume_repairs_precommit_final_progress(tmp_path):
    from mub.vnext.runtime.run_v3 import (
        ExternalRunProgressV1,
        ExternalRunWriterV1,
    )

    output = tmp_path / "run"
    writer = ExternalRunWriterV1.create(output, _config())
    writer.append(_row("task-1"))
    writer.append(_row("task-2"))
    writer._write_progress(finalized=True)
    assert not (output / "run_manifest.json").exists()

    resumed = ExternalRunWriterV1.resume(output, _config())
    progress = ExternalRunProgressV1.model_validate_json(
        (output / "progress.json").read_bytes()
    )
    assert progress.finalized is False
    assert tuple(row.task_id for row in resumed.rows) == (
        "task-1",
        "task-2",
    )


def test_external_run_resume_rejects_extra_or_multilink_artifacts(tmp_path):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    output = tmp_path / "extra"
    ExternalRunWriterV1.create(output, _config())
    (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="run tree"):
        ExternalRunWriterV1.resume(output, _config())

    hardlink_output = tmp_path / "hardlink"
    ExternalRunWriterV1.create(hardlink_output, _config())
    task_runs = hardlink_output / "task_runs.jsonl"
    external = tmp_path / "external-task-runs.jsonl"
    task_runs.replace(external)
    try:
        task_runs.hardlink_to(external)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    with pytest.raises(ValueError, match="single-link"):
        ExternalRunWriterV1.resume(hardlink_output, _config())


def test_external_run_finalize_reauthenticates_identity_artifact(tmp_path):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    output = tmp_path / "run"
    writer = ExternalRunWriterV1.create(output, _config())
    writer.append(_row("task-1"))
    writer.append(_row("task-2"))
    (output / "run_identity.json").write_bytes(b"{}")

    with pytest.raises(ValueError, match="run identity"):
        writer.finalize()
    assert not (output / "run_manifest.json").exists()


def test_external_run_postrow_finalize_failure_leaves_no_manifest(
    tmp_path,
    monkeypatch,
):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    output = tmp_path / "run"
    writer = ExternalRunWriterV1.create(output, _config())
    writer.append(_row("task-1"))
    writer.append(_row("task-2"))
    original_write_progress = writer._write_progress

    def fail_final_progress(*, finalized: bool):
        if finalized:
            raise OSError("injected final progress failure")
        original_write_progress(finalized=finalized)

    monkeypatch.setattr(writer, "_write_progress", fail_final_progress)
    with pytest.raises(OSError, match="final progress"):
        writer.finalize()
    assert not (output / "run_manifest.json").exists()
    assert not tuple(output.glob(".run_manifest.json.tmp-*"))


def test_external_run_manifest_fsync_failure_rolls_back_commit(
    tmp_path,
    monkeypatch,
):
    import mub.vnext.runtime.run_v3 as run_module

    output = tmp_path / "run"
    writer = run_module.ExternalRunWriterV1.create(output, _config())
    writer.append(_row("task-1"))
    writer.append(_row("task-2"))
    original_fsync_directory = run_module._fsync_directory

    def fail_manifest_fsync(path):
        if (path / "run_manifest.json").exists():
            raise OSError("injected manifest fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        run_module,
        "_fsync_directory",
        fail_manifest_fsync,
    )
    with pytest.raises(OSError, match="manifest fsync"):
        writer.finalize()
    assert not (output / "run_manifest.json").exists()
    assert not tuple(output.glob(".run_manifest.json.tmp-*"))


def test_external_run_finalize_failure_leaves_no_manifest_or_temp(
    tmp_path,
    monkeypatch,
):
    import mub.vnext.runtime.run_v3 as run_module

    output = tmp_path / "run"
    writer = run_module.ExternalRunWriterV1.create(output, _config())
    writer.append(_row("task-1"))
    writer.append(_row("task-2"))

    def fail_link(source, destination):
        raise OSError("injected atomic publication failure")

    monkeypatch.setattr(run_module.os, "link", fail_link)
    with pytest.raises(OSError, match="atomic no-replace"):
        writer.finalize()
    assert not (output / "run_manifest.json").exists()
    assert not tuple(output.glob(".run_manifest.json.tmp-*"))


def test_external_run_finalize_rejects_missing_failed_or_partial_rows(tmp_path):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    missing = ExternalRunWriterV1.create(tmp_path / "missing", _config())
    missing.append(_row("task-1"))
    with pytest.raises(ValueError, match="complete ordered coverage"):
        missing.finalize()

    for status in (CompletionStatus.FAILED, CompletionStatus.PARTIAL):
        output = tmp_path / status.value
        writer = ExternalRunWriterV1.create(output, _config())
        writer.append(_row("task-1", status=status))
        writer.append(_row("task-2"))
        with pytest.raises(ValueError, match="FAILED or PARTIAL"):
            writer.finalize()
        assert not (output / "run_manifest.json").exists()


def test_external_run_public_rows_reject_private_paths_state_and_secrets(tmp_path):
    from mub.vnext.runtime.run_v3 import ExternalRunWriterV1

    cases = (
        (_row("task-1", raw_path="private/provider.json"), "private raw paths"),
        (_row("task-1", raw_state={"provider": "raw"}), "raw adapter state"),
    )
    for index, (row, message) in enumerate(cases):
        writer = ExternalRunWriterV1.create(
            tmp_path / f"case-{index}",
            _config(),
        )
        with pytest.raises(ValueError, match=message):
            writer.append(row)

    secret_row = _row("task-1").validated_replace(
        system_events=({"error": "client_secret=hunter2"},)
    )
    writer = ExternalRunWriterV1.create(tmp_path / "secret", _config())
    with pytest.raises(ValueError, match="security scan") as exc_info:
        writer.append(secret_row)
    assert "hunter2" not in str(exc_info.value)


def test_external_run_identity_excludes_output_path_and_binds_configuration():
    from mub.vnext.runtime.run_v3 import compute_external_run_identity

    config = _config()
    first = compute_external_run_identity(config)
    second = compute_external_run_identity(config)
    assert first == second
    assert first != compute_external_run_identity(
        _config(repetition_index=1, repetition_count=3)
    )


def test_external_run_config_requires_redistributable_normalized_license():
    from mub.vnext.external.artifacts import RawPayloadLicenseStatus

    with pytest.raises(ValueError, match="normalized license"):
        _config(
            normalized_license_status=RawPayloadLicenseStatus.PRIVATE
        )


def test_external_run_writer_requires_authenticated_factory(tmp_path):
    from mub.vnext.runtime.run_v3 import (
        ExternalRunWriterV1,
        compute_external_run_identity,
    )

    config = _config()
    with pytest.raises(ValueError, match="create or resume"):
        ExternalRunWriterV1(
            tmp_path / "forged",
            config,
            compute_external_run_identity(config),
            (),
        )


def test_external_run_rejects_constructed_config_and_record(tmp_path):
    from mub.vnext.runtime.run_v3 import (
        ExternalRunConfigV1,
        ExternalRunWriterV1,
    )

    valid = _config()
    forged_config = ExternalRunConfigV1.model_construct(
        **{
            **valid.model_dump(mode="python"),
            "repetition_count": 0,
        }
    )
    with pytest.raises(ValueError, match="trust-boundary"):
        ExternalRunWriterV1.create(tmp_path / "config", forged_config)

    writer = ExternalRunWriterV1.create(tmp_path / "row", valid)
    valid_row = _row("task-1")
    forged_row = TaskRunRecordV3.model_construct(
        **{
            **valid_row.model_dump(mode="python"),
            "completion_status": "completed",
        }
    )
    with pytest.raises(ValueError, match="trust-boundary"):
        writer.append(forged_row)
