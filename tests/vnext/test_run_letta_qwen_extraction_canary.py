from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.vnext_run_letta_qwen_extraction_canary import (
    MODEL_ID,
    MODEL_REVISION,
    TASK_SHA256,
    build_extraction_prompt,
    canonical_json_bytes,
    safe_worker_environment,
    validate_loopback_url,
    validate_qualification_artifacts,
)


def _binding_for_snapshot(module, snapshot: Path, *, tree_hash: str = "e" * 64) -> dict:
    payload = (snapshot / "config.json").read_bytes()
    binding = {
        "schema_version": "memupdatebench.post-core.shared-snapshot-binding.v1",
        "repo": MODEL_ID,
        "revision": MODEL_REVISION,
        "shared_snapshot_path": "/NAS/HuggingFaceModels/Qwen3.5-9B",
        "tree_sha256": tree_hash,
        "file_count": 1,
        "total_bytes": len(payload),
        "entries": [{"path": "config.json", "sha256": module.sha256_bytes(payload), "bytes": len(payload)}],
    }
    binding["receipt_payload_sha256"] = module.sha256_bytes(module.canonical_json_bytes(binding))
    return binding


def test_extraction_prompt_contains_only_visible_text_and_admitted_attribute() -> None:
    prompt = build_extraction_prompt("Alice lives in Paris.", "city")
    assert "Alice lives in Paris." in prompt
    assert '"city"' in prompt
    assert "gold" not in prompt.lower()
    assert "object_id" not in prompt
    assert "normalized" not in prompt.lower()


def test_loopback_url_rejects_credentials_paths_and_https() -> None:
    assert validate_loopback_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    for url in ("https://127.0.0.1:8000", "http://10.0.0.1:8000", "http://user@127.0.0.1:8000", "http://127.0.0.1:8000/v1"):
        with pytest.raises(ValueError):
            validate_loopback_url(url)


def test_worker_environment_forwards_only_validated_runtime_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LETTA_TOKEN", "must-not-forward")
    monkeypatch.setenv("PGPASSWORD", "must-not-forward")
    env = safe_worker_environment("http://127.0.0.1:8000", tmp_path)
    assert env["LETTA_NATIVE_API_BASE_URL"] == "http://127.0.0.1:8000"
    assert env["PYTHONPATH"] == str(tmp_path.resolve())
    assert "LETTA_TOKEN" not in env and "PGPASSWORD" not in env


def test_qualification_validation_requires_passed_triplet(tmp_path: Path) -> None:
    closure = {"outcome": "PASS", "identity": {"package_name": "letta", "package_version": "0.16.8", "source_commit": "1131535716e8a31c9a437f8695e25ac98f203a24"}, "runtime": {"loopback_only": True}}
    preflight = {"outcome": "pass"}
    admission = {"outcome": "pass", "admitted": True}
    for name, value in (("letta_runtime_qualification.json", closure), ("letta_runtime_preflight.json", preflight), ("letta_runtime_admission.json", admission)):
        (tmp_path / name).write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError):
        validate_qualification_artifacts(tmp_path)


def test_model_provenance_constants_are_frozen() -> None:
    assert MODEL_ID == "Qwen/Qwen3.5-9B"
    assert MODEL_REVISION == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert len(TASK_SHA256) == 64


def test_qualification_validation_rejects_minimal_forged_triplet(tmp_path: Path) -> None:
    closure = {
        "schema_version": "memupdatebench.external.letta.runtime_qualification.v1",
        "candidate_id": "letta_0_16_8_song1_local_linux",
        "mode": "direct_block_profile",
        "outcome": "PASS",
        "identity": {"package_name": "letta", "package_version": "0.16.8", "source_commit": "1131535716e8a31c9a437f8695e25ac98f203a24"},
        "runtime": {"loopback_only": True},
        "boundary": {"llm_used": False, "api_used": False, "gpu_used": False, "network_credential_inputs": False},
        "cleanup": {"status": "PASS"},
    }
    preflight = {"schema_version": "memupdatebench.external.letta.preflight.v2", "candidate_id": "letta_0_16_8_profile", "mode": "profile_single_record_runtime", "outcome": "pass", "passed": True}
    admission = {"schema_version": "memupdatebench.external.letta.admission.v2", "candidate_id": "letta_0_16_8_profile", "outcome": "pass", "admitted": True}
    for name, value in (("letta_runtime_qualification.json", closure), ("letta_runtime_preflight.json", preflight), ("letta_runtime_admission.json", admission)):
        (tmp_path / name).write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="hash|identity|runtime|qualification"):
        validate_qualification_artifacts(tmp_path)


def test_validate_extraction_value_semantics() -> None:
    from scripts.vnext_run_letta_qwen_extraction_canary import validate_extraction
    for value in ("Paris", 3, 2.5, True):
        assert validate_extraction({"operation": "add", "value": value})["value"] == value
    for payload in ({"operation": "add", "value": None}, {"operation": "update", "value": {}}, {"operation": "update", "value": float("nan")}, {"operation": "noop", "value": "x"}, {"operation": "delete", "value": 1}):
        with pytest.raises(ValueError):
            validate_extraction(payload)


def test_model_snapshot_requires_exact_tree_and_runtime_receipt(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_qwen_extraction_canary import verify_model_provenance
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    receipt = tmp_path / "runtime.json"
    receipt.write_bytes(canonical_json_bytes({"model_id": MODEL_ID, "revision": MODEL_REVISION, "tree_sha256": "bad", "runtime_identity": "qwen"}))
    with pytest.raises(ValueError, match="binding"):
        verify_model_provenance(snapshot, receipt)


def test_model_runtime_source_receipt_schema_is_accepted(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    payload = b"{}"
    (snapshot / "config.json").write_bytes(payload)
    tree_hash = module.stable_tree_sha256(snapshot)
    binding = {
        "schema_version": "memupdatebench.post-core.shared-snapshot-binding.v1",
        "repo": MODEL_ID,
        "revision": MODEL_REVISION,
        "shared_snapshot_path": "/NAS/HuggingFaceModels/Qwen3.5-9B",
        "tree_sha256": tree_hash,
        "file_count": 1,
        "total_bytes": len(payload),
        "entries": [{"path": "config.json", "sha256": module.sha256_bytes(payload), "bytes": len(payload)}],
    }
    binding["receipt_payload_sha256"] = module.sha256_bytes(module.canonical_json_bytes(binding))
    receipt_value = {"schema_version":"memupdatebench.post-core.qwen-runtime-source-receipt.v1","load_status":"PASS","generation_status":"PASS","determinism_status":"PASS","unload_status":"PASS","provider_calls":0,"network_calls":0,"benchmark_generations":0,"gpu_index":3,"node":"Tang-1-Wu"}
    receipt = tmp_path / "runtime.json"
    receipt.write_bytes(canonical_json_bytes(receipt_value))
    monkeypatch.setattr(module, "MODEL_TREE_SHA256", tree_hash)
    monkeypatch.setattr(module, "MODEL_RUNTIME_RECEIPT_SHA256", module.sha256_bytes(receipt.read_bytes()))
    monkeypatch.setattr(module, "snapshot_tree_sha256_v3", lambda **_: pytest.fail("authoritative binding must supply tree hash"))
    binding_raw = canonical_json_bytes(binding)
    binding_receipt = tmp_path / "binding.json"
    binding_receipt.write_bytes(binding_raw)
    result = module.verify_model_provenance(
        snapshot,
        receipt,
        binding,
        binding_raw=binding_raw,
        binding_path=binding_receipt,
    )
    assert result["tree_sha256"] == tree_hash
    assert result["runtime_receipt_sha256"] == module.MODEL_RUNTIME_RECEIPT_SHA256
    assert result["snapshot_binding"]["repo"] == MODEL_ID
    assert result["snapshot_binding"]["receipt_payload_sha256"] == binding["receipt_payload_sha256"]
    assert result["snapshot_binding_receipt_sha256"] == module.sha256_bytes(binding_raw)
    assert result["snapshot_binding_payload_sha256"] == binding["receipt_payload_sha256"]
    assert result["repo"] == MODEL_ID
    assert result["revision"] == MODEL_REVISION
    assert result["shared_snapshot_path"] == "/NAS/HuggingFaceModels/Qwen3.5-9B"
    assert result["file_count"] == 1
    assert result["total_bytes"] == len(payload)
def test_snapshot_hash_delegates_to_canonical_helper(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    observed = {}
    def canonical(**kwargs):
        observed.update(kwargs)
        return "f" * 64
    monkeypatch.setattr(module, "snapshot_tree_sha256_v3", canonical)
    assert module.stable_tree_sha256(snapshot) == "f" * 64
    assert observed["model_id"] == MODEL_ID and observed["revision"] == MODEL_REVISION


def test_snapshot_binding_accepts_authoritative_receipt_audit_metadata(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    payload = b"model-bytes"
    (snapshot / "config.json").write_bytes(payload)
    binding = _binding_for_snapshot(module, snapshot)
    binding.update(
        {
            "available_bytes_after": 90,
            "available_bytes_before": 100,
            "model_loads": 0,
            "operation_id": "cleanup-qwen-v1",
            "provider_calls": 0,
            "remaining_stage_roots": ["/NAS/staging/qwen"],
            "removed_allocated_bytes": 10,
            "removed_duplicate_path": "/NAS/staging/qwen-duplicate",
            "removed_logical_bytes": 10,
        }
    )
    binding["receipt_payload_sha256"] = module.sha256_bytes(
        module.canonical_json_bytes(
            {key: value for key, value in binding.items() if key != "receipt_payload_sha256"}
        )
    )

    provenance = module.validate_snapshot_binding(snapshot, binding)

    assert provenance["repo"] == MODEL_ID
    assert provenance["revision"] == MODEL_REVISION
    assert provenance["tree_sha256"] == "e" * 64
    assert provenance["file_count"] == 1


def test_snapshot_binding_rejects_audit_metadata_path_traversal(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"model-bytes")
    binding = _binding_for_snapshot(module, snapshot)
    binding["removed_duplicate_path"] = r"..\secret"
    binding["receipt_payload_sha256"] = module.sha256_bytes(
        module.canonical_json_bytes(
            {key: value for key, value in binding.items() if key != "receipt_payload_sha256"}
        )
    )

    with pytest.raises(ValueError, match="path traversal"):
        module.validate_snapshot_binding(snapshot, binding)


def test_snapshot_binding_accepts_authoritative_entry_source_metadata(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"model-bytes")
    binding = _binding_for_snapshot(module, snapshot)
    binding["entries"][0].update(
        {
            "source_digest": "a" * 40,
            "source_digest_kind": "git",
        }
    )
    binding["receipt_payload_sha256"] = module.sha256_bytes(
        module.canonical_json_bytes(
            {key: value for key, value in binding.items() if key != "receipt_payload_sha256"}
        )
    )

    assert module.validate_snapshot_binding(snapshot, binding)["file_count"] == 1


def test_snapshot_binding_rejects_invalid_entry_source_metadata(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"model-bytes")
    binding = _binding_for_snapshot(module, snapshot)
    binding["entries"][0].update(
        {
            "source_digest": "a" * 40,
            "source_digest_kind": [],
        }
    )
    binding["receipt_payload_sha256"] = module.sha256_bytes(
        module.canonical_json_bytes(
            {key: value for key, value in binding.items() if key != "receipt_payload_sha256"}
        )
    )

    with pytest.raises(ValueError, match="source_digest_kind"):
        module.validate_snapshot_binding(snapshot, binding)


def test_snapshot_binding_receipt_validates_authoritative_entries_shape(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    payload = b"model-bytes"
    (snapshot / "config.json").write_bytes(payload)
    digest = module.sha256_bytes(payload)
    binding = {
        "schema_version": "memupdatebench.post-core.shared-snapshot-binding.v1",
        "repo": MODEL_ID,
        "revision": MODEL_REVISION,
        "shared_snapshot_path": "/NAS/HuggingFaceModels/Qwen3.5-9B",
        "tree_sha256": "e" * 64,
        "file_count": 1,
        "total_bytes": len(payload),
        "entries": [{"path": "config.json", "sha256": digest, "bytes": len(payload)}],
    }
    binding["receipt_payload_sha256"] = module.sha256_bytes(
        module.canonical_json_bytes(binding)
    )
    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    assert module.validate_snapshot_binding(snapshot, binding)["file_count"] == 1


def test_snapshot_binding_rejects_duplicate_entries_and_extra_snapshot_files(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"model-bytes")
    binding = _binding_for_snapshot(module, snapshot)
    binding["entries"].append(dict(binding["entries"][0]))
    binding["file_count"] = 2
    binding["total_bytes"] *= 2
    binding["receipt_payload_sha256"] = module.sha256_bytes(module.canonical_json_bytes({key: value for key, value in binding.items() if key != "receipt_payload_sha256"}))
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_snapshot_binding(snapshot, binding)

    binding = _binding_for_snapshot(module, snapshot)
    (snapshot / "extra.json").write_bytes(b"extra")
    with pytest.raises(ValueError, match="exactly match"):
        module.validate_snapshot_binding(snapshot, binding)


def test_snapshot_binding_rejects_payload_and_entry_shape_tampering(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"model-bytes")
    binding = _binding_for_snapshot(module, snapshot)
    binding["receipt_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload hash"):
        module.validate_snapshot_binding(snapshot, binding)

    binding = _binding_for_snapshot(module, snapshot)
    binding["entries"][0]["size_bytes"] = binding["entries"][0]["bytes"]
    binding["receipt_payload_sha256"] = module.sha256_bytes(module.canonical_json_bytes({key: value for key, value in binding.items() if key != "receipt_payload_sha256"}))
    with pytest.raises(ValueError, match="entry fields"):
        module.validate_snapshot_binding(snapshot, binding)


def test_snapshot_binding_rejects_snapshot_symlink(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    monkeypatch.setattr(module, "MODEL_TREE_SHA256", "e" * 64)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"model-bytes")
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    link = snapshot / "link.bin"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink|reparse"):
        module.validate_snapshot_binding(snapshot, _binding_for_snapshot(module, snapshot))


    from scripts.vnext_run_letta_qwen_extraction_canary import validate_output_root
    frozen = tmp_path / "core"
    frozen.mkdir()
    with pytest.raises(ValueError, match="frozen|overlap"):
        validate_output_root(frozen, frozen_roots=(frozen,))
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink|reparse"):
        validate_output_root(link, frozen_roots=())
