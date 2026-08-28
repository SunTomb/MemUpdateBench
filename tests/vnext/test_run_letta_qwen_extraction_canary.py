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
    with pytest.raises(ValueError, match="tree|runtime"):
        verify_model_provenance(snapshot, receipt)


def test_model_runtime_source_receipt_schema_is_accepted(tmp_path: Path, monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_extraction_canary as module
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    tree_hash = module.stable_tree_sha256(snapshot)
    receipt_value = {"schema_version":"memupdatebench.post-core.qwen-runtime-source-receipt.v1","load_status":"PASS","generation_status":"PASS","determinism_status":"PASS","unload_status":"PASS","provider_calls":0,"network_calls":0,"benchmark_generations":0,"gpu_index":3,"node":"Tang-1-Wu"}
    receipt = tmp_path / "runtime.json"
    receipt.write_bytes(canonical_json_bytes(receipt_value))
    monkeypatch.setattr(module, "MODEL_TREE_SHA256", tree_hash)
    monkeypatch.setattr(module, "MODEL_RUNTIME_RECEIPT_SHA256", module.sha256_bytes(receipt.read_bytes()))
    result = module.verify_model_provenance(snapshot, receipt)
    assert result["tree_sha256"] == tree_hash
    assert result["runtime_receipt_sha256"] == module.MODEL_RUNTIME_RECEIPT_SHA256
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


def test_output_root_rejects_frozen_or_symlink_paths(tmp_path: Path) -> None:
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
