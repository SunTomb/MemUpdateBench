from __future__ import annotations

import pytest


_EXPECTED_MANAGER_RUNNER_SHA256 = "2c4eb3656a62b4041de33b8e7b425a6f0550b68190b9d967886df97d46b943d1"


def test_manager_runner_source_hash_is_hardcoded_to_current_runner() -> None:
    import hashlib
    from pathlib import Path

    from scripts import vnext_attest_main_track_factorial_manager as attestor
    from scripts import vnext_run_main_track_factorial_manager as manager_runner

    assert attestor.EXPECTED_MANAGER_RUNNER_SOURCE_SHA256 == _EXPECTED_MANAGER_RUNNER_SHA256
    source_bytes = Path(manager_runner.__file__).read_bytes()
    assert hashlib.sha256(source_bytes.replace(b"\r\n", b"\n")).hexdigest() == (
        attestor.EXPECTED_MANAGER_RUNNER_SOURCE_SHA256
    )


def test_attestation_schema_is_stable() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    assert attestor.RECEIPT_SCHEMA == "memupdatebench.main-track.manager-fixture-execution-receipt.v1"


def _snapshot(attestor):
    return {
        "fixture_hashes": {name: chr(97 + index) * 64 for index, name in enumerate(attestor._ARTIFACTS)},
        "fixture_root_digest": "d" * 64,
        "manifest_sha256": "e" * 64,
        "candidate_hashes": {"release_index.json": "f" * 64, "tasks.jsonl": "0" * 64},
        "candidate_release_index_sha256": "1" * 64,
        "audit_sha256": "2" * 64,
        "cell_id": "letta_profile__qwen35_answer",
        "manager_kind": "letta",
        "manager_id": "letta_0_16_8_block_profile",
        "scope": "full240",
        "runner_source_sha256": _EXPECTED_MANAGER_RUNNER_SHA256,
        "execution_boundary": {
            "provider_calls": 0,
            "model_loads": 1,
            "database_accesses": 0,
            "network_calls": 0,
            "gpu_calls": 8,
            "executable_calls": 0,
            "remote_operations": 0,
        },
    }


def test_receipt_binds_hashes_and_producer_without_paths() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    snapshot = _snapshot(attestor)
    receipt = attestor._build_receipt(
        snapshot,
        producer_id="letta-production-fixture",
        producer_revision="rev-1",
        runtime_identity={"system": "offline", "runtime": "fixture-v1"},
        allow_relocated_authenticated_inputs=True,
    )
    assert receipt["candidate_artifact_hashes"] == snapshot["candidate_hashes"]
    assert receipt["manager_fixture_artifact_hashes"] == snapshot["fixture_hashes"]
    assert receipt["manager_fixture_root_digest"] == snapshot["fixture_root_digest"]
    assert receipt["execution_boundary"] == snapshot["execution_boundary"]
    assert receipt["producer"] == {
        "producer_id": "letta-production-fixture",
        "producer_revision": "rev-1",
        "runtime_identity": {"system": "offline", "runtime": "fixture-v1"},
        "manager_id": snapshot["manager_id"],
        "cell_id": snapshot["cell_id"],
    }
    assert receipt["producer_source_sha256"] == _EXPECTED_MANAGER_RUNNER_SHA256
    assert attestor.receipt_sha256(receipt) == attestor.sha256_bytes(
        attestor.canonical_json_bytes(receipt)
    )


def test_receipt_records_normalized_runner_source_hash_for_portable_checkout() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    snapshot = _snapshot(attestor)
    snapshot["runner_source_sha256"] = "a" * 64
    snapshot["runner_source_sha256_normalized"] = _EXPECTED_MANAGER_RUNNER_SHA256
    receipt = attestor._build_receipt(
        snapshot,
        producer_id="producer",
        producer_revision="revision",
        runtime_identity="fixture-runtime",
        allow_relocated_authenticated_inputs=False,
    )
    assert receipt["runner_source_sha256"] == "a" * 64
    assert receipt["runner_source_sha256_normalized"] == _EXPECTED_MANAGER_RUNNER_SHA256
    assert receipt["producer_source_sha256"] == _EXPECTED_MANAGER_RUNNER_SHA256



def test_incomplete_producer_is_rejected_before_fixture_validation(tmp_path, monkeypatch) -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    calls = []
    monkeypatch.setattr(attestor, "_validate_snapshot", lambda **_: calls.append("snapshot"))
    with __import__("pytest").raises(ValueError, match="producer_id"):
        attestor.run(
            manager_fixture_root=tmp_path / "fixture",
            candidate_root=tmp_path / "candidate",
            audit_attestation=tmp_path / "audit.json",
            manager_kind="letta",
            cell_id="letta_profile__qwen35_answer",
            producer_id="",
            producer_revision="rev-1",
            runtime_identity="fixture-runtime",
            output_receipt=tmp_path / "receipt.json",
        )
    assert calls == []


def test_runtime_identity_absolute_path_is_rejected(tmp_path) -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    with pytest.raises(ValueError, match="absolute path"):
        attestor._build_receipt(
            _snapshot(attestor),
            producer_id="producer",
            producer_revision="revision",
            runtime_identity={"python_executable": "C:\\\\private\\python.exe"},
            allow_relocated_authenticated_inputs=False,
        )


def test_publication_is_no_replace_and_emits_exact_receipt(tmp_path, monkeypatch) -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for name in attestor._ARTIFACTS:
        (fixture / name).write_bytes(b"fixture")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}")
    monkeypatch.setattr(attestor, "_validate_snapshot", lambda **_: _snapshot(attestor))
    output = tmp_path / "receipt.json"
    receipt = attestor.run(
        manager_fixture_root=fixture,
        manifest=manifest,
        candidate_root=candidate,
        audit_attestation=audit,
        manager_kind="letta",
        cell_id="letta_profile__qwen35_answer",
        producer_id="producer",
        producer_revision="revision",
        runtime_identity="fixture-runtime",
        output_receipt=output,
    )
    raw = output.read_bytes()
    assert raw == attestor.canonical_json_bytes(receipt)
    assert attestor.sha256_bytes(raw) == attestor.receipt_sha256(receipt)
    with __import__("pytest").raises(FileExistsError):
        attestor.run(
            manager_fixture_root=fixture,
            manifest=manifest,
            candidate_root=candidate,
            audit_attestation=audit,
            manager_kind="letta",
            cell_id="letta_profile__qwen35_answer",
            producer_id="producer",
            producer_revision="revision",
            runtime_identity="fixture-runtime",
            output_receipt=output,
        )


def test_prepublication_tamper_leaves_no_receipt(tmp_path, monkeypatch) -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for name in attestor._ARTIFACTS:
        (fixture / name).write_bytes(b"fixture")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}")
    snapshots = [_snapshot(attestor), {**_snapshot(attestor), "fixture_root_digest": "4" * 64}]
    monkeypatch.setattr(attestor, "_validate_snapshot", lambda **_: snapshots.pop(0))
    output = tmp_path / "nested" / "receipt.json"
    with pytest.raises(ValueError, match="changed before publication"):
        attestor.run(
            manager_fixture_root=fixture,
            manifest=manifest,
            candidate_root=candidate,
            audit_attestation=audit,
            manager_kind="letta",
            cell_id="letta_profile__qwen35_answer",
            producer_id="producer",
            producer_revision="revision",
            runtime_identity="fixture-runtime",
            output_receipt=output,
        )
    assert not output.exists()
    assert not output.parent.exists()


def test_output_root_uses_stable_receipt_name_and_cli_parses_identity(tmp_path) -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    parser = attestor.build_arg_parser()
    args = parser.parse_args(
        [
            "--manager-fixture-root", "fixture",
            "--manifest", "manifest.json",
            "--candidate-root", "candidate",
            "--audit-attestation", "audit.json",
            "--manager-kind", "letta",
            "--cell-id", "letta_profile__qwen35_answer",
            "--producer-id", "producer",
            "--producer-revision", "revision",
            "--runtime-identity", '{"runtime":"fixture"}',
            "--output-root", str(tmp_path / "out"),
        ]
    )
    assert args.runtime_identity == {"runtime": "fixture"}
    assert args.output_root == tmp_path / "out"


def _summary_and_index(attestor, snapshot):
    selected = [f"task-{index}" for index in range(240)]
    selected_hash = attestor.sha256_bytes(attestor.canonical_json_bytes(selected))
    summary = {
        "status": "PASS",
        "execution_mode": attestor.EXECUTION_MODE,
        "evidence_class": attestor.EVIDENCE_CLASS,
        "scientific_evidence": True,
        "scope": "full240",
        "cell_id": snapshot["cell_id"],
        "manager_kind": snapshot["manager_kind"],
        "manager_id": snapshot["manager_id"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "candidate_artifact_hashes": dict(snapshot["candidate_hashes"]),
        "audit_attestation_sha256": snapshot["audit_sha256"],
        "failed": 0,
        "supported": 240,
        "unsupported": 480,
        "eligible_supported_count": 240,
        "executed_supported_count": 240,
        "not_requested_supported_count": 0,
        "requested_task_count": 720,
        "terminal_rows": 720,
        "execution_boundary_observed": True,
        "execution_accounting_observed": True,
        "execution_accounting_source": "production_runtime_profile",
        "runner_source_sha256": snapshot["runner_source_sha256"],
        "selected_supported_task_ids": selected,
        "selected_supported_task_ids_sha256": selected_hash,
        "execution_boundary": dict(snapshot["execution_boundary"]),
    }
    index = {
        "status": "PASS",
        "execution_mode": attestor.EXECUTION_MODE,
        "evidence_class": attestor.EVIDENCE_CLASS,
        "scientific_evidence": True,
        "scope": "full240",
        "cell_id": snapshot["cell_id"],
        "manager_kind": snapshot["manager_kind"],
        "manager_id": snapshot["manager_id"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "candidate_artifact_hashes": dict(snapshot["candidate_hashes"]),
        "audit_attestation_sha256": snapshot["audit_sha256"],
        "execution_boundary_observed": True,
        "runner_source_sha256": snapshot["runner_source_sha256"],
        "execution_boundary": dict(snapshot["execution_boundary"]),
        "execution_accounting_observed": True,
        "execution_accounting_source": "production_runtime_profile",
        "executed_supported_count": 240,
        "unsupported_count": 480,
        "selected_supported_task_ids": selected,
        "selected_supported_task_ids_sha256": selected_hash,
    }
    return summary, index


def test_summary_contract_rejects_nonproduction_and_incomplete_counts() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    snapshot = _snapshot(attestor)
    for field, value in (
        ("status", "FAIL"),
        ("execution_mode", "injected_test_only"),
        ("evidence_class", "manager_state_retrieval_fixture_test_only"),
        ("scientific_evidence", False),
        ("scope", "canary8"),
        ("failed", 1),
        ("supported", 239),
        ("unsupported", 481),
    ):
        summary, index = _summary_and_index(attestor, snapshot)
        summary[field] = value
        with pytest.raises(attestor.AttestationError):
            attestor._validate_summary_and_index(
                summary,
                index,
                expected_scope="full240",
                expected_supported_count=240,
                expected_unsupported_count=480,
                manager_kind=snapshot["manager_kind"],
                manager_id=snapshot["manager_id"],
                cell_id=snapshot["cell_id"],
                manifest_sha256=snapshot["manifest_sha256"],
                candidate_hashes=snapshot["candidate_hashes"],
                audit_sha=snapshot["audit_sha256"],
                runner_source_sha256=snapshot["runner_source_sha256"],
            )


def test_summary_contract_rejects_runner_source_hash_mismatch() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    snapshot = _snapshot(attestor)
    summary, index = _summary_and_index(attestor, snapshot)
    summary["runner_source_sha256"] = "4" * 64
    with pytest.raises(attestor.AttestationError, match="runner_source_sha256"):
        attestor._validate_summary_and_index(
            summary,
            index,
            expected_scope="full240",
            expected_supported_count=240,
            expected_unsupported_count=480,
            manager_kind=snapshot["manager_kind"],
            manager_id=snapshot["manager_id"],
            cell_id=snapshot["cell_id"],
            manifest_sha256=snapshot["manifest_sha256"],
            candidate_hashes=snapshot["candidate_hashes"],
            audit_sha=snapshot["audit_sha256"],
            runner_source_sha256=snapshot["runner_source_sha256"],
        )


def test_root_digest_and_receipt_are_compatible_with_answer_runner(tmp_path) -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    from scripts import vnext_run_main_track_factorial_answer as answer_runner

    snapshot = _snapshot(attestor)
    snapshot = {
        **snapshot,
        "fixture_root_digest": answer_runner._fixture_root_digest(snapshot["fixture_hashes"]),
    }
    receipt = attestor._build_receipt(
        snapshot,
        producer_id="producer",
        producer_revision="revision",
        runtime_identity={"runtime": "fixture"},
        allow_relocated_authenticated_inputs=False,
    )
    assert receipt["manager_fixture_root_digest"] == answer_runner._fixture_root_digest(
        snapshot["fixture_hashes"]
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(attestor.canonical_json_bytes(receipt))
    result = answer_runner._validate_manager_fixture_attestation(
        receipt_path,
        manager_fixture_attestation_sha256=attestor.receipt_sha256(receipt),
        fixture_root=tmp_path / "fixture",
        fixture_hashes=snapshot["fixture_hashes"],
        summary={
            "evidence_class": attestor.EVIDENCE_CLASS,
            "cell_id": snapshot["cell_id"],
            "scope": snapshot["scope"],
            "manager_id": snapshot["manager_id"],
        },
        index={},
        manifest_sha256=snapshot["manifest_sha256"],
        candidate_hashes=snapshot["candidate_hashes"],
        audit_sha=snapshot["audit_sha256"],
    )
    assert result["receipt_sha256"] == attestor.receipt_sha256(receipt)
    assert result["producer_id"] == "producer"


def test_execution_boundary_rejects_provider_network_database_and_remote_effects() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    snapshot = _snapshot(attestor)
    for key in ("provider_calls", "network_calls", "database_accesses", "remote_operations"):
        summary, index = _summary_and_index(attestor, snapshot)
        summary["execution_boundary"][key] = 1
        index["execution_boundary"][key] = 1
        with pytest.raises(attestor.AttestationError, match=key):
            attestor._validate_summary_and_index(
                summary,
                index,
                expected_scope="full240",
                expected_supported_count=240,
                expected_unsupported_count=480,
                manager_kind=snapshot["manager_kind"],
                manager_id=snapshot["manager_id"],
                cell_id=snapshot["cell_id"],
                manifest_sha256=snapshot["manifest_sha256"],
                candidate_hashes=snapshot["candidate_hashes"],
                audit_sha=snapshot["audit_sha256"],
                runner_source_sha256=snapshot["runner_source_sha256"],
            )


def test_receipt_requires_nonblank_snapshot_manager_and_cell_identity() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    for field in ("manager_id", "cell_id"):
        snapshot = _snapshot(attestor)
        snapshot[field] = ""
        with pytest.raises(attestor.AttestationError, match=field):
            attestor._build_receipt(
                snapshot,
                producer_id="producer",
                producer_revision="revision",
                runtime_identity="fixture-runtime",
                allow_relocated_authenticated_inputs=False,
            )


def test_output_target_rejects_frozen_immutable_roots() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    for root_name in ("core/v3", "pilot"):
        frozen_root = attestor.ROOT / "data" / "vnext" / root_name
        with pytest.raises(attestor.AttestationError, match="frozen immutable"):
            attestor._validate_output_target(frozen_root / "receipt.json", ())


def test_execution_boundary_allows_nonnegative_local_model_and_compute_counts() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    boundary = _snapshot(attestor)["execution_boundary"]
    boundary.update({"model_loads": 2, "gpu_calls": 3, "executable_calls": 4})
    assert attestor._validate_execution_boundary(boundary) == boundary


def test_runtime_identity_credential_url_is_rejected() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    with pytest.raises(ValueError, match="credential URL"):
        attestor._build_receipt(
            _snapshot(attestor),
            producer_id="producer",
            producer_revision="revision",
            runtime_identity={"endpoint": "https://user:password@example.test/api"},
            allow_relocated_authenticated_inputs=False,
        )


def test_output_cannot_be_published_inside_frozen_fixture_root(tmp_path, monkeypatch) -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor
    import pytest

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for name in attestor._ARTIFACTS:
        (fixture / name).write_bytes(b"fixture")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}")
    monkeypatch.setattr(attestor, "_validate_snapshot", lambda **_: _snapshot(attestor))
    with pytest.raises(attestor.AttestationError, match="separate from inputs"):
        attestor.run(
            manager_fixture_root=fixture,
            manifest=manifest,
            candidate_root=candidate,
            audit_attestation=audit,
            manager_kind="letta",
            cell_id="letta_profile__qwen35_answer",
            producer_id="producer",
            producer_revision="revision",
            runtime_identity="fixture-runtime",
            output_receipt=fixture / "receipt.json",
        )
    assert not (fixture / "receipt.json").exists()


def test_injected_fixture_cannot_be_attested_as_production(tmp_path) -> None:
    import pytest

    from tests.vnext import test_main_track_factorial_manager as manager_tests
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    manager_tests._run(tmp_path)
    fixture = tmp_path / "out"
    with pytest.raises(ValueError, match="execution mode binding"):
        attestor.run(
            manager_fixture_root=fixture,
            manifest=manager_tests.MANIFEST,
            candidate_root=manager_tests.CANDIDATE,
            audit_attestation=manager_tests.ATTESTATION,
            manager_kind="letta",
            cell_id="letta_profile__qwen35_answer",
            producer_id="fake-production-fixture",
            producer_revision="test-revision",
            runtime_identity="fake-runtime",
            output_receipt=tmp_path / "receipt.json",
        )
    assert not (tmp_path / "receipt.json").exists()


def test_execution_accounting_requires_production_values_and_exact_summary_index_equality() -> None:
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    snapshot = _snapshot(attestor)
    for field, summary_value, index_value in (
        ("execution_accounting_observed", False, False),
        ("execution_accounting_observed", True, False),
        ("execution_accounting_source", "injected_or_unavailable", "injected_or_unavailable"),
        ("execution_accounting_source", "production_runtime_profile", "injected_or_unavailable"),
    ):
        summary, index = _summary_and_index(attestor, snapshot)
        summary[field] = summary_value
        index[field] = index_value
        with pytest.raises(attestor.AttestationError, match="execution_accounting"):
            attestor._validate_summary_and_index(
                summary,
                index,
                expected_scope="full240",
                expected_supported_count=240,
                expected_unsupported_count=480,
                manager_kind=snapshot["manager_kind"],
                manager_id=snapshot["manager_id"],
                cell_id=snapshot["cell_id"],
                manifest_sha256=snapshot["manifest_sha256"],
                candidate_hashes=snapshot["candidate_hashes"],
                audit_sha=snapshot["audit_sha256"],
                runner_source_sha256=snapshot["runner_source_sha256"],
            )


def test_relabelled_injected_fixture_cannot_be_attested_as_production(tmp_path) -> None:
    import json
    import shutil

    from tests.vnext import test_main_track_factorial_manager as manager_tests
    from scripts import vnext_attest_main_track_factorial_manager as attestor

    manager_tests._run(tmp_path)
    injected = tmp_path / "out"
    fixture = tmp_path / "relabeled"
    shutil.copytree(injected, fixture)
    summary_path = fixture / "manager_summary.json"
    index_path = fixture / "artifact_index.json"
    summary = json.loads(summary_path.read_bytes())
    index = json.loads(index_path.read_bytes())
    production_boundary = dict(summary["execution_boundary"])
    production_boundary.update({"model_loads": 1, "gpu_calls": 1})
    summary.update(
        {
            "execution_mode": "production",
            "evidence_class": attestor.EVIDENCE_CLASS,
            "scientific_evidence": True,
            "execution_boundary_observed": True,
            "execution_boundary": production_boundary,
        }
    )
    index.update(
        {
            "execution_mode": "production",
            "evidence_class": attestor.EVIDENCE_CLASS,
            "scientific_evidence": True,
            "execution_boundary_observed": True,
            "execution_boundary": production_boundary,
        }
    )
    summary_raw = attestor.canonical_json_bytes(summary)
    summary_path.write_bytes(summary_raw)
    index["artifacts"]["manager_summary.json"].update(
        {"sha256": attestor.sha256_bytes(summary_raw), "bytes": len(summary_raw)}
    )
    index_path.write_bytes(attestor.canonical_json_bytes(index))

    with pytest.raises(attestor.AttestationError, match="execution_accounting"):
        attestor.run(
            manager_fixture_root=fixture,
            manifest=manager_tests.MANIFEST,
            candidate_root=manager_tests.CANDIDATE,
            audit_attestation=manager_tests.ATTESTATION,
            manager_kind="letta",
            cell_id="letta_profile__qwen35_answer",
            producer_id="fake-production-fixture",
            producer_revision="test-revision",
            runtime_identity="fake-runtime",
            output_receipt=tmp_path / "receipt.json",
        )
    assert not (tmp_path / "receipt.json").exists()
