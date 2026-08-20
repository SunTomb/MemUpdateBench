from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mub.vnext.release.task14_sources import (
    TASK14_EXPECTED_FILE_HASHES,
    TASK14_EXPECTED_TASK13_INDEX_SHA256,
    Task14SourcePathsV1,
    load_task14_sources_v1,
    revalidate_task14_sources_v1,
    snapshot_task14_root_v1,
)


REPOSITORY = Path(__file__).resolve().parents[2]
CORE = REPOSITORY / "data" / "vnext" / "core" / "v3"
EVIDENCE = REPOSITORY / "results" / "vnext" / "core_task14_evidence"
TASK13 = REPOSITORY / "results" / "vnext" / "core_task13_bc82566_v1"
TASK13_AUDIT = REPOSITORY / "results" / "vnext" / "core_task13_bc82566_v1_audit.json"
REMOTE_STAGE = "/NAS/yesh/MemUpdateBench/results/vnext/.mub-task13-stage-1a791f4cbfdd471aa6a8bd45ab6432d4"


def paths(*, evidence: Path = EVIDENCE, remote: str = REMOTE_STAGE) -> Task14SourcePathsV1:
    return Task14SourcePathsV1(
        core_root=CORE,
        evidence_root=evidence,
        task13_root=TASK13,
        task13_audit_path=TASK13_AUDIT,
        repository_root=REPOSITORY,
        remote_task13_staging_path=remote,
    )


def test_real_task14_source_inventory_closes_all_frozen_hashes() -> None:
    loaded = load_task14_sources_v1(paths())
    assert set(loaded.artifacts) == set(TASK14_EXPECTED_FILE_HASHES)
    assert loaded.artifacts["task13/task13_artifact_index.json"].sha256 == TASK14_EXPECTED_TASK13_INDEX_SHA256
    assert len(loaded.root_snapshots) == 3
    assert revalidate_task14_sources_v1(loaded)


def test_revalidation_rejects_mutated_loaded_payloads() -> None:
    loaded = load_task14_sources_v1(paths())
    loaded.json_payloads["task13_audit/core_task13_bc82566_v1_audit.json"][
        "status"
    ] = "forged"
    assert not revalidate_task14_sources_v1(loaded)


def test_source_inventory_rejects_hash_tamper(tmp_path: Path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    target = copied / "task10" / "admission_decision.json"
    raw = target.read_bytes()
    target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_task14_sources_v1(paths(evidence=copied))


def test_source_inventory_rejects_root_and_file_aliases(tmp_path: Path) -> None:
    duplicated_roots = Task14SourcePathsV1(
        core_root=CORE,
        evidence_root=CORE,
        task13_root=TASK13,
        task13_audit_path=TASK13_AUDIT,
        repository_root=REPOSITORY,
        remote_task13_staging_path=REMOTE_STAGE,
    )
    with pytest.raises(ValueError, match="aliases"):
        load_task14_sources_v1(duplicated_roots)

    overlapping_audit = Task14SourcePathsV1(
        core_root=CORE,
        evidence_root=EVIDENCE,
        task13_root=TASK13,
        task13_audit_path=TASK13 / "statistics_receipt.json",
        repository_root=REPOSITORY,
        remote_task13_staging_path=REMOTE_STAGE,
    )
    with pytest.raises(ValueError, match="overlaps"):
        load_task14_sources_v1(overlapping_audit)


def test_remote_nfs_staging_cannot_be_relabeled_final() -> None:
    with pytest.raises(ValueError, match="NFS staging"):
        load_task14_sources_v1(paths(remote="results/vnext/core_task13_bc82566_v1"))


def test_root_snapshot_changes_on_same_size_replacement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "evidence.json"
    source.write_bytes(b"old!")
    before = snapshot_task14_root_v1(root, "fixture")
    source.write_bytes(b"new!")
    after = snapshot_task14_root_v1(root, "fixture")
    assert before.tree_sha256 != after.tree_sha256


def test_frozen_semantic_anchors_reject_cross_layer_rebinding() -> None:
    from mub.vnext.release.task14_sources import _validate_anchor_semantics

    loaded = load_task14_sources_v1(paths())

    task10 = dict(loaded.json_payloads)
    report = dict(task10["task10/external_admission_report.json"])
    gates = [dict(item) for item in report["gates"]]
    gates[0]["status"] = "fail"
    report["gates"] = gates
    task10["task10/external_admission_report.json"] = report
    with pytest.raises(ValueError, match="14/14"):
        _validate_anchor_semantics(task10)

    task11 = dict(loaded.json_payloads)
    qualification = dict(task11["task11/qualification_report.json"])
    slots = [dict(item) for item in qualification["slots"]]
    slots[1]["revision"] = "0" * 40
    qualification["slots"] = slots
    task11["task11/qualification_report.json"] = qualification
    with pytest.raises(ValueError, match="slots"):
        _validate_anchor_semantics(task11)

    task12 = dict(loaded.json_payloads)
    manifest = dict(task12["task12/matrix_bundle_manifest.json"])
    bundles = [dict(item) for item in manifest["run_bundles"]]
    bundles[0]["bundle_leaf"] = "fake-offline"
    manifest["run_bundles"] = bundles
    task12["task12/matrix_bundle_manifest.json"] = manifest
    with pytest.raises(ValueError, match="prompted-answer"):
        _validate_anchor_semantics(task12)

    task13 = dict(loaded.json_payloads)
    audit = dict(task13["task13_audit/core_task13_bc82566_v1_audit.json"])
    rejoin = dict(audit["matrix_case_rejoin"])
    rejoin["observations"] = 1439
    audit["matrix_case_rejoin"] = rejoin
    task13["task13_audit/core_task13_bc82566_v1_audit.json"] = audit
    with pytest.raises(ValueError, match="rejoin"):
        _validate_anchor_semantics(task13)


def test_root_snapshot_rejects_symlink_member(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("value", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="reparse"):
        snapshot_task14_root_v1(root, "fixture")
