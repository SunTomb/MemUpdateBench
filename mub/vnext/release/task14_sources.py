from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import MappingProxyType
from typing import Mapping, Sequence

from mub.vnext.release.task14_contracts import (
    Task14ArtifactRefV1,
    Task14RootEntryV1,
    Task14RootSnapshotV1,
)


TASK14_TASK9_IMPLEMENTATION_REVISION = "9118d491fb3f13a2b4278f131fd2520f9c4fe809"
TASK14_EXPECTED_TASK13_INDEX_SHA256 = (
    "da02787276dd171cce716258ec071947ae99fb047a607df983f52125a20937aa"
)
TASK14_EXPECTED_CORE_RELEASE_MANIFEST_SHA256 = (
    "dd5ea033fd1bb7353f4c7f443c6a1e14ed44fb9e8641f8e05838b4147d3ec13b"
)
TASK14_EXPECTED_FILE_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "core/task_release_manifest.json": TASK14_EXPECTED_CORE_RELEASE_MANIFEST_SHA256,
        "core/audit/gate_report.json": "5edb18bbe34c0b903b413a1564a5e85ca3dea5a0ff2365001dba13c154927665",
        "core/audit/gate_verification_attestation.json": "1bf15c7b314505de175e1108a0f358684c22d378709671ab9ecab7246a00237e",
        "task9/evaluation_configuration.json": "543881d8a6a1d16e5e4c5e5a3db655c4dda557d618736b00aee6e828d3003c7e",
        "task9/model_provenance.json": "8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e",
        "task10/adapter_configuration.json": "d72bd37887267e390484fa1b04cadba204108584a761fb5ee0ded30e9de26662",
        "task10/admission_decision.json": "c4355fdd1149325306eecf3242eeaf4e3e47a0d9ee616b0f9777058529e04f1c",
        "task10/canary_terminal_rows.json": "5bfebc4aa8f18ac5e0d0689c8d2deef6d55003d50d8a5194d8bb2daba2666fec",
        "task10/capability_verification.json": "b44a43cc4667388e7d5935759ca50b20badbbc9f9008622f47769709779a9626",
        "task10/evaluation_configuration.json": "37d923c10d070b894f5d5f00d7ff9cfbc15622f5890e5e5a0fca66301515cf83",
        "task10/external_admission_report.json": "2a00a350c750fc02f727af188a8f3d63f68df474e55a53a0710b5b62c6b43fae",
        "task10/package_provenance.json": "03c9ecdb12ba70560041826421319f56575c4a415592572846dab2697b02ae33",
        "task10/probe.json": "e8f8a05faa291618be5eed5d076abff8e1ce918929b34eb5963ae61b1eaec10e",
        "task11/mistral_snapshot_provenance.json": "0fc48730152bafa005e3f18b12861bec295db02d9ff221ff7b0871cb9bf409da",
        "task11/qualification_report.json": "00699e0d7a027d9bb63dca52753d53fe06bcdd0f7c87535aff6f25a7cb496672",
        "task12/matrix_bundle_manifest.json": "85145a8a460ee6cec3785926f9aaa85c8bee8cd41d4ad0582d2b0333b8cf10d2",
        "task12/matrix_integrity_audit.json": "bfc85922c36dcc87deca983ce39ff395b10da00c2ee91c8aba7a6c02c3f04f60",
        "task12/matrix_run_summary.json": "a1c4f89af2b9f39de9791ce9c6348c24b4c81474abf3da865f22e5dfe68f1f15",
        "task13/bootstrap_indices.bin": "0d8faf77bc7e4d138f0f9dd3db85ab136f99884906298984202c8dc38c0bbd53",
        "task13/cell_statistics.jsonl": "e4f25e3a7fc9795a93e8007acb1131dc84bb24fcdaf4867ac65042683bf0036b",
        "task13/paired_contrasts.jsonl": "517d426b86e415467ab72e4655d9fe7972ca1218d3765141bc210d3b28120e47",
        "task13/statistics_receipt.json": "398914d52b22c9c2bb71fc548e1f4239cf15cbf99d7cae2cd53e86b4fdcf9451",
        "task13/cases.jsonl": "af863aa24f90851a6b7149b5cefbceafdbb8c3987bd7be439768386a4bdfdb80",
        "task13/case_index.json": "8c97243db3265cb39f7048ea4e825d49aead50da94e122fd9c8e638360f2ed36",
        "task13/claim_ledger.jsonl": "9f486dd90361dd8b70ed8cc2fa0c5a552dbf37f88b55addde71456347a4d0273",
        "task13/task13_artifact_index.json": TASK14_EXPECTED_TASK13_INDEX_SHA256,
        "task13_audit/core_task13_bc82566_v1_audit.json": "c60c49d917c582506e262534a6c48bb68668027e428ba0c06557ae8381982145",
    }
)


@dataclass(frozen=True)
class Task14SourcePathsV1:
    core_root: Path
    evidence_root: Path
    task13_root: Path
    task13_audit_path: Path
    repository_root: Path
    remote_task13_staging_path: str


@dataclass(frozen=True)
class Task14LoadedSourcesV1:
    paths: Task14SourcePathsV1
    artifacts: Mapping[str, Task14ArtifactRefV1]
    payloads: Mapping[str, bytes]
    json_payloads: Mapping[str, object]
    root_snapshots: tuple[Task14RootSnapshotV1, ...]
    aggregate_snapshot_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unsafe_reparse(path: Path) -> bool:
    metadata = path.lstat()
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _checked_root(path: Path) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    current = Path(selected.anchor)
    for part in selected.parts[1:]:
        current = current / part
        if current.exists() and _unsafe_reparse(current):
            raise ValueError(f"Task 14 path contains a reparse point: {current}")
    resolved = selected.resolve(strict=True)
    if _unsafe_reparse(resolved) or not resolved.is_dir():
        raise ValueError(f"Task 14 source root is unsafe: {resolved}")
    return resolved


def _checked_file(path: Path) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    current = Path(selected.anchor)
    for part in selected.parts[1:]:
        current = current / part
        if current.exists() and _unsafe_reparse(current):
            raise ValueError(f"Task 14 path contains a reparse point: {current}")
    resolved = selected.resolve(strict=True)
    metadata = resolved.lstat()
    if _unsafe_reparse(resolved) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Task 14 source is not a regular file: {resolved}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"Task 14 source must be single-link: {resolved}")
    return resolved


def _canonical_plain_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_task14_root_v1(root: Path, root_id: str) -> Task14RootSnapshotV1:
    checked = _checked_root(root)
    entries: list[Task14RootEntryV1] = []
    for current, directories, files in os.walk(checked, followlinks=False):
        current_path = Path(current)
        if _unsafe_reparse(current_path):
            raise ValueError("Task 14 root contains a reparse-point directory")
        directories.sort(key=os.fsencode)
        files.sort(key=os.fsencode)
        for name in directories:
            if _unsafe_reparse(current_path / name):
                raise ValueError("Task 14 root contains a reparse-point member")
        for name in files:
            member = _checked_file(current_path / name)
            relative = member.relative_to(checked).as_posix()
            entries.append(
                Task14RootEntryV1(
                    relative_path=relative,
                    byte_count=member.stat().st_size,
                    sha256=_sha256_file(member),
                )
            )
    ordered = tuple(sorted(entries, key=lambda item: item.relative_path.encode("utf-8")))
    binding = {
        "root_id": root_id,
        "entries": [item.model_dump(mode="json") for item in ordered],
    }
    tree_sha = hashlib.sha256(_canonical_plain_bytes(binding)).hexdigest()
    metadata = checked.stat()
    identity = f"{metadata.st_dev}:{metadata.st_ino}"
    return Task14RootSnapshotV1(
        root_id=root_id,
        root_path=str(checked),
        filesystem_identity=identity,
        entries=ordered,
        tree_sha256=tree_sha,
    )


def task14_source_snapshot_hash_v1(
    snapshots: Sequence[Task14RootSnapshotV1],
) -> str:
    payload = [item.model_dump(mode="json") for item in snapshots]
    return hashlib.sha256(_canonical_plain_bytes(payload)).hexdigest()


def _verify_repository_history(repository_root: Path) -> Path:
    root = _checked_root(repository_root)
    top = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "--show-toplevel"),
        check=False,
        capture_output=True,
        text=True,
    )
    if top.returncode != 0 or Path(top.stdout.strip()).resolve(strict=True) != root:
        raise ValueError("Task 14 repository root is not the Git worktree root")
    ancestor = subprocess.run(
        (
            "git", "-C", str(root), "merge-base", "--is-ancestor",
            TASK14_TASK9_IMPLEMENTATION_REVISION, "HEAD",
        ),
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("Task 14 source history omits the frozen Task 9 implementation")
    return root


def _source_locations(paths: Task14SourcePathsV1) -> dict[str, Path]:
    core = _checked_root(paths.core_root)
    evidence = _checked_root(paths.evidence_root)
    task13 = _checked_root(paths.task13_root)
    return {
        "core/task_release_manifest.json": core / "task_release_manifest.json",
        "core/audit/gate_report.json": core / "audit" / "gate_report.json",
        "core/audit/gate_verification_attestation.json": core / "audit" / "gate_verification_attestation.json",
        **{
            key: evidence / key
            for key in TASK14_EXPECTED_FILE_HASHES
            if key.startswith(("task9/", "task10/", "task11/", "task12/"))
        },
        **{
            key: task13 / key.removeprefix("task13/")
            for key in TASK14_EXPECTED_FILE_HASHES
            if key.startswith("task13/")
        },
        "task13_audit/core_task13_bc82566_v1_audit.json": Path(paths.task13_audit_path),
    }


def _parse_json(raw: bytes, role: str) -> object:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Task 14 JSON artifact is invalid: {role}") from exc


def _validate_anchor_semantics(json_payloads: Mapping[str, object]) -> None:
    release = json_payloads["core/task_release_manifest.json"]
    if not isinstance(release, dict) or any(
        release.get(key) != value
        for key, value in {
            "release_status": "FINAL_APPROVED",
            "release_manifest_hash": "f953283a10dd45d3f9d1de066570a9c09b9d132ed458f8dea3c948641b89e99d",
            "candidate_root_digest": "71a6beb3ac8a28dabc753c969e96a47a59f92031d217bebf0fa63d6061012af1",
            "gate_attestation_hash": "45461659ab3f65a0a559897e50340a470f27cdecf55b999a1431988567cf00c2",
            "task_count": 12000,
            "semantic_core_count": 3000,
            "hard_suite_task_count": 560,
        }.items()
    ):
        raise ValueError("Task 14 immutable Core release anchors do not match")

    decision = json_payloads["task10/admission_decision.json"]
    report = json_payloads["task10/external_admission_report.json"]
    rows = json_payloads["task10/canary_terminal_rows.json"]
    if not isinstance(decision, dict) or decision.get("status") != "admitted":
        raise ValueError("Task 14 Mem0 admission decision is not admitted")
    if not isinstance(report, dict) or report.get("outcome") != "pass":
        raise ValueError("Task 14 Mem0 admission report is not PASS")
    gates = report.get("gates", ())
    if len(gates) != 14 or any(item.get("status") != "pass" for item in gates):
        raise ValueError("Task 14 Mem0 admission gates are not 14/14 PASS")
    if decision.get("reasons") != ["admitted_mem0_primary"] or len(decision.get("reports", ())) != 1:
        raise ValueError("Task 14 Mem0 fallback/admission boundary mismatch")
    terminal_rows = rows.get("terminal_rows", ()) if isinstance(rows, dict) else ()
    if not isinstance(rows, dict) or rows.get("expected_rows") != 128 or len(terminal_rows) != 128:
        raise ValueError("Task 14 Mem0 canary terminal-row cardinality mismatch")
    if any(item.get("completion_status") != "not_supported" for item in terminal_rows):
        raise ValueError("Task 14 Mem0 canary contains non-NOT_SUPPORTED rows")
    if any(item.get("completion_status") in {"failed", "partial"} for item in terminal_rows):
        raise ValueError("Task 14 Mem0 canary contains FAILED/PARTIAL rows")

    qualification = json_payloads["task11/qualification_report.json"]
    if not isinstance(qualification, dict) or qualification.get("status") != "qualified":
        raise ValueError("Task 14 Task 11 qualification is not qualified")
    slots = qualification.get("slots", ())
    expected_slots = {
        "answer_model_a": (
            "Qwen/Qwen2.5-7B-Instruct",
            "a09a35458c702b33eeacc393d103063234e8bc28",
            "5c5fc08ade3cfa718521bbb2206deb1f0249527b8f210c95a4db9140460154ca",
        ),
        "answer_model_b": (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "c170c708c41dac9275d15a8fff4eca08d52bab71",
            "31a92a122692365f74cc64939cc948fb21f1efa1d500afd3d92332ad319db015",
        ),
    }
    observed_slots = {
        item.get("slot_id"): (
            item.get("model_id"), item.get("revision"), item.get("tree_manifest_sha256")
        )
        for item in slots
    }
    if observed_slots != expected_slots:
        raise ValueError("Task 14 Task 11 frozen answer-model slots mismatch")

    summary = json_payloads["task12/matrix_run_summary.json"]
    audit = json_payloads["task12/matrix_integrity_audit.json"]
    manifest = json_payloads["task12/matrix_bundle_manifest.json"]
    if not all(isinstance(item, dict) for item in (summary, audit, manifest)):
        raise ValueError("Task 14 Task 12 controls must be JSON objects")
    bundles = manifest.get("run_bundles", ())
    completed = summary.get("completed_runs", ())
    expected_pairs = {
        (f"raw-add-{condition}-k{k:02d}", slot)
        for condition in (
            "chronological-none",
            "reverse-none",
            "reverse-version-labeled",
        )
        for k in (4, 8, 16)
        for slot in ("answer_model_a", "answer_model_b")
    }
    bundle_pairs = {(item.get("cell_id"), item.get("answer_model_slot")) for item in bundles}
    completed_pairs = {(item.get("cell_id"), item.get("answer_model_slot")) for item in completed}
    if (
        manifest.get("bundle_count") != 18
        or len(bundles) != 18
        or bundle_pairs != expected_pairs
        or len(completed) != 18
        or completed_pairs != expected_pairs
        or any(not str(item.get("bundle_leaf", "")).startswith("raw-add-") for item in bundles)
        or any(item.get("output_leaf") != "run" for item in bundles)
        or any(item.get("task_count") != 80 or item.get("score_count") != 80 for item in completed)
    ):
        raise ValueError("Task 14 Task 12 real prompted-answer run scope mismatch")
    if (
        summary.get("total_task_rows") != 1440
        or summary.get("total_score_rows") != 1440
        or len(summary.get("completed_runs", [])) != 18
        or audit.get("run_count") != 18
        or audit.get("failed_or_partial_rows") != 0
        or audit.get("total_task_rows") != 1440
        or audit.get("total_score_rows") != 1440
        or audit.get("status") != "verified"
    ):
        raise ValueError("Task 14 Task 12 matrix completeness mismatch")

    task13_audit = json_payloads["task13_audit/core_task13_bc82566_v1_audit.json"]
    if not isinstance(task13_audit, dict) or task13_audit.get("status") != "verified":
        raise ValueError("Task 14 Task 13 independent audit is not verified")
    if task13_audit.get("remote_final_root_absent") is not True:
        raise ValueError("Task 14 Task 13 NFS final-root boundary is not preserved")
    counts = task13_audit.get("counts", {})
    expected_counts = {
        "cell_statistics": 126,
        "paired_contrasts": 84,
        "claim_ledger": 210,
        "cases": 57,
        "case_runs": 18,
        "bootstrap_bytes": 200000,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("Task 14 Task 13 audit counts mismatch")
    rejoin = task13_audit.get("matrix_case_rejoin", {})
    if rejoin != {
        "status": "verified",
        "runs": 18,
        "cases": 57,
        "observations": 1440,
    }:
        raise ValueError("Task 14 Task 13 matrix/case rejoin mismatch")
    if task13_audit.get("artifact_sha256", {}).get("bootstrap_indices.bin") != TASK14_EXPECTED_FILE_HASHES["task13/bootstrap_indices.bin"]:
        raise ValueError("Task 14 Task 13 frozen bootstrap binding mismatch")
    unsupported = task13_audit.get("metric_status", {}).get("unsupported", {})
    if set(unsupported) != {
        "answer_scores.gold_retrieved_wrong_answer",
        "retrieval_scores.stale_count_in_context",
        "retrieval_scores.stale_exposure_rate",
    } or any(value != 18 for value in unsupported.values()):
        raise ValueError("Task 14 Task 13 unsupported/null policy mismatch")


def load_task14_sources_v1(paths: Task14SourcePathsV1) -> Task14LoadedSourcesV1:
    if type(paths) is not Task14SourcePathsV1:
        raise TypeError("Task 14 paths must be Task14SourcePathsV1")
    _verify_repository_history(paths.repository_root)
    if not paths.remote_task13_staging_path.startswith("/NAS/") or ".mub-task13-stage-" not in paths.remote_task13_staging_path:
        raise ValueError("Task 14 remote Task 13 path must remain NFS staging evidence")
    locations = _source_locations(paths)
    if set(locations) != set(TASK14_EXPECTED_FILE_HASHES):
        raise AssertionError("Task 14 source location inventory is incomplete")
    artifacts: dict[str, Task14ArtifactRefV1] = {}
    payloads: dict[str, bytes] = {}
    json_payloads: dict[str, object] = {}
    for role in sorted(locations, key=lambda value: value.encode("utf-8")):
        path = _checked_file(locations[role])
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        expected = TASK14_EXPECTED_FILE_HASHES[role]
        if observed != expected:
            raise ValueError(f"Task 14 source hash mismatch: {role}")
        payloads[role] = raw
        if not role.endswith((".bin", ".jsonl")):
            json_payloads[role] = _parse_json(raw, role)
        record_count = None
        if role.endswith(".jsonl"):
            record_count = len(raw.splitlines())
        artifacts[role] = Task14ArtifactRefV1(
            artifact_id=role.replace("/", ":"),
            path=role,
            sha256=observed,
            media_type=(
                "application/octet-stream"
                if role.endswith(".bin")
                else "application/x-ndjson"
                if role.endswith(".jsonl")
                else "application/json"
            ),
            byte_count=len(raw),
            record_count=record_count,
            root_kind=(
                "immutable_local"
                if role.startswith("core/")
                else "published_local"
                if role.startswith(("task13/", "task13_audit/"))
                else "remote_evidence"
            ),
        )
    _validate_anchor_semantics(json_payloads)
    snapshots = (
        snapshot_task14_root_v1(paths.core_root, "immutable_core"),
        snapshot_task14_root_v1(paths.evidence_root, "task9_task12_evidence"),
        snapshot_task14_root_v1(paths.task13_root, "task13_local_final"),
    )
    return Task14LoadedSourcesV1(
        paths=paths,
        artifacts=MappingProxyType(artifacts),
        payloads=MappingProxyType(payloads),
        json_payloads=MappingProxyType(json_payloads),
        root_snapshots=snapshots,
        aggregate_snapshot_sha256=task14_source_snapshot_hash_v1(snapshots),
    )


def revalidate_task14_sources_v1(loaded: Task14LoadedSourcesV1) -> bool:
    if type(loaded) is not Task14LoadedSourcesV1:
        return False
    try:
        refreshed = load_task14_sources_v1(loaded.paths)
        return bool(
            refreshed.aggregate_snapshot_sha256 == loaded.aggregate_snapshot_sha256
            and dict(refreshed.artifacts) == dict(loaded.artifacts)
        )
    except Exception:
        return False


__all__ = [
    "TASK14_TASK9_IMPLEMENTATION_REVISION",
    "TASK14_EXPECTED_CORE_RELEASE_MANIFEST_SHA256",
    "TASK14_EXPECTED_FILE_HASHES",
    "TASK14_EXPECTED_TASK13_INDEX_SHA256",
    "Task14LoadedSourcesV1",
    "Task14SourcePathsV1",
    "load_task14_sources_v1",
    "revalidate_task14_sources_v1",
    "snapshot_task14_root_v1",
    "task14_source_snapshot_hash_v1",
]
