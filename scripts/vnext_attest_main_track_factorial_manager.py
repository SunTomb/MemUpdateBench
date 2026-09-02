from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.io.atomic import publish_files_atomically
from scripts import vnext_plan_main_track_factorial as planner
from scripts import vnext_run_main_track_factorial_answer as answer_runner
from scripts import vnext_run_main_track_factorial_manager as manager_runner


RECEIPT_SCHEMA = "memupdatebench.main-track.manager-fixture-execution-receipt.v1"
EVIDENCE_CLASS = "manager_state_retrieval_fixture"
EXECUTION_MODE = "production"
DEFAULT_MANIFEST = planner.DEFAULT_OUTPUT
DEFAULT_RECEIPT_NAME = "manager_fixture_attestation.json"
_ARTIFACTS = ("manager_rows.jsonl", "manager_summary.json", "artifact_index.json")
_BOUNDARY_KEYS = (
    "provider_calls",
    "model_loads",
    "database_accesses",
    "network_calls",
    "gpu_calls",
    "executable_calls",
    "remote_operations",
)
_ZERO_BOUNDARY_KEYS = (
    "provider_calls",
    "database_accesses",
    "network_calls",
    "remote_operations",
)
_PRODUCTION_ACCOUNTING_SOURCE = "production_runtime_profile"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s?#]*@", re.IGNORECASE)
# Pin the source after normalizing checkout-specific line endings to LF.
EXPECTED_MANAGER_RUNNER_SOURCE_SHA256 = "b290fcd29cab699ca1e7e1e2d97c1d85878104ada91e5b545a6f901422c7ab8d"
_FROZEN_IMMUTABLE_ROOTS = (
    ROOT / "data" / "vnext" / "core" / "v3",
    ROOT / "data" / "vnext" / "pilot",
)


class AttestationError(ValueError):
    """Raised when a manager fixture cannot be independently attested."""


def canonical_json_bytes(value: Any) -> bytes:
    return answer_runner.canonical_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return answer_runner.sha256_bytes(value)


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(receipt))


def _require_sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise AttestationError(f"{label} must be lowercase SHA-256")
    return value


def _require_nonblank_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise AttestationError(f"{label} must be a nonblank string")
    return value


def _validate_runner_source_sha(
    value: Any, *, expected: str | None = EXPECTED_MANAGER_RUNNER_SOURCE_SHA256
) -> str:
    observed = _require_sha(value, "runner_source_sha256")
    if expected is not None and observed != expected:
        raise AttestationError(
            "runner_source_sha256 does not match expected manager runner source digest"
        )
    return observed


def _normalize_source_line_endings(raw: bytes) -> bytes:
    """Normalize CRLF and legacy CR source checkouts to canonical LF bytes."""
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _manager_runner_source_hashes(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    normalized_sha256 = _validate_runner_source_sha(
        sha256_bytes(_normalize_source_line_endings(raw)),
        expected=EXPECTED_MANAGER_RUNNER_SOURCE_SHA256,
    )
    return sha256_bytes(raw), normalized_sha256


def _validate_runtime_identity(value: Any) -> str | dict[str, Any]:
    if type(value) is str:
        if not value.strip():
            raise AttestationError("runtime_identity must be nonblank")
        result: str | dict[str, Any] = value
    elif isinstance(value, Mapping) and value:
        result = dict(value)
    else:
        raise AttestationError("runtime_identity must be a nonempty string or object")
    _validate_no_credential_urls(result)
    answer_runner.validate_public_payload(result)
    return result


def _validate_no_credential_urls(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_no_credential_urls(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_no_credential_urls(child)
    elif type(value) is str and _CREDENTIAL_URL.search(value):
        raise AttestationError("runtime_identity contains a credential URL")


def _validate_execution_boundary(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_BOUNDARY_KEYS):
        raise AttestationError("fixture execution boundary is incomplete")
    result: dict[str, int] = {}
    for key in _BOUNDARY_KEYS:
        item = value.get(key)
        if type(item) is not int or item < 0:
            raise AttestationError(f"fixture execution boundary {key} is invalid")
        if key in _ZERO_BOUNDARY_KEYS and item != 0:
            raise AttestationError(f"fixture execution boundary {key} must be zero")
        result[key] = item
    return result


def _resolve_input(path: str | Path, label: str) -> Path:
    value = Path(path)
    assert_no_reparse_components(value)
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise AttestationError(f"{label} is unavailable") from exc
    return resolved


def _validate_summary_and_index(
    summary: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    expected_scope: str,
    expected_supported_count: int,
    expected_unsupported_count: int,
    manager_kind: str,
    manager_id: str,
    cell_id: str,
    manifest_sha256: str,
    candidate_hashes: Mapping[str, str],
    audit_sha: str,
    runner_source_sha256: str,
) -> dict[str, int]:
    runner_source_sha256 = _validate_runner_source_sha(
        runner_source_sha256, expected=runner_source_sha256
    )
    for key, expected in (
        ("execution_accounting_observed", True),
        ("execution_accounting_source", _PRODUCTION_ACCOUNTING_SOURCE),
    ):
        if summary.get(key) != index.get(key):
            raise AttestationError(f"manager fixture summary/index {key} mismatch")
        if summary.get(key) != expected:
            raise AttestationError(f"manager fixture {key} is not production accounting")
    required_summary = {
        "status": "PASS",
        "execution_mode": EXECUTION_MODE,
        "evidence_class": EVIDENCE_CLASS,
        "scientific_evidence": True,
        "scope": expected_scope,
        "cell_id": cell_id,
        "manager_kind": manager_kind,
        "manager_id": manager_id,
        "manifest_sha256": manifest_sha256,
        "candidate_artifact_hashes": dict(candidate_hashes),
        "audit_attestation_sha256": audit_sha,
        "failed": 0,
        "supported": expected_supported_count,
        "unsupported": expected_unsupported_count,
        "eligible_supported_count": expected_supported_count,
        "executed_supported_count": expected_supported_count,
        "not_requested_supported_count": 0,
        "requested_task_count": expected_supported_count + expected_unsupported_count,
        "terminal_rows": expected_supported_count + expected_unsupported_count,
        "execution_boundary_observed": True,
        "runner_source_sha256": runner_source_sha256,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise AttestationError(f"manager fixture summary {key} binding mismatch")
    selected = summary.get("selected_supported_task_ids")
    if not isinstance(selected, list) or len(selected) != expected_supported_count:
        raise AttestationError("manager fixture summary supported membership is incomplete")
    if summary.get("selected_supported_task_ids_sha256") != sha256_bytes(canonical_json_bytes(selected)):
        raise AttestationError("manager fixture summary supported membership hash mismatch")
    boundary = _validate_execution_boundary(summary.get("execution_boundary"))
    if expected_scope == "full240" and cell_id.endswith("__qwen35_answer"):
        for key in ("model_loads", "gpu_calls"):
            if boundary[key] < 1:
                raise AttestationError(
                    f"full production Qwen fixture execution boundary {key} must be positive"
                )
    if "runner_source_sha256_normalized" in summary or "runner_source_sha256_normalized" in index:
        if summary.get("runner_source_sha256_normalized") != index.get(
            "runner_source_sha256_normalized"
        ):
            raise AttestationError(
                "manager fixture summary/index runner_source_sha256_normalized mismatch"
            )
        _validate_runner_source_sha(
            summary.get("runner_source_sha256_normalized")
        )
    for key in ("extractor_identity", "extractor_source_sha256"):
        if key in summary or key in index:
            if key not in summary or key not in index or summary.get(key) != index.get(key):
                raise AttestationError(f"manager fixture summary/index {key} mismatch")
            if key == "extractor_source_sha256":
                _require_sha(summary.get(key), key)

    required_index = {
        "status": "PASS",
        "execution_mode": EXECUTION_MODE,
        "evidence_class": EVIDENCE_CLASS,
        "scientific_evidence": True,
        "scope": expected_scope,
        "manager_kind": manager_kind,
        "manager_id": manager_id,
        "manifest_sha256": manifest_sha256,
        "candidate_artifact_hashes": dict(candidate_hashes),
        "audit_attestation_sha256": audit_sha,
        "execution_boundary_observed": True,
        "runner_source_sha256": runner_source_sha256,
        "execution_boundary": boundary,
    }
    for key, expected in required_index.items():
        if index.get(key) != expected:
            raise AttestationError(f"manager fixture index {key} binding mismatch")
    if index.get("cell_id") != cell_id:
        raise AttestationError("manager fixture index cell_id binding mismatch")
    if index.get("executed_supported_count") != expected_supported_count:
        raise AttestationError("manager fixture index supported count mismatch")
    if index.get("unsupported_count") not in (None, expected_unsupported_count):
        raise AttestationError("manager fixture index unsupported count mismatch")
    if index.get("selected_supported_task_ids") != selected:
        raise AttestationError("manager fixture index supported membership mismatch")
    if index.get("selected_supported_task_ids_sha256") != summary.get("selected_supported_task_ids_sha256"):
        raise AttestationError("manager fixture index supported membership hash mismatch")
    return boundary


def _validate_snapshot(
    *,
    fixture_root: Path,
    manifest_path: Path,
    candidate_root: Path,
    audit_path: Path,
    manager_kind: str,
    cell_id: str,
    allow_relocated_authenticated_inputs: bool,
) -> dict[str, Any]:
    rows, summary, index, fixture_hashes = answer_runner._load_fixture(fixture_root)
    loaded_manifest, manifest_raw, provenance, tasks, _ = answer_runner._validate_manifest_and_candidate(
        manifest_path,
        candidate_root,
        audit_path,
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
    )
    manifest_sha256 = sha256_bytes(manifest_raw)
    cell = answer_runner._expected_cell(loaded_manifest, cell_id, tasks=tasks)
    expected_manager_kind, _ = planner._cell_spec(cell_id)
    if manager_kind != expected_manager_kind:
        raise AttestationError("manager_kind does not match factorial cell")
    expected_manager_id = cell.get("manager_id")
    if type(expected_manager_id) is not str or not expected_manager_id:
        raise AttestationError("factorial cell manager_id is incomplete")
    expected_supported_count = len(cell.get("supported_task_ids", ()))
    expected_unsupported_count = len(cell.get("unsupported_tasks", ()))
    expected_scope = f"full{expected_supported_count}"
    validated_rows, scope_info = answer_runner._validate_fixture_rows(
        rows,
        summary,
        index,
        loaded_manifest,
        cell,
        tasks,
        provenance,
        provenance["audit_attestation_sha256"],
        manifest_sha256=manifest_sha256,
        execution_mode=EXECUTION_MODE,
    )
    if scope_info.get("scope") != expected_scope or scope_info.get("scientific_evidence") is not True:
        raise AttestationError("manager fixture is not a full scientific production fixture")
    if summary.get("manager_kind") != manager_kind or summary.get("manager_id") != expected_manager_id:
        raise AttestationError("manager fixture manager identity binding mismatch")
    if summary.get("cell_id") != cell_id:
        raise AttestationError("manager fixture cell binding mismatch")
    runner_source_path = Path(manager_runner.__file__).resolve(strict=True)
    runner_source_sha256, runner_source_sha256_normalized = _manager_runner_source_hashes(
        runner_source_path
    )
    boundary = _validate_summary_and_index(
        summary,
        index,
        expected_scope=expected_scope,
        expected_supported_count=expected_supported_count,
        expected_unsupported_count=expected_unsupported_count,
        manager_kind=manager_kind,
        manager_id=expected_manager_id,
        cell_id=cell_id,
        manifest_sha256=manifest_sha256,
        candidate_hashes=provenance["artifact_hashes"],
        audit_sha=provenance["audit_attestation_sha256"],
        runner_source_sha256=runner_source_sha256,
    )
    return {
        "rows": validated_rows,
        "summary": dict(summary),
        "index": dict(index),
        "fixture_hashes": dict(fixture_hashes),
        "fixture_root_digest": answer_runner._fixture_root_digest(fixture_hashes),
        "manifest_sha256": manifest_sha256,
        "candidate_hashes": dict(provenance["artifact_hashes"]),
        "candidate_release_index_sha256": provenance["release_index_sha256"],
        "audit_sha256": provenance["audit_attestation_sha256"],
        "cell_id": cell_id,
        "manager_kind": manager_kind,
        "manager_id": expected_manager_id,
        "scope": expected_scope,
        "runner_source_sha256": runner_source_sha256,
        "runner_source_sha256_normalized": runner_source_sha256_normalized,
        "execution_boundary": boundary,
    }


def _build_receipt(
    snapshot: Mapping[str, Any],
    *,
    producer_id: str,
    producer_revision: str,
    runtime_identity: str | Mapping[str, Any],
    allow_relocated_authenticated_inputs: bool,
) -> dict[str, Any]:
    producer_id = _require_nonblank_string(producer_id, "producer_id")
    producer_revision = _require_nonblank_string(producer_revision, "producer_revision")
    runtime_identity = _validate_runtime_identity(runtime_identity)
    manager_id = _require_nonblank_string(snapshot.get("manager_id"), "producer.manager_id")
    cell_id = _require_nonblank_string(snapshot.get("cell_id"), "producer.cell_id")
    runner_source_sha256 = _validate_runner_source_sha(
        snapshot.get("runner_source_sha256"), expected=snapshot.get("runner_source_sha256")
    )
    runner_source_sha256_normalized = _validate_runner_source_sha(
        snapshot.get("runner_source_sha256_normalized", runner_source_sha256)
    )
    producer = {
        "producer_id": producer_id,
        "producer_revision": producer_revision,
        "runtime_identity": runtime_identity,
        "manager_id": manager_id,
        "cell_id": cell_id,
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "attestation_status": "PASS",
        "authentication_method": "hash_bound_release_attestation",
        "scientific_evidence": False,
        "execution_mode": EXECUTION_MODE,
        "evidence_class": EVIDENCE_CLASS,
        "manifest_sha256": snapshot["manifest_sha256"],
        "candidate_release_index_sha256": snapshot["candidate_release_index_sha256"],
        "candidate_artifact_hashes": dict(snapshot["candidate_hashes"]),
        "audit_attestation_sha256": snapshot["audit_sha256"],
        "manager_fixture_artifact_hashes": dict(snapshot["fixture_hashes"]),
        "manager_fixture_root_digest": snapshot["fixture_root_digest"],
        "cell_id": cell_id,
        "scope": snapshot["scope"],
        "manager_kind": snapshot["manager_kind"],
        "manager_id": manager_id,
        "producer": producer,
        "producer_manager_id": manager_id,
        "producer_cell_id": cell_id,
        "producer_source_sha256": runner_source_sha256_normalized,
        "runner_source_sha256": runner_source_sha256,
        "runner_source_sha256_normalized": runner_source_sha256_normalized,
        "execution_boundary": dict(snapshot["execution_boundary"]),
        "input_relocation": {
            "enabled": bool(allow_relocated_authenticated_inputs),
            "authenticated_equivalence": True,
        },
    }
    _validate_no_credential_urls(receipt)
    answer_runner.validate_public_payload(receipt)
    return receipt


def _resolve_output_path(
    *, output_receipt: str | Path | None, output_root: str | Path | None
) -> Path:
    if (output_receipt is None) == (output_root is None):
        raise AttestationError("provide exactly one of output_receipt or output_root")
    if output_root is not None:
        root = Path(output_root)
        if not root.is_absolute():
            raise AttestationError("output_root must be absolute")
        return root / DEFAULT_RECEIPT_NAME
    output = Path(output_receipt)  # type: ignore[arg-type]
    if not output.is_absolute():
        raise AttestationError("output_receipt must be absolute")
    return output


def _validate_output_target(output: Path, source_paths: Sequence[Path]) -> None:
    assert_no_reparse_components(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    resolved = output.resolve(strict=False)
    for frozen_root in _FROZEN_IMMUTABLE_ROOTS:
        frozen_resolved = frozen_root.resolve(strict=False)
        if resolved == frozen_resolved or frozen_resolved in resolved.parents:
            raise AttestationError(
                "output receipt must be outside frozen immutable roots"
            )
    for source in source_paths:
        source_resolved = source.resolve(strict=False)
        if resolved == source_resolved or source_resolved in resolved.parents or resolved in source_resolved.parents:
            raise AttestationError("output receipt must be separate from inputs")


def _validate_staged_receipt(path: Path, expected: bytes) -> None:
    if path.read_bytes() != expected:
        raise AttestationError("staged release attestation bytes changed")
    value, _ = answer_runner._load_json(path, "manager fixture release attestation")
    answer_runner.validate_public_payload(value)


def _revalidate_snapshot(
    *,
    initial: Mapping[str, Any],
    fixture_root: Path,
    manifest_path: Path,
    candidate_root: Path,
    audit_path: Path,
    manager_kind: str,
    cell_id: str,
    allow_relocated_authenticated_inputs: bool,
) -> None:
    observed = _validate_snapshot(
        fixture_root=fixture_root,
        manifest_path=manifest_path,
        candidate_root=candidate_root,
        audit_path=audit_path,
        manager_kind=manager_kind,
        cell_id=cell_id,
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
    )
    for key in (
        "fixture_hashes",
        "fixture_root_digest",
        "manifest_sha256",
        "candidate_hashes",
        "candidate_release_index_sha256",
        "audit_sha256",
        "scope",
        "runner_source_sha256",
        "runner_source_sha256_normalized",
        "execution_boundary",
    ):
        if key in initial or key in observed:
            if observed.get(key) != initial.get(key):
                raise AttestationError("manager fixture or source changed before publication")


def run(
    *,
    manager_fixture_root: str | Path,
    manifest: str | Path = DEFAULT_MANIFEST,
    candidate_root: str | Path,
    audit_attestation: str | Path,
    manager_kind: str,
    cell_id: str,
    producer_id: str,
    producer_revision: str,
    runtime_identity: str | Mapping[str, Any],
    output_receipt: str | Path | None = None,
    output_root: str | Path | None = None,
    allow_relocated_authenticated_inputs: bool = False,
) -> dict[str, Any]:
    producer_id = _require_nonblank_string(producer_id, "producer_id")
    producer_revision = _require_nonblank_string(producer_revision, "producer_revision")
    runtime_identity = _validate_runtime_identity(runtime_identity)
    fixture_path = _resolve_input(manager_fixture_root, "manager fixture root")
    manifest_path = _resolve_input(manifest, "factorial manifest")
    candidate_path = _resolve_input(candidate_root, "candidate root")
    audit_path = _resolve_input(audit_attestation, "audit attestation")
    output = _resolve_output_path(output_receipt=output_receipt, output_root=output_root)
    source_paths = (
        manifest_path,
        candidate_path,
        audit_path,
        fixture_path,
        *(fixture_path / name for name in _ARTIFACTS),
        Path(manager_runner.__file__).resolve(strict=True),
    )
    _validate_output_target(output, source_paths)
    initial = _validate_snapshot(
        fixture_root=fixture_path,
        manifest_path=manifest_path,
        candidate_root=candidate_path,
        audit_path=audit_path,
        manager_kind=manager_kind,
        cell_id=cell_id,
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
    )
    receipt = _build_receipt(
        initial,
        producer_id=producer_id,
        producer_revision=producer_revision,
        runtime_identity=runtime_identity,
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
    )
    raw = canonical_json_bytes(receipt)
    parent = output.parent
    parent_preexisted = parent.exists()
    try:
        publish_files_atomically(
            {output: raw},
            overwrite=False,
            source_paths=source_paths,
            validators={output: lambda staged: _validate_staged_receipt(staged, raw)},
            pre_publish=lambda: _revalidate_snapshot(
                initial=initial,
                fixture_root=fixture_path,
                manifest_path=manifest_path,
                candidate_root=candidate_path,
                audit_path=audit_path,
                manager_kind=manager_kind,
                cell_id=cell_id,
                allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
            ),
        )
    except BaseException:
        if not parent_preexisted and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
        raise
    return receipt


def _parse_runtime_identity(value: str) -> str | dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if type(parsed) is str:
        return parsed
    if isinstance(parsed, Mapping):
        return dict(parsed)
    raise argparse.ArgumentTypeError("runtime identity must be a string or JSON object")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently attest a completed production main-track manager fixture."
    )
    parser.add_argument("--manager-fixture-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--audit-attestation", type=Path, required=True)
    parser.add_argument("--manager-kind", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--producer-id", required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--runtime-identity", type=_parse_runtime_identity, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-receipt", type=Path)
    output.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-relocated-authenticated-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        receipt = run(**vars(args))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {"status": receipt["status"], "receipt_sha256": receipt_sha256(receipt)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
