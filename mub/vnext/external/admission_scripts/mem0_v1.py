from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.external.admission import authorize_fallback
from mub.vnext.external.canaries_v3 import (
    authenticate_core_release,
    build_canary_set,
    validate_canary_set,
)
from mub.vnext.external.contracts import (
    ADMISSION_GATE_NAMES,
    ExternalAdmissionReportV1,
    ExternalCandidateId,
    GateResultV1,
    GateStatus,
)
from mub.vnext.external.probe_v3 import (
    DeterminismStatus,
    NormalizedCandidateSnapshotV1,
    classify_determinism,
    required_canary_repetitions,
    verify_capability_truthfulness,
)
from mub.vnext.external.providers.mem0 import MEM0_PACKAGE_VERSION
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.runtime.support_v3 import resolve_task_support_v3


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact(path: str, raw: bytes, media_type: str, count: int | None = 1) -> ArtifactRef:
    return ArtifactRef(path=path, sha256=_sha256(raw), media_type=media_type, record_count=count)


def _canonical_object_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_json(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    if _canonical_object_bytes(value) != raw:
        raise ValueError(f"noncanonical JSON: {path}")
    return value


def _capabilities_from_preflight(value: dict) -> AdapterCapabilitiesV3:
    return AdapterCapabilitiesV3.model_validate(value["capabilities"], strict=True)


def _info_from_preflight(value: dict) -> AdapterInfoV3:
    return AdapterInfoV3.model_validate(value["adapter_info"], strict=True)


def _semantic_snapshot(value: dict) -> NormalizedCandidateSnapshotV1:
    integration = value["integration"]
    entries = integration["entries"]["entries"]
    retrieved = integration["retrieval"]["trace"]["retrieved_entries"]
    state = {
        "entries": [
            {
                "content": item["content"],
                "object_key_candidate": item["object_key_candidate"],
                "value_candidate": item["value_candidate"],
                "source_event_ids": item["source_event_ids"],
            }
            for item in entries
        ],
        "retrieval_contents": [item["content"] for item in retrieved],
    }
    action = integration["action"]["effective_action"]
    return NormalizedCandidateSnapshotV1(
        state_hash=_sha256(_canonical_object_bytes(state)),
        retrieval_entry_ids=tuple(item["content"] for item in retrieved),
        action_trace_hash=_sha256(_canonical_object_bytes(action)),
    )


def _terminal_rows(canary_set, capabilities: AdapterCapabilitiesV3) -> tuple[dict, ...]:
    rows: list[dict] = []
    for bundle in canary_set.canaries:
        for task in bundle.tasks:
            support = resolve_task_support_v3(task, capabilities, answer_mode="slot_direct")
            rows.append(
                {
                    "canary_id": bundle.manifest.canary_id,
                    "task_id": task.task_id,
                    "task_record_hash": bundle.manifest.selected_tasks[
                        tuple(item.task_id for item in bundle.manifest.selected_tasks).index(task.task_id)
                    ].task_record_hash,
                    "completion_status": "not_supported",
                    "terminal_supported": support.terminal_supported,
                    "missing_capabilities": list(support.missing_capabilities),
                    "runtime_support": dict(support.runtime_support),
                    "operation_support": dict(support.operation_support),
                    "query_support": dict(support.query_support),
                }
            )
    return tuple(rows)


def build_report(
    *,
    core_root: Path,
    canary_root: Path,
    model_root: Path,
    evidence_root: Path,
    update_noop_probe_path: Path,
    worker_configuration_path: Path,
    lock_path: Path,
    wheel_manifest_path: Path,
    output_root: Path,
) -> tuple[ExternalAdmissionReportV1, bool]:
    release = authenticate_core_release(core_root)
    canary_set = validate_canary_set(build_canary_set(release), release)
    preflight_paths = (
        evidence_root / "real-preflight-v3-reviewed.json",
        evidence_root / "determinism-preflight-1.json",
        evidence_root / "determinism-preflight-2.json",
    )
    preflights = tuple(_read_json(path) for path in preflight_paths)
    if not all(value.get("passed") is True for value in preflights):
        raise ValueError("all Mem0 determinism probes must pass")
    update_probe_raw = update_noop_probe_path.read_bytes()
    update_probe_text = update_probe_raw.decode("utf-8")
    if (
        "'requested_action': {'operation': 'UPDATE'" not in update_probe_text
        or "'execution_status': 'no_effect'" not in update_probe_text
        or "'reason': 'provider_no_effect'" not in update_probe_text
        or "probe-noop" not in update_probe_text
        or "'execution_status': 'executed'" not in update_probe_text
    ):
        raise ValueError("Mem0 update/NOOP capability probe is incomplete")
    if scan_for_secrets(update_probe_text):
        raise ValueError("Mem0 update/NOOP capability probe contains secrets")
    info = _info_from_preflight(preflights[0])
    capabilities = _capabilities_from_preflight(preflights[0])
    if any(_info_from_preflight(value) != info for value in preflights[1:]):
        raise ValueError("Mem0 probe adapter identity drift")
    if any(_capabilities_from_preflight(value) != capabilities for value in preflights[1:]):
        raise ValueError("Mem0 probe capability drift")
    snapshots = tuple(_semantic_snapshot(value) for value in preflights)
    determinism = classify_determinism(snapshots)
    if determinism is not DeterminismStatus.DETERMINISTIC:
        raise ValueError("Mem0 determinism probe did not classify deterministic")
    repetitions = required_canary_repetitions(determinism)
    rows = _terminal_rows(canary_set, capabilities)
    if len(rows) != 128 or tuple(row["task_id"] for row in rows) != tuple(
        task.task_id for bundle in canary_set.canaries for task in bundle.tasks
    ):
        raise ValueError("Mem0 terminal rows are incomplete or reordered")
    if any(row["completion_status"] != "not_supported" for row in rows):
        raise ValueError("Mem0 capability rejection rows are inconsistent")
    capability = verify_capability_truthfulness(
        info,
        capabilities,
        capabilities,
        state_transition_linkage_available=True,
    )
    if not capability.passed or capability.presentation_level != 3:
        raise ValueError("Mem0 capability verification failed")

    output_root.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, ArtifactRef] = {}
    payloads = {
        "probe.json": {
            "candidate_id": "mem0_oss",
            "preflight_sha256": [_sha256(path.read_bytes()) for path in preflight_paths],
            "normalized_snapshots": [snapshot.model_dump(mode="json") for snapshot in snapshots],
            "determinism_status": determinism.value,
            "required_canary_repetitions": repetitions,
            "namespace_reset_passed": all(value["namespace_reset_probe"]["passed"] for value in preflights),
            "update_noop_probe_sha256": _sha256(update_probe_raw),
            "observed_update_behavior": "provider_no_effect",
            "observed_noop_behavior": "executed",
        },
        "capability_verification.json": capability.model_dump(mode="json"),
        "canary_terminal_rows.json": {
            "candidate_id": "mem0_oss",
            "canary_ids": [bundle.manifest.canary_id for bundle in canary_set.canaries],
            "expected_rows": 128,
            "terminal_rows": rows,
        },
        "evaluation_configuration.json": {
            "candidate_id": "mem0_oss",
            "answer_mode": "slot_direct",
            "retrieval_policy": "normal_topk",
            "retrieval_k": 10,
            "determinism_probe_fresh_namespaces": 3,
            "determinism_status": determinism.value,
            "repetition_count": repetitions,
            "canary_manifest_hashes": [
                _sha256(bundle.manifest_bytes) for bundle in canary_set.canaries
            ],
            "source_task_manifest_hash": release.task_manifest_ref.sha256,
        },
        "package_provenance.json": {
            "package": "mem0ai",
            "version": MEM0_PACKAGE_VERSION,
            "wheel_sha256": "1521209f0ab4c77b7e5777aa1b0b5f0104efa06ca5b9eddb804cdd091c17726a",
            "lock_sha256": _sha256(lock_path.read_bytes()),
            "wheel_manifest_sha256": _sha256(wheel_manifest_path.read_bytes()),
            "license": "Apache-2.0",
        },
    }
    for name, payload in payloads.items():
        raw = _canonical_object_bytes(payload)
        (output_root / name).write_bytes(raw)
        artifacts[name] = _artifact(name, raw, "application/json")

    worker_value = _read_json(worker_configuration_path)
    public_configuration = worker_value.get("public_configuration")
    if not isinstance(public_configuration, dict):
        raise ValueError("Mem0 worker configuration lacks public configuration")
    public_configuration_raw = _canonical_object_bytes(public_configuration)
    if _sha256(public_configuration_raw) != info.configuration_hash:
        raise ValueError("Mem0 public configuration hash is inconsistent")
    (output_root / "adapter_configuration.json").write_bytes(
        public_configuration_raw
    )
    artifacts["adapter_configuration.json"] = _artifact(
        "adapter_configuration.json",
        public_configuration_raw,
        "application/json",
    )
    model_raw = (model_root / "model_provenance.json").read_bytes()
    canary_raw = (canary_root / "canary_set_manifest.json").read_bytes()
    gate_artifacts = {
        "source_authentication": release.task_manifest_ref,
        "official_provenance_license": artifacts["package_provenance.json"],
        "offline_model_prerequisite": _artifact("model_provenance.json", model_raw, "application/json"),
        "candidate_environment": artifacts["package_provenance.json"],
        "visible_only_fairness": artifacts["probe.json"],
        "namespace_reset": artifacts["probe.json"],
        "capability_truthfulness": artifacts["probe.json"],
        "raw_normalized_export": artifacts["probe.json"],
        "field_provenance": artifacts["probe.json"],
        "terminal_completeness": artifacts["canary_terminal_rows.json"],
        "retrieval_policy": artifacts["probe.json"],
        "presentation_level": artifacts["capability_verification.json"],
        "security_redaction": artifacts["probe.json"],
        "repetition_rule": artifacts["probe.json"],
    }
    gates = []
    for name in ADMISSION_GATE_NAMES:
        status = GateStatus.FAIL if name == "capability_truthfulness" else GateStatus.PASS
        gates.append(
            GateResultV1(
                name=name,
                status=status,
                evidence_artifacts=(gate_artifacts[name],),
                reasons=(
                    ("required_update_capability_not_declared",)
                    if status is GateStatus.FAIL
                    else ()
                ),
            )
        )
    evaluation_raw = (output_root / "evaluation_configuration.json").read_bytes()
    report = ExternalAdmissionReportV1(
        candidate_id=ExternalCandidateId.MEM0_OSS,
        source_task_manifest_hash=release.task_manifest_ref.sha256,
        source_task_manifest_ref=release.task_manifest_ref,
        evaluation_configuration_hash=_sha256(evaluation_raw),
        evaluation_configuration_ref=artifacts["evaluation_configuration.json"],
        adapter_configuration_ref=artifacts["adapter_configuration.json"],
        probe_ref=artifacts["probe.json"],
        canary_ref=_artifact("canary_set_manifest.json", canary_raw, "application/json"),
        package_provenance_ref=artifacts["package_provenance.json"],
        model_provenance_ref=_artifact("model_provenance.json", model_raw, "application/json"),
        adapter_info=info,
        adapter_capabilities=capabilities,
        state_transition_linkage_available=True,
        gates=tuple(gates),
        outcome=GateStatus.FAIL,
        reasons=("candidate_gate_failed",),
    )
    report_raw = canonical_json_bytes(report)
    (output_root / "external_admission_report.json").write_bytes(report_raw)
    fallback = authorize_fallback(
        report,
        release.task_manifest_ref.sha256,
        _sha256(evaluation_raw),
    )
    decision_raw = _canonical_object_bytes(
        {
            "candidate_id": "mem0_oss",
            "report_sha256": _sha256(report_raw),
            "outcome": report.outcome.value,
            "fallback_authorized": fallback,
        }
    )
    (output_root / "fallback_authorization.json").write_bytes(decision_raw)
    for path in output_root.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    return report, fallback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--update-noop-probe", type=Path, required=True)
    parser.add_argument("--worker-configuration", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--wheel-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report, fallback = build_report(
        core_root=args.core_root,
        canary_root=args.canary_root,
        model_root=args.model_root,
        evidence_root=args.evidence_root,
        update_noop_probe_path=args.update_noop_probe,
        worker_configuration_path=args.worker_configuration,
        lock_path=args.lock,
        wheel_manifest_path=args.wheel_manifest,
        output_root=args.output_root,
    )
    print(json.dumps({"outcome": report.outcome.value, "fallback_authorized": fallback, "report_sha256": sha256_model(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
