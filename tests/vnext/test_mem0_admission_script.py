from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import mub.vnext.external.admission_scripts.mem0_v1 as mem0_admission
from mub.vnext.contracts import ArtifactRef
from mub.vnext.contracts.v3 import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.external.admission import (
    EXTERNAL_ADMISSION_POLICY_VERSION,
    evaluate_candidate_admission,
)
from mub.vnext.external.admission_scripts.mem0_v1 import (
    _evaluation_configuration,
    _parse_update_noop_probe,
    _passing_gates,
    _public_configuration_bytes,
    build_report,
)
from mub.vnext.external.contracts import (
    ADMISSION_GATE_NAMES,
    ExternalAdmissionReportV1,
    ExternalCandidateId,
    GateStatus,
)


def _ref(name: str) -> ArtifactRef:
    return ArtifactRef(
        path=f"evidence/{name}.json",
        sha256="a" * 64,
        media_type="application/json",
    )


def test_mem0_admission_script_help_prioritizes_project_root(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[2]
    script = project / "mub/vnext/external/admission_scripts/mem0_v1.py"
    shadow = tmp_path / "shadow"
    (shadow / "mub").mkdir(parents=True)
    (shadow / "mub/__init__.py").write_text("", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(shadow), str(project)))

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _probe_line(label: str, payload: dict[str, object]) -> str:
    return f"{label} {payload!r}"


def test_update_noop_probe_requires_bound_typed_records() -> None:
    update = {
        "event_id": "probe-update",
        "requested_action": {
            "operation": "UPDATE",
            "scope": "object",
            "target_object_keys": [
                {
                    "object_type": "profile",
                    "namespace": "default",
                    "entity": "alice",
                    "attribute": "city",
                    "subkey": None,
                }
            ],
            "value": "Prague",
        },
        "effective_action": {
            "operation": "NOOP",
            "scope": None,
            "target_object_keys": [],
            "value": None,
        },
        "execution_status": "no_effect",
        "reason": "provider_no_effect",
        "error": None,
        "affected_entry_ids": [],
        "raw_result": None,
    }
    noop = {
        "event_id": "probe-noop",
        "requested_action": {
            "operation": "NOOP",
            "scope": None,
            "target_object_keys": [],
            "value": None,
        },
        "effective_action": {
            "operation": "NOOP",
            "scope": None,
            "target_object_keys": [],
            "value": None,
        },
        "execution_status": "executed",
        "reason": None,
        "error": None,
        "affected_entry_ids": [],
        "raw_result": None,
    }

    observed = _parse_update_noop_probe(
        "\n".join(
            (_probe_line("probe-update", update), _probe_line("probe-noop", noop))
        )
    )
    assert observed == {
        "observed_update_behavior": "provider_no_effect",
        "observed_noop_behavior": "executed",
    }

    wrong_target = json.loads(json.dumps(update))
    wrong_target["requested_action"]["target_object_keys"][0]["entity"] = "bob"
    with pytest.raises(ValueError, match="UPDATE record"):
        _parse_update_noop_probe(
            "\n".join(
                (
                    _probe_line("probe-update", wrong_target),
                    _probe_line("probe-noop", noop),
                )
            )
        )

    corrupted = dict(update)
    corrupted["execution_status"] = "executed"
    with pytest.raises(ValueError, match="probe-update record"):
        _parse_update_noop_probe(
            "\n".join(
                (
                    _probe_line("probe-update", corrupted),
                    _probe_line("unrelated", {"execution_status": "no_effect", "reason": "provider_no_effect"}),
                    _probe_line("probe-noop", noop),
                )
            )
        )


def test_public_configuration_is_strictly_typed_and_scanned() -> None:
    project = Path(__file__).resolve().parents[2]
    valid = (
        project
        / "results/vnext/core_task10_mem0_admission_v2/adapter_configuration.json"
    ).read_bytes()
    payload = json.loads(valid)
    assert _public_configuration_bytes(
        {"public_configuration": payload}
    ) == valid

    payload["api_key"] = "secret"
    with pytest.raises(ValueError, match="public configuration"):
        _public_configuration_bytes({"public_configuration": payload})


def test_mem0_admission_script_does_not_fail_missing_native_update() -> None:
    gates = _passing_gates({name: _ref(name) for name in ADMISSION_GATE_NAMES})
    configuration_hash = "b" * 64
    capabilities = AdapterCapabilitiesV3(
        supports_event_ingest=True,
        supports_add=True,
        supports_update=False,
        supports_isolated_reset=True,
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
        exports_action_trace=True,
        requires_evaluation_extractor=True,
        extractor_version="v1",
    )
    report = ExternalAdmissionReportV1(
        candidate_id=ExternalCandidateId.MEM0_OSS,
        source_task_manifest_hash="c" * 64,
        source_task_manifest_ref=_ref("source").model_copy(
            update={"sha256": "c" * 64}
        ),
        evaluation_configuration_hash="d" * 64,
        evaluation_configuration_ref=_ref("evaluation").model_copy(
            update={"sha256": "d" * 64}
        ),
        adapter_configuration_ref=_ref("configuration").model_copy(
            update={"sha256": configuration_hash}
        ),
        probe_ref=_ref("probe"),
        canary_ref=_ref("canary"),
        package_provenance_ref=_ref("package"),
        model_provenance_ref=_ref("model"),
        adapter_info=AdapterInfoV3(
            adapter_id="mem0_oss",
            adapter_version="1",
            system_name="mem0_oss",
            system_version="2.0.17",
            configuration_hash=configuration_hash,
            extractor_id="visible_extractor",
            extractor_version="v1",
        ),
        adapter_capabilities=capabilities,
        state_transition_linkage_available=True,
        gates=gates,
        outcome=GateStatus.PASS,
    )

    assert tuple(gate.name.value for gate in gates) == ADMISSION_GATE_NAMES
    assert all(gate.status is GateStatus.PASS for gate in gates)
    assert all(not gate.reasons for gate in gates)
    assert evaluate_candidate_admission(report)


def test_mem0_evaluation_configuration_binds_admission_policy_version() -> None:
    configuration = _evaluation_configuration(
        canary_manifest_hashes=("1" * 64, "2" * 64),
        source_task_manifest_hash="3" * 64,
        repetition_count=1,
    )

    assert configuration["admission_policy_version"] == (
        EXTERNAL_ADMISSION_POLICY_VERSION
    )


def test_admission_publication_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Path(__file__).resolve().parents[2]
    evidence_root = project / "external/mem0_2_0_17/evidence"
    required = (
        evidence_root / "real-preflight-v3-reviewed.json",
        evidence_root / "determinism-preflight-1.json",
        evidence_root / "determinism-preflight-2.json",
        evidence_root / "update-noop-probe.log",
        project / "external/mem0_2_0_17/worker-configuration-v3.json",
    )
    if any(not path.is_file() for path in required):
        pytest.skip("private Mem0 admission inputs are unavailable locally")
    output_root = tmp_path / "admission"
    real_select = mem0_admission.select_single_admitted_candidate
    monkeypatch.setattr(
        mem0_admission,
        "select_single_admitted_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected decision failure")
        ),
    )
    with pytest.raises(RuntimeError, match="injected decision failure"):
        build_report(
            core_root=project / "data/vnext/core/v3",
            canary_root=project / "results/vnext/core_task10_canaries_v1",
            model_root=project / "results/vnext/core_task10_model_provenance_v3",
            evidence_root=evidence_root,
            update_noop_probe_path=evidence_root / "update-noop-probe.log",
            worker_configuration_path=(
                project / "external/mem0_2_0_17/worker-configuration-v3.json"
            ),
            lock_path=(
                project / "requirements/external/mem0-2.0.17-linux-py310.lock"
            ),
            wheel_manifest_path=(
                project
                / "requirements/external/mem0-2.0.17-linux-py310.wheels.sha256"
            ),
            output_root=output_root,
        )
    assert not output_root.exists()
    monkeypatch.setattr(
        mem0_admission,
        "select_single_admitted_candidate",
        real_select,
    )
    report, fallback = build_report(
        core_root=project / "data/vnext/core/v3",
        canary_root=project / "results/vnext/core_task10_canaries_v1",
        model_root=project / "results/vnext/core_task10_model_provenance_v3",
        evidence_root=evidence_root,
        update_noop_probe_path=evidence_root / "update-noop-probe.log",
        worker_configuration_path=(
            project / "external/mem0_2_0_17/worker-configuration-v3.json"
        ),
        lock_path=project / "requirements/external/mem0-2.0.17-linux-py310.lock",
        wheel_manifest_path=(
            project / "requirements/external/mem0-2.0.17-linux-py310.wheels.sha256"
        ),
        output_root=output_root,
    )
    assert report.outcome is GateStatus.PASS
    assert not fallback
    assert (output_root / "admission_decision.json").is_file()


def test_mem0_admission_script_builds_a_pass_report_and_decision(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[2]
    evidence_root = project / "external/mem0_2_0_17/evidence"
    required = (
        evidence_root / "real-preflight-v3-reviewed.json",
        evidence_root / "determinism-preflight-1.json",
        evidence_root / "determinism-preflight-2.json",
        evidence_root / "update-noop-probe.log",
        project / "external/mem0_2_0_17/worker-configuration-v3.json",
    )
    if any(not path.is_file() for path in required):
        pytest.skip("private Mem0 admission inputs are unavailable locally")
    output_root = tmp_path / "admission"

    report, fallback = build_report(
        core_root=project / "data/vnext/core/v3",
        canary_root=project / "results/vnext/core_task10_canaries_v1",
        model_root=project / "results/vnext/core_task10_model_provenance_v3",
        evidence_root=evidence_root,
        update_noop_probe_path=(
            project / "external/mem0_2_0_17/evidence/update-noop-probe.log"
        ),
        worker_configuration_path=(
            project / "external/mem0_2_0_17/worker-configuration-v3.json"
        ),
        lock_path=(
            project / "requirements/external/mem0-2.0.17-linux-py310.lock"
        ),
        wheel_manifest_path=(
            project
            / "requirements/external/mem0-2.0.17-linux-py310.wheels.sha256"
        ),
        output_root=output_root,
    )

    assert report.outcome is GateStatus.PASS
    assert not fallback
    assert (output_root / "admission_decision.json").is_file()
    assert not (output_root / "fallback_authorization.json").exists()
