from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import ArtifactRef
from mub.vnext.contracts.v3 import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.io import sha256_model
import mub.vnext.external.admission as admission_module
from mub.vnext.external import (
    ADMISSION_GATE_NAMES,
    AdmissionDecisionStatus,
    AdmissionDecisionV1,
    CandidateReportRefV1,
    ExternalAdmissionReportV1,
    ExternalCandidateId,
    GateResultV1,
    GateStatus,
    authorize_fallback,
    evaluate_candidate_admission,
    select_single_admitted_candidate,
    validate_artifact_provenance,
)

H_MANIFEST = "1" * 64
H_EVALUATION_CONFIG = "2" * 64
H_MEM0_CONFIG = "7" * 64
H_LANGGRAPH_CONFIG = "8" * 64
H_PROBE = "3" * 64
H_CANARY = "4" * 64


def _ref(name: str, digest: str) -> ArtifactRef:
    return ArtifactRef(path=f"evidence/{name}.json", sha256=digest, media_type="application/json")


def _capabilities(level: int, **changes: object) -> AdapterCapabilitiesV3:
    values: dict[str, object] = {
        "supports_event_ingest": True,
        "supports_add": True,
        "supports_update": True,
    }
    if level == 0:
        values["supports_native_answer"] = True
    elif level == 1:
        values["exports_entries"] = True
    elif level == 2:
        values.update(
            supports_isolated_reset=True,
            exports_entries=True,
            exports_object_keys=True,
            exports_values=True,
        )
    elif level == 3:
        values.update(
            supports_isolated_reset=True,
            exports_entries=True,
            exports_object_keys=True,
            exports_values=True,
            exports_action_trace=True,
        )
    else:
        raise AssertionError(level)
    values.update(changes)
    return AdapterCapabilitiesV3(**values)


def _gates(overrides: Mapping[str, GateStatus] | None = None) -> tuple[GateResultV1, ...]:
    statuses = dict.fromkeys(ADMISSION_GATE_NAMES, GateStatus.PASS)
    statuses.update(overrides or {})
    return tuple(
        GateResultV1(
            name=name,
            status=statuses[name],
            evidence_artifacts=(_ref(f"gate-{name}", "a" * 64),),
            reasons=(() if statuses[name] is GateStatus.PASS else (f"{name}_did_not_pass",)),
        )
        for name in ADMISSION_GATE_NAMES
    )


def _outcome(gates: tuple[GateResultV1, ...]) -> GateStatus:
    statuses = {gate.status for gate in gates}
    for status in (GateStatus.FAIL, GateStatus.BLOCKED, GateStatus.NOT_RUN):
        if status in statuses:
            return status
    return GateStatus.PASS


def _report(
    candidate: ExternalCandidateId = ExternalCandidateId.MEM0_OSS,
    *,
    level: int = 2,
    linkage: bool = False,
    gate_overrides: Mapping[str, GateStatus] | None = None,
    manifest_hash: str = H_MANIFEST,
    evaluation_configuration_hash: str = H_EVALUATION_CONFIG,
    adapter_configuration_hash: str | None = None,
    capabilities: AdapterCapabilitiesV3 | None = None,
    extractor_id: str | None = None,
    extractor_version: str | None = None,
) -> ExternalAdmissionReportV1:
    gates = _gates(gate_overrides)
    if adapter_configuration_hash is None:
        adapter_configuration_hash = (
            H_MEM0_CONFIG
            if candidate is ExternalCandidateId.MEM0_OSS
            else H_LANGGRAPH_CONFIG
        )
    return ExternalAdmissionReportV1(
        candidate_id=candidate,
        source_task_manifest_hash=manifest_hash,
        source_task_manifest_ref=_ref("source-task-manifest", manifest_hash),
        evaluation_configuration_hash=evaluation_configuration_hash,
        evaluation_configuration_ref=_ref(
            "evaluation-configuration", evaluation_configuration_hash
        ),
        adapter_configuration_ref=_ref(
            f"{candidate.value}-adapter-configuration", adapter_configuration_hash
        ),
        probe_ref=_ref("probe", H_PROBE),
        canary_ref=_ref("canary", H_CANARY),
        package_provenance_ref=_ref("package-provenance", "5" * 64),
        model_provenance_ref=_ref("model-provenance", "6" * 64),
        adapter_info=AdapterInfoV3(
            adapter_id=candidate.value,
            adapter_version="1",
            system_name=candidate.value,
            system_version="1",
            configuration_hash=adapter_configuration_hash,
            extractor_id=extractor_id,
            extractor_version=extractor_version,
        ),
        adapter_capabilities=capabilities or _capabilities(level),
        state_transition_linkage_available=linkage,
        gates=gates,
        outcome=_outcome(gates),
        reasons=(() if _outcome(gates) is GateStatus.PASS else ("candidate_gate_failed",)),
    )


def _constructed_report(
    report: ExternalAdmissionReportV1, **changes: object
) -> ExternalAdmissionReportV1:
    fields = dict(report.__dict__)
    fields.update(changes)
    return ExternalAdmissionReportV1.model_construct(**fields)


def _constructed_gate(gate: GateResultV1, **changes: object) -> GateResultV1:
    fields = dict(gate.__dict__)
    fields.update(changes)
    return GateResultV1.model_construct(**fields)


def test_gate_contract_has_exact_names_statuses_strict_ids_and_is_immutable() -> None:
    assert ADMISSION_GATE_NAMES == (
        "source_authentication",
        "official_provenance_license",
        "offline_model_prerequisite",
        "candidate_environment",
        "visible_only_fairness",
        "namespace_reset",
        "capability_truthfulness",
        "raw_normalized_export",
        "field_provenance",
        "terminal_completeness",
        "retrieval_policy",
        "presentation_level",
        "security_redaction",
        "repetition_rule",
    )
    assert {status.value for status in GateStatus} == {"pass", "fail", "blocked", "not_run"}
    gate = _gates()[0]
    with pytest.raises(ValidationError):
        GateResultV1(name="unknown_gate", status="pass")
    with pytest.raises(ValidationError):
        GateResultV1(name="source_authentication", status="skipped")
    with pytest.raises(ValidationError):
        GateResultV1(name="source_authentication", status="pass", reasons=(1,))
    with pytest.raises(ValidationError):
        gate.status = GateStatus.FAIL  # type: ignore[misc]


def test_report_requires_every_gate_once_and_binds_authenticated_hashes_and_refs() -> None:
    report = _report()
    with pytest.raises(ValidationError, match="all fixed admission gates"):
        ExternalAdmissionReportV1.model_validate(
            {**report.model_dump(mode="python"), "gates": report.gates[:-1]}
        )
    with pytest.raises(ValidationError, match="all fixed admission gates"):
        ExternalAdmissionReportV1.model_validate(
            {**report.model_dump(mode="python"), "gates": report.gates[:-1] + (report.gates[0],)}
        )
    with pytest.raises(ValidationError, match="evaluation configuration hash"):
        report.model_copy(update={"evaluation_configuration_hash": "9" * 64})
    with pytest.raises(ValidationError, match="adapter configuration ref"):
        report.model_copy(
            update={
                "adapter_configuration_ref": _ref(
                    "adapter-configuration", "9" * 64
                )
            }
        )
    with pytest.raises(ValidationError):
        report.model_copy(update={"source_task_manifest_hash": "not-a-hash"})
    with pytest.raises(ValidationError):
        report.model_copy(update={"candidate_id": "memory_r1"})
    with pytest.raises(ValidationError):
        report.gates += report.gates  # type: ignore[misc]
    assert (
        report.evaluation_configuration_ref.sha256
        == report.evaluation_configuration_hash
    )
    assert (
        report.adapter_configuration_ref.sha256
        == report.adapter_info.configuration_hash
    )
    assert sha256_model(report) == sha256_model(
        ExternalAdmissionReportV1.model_validate(report.model_dump(mode="json"))
    )


def test_report_contract_requires_source_package_and_model_provenance_refs() -> None:
    required_refs = {
        "source_task_manifest_ref",
        "package_provenance_ref",
        "model_provenance_ref",
    }
    assert required_refs <= set(ExternalAdmissionReportV1.model_fields)
    payload = _report().model_dump(mode="python")
    for field in required_refs:
        incomplete = dict(payload)
        incomplete.pop(field)
        with pytest.raises(ValidationError, match="Field required"):
            ExternalAdmissionReportV1.model_validate(incomplete)


def test_source_task_manifest_ref_is_hash_bound() -> None:
    report = _report()
    with pytest.raises(ValidationError, match="source task manifest hash"):
        report.model_copy(
            update={
                "source_task_manifest_ref": _ref("source-task-manifest", "9" * 64)
            }
        )


def test_public_provenance_and_report_revalidation_reject_constructed_and_subclass_spoofs() -> None:
    report = _report()
    bad_sha_ref = ArtifactRef.model_construct(
        path="evidence/bad-sha.json",
        sha256="not-a-hash",
        media_type="application/json",
        record_count=None,
    )
    bad_media_ref = ArtifactRef.model_construct(
        path="evidence/bad-media.json",
        sha256="a" * 64,
        media_type=7,
        record_count=None,
    )
    for bad_ref in (bad_sha_ref, bad_media_ref):
        with pytest.raises((ValidationError, ValueError)):
            validate_artifact_provenance(bad_ref)
    with pytest.raises((ValidationError, ValueError)):
        GateResultV1(
            name="source_authentication",
            status="pass",
            evidence_artifacts=(bad_sha_ref,),
        )
    with pytest.raises((ValidationError, ValueError)):
        report.model_copy(update={"probe_ref": bad_sha_ref})

    raw_gate = GateResultV1.model_construct(
        contract_version="1.0.0",
        name="source_authentication",
        status="pass",
        evidence_artifacts=(report.gates[0].evidence_artifacts[0],),
        reasons=(),
    )
    with pytest.raises((ValidationError, ValueError)):
        report.model_copy(update={"gates": (raw_gate, *report.gates[1:])})

    class ArtifactRefSubclass(ArtifactRef):
        pass

    with pytest.raises(ValueError, match="exact ArtifactRef"):
        validate_artifact_provenance(
            ArtifactRefSubclass(
                path="evidence/subclass.json",
                sha256="a" * 64,
                media_type="application/json",
            )
        )

    class ReportSubclass(ExternalAdmissionReportV1):
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            return report.model_dump(mode="python")

    subclass = ReportSubclass.model_validate(report.model_dump(mode="python"))
    with pytest.raises(ValueError, match="exact ExternalAdmissionReportV1"):
        admission_module._revalidate_report(subclass)


def test_all_public_admission_apis_reject_malformed_adapter_nested_contracts() -> None:
    report = _report()
    capability_fields = dict(report.adapter_capabilities.__dict__)
    info_fields = dict(report.adapter_info.__dict__)

    malformed_capabilities = [
        AdapterCapabilitiesV3.model_construct(
            **{**capability_fields, "supports_add": "true"}
        ),
    ]
    missing_capability_fields = dict(capability_fields)
    missing_capability_fields.pop("supports_noop")
    missing_capabilities = AdapterCapabilitiesV3.model_construct(
        **missing_capability_fields
    )
    missing_capabilities.__dict__.pop("supports_noop", None)
    malformed_capabilities.append(missing_capabilities)

    class CapabilitiesSubclass(AdapterCapabilitiesV3):
        pass

    malformed_capabilities.append(
        CapabilitiesSubclass.model_validate(
            report.adapter_capabilities.model_dump(mode="python")
        )
    )

    malformed_infos = [
        AdapterInfoV3.model_construct(
            **{**info_fields, "configuration_hash": "bad"}
        ),
        AdapterInfoV3.model_construct(
            **{**info_fields, "adapter_version": " "}
        ),
    ]
    missing_info_fields = dict(info_fields)
    missing_info_fields.pop("system_version")
    missing_info = AdapterInfoV3.model_construct(**missing_info_fields)
    missing_info.__dict__.pop("system_version", None)
    malformed_infos.append(missing_info)

    class AdapterInfoSubclass(AdapterInfoV3):
        pass

    malformed_infos.append(
        AdapterInfoSubclass.model_validate(
            report.adapter_info.model_dump(mode="python")
        )
    )

    malformed_reports = [
        *(
            _constructed_report(
                report,
                adapter_capabilities=capabilities,
            )
            for capabilities in malformed_capabilities
        ),
        *(
            _constructed_report(report, adapter_info=adapter_info)
            for adapter_info in malformed_infos
        ),
    ]
    for malformed_report in malformed_reports:
        with pytest.raises((ValidationError, ValueError)):
            evaluate_candidate_admission(malformed_report)
        with pytest.raises((ValidationError, ValueError)):
            authorize_fallback(
                malformed_report,
                H_MANIFEST,
                H_EVALUATION_CONFIG,
            )
        with pytest.raises((ValidationError, ValueError)):
            select_single_admitted_candidate(
                (malformed_report,),
                current_manifest_hash=H_MANIFEST,
                current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
            )


def test_prior_aliases_are_rejected_across_all_evidence_trust_boundaries() -> None:
    alias_refs = (
        ArtifactRef(
            path="evidence/memory_r1.json.bak",
            sha256="d" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="prefix/scripts/eval_mem0_baseline.py.tar.gz",
            sha256="f" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="scripts/eval_mem0_baseline.py/results.json",
            sha256="f" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="prefix/baselines/memory_r1_agent.py/results.json",
            sha256="f" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="scripts/EVAL_M~1.PY",
            sha256="7" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="prefix/scripts/eval_m~1.py",
            sha256="8" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="prefix/SCRIPTS/EvAl_M~1.Py",
            sha256="b" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="CONIN$.json",
            sha256="1" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="conout$.txt",
            sha256="2" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_​r1/result.json",
            sha256="3" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_⁠r1/result.json",
            sha256="4" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_­r1/result.json",
            sha256="5" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_͏r1/result.json",
            sha256="6" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_️r1/result.json",
            sha256="c" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_⁥r1/result.json",
            sha256="0" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_￰r1/result.json",
            sha256="d" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/memory_￸r1/result.json",
            sha256="e" * 64,
            media_type="application/json",
        ),
    )
    allowed_refs = (
        ArtifactRef(
            path="evidence/memory_r1_rejection_analysis.json.bak",
            sha256="e" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="scripts/eval_mem0_baseline_rejection_analysis.py.bak",
            sha256="9" * 64,
            media_type="application/json",
        ),
        ArtifactRef(
            path="evidence/café.json",
            sha256="a" * 64,
            media_type="application/json",
        ),
    )
    for allowed_ref in allowed_refs:
        assert validate_artifact_provenance(allowed_ref) == allowed_ref

    report = _report()
    rejection_pattern = (
        "denied prior-system evidence|portable canonical relative path"
    )
    for alias_ref in alias_refs:
        with pytest.raises(ValueError, match=rejection_pattern):
            validate_artifact_provenance(alias_ref)
        with pytest.raises(ValidationError, match=rejection_pattern):
            GateResultV1(
                name="source_authentication",
                status="pass",
                evidence_artifacts=(alias_ref,),
            )
        with pytest.raises(ValidationError, match=rejection_pattern):
            report.model_copy(update={"probe_ref": alias_ref})

        constructed_report = _constructed_report(report, probe_ref=alias_ref)
        with pytest.raises(ValidationError, match=rejection_pattern):
            evaluate_candidate_admission(constructed_report)
        with pytest.raises(ValidationError, match=rejection_pattern):
            authorize_fallback(
                constructed_report,
                H_MANIFEST,
                H_EVALUATION_CONFIG,
            )
        with pytest.raises(ValidationError, match=rejection_pattern):
            select_single_admitted_candidate(
                (constructed_report,),
                current_manifest_hash=H_MANIFEST,
                current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
            )


def test_report_reasons_are_canonical_for_aggregate_outcome() -> None:
    with pytest.raises(ValidationError, match="reasons"):
        _report().model_copy(update={"reasons": ("candidate_gate_failed",)})

    failed = _report(gate_overrides={"presentation_level": GateStatus.FAIL})
    for reasons in ((), ("other_reason",), ("candidate_gate_failed", "other_reason")):
        with pytest.raises(ValidationError, match="reasons"):
            failed.model_copy(update={"reasons": reasons})


def test_report_rejects_candidate_identity_spoofing() -> None:
    for candidate, wrong_identity in (
        (ExternalCandidateId.MEM0_OSS, "local_approximation"),
        (
            ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
            ExternalCandidateId.MEM0_OSS.value,
        ),
    ):
        report = _report(candidate)
        spoofed_info = report.adapter_info.model_copy(
            update={
                "adapter_id": wrong_identity,
                "system_name": wrong_identity,
            }
        )
        with pytest.raises(ValidationError, match="candidate identity"):
            report.model_copy(update={"adapter_info": spoofed_info})


def test_report_rejects_denied_direct_and_gate_evidence_paths() -> None:
    report = _report()
    denied_ref = ArtifactRef(
        path="evidence/local_approximation/config.json",
        sha256="b" * 64,
        media_type="application/json",
    )
    for field in (
        "source_task_manifest_ref",
        "evaluation_configuration_ref",
        "adapter_configuration_ref",
        "probe_ref",
        "canary_ref",
        "package_provenance_ref",
        "model_provenance_ref",
    ):
        digest = denied_ref.sha256
        if field == "source_task_manifest_ref":
            digest = report.source_task_manifest_hash
        elif field == "evaluation_configuration_ref":
            digest = report.evaluation_configuration_hash
        elif field == "adapter_configuration_ref":
            digest = report.adapter_info.configuration_hash
        replacement = ArtifactRef(
            path=denied_ref.path,
            sha256=digest,
            media_type=denied_ref.media_type,
        )
        with pytest.raises(ValidationError, match="denied prior-system evidence"):
            report.model_copy(update={field: replacement})

    for denied_gate_path in (
        "evidence/Memory-R1/probe.json",
        "evidence/local approximation/probe.json",
        "scripts/eval_mem0_baseline.py",
    ):
        with pytest.raises(ValidationError, match="denied prior-system evidence"):
            report.gates[0].model_copy(
                update={
                    "evidence_artifacts": (
                        ArtifactRef(
                            path=denied_gate_path,
                            sha256="c" * 64,
                            media_type="application/json",
                        ),
                    )
                }
            )


def test_artifact_paths_must_be_portable_canonical_relative_paths() -> None:
    report = _report()
    reserved_device_names = (
        "CON",
        "prn",
        "Aux",
        "nul",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    )
    default_ignorable_characters = tuple(
        chr(code_point)
        for code_point in (
            0x0085,
            0x00AD,
            0x034F,
            0x115F,
            0x1160,
            0x17B4,
            0x17B5,
            0x180B,
            0x180F,
            0x200B,
            0x2060,
            0x2065,
            0x3164,
            0xFE00,
            0xFE0F,
            0xFEFF,
            0xFFA0,
            0xFFF0,
            0xFFF8,
            0x1BCA0,
            0x1BCA3,
            0x1D173,
            0x1D17A,
            0xE0000,
            0xE0FFF,
        )
    )
    invalid_paths = (
        "scripts/./eval_mem0_baseline.py",
        "scripts./eval_mem0_baseline.py",
        "scripts /eval_mem0_baseline.py",
        "evidence/.. /probe.json",
        "evidence/. /probe.json",
        *(f"evidence/{name}.json" for name in reserved_device_names),
        "evidence/CON .json",
        "evidence/com¹.json",
        "evidence/CoM².json",
        "evidence/COM³.json",
        "evidence/lpt¹.json",
        "evidence/LpT².json",
        "evidence/LPT³.json",
        "scripts//eval_mem0_baseline.py",
        "evidence/../probe.json",
        "../probe.json",
        "C:/evidence/probe.json",
        "C:\\evidence\\probe.json",
        "\\\\server\\share\\probe.json",
        "//server/share/probe.json",
        "evidence\\..\\probe.json",
        "/absolute/probe.json",
        "evidence/",
        "./evidence/probe.json",
        "evidence/cafe\u0301.json",
        *(
            f"evidence/default{character}.json"
            for character in default_ignorable_characters
        ),
        "evidence/bad<name.json",
        "evidence/bad>name.json",
        "evidence/bad\"name.json",
        "evidence/bad|name.json",
        "evidence/bad?name.json",
        "evidence/bad*name.json",
        "evidence/del\x7f.json",
        "evidence/lone\ud800.json",
    )
    for path in invalid_paths:
        ref = ArtifactRef(
            path=path,
            sha256="d" * 64,
            media_type="application/json",
        )
        with pytest.raises(ValidationError, match="portable canonical relative path"):
            report.model_copy(update={"probe_ref": ref})
        with pytest.raises(ValidationError, match="portable canonical relative path"):
            report.gates[0].model_copy(update={"evidence_artifacts": (ref,)})

    composed = ArtifactRef(
        path="evidence/caf\u00e9.json",
        sha256="d" * 64,
        media_type="application/json",
    )
    assert report.model_copy(update={"probe_ref": composed}).probe_ref == composed

    for path in (
        "evidence/COM10.json",
        "evidence/LPT10.json",
        "evidence/console.json",
    ):
        ref = ArtifactRef(
            path=path,
            sha256="d" * 64,
            media_type="application/json",
        )
        assert report.model_copy(update={"probe_ref": ref}).probe_ref == ref


def test_all_pass_report_cannot_use_zero_evidence_gates() -> None:
    report = _report()
    with pytest.raises(ValidationError, match="PASS gates require evidence"):
        tuple(
            gate.model_copy(update={"evidence_artifacts": ()})
            for gate in report.gates
        )


def test_gate_result_status_controls_required_evidence_and_reasons() -> None:
    evidence = (_ref("gate-evidence", "a" * 64),)
    invalid = (
        {"status": GateStatus.PASS, "evidence_artifacts": (), "reasons": ()},
        {"status": GateStatus.PASS, "evidence_artifacts": evidence, "reasons": ("unexpected",)},
        {"status": GateStatus.FAIL, "evidence_artifacts": (), "reasons": ("failed",)},
        {"status": GateStatus.FAIL, "evidence_artifacts": evidence, "reasons": ()},
        {"status": GateStatus.BLOCKED, "evidence_artifacts": (), "reasons": ()},
        {"status": GateStatus.NOT_RUN, "evidence_artifacts": evidence, "reasons": ()},
    )
    for fields in invalid:
        with pytest.raises(ValidationError):
            GateResultV1(name="source_authentication", **fields)

    for status in (GateStatus.BLOCKED, GateStatus.NOT_RUN):
        gate = GateResultV1(
            name="source_authentication",
            status=status,
            reasons=("not_completed",),
        )
        assert gate.evidence_artifacts == ()


def test_report_outcome_must_match_fixed_gate_results() -> None:
    passed = _report()
    with pytest.raises(ValidationError, match="outcome must equal"):
        passed.model_copy(update={"outcome": GateStatus.FAIL})


def test_evaluate_revalidates_constructed_top_level_report() -> None:
    report = _constructed_report(
        _report(),
        candidate_id=ExternalCandidateId.MEM0_OSS.value,
    )
    with pytest.raises(ValidationError):
        evaluate_candidate_admission(report)


def test_authorize_fallback_revalidates_constructed_top_level_report() -> None:
    report = _constructed_report(
        _report(gate_overrides={"presentation_level": GateStatus.FAIL}),
        candidate_id=ExternalCandidateId.MEM0_OSS.value,
    )
    with pytest.raises(ValidationError):
        authorize_fallback(report, H_MANIFEST, H_EVALUATION_CONFIG)


def test_selection_revalidates_constructed_top_level_report() -> None:
    report = _constructed_report(
        _report(ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE),
        candidate_id=ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE.value,
    )
    with pytest.raises(ValidationError):
        select_single_admitted_candidate(
            (report,),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )


def test_selection_revalidates_unknown_constructed_candidate_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _constructed_report(_report(), candidate_id="unknown")
    calls = 0
    original = admission_module._revalidate_report

    def counted_revalidate(
        candidate_report: ExternalAdmissionReportV1,
    ) -> ExternalAdmissionReportV1:
        nonlocal calls
        calls += 1
        return original(candidate_report)

    monkeypatch.setattr(admission_module, "_revalidate_report", counted_revalidate)

    def unvalidated_sort_key_must_not_run(
        candidate_report: ExternalAdmissionReportV1,
    ) -> int:
        raise AssertionError("unvalidated reports must not reach canonical sorting")

    monkeypatch.setattr(
        admission_module, "_candidate_sort_key", unvalidated_sort_key_must_not_run
    )
    with pytest.raises(ValidationError):
        admission_module.select_single_admitted_candidate(
            (report,),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )
    assert calls == 1


def _report_with_invalid_constructed_pass_gate(
    report: ExternalAdmissionReportV1,
) -> ExternalAdmissionReportV1:
    invalid_gate = _constructed_gate(
        report.gates[0],
        evidence_artifacts=(),
    )
    return _constructed_report(
        report,
        gates=(invalid_gate, *report.gates[1:]),
    )


def test_evaluate_revalidates_constructed_nested_model() -> None:
    report = _report_with_invalid_constructed_pass_gate(_report())
    with pytest.raises(ValidationError, match="PASS gates require evidence"):
        evaluate_candidate_admission(report)


def test_selection_revalidates_constructed_nested_model() -> None:
    report = _report_with_invalid_constructed_pass_gate(_report())
    with pytest.raises(ValidationError, match="PASS gates require evidence"):
        select_single_admitted_candidate(
            (report,),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )


def test_authorize_fallback_revalidates_constructed_nested_model() -> None:
    report = _report_with_invalid_constructed_pass_gate(
        _report(gate_overrides={"presentation_level": GateStatus.FAIL})
    )
    with pytest.raises(ValidationError, match="PASS gates require evidence"):
        authorize_fallback(report, H_MANIFEST, H_EVALUATION_CONFIG)


def test_candidate_admission_recomputes_level_and_requires_every_gate_pass() -> None:
    assert not evaluate_candidate_admission(_report(level=0))
    assert not evaluate_candidate_admission(_report(level=1))
    assert evaluate_candidate_admission(_report(level=2))
    assert evaluate_candidate_admission(_report(level=3, linkage=True))
    assert not evaluate_candidate_admission(
        _report(level=3, linkage=False, gate_overrides={"namespace_reset": GateStatus.FAIL})
    )


def test_candidate_admission_requires_ingest_add_update_and_coherent_extractor() -> None:
    base = _capabilities(2)
    for field in ("supports_event_ingest", "supports_add", "supports_update"):
        assert not evaluate_candidate_admission(
            _report(capabilities=base.model_copy(update={field: False}))
        )

    for capability_version, extractor_id, info_version in (
        (None, "extractor", "v1"),
        ("", "extractor", "v1"),
        ("v1", None, "v1"),
        ("v1", "extractor", None),
        ("v1", "extractor", "v2"),
        (" v1 ", "extractor", " v1 "),
        ("v1", " extractor ", "v1"),
        ("v1", "extractor", " v1 "),
    ):
        capabilities = _capabilities(
            2,
            requires_evaluation_extractor=True,
            extractor_version=capability_version,
        )
        assert not evaluate_candidate_admission(
            _report(
                capabilities=capabilities,
                extractor_id=extractor_id,
                extractor_version=info_version,
            )
        )

    assert evaluate_candidate_admission(
        _report(
            capabilities=_capabilities(
                2,
                requires_evaluation_extractor=True,
                extractor_version="v1",
            ),
            extractor_id="extractor",
            extractor_version="v1",
        )
    )


def test_only_authenticated_same_manifest_and_configuration_mem0_failure_authorizes_fallback() -> None:
    failed = _report(gate_overrides={"presentation_level": GateStatus.FAIL})
    assert authorize_fallback(failed, H_MANIFEST, H_EVALUATION_CONFIG)
    assert not authorize_fallback(_report(), H_MANIFEST, H_EVALUATION_CONFIG)
    assert not authorize_fallback(
        _report(gate_overrides={"offline_model_prerequisite": GateStatus.FAIL}),
        H_MANIFEST,
        H_EVALUATION_CONFIG,
    )
    for invariant_gate in (
        "source_authentication",
        "official_provenance_license",
        "offline_model_prerequisite",
        "security_redaction",
        "repetition_rule",
    ):
        assert not authorize_fallback(
            _report(
                gate_overrides={
                    invariant_gate: GateStatus.FAIL,
                    "presentation_level": GateStatus.FAIL,
                }
            ),
            H_MANIFEST,
            H_EVALUATION_CONFIG,
        )
    assert not authorize_fallback(
        _report(gate_overrides={"presentation_level": GateStatus.BLOCKED}), H_MANIFEST, H_EVALUATION_CONFIG
    )
    assert not authorize_fallback(
        _report(gate_overrides={"presentation_level": GateStatus.NOT_RUN}), H_MANIFEST, H_EVALUATION_CONFIG
    )
    assert not authorize_fallback(
        _report(
            gate_overrides={
                "presentation_level": GateStatus.FAIL,
                "candidate_environment": GateStatus.BLOCKED,
            }
        ),
        H_MANIFEST,
        H_EVALUATION_CONFIG,
    )
    assert not authorize_fallback(
        _report(
            gate_overrides={
                "presentation_level": GateStatus.FAIL,
                "candidate_environment": GateStatus.NOT_RUN,
            }
        ),
        H_MANIFEST,
        H_EVALUATION_CONFIG,
    )
    for gate_name in ADMISSION_GATE_NAMES:
        for status in (GateStatus.BLOCKED, GateStatus.NOT_RUN):
            overrides = {
                "presentation_level": GateStatus.FAIL,
                gate_name: status,
            }
            assert not authorize_fallback(
                _report(gate_overrides=overrides),
                H_MANIFEST,
                H_EVALUATION_CONFIG,
            )
    assert not authorize_fallback(failed, "8" * 64, H_EVALUATION_CONFIG)
    assert not authorize_fallback(failed, H_MANIFEST, "8" * 64)
    unauthenticated = _report(
        gate_overrides={
            "source_authentication": GateStatus.FAIL,
            "presentation_level": GateStatus.FAIL,
        }
    )
    assert not authorize_fallback(unauthenticated, H_MANIFEST, H_EVALUATION_CONFIG)
    with pytest.raises(ValueError, match="Mem0"):
        authorize_fallback(
            _report(
                ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
                gate_overrides={"presentation_level": GateStatus.FAIL},
            ),
            H_MANIFEST,
            H_EVALUATION_CONFIG,
        )


def test_selection_requires_strict_authenticated_current_hashes_and_matching_reports() -> None:
    for manifest_hash, configuration_hash in (
        ("not-a-hash", H_EVALUATION_CONFIG),
        (H_MANIFEST, "A" * 64),
    ):
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            select_single_admitted_candidate(
                (),
                current_manifest_hash=manifest_hash,
                current_evaluation_configuration_hash=configuration_hash,
            )

    with pytest.raises(ValueError, match="current manifest/evaluation configuration"):
        select_single_admitted_candidate(
            (_report(manifest_hash="8" * 64),),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )
    with pytest.raises(ValueError, match="current manifest/evaluation configuration"):
        select_single_admitted_candidate(
            (_report(evaluation_configuration_hash="8" * 64),),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )


def test_fallback_cannot_authenticate_itself_with_selected_report_hashes() -> None:
    forged_manifest = "8" * 64
    forged_configuration = "9" * 64
    failed_mem0 = _report(
        gate_overrides={"presentation_level": GateStatus.FAIL},
        manifest_hash=forged_manifest,
        evaluation_configuration_hash=forged_configuration,
    )
    langgraph = _report(
        ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
        manifest_hash=forged_manifest,
        evaluation_configuration_hash=forged_configuration,
    )
    with pytest.raises(ValueError, match="current manifest/evaluation configuration"):
        select_single_admitted_candidate(
            (failed_mem0, langgraph),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )


def test_selection_rejects_size_duplicates_and_binding_before_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = _report(gate_overrides={"presentation_level": GateStatus.BLOCKED})

    def hash_must_not_run(report: ExternalAdmissionReportV1) -> str:
        raise AssertionError("invalid selection input must fail before hashing")

    monkeypatch.setattr(admission_module, "sha256_model", hash_must_not_run)
    with pytest.raises(ValueError, match="at most two"):
        select_single_admitted_candidate(
            (blocked, blocked, blocked),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )
    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        select_single_admitted_candidate(
            (blocked, blocked),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )
    with pytest.raises(ValueError, match="current manifest/evaluation configuration"):
        select_single_admitted_candidate(
            (_report(manifest_hash="8" * 64),),
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )


def test_selection_rejects_duplicate_candidate_ids_for_all_outcomes() -> None:
    passed = _report()
    for status in (GateStatus.FAIL, GateStatus.BLOCKED, GateStatus.NOT_RUN):
        other = _report(gate_overrides={"presentation_level": status})
        assert sha256_model(passed) != sha256_model(other)
        with pytest.raises(ValueError, match="duplicate candidate IDs"):
            select_single_admitted_candidate(
                (passed, other),
                current_manifest_hash=H_MANIFEST,
                current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
            )


def test_selection_admits_exactly_one_candidate_or_stops_release() -> None:
    mem0 = _report()
    decision = select_single_admitted_candidate(
        (mem0,),
        current_manifest_hash=H_MANIFEST,
        current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
    )
    assert decision.status is AdmissionDecisionStatus.ADMITTED
    assert decision.source_task_manifest_hash == H_MANIFEST
    assert decision.evaluation_configuration_hash == H_EVALUATION_CONFIG
    assert decision.admitted_candidate_id is ExternalCandidateId.MEM0_OSS
    assert decision.report_hashes == (sha256_model(mem0),)
    assert decision.admitted_report_hash == sha256_model(mem0)

    failed_mem0 = _report(gate_overrides={"presentation_level": GateStatus.FAIL})
    langgraph = _report(ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE)
    fallback = select_single_admitted_candidate(
        (failed_mem0, langgraph),
        current_manifest_hash=H_MANIFEST,
        current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
    )
    assert fallback.status is AdmissionDecisionStatus.ADMITTED
    assert fallback.admitted_candidate_id is ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE
    assert fallback.admitted_report_hash == sha256_model(langgraph)

    for reports in ((), (failed_mem0,), (langgraph,)):
        stopped = select_single_admitted_candidate(
            reports,
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )
        assert stopped.status is AdmissionDecisionStatus.RELEASE_STOPPED
        assert stopped.admitted_candidate_id is None
        assert stopped.admitted_report_hash is None


def test_langgraph_participation_always_requires_authorized_mem0_fallback() -> None:
    mem0 = _report()
    passing_langgraph = _report(
        ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE
    )
    failing_langgraph = _report(
        ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
        gate_overrides={"presentation_level": GateStatus.FAIL},
    )
    assert not authorize_fallback(mem0, H_MANIFEST, H_EVALUATION_CONFIG)

    for reports in (
        (mem0, passing_langgraph),
        (mem0, failing_langgraph),
        (passing_langgraph,),
        (
            _report(
                gate_overrides={
                    "source_authentication": GateStatus.FAIL,
                    "presentation_level": GateStatus.FAIL,
                }
            ),
            passing_langgraph,
        ),
        (
            _report(gate_overrides={"presentation_level": GateStatus.BLOCKED}),
            passing_langgraph,
        ),
    ):
        stopped = select_single_admitted_candidate(
            reports,
            current_manifest_hash=H_MANIFEST,
            current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
        )
        assert stopped.status is AdmissionDecisionStatus.RELEASE_STOPPED
        assert stopped.reasons == ("langgraph_fallback_not_authorized",)

    failed_mem0 = _report(
        gate_overrides={"presentation_level": GateStatus.FAIL}
    )
    admitted = select_single_admitted_candidate(
        (failed_mem0, passing_langgraph),
        current_manifest_hash=H_MANIFEST,
        current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
    )
    assert admitted.status is AdmissionDecisionStatus.ADMITTED
    assert admitted.admitted_candidate_id is ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE

    no_candidate = select_single_admitted_candidate(
        (failed_mem0, failing_langgraph),
        current_manifest_hash=H_MANIFEST,
        current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
    )
    assert no_candidate.status is AdmissionDecisionStatus.RELEASE_STOPPED
    assert no_candidate.reasons == ("no_candidate_admitted",)


def test_admission_decision_uses_typed_canonical_report_bindings() -> None:
    mem0_ref = CandidateReportRefV1(
        candidate_id=ExternalCandidateId.MEM0_OSS,
        report_hash="a" * 64,
    )
    langgraph_ref = CandidateReportRefV1(
        candidate_id=ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
        report_hash="b" * 64,
    )
    decision_context = {
        "source_task_manifest_hash": H_MANIFEST,
        "evaluation_configuration_hash": H_EVALUATION_CONFIG,
    }
    decision = AdmissionDecisionV1(
        status=AdmissionDecisionStatus.ADMITTED,
        **decision_context,
        reports=(mem0_ref, langgraph_ref),
        admitted_report=langgraph_ref,
        reasons=("admitted_langgraph_fallback",),
    )
    assert decision.admitted_candidate_id is langgraph_ref.candidate_id
    assert decision.admitted_report_hash == langgraph_ref.report_hash
    assert decision.report_hashes == (mem0_ref.report_hash, langgraph_ref.report_hash)

    with pytest.raises(ValidationError, match="canonical candidate order"):
        AdmissionDecisionV1(
            status="admitted",
            **decision_context,
            reports=(langgraph_ref, mem0_ref),
            admitted_report=langgraph_ref,
        )
    with pytest.raises(ValidationError, match="unique candidate IDs"):
        AdmissionDecisionV1(
            status="release_stopped",
            **decision_context,
            reports=(mem0_ref, mem0_ref.model_copy(update={"report_hash": "c" * 64})),
        )
    with pytest.raises(ValidationError, match="unique report hashes"):
        AdmissionDecisionV1(
            status="release_stopped",
            **decision_context,
            reports=(mem0_ref, langgraph_ref.model_copy(update={"report_hash": "a" * 64})),
        )

    mismatched_binding = CandidateReportRefV1(
        candidate_id=ExternalCandidateId.MEM0_OSS,
        report_hash=langgraph_ref.report_hash,
    )
    with pytest.raises(ValidationError, match="must identify one participating report"):
        AdmissionDecisionV1(
            status="admitted",
            **decision_context,
            reports=(mem0_ref, langgraph_ref),
            admitted_report=mismatched_binding,
        )
    with pytest.raises(ValidationError, match="cannot carry admitted_report"):
        AdmissionDecisionV1(
            status="release_stopped",
            **decision_context,
            reports=(mem0_ref,),
            admitted_report=mem0_ref,
        )
    with pytest.raises(ValidationError):
        AdmissionDecisionV1(
            status="admitted",
            **decision_context,
            admitted_candidate_id="mem0_oss",
            admitted_report_hash="a" * 64,
            report_hashes=("a" * 64,),
        )


def test_admission_decision_revalidates_refs_context_and_canonical_reasons() -> None:
    context = {
        "source_task_manifest_hash": H_MANIFEST,
        "evaluation_configuration_hash": H_EVALUATION_CONFIG,
    }
    mem0_ref = CandidateReportRefV1(
        candidate_id=ExternalCandidateId.MEM0_OSS,
        report_hash="a" * 64,
    )
    langgraph_ref = CandidateReportRefV1(
        candidate_id=ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
        report_hash="b" * 64,
    )
    bad_refs = (
        CandidateReportRefV1.model_construct(
            contract_version="1.0.0",
            candidate_id="mem0_oss",
            report_hash="a" * 64,
        ),
        CandidateReportRefV1.model_construct(
            contract_version="1.0.0",
            candidate_id=ExternalCandidateId.MEM0_OSS,
            report_hash="bad",
        ),
    )
    for bad_ref in bad_refs:
        with pytest.raises(ValidationError):
            AdmissionDecisionV1(
                status="release_stopped",
                **context,
                reports=(bad_ref,),
                reasons=("no_candidate_admitted",),
            )
    with pytest.raises(ValidationError, match="both Mem0 and LangGraph"):
        AdmissionDecisionV1(
            status="admitted",
            **context,
            reports=(langgraph_ref,),
            admitted_report=langgraph_ref,
            reasons=("admitted_langgraph_fallback",),
        )
    with pytest.raises(ValidationError, match="reasons"):
        AdmissionDecisionV1(
            status="admitted",
            **context,
            reports=(mem0_ref,),
            admitted_report=mem0_ref,
            reasons=("admitted_langgraph_fallback",),
        )
    with pytest.raises(ValidationError, match="reasons"):
        AdmissionDecisionV1(
            status="release_stopped",
            **context,
            reports=(mem0_ref,),
            reasons=("other_reason",),
        )
    with pytest.raises(ValidationError, match="LangGraph report"):
        AdmissionDecisionV1(
            status="release_stopped",
            **context,
            reports=(),
            reasons=("langgraph_fallback_not_authorized",),
        )
    with pytest.raises(ValidationError, match="LangGraph report"):
        AdmissionDecisionV1(
            status="release_stopped",
            **context,
            reports=(mem0_ref,),
            reasons=("langgraph_fallback_not_authorized",),
        )
    with pytest.raises(ValidationError):
        AdmissionDecisionV1(
            status="release_stopped",
            source_task_manifest_hash="bad",
            evaluation_configuration_hash=H_EVALUATION_CONFIG,
            reasons=("no_candidate_admitted",),
        )


def test_fallback_gate_policy_partition_is_disjoint_and_exhaustive() -> None:
    assert admission_module._FALLBACK_ELIGIBLE_GATES.isdisjoint(
        admission_module._FALLBACK_INVARIANT_GATES
    )
    assert (
        admission_module._FALLBACK_ELIGIBLE_GATES
        | admission_module._FALLBACK_INVARIANT_GATES
    ) == frozenset(admission_module.ExternalGateName)


def test_selection_canonicalizes_input_and_revalidates_and_hashes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_mem0 = _report(gate_overrides={"presentation_level": GateStatus.FAIL})
    langgraph = _report(ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE)
    assert (
        failed_mem0.adapter_info.configuration_hash
        != langgraph.adapter_info.configuration_hash
    )
    canonical = select_single_admitted_candidate(
        (failed_mem0, langgraph),
        current_manifest_hash=H_MANIFEST,
        current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
    )

    revalidated: list[ExternalCandidateId] = []
    hashed: list[ExternalCandidateId] = []
    original_revalidate = admission_module._revalidate_report
    original_hash = admission_module.sha256_model

    def counted_revalidate(report: ExternalAdmissionReportV1) -> ExternalAdmissionReportV1:
        revalidated.append(report.candidate_id)
        return original_revalidate(report)

    def counted_hash(report: ExternalAdmissionReportV1) -> str:
        hashed.append(report.candidate_id)
        return original_hash(report)

    def public_helper_must_not_be_called(*args: object, **kwargs: object) -> bool:
        raise AssertionError("selection must use private validated helpers")

    monkeypatch.setattr(admission_module, "_revalidate_report", counted_revalidate)
    monkeypatch.setattr(admission_module, "sha256_model", counted_hash)
    monkeypatch.setattr(
        admission_module, "evaluate_candidate_admission", public_helper_must_not_be_called
    )
    monkeypatch.setattr(admission_module, "authorize_fallback", public_helper_must_not_be_called)

    reversed_decision = admission_module.select_single_admitted_candidate(
        (langgraph, failed_mem0),
        current_manifest_hash=H_MANIFEST,
        current_evaluation_configuration_hash=H_EVALUATION_CONFIG,
    )
    assert reversed_decision == canonical
    assert sha256_model(reversed_decision) == sha256_model(canonical)
    assert revalidated == [
        ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
        ExternalCandidateId.MEM0_OSS,
    ]
    assert hashed == [
        ExternalCandidateId.MEM0_OSS,
        ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
    ]
