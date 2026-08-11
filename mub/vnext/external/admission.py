from __future__ import annotations

import re

from mub.vnext.contracts.common import SHA256_PATTERN
from mub.vnext.external.contracts import (
    AdmissionDecisionStatus,
    AdmissionDecisionV1,
    CandidateReportRefV1,
    ExternalAdmissionReportV1,
    ExternalCandidateId,
    ExternalGateName,
    GateStatus,
)
from mub.vnext.io.canonical import sha256_model

_SHA256_RE = re.compile(SHA256_PATTERN)
_CANDIDATE_ORDER = {
    candidate_id: index for index, candidate_id in enumerate(ExternalCandidateId)
}
_FALLBACK_ELIGIBLE_GATES = frozenset(
    {
        ExternalGateName.CANDIDATE_ENVIRONMENT,
        ExternalGateName.VISIBLE_ONLY_FAIRNESS,
        ExternalGateName.NAMESPACE_RESET,
        ExternalGateName.CAPABILITY_TRUTHFULNESS,
        ExternalGateName.RAW_NORMALIZED_EXPORT,
        ExternalGateName.FIELD_PROVENANCE,
        ExternalGateName.TERMINAL_COMPLETENESS,
        ExternalGateName.RETRIEVAL_POLICY,
        ExternalGateName.PRESENTATION_LEVEL,
    }
)
_FALLBACK_INVARIANT_GATES = frozenset(
    {
        ExternalGateName.SOURCE_AUTHENTICATION,
        ExternalGateName.OFFICIAL_PROVENANCE_LICENSE,
        ExternalGateName.OFFLINE_MODEL_PREREQUISITE,
        ExternalGateName.SECURITY_REDACTION,
        ExternalGateName.REPETITION_RULE,
    }
)
if (
    not _FALLBACK_ELIGIBLE_GATES.isdisjoint(_FALLBACK_INVARIANT_GATES)
    or _FALLBACK_ELIGIBLE_GATES | _FALLBACK_INVARIANT_GATES
    != frozenset(ExternalGateName)
):
    raise RuntimeError("fallback gate policy must be disjoint and exhaustive")


def _require_sha256(value: str, name: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _revalidate_report(
    report: ExternalAdmissionReportV1,
) -> ExternalAdmissionReportV1:
    if type(report) is not ExternalAdmissionReportV1:
        raise ValueError(
            "admission requires an exact ExternalAdmissionReportV1"
        )
    try:
        payload = {
            field_name: report.__dict__[field_name]
            for field_name in ExternalAdmissionReportV1.model_fields
        }
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "ExternalAdmissionReportV1 stored fields are incomplete"
        ) from exc
    return ExternalAdmissionReportV1.model_validate(payload, strict=True)


def _gate_statuses(
    report: ExternalAdmissionReportV1,
) -> dict[ExternalGateName, GateStatus]:
    return {gate.name: gate.status for gate in report.gates}


def _nonempty(value: str | None) -> bool:
    return (
        value is not None
        and bool(value)
        and value == value.strip()
    )


def _evaluate_validated_candidate(report: ExternalAdmissionReportV1) -> bool:
    capabilities = report.adapter_capabilities
    level = capabilities.presentation_level(
        state_transition_linkage_available=(
            report.state_transition_linkage_available
        )
    )
    if (
        report.outcome is not GateStatus.PASS
        or any(gate.status is not GateStatus.PASS for gate in report.gates)
        or level not in {2, 3}
        or not capabilities.supports_event_ingest
        or not capabilities.supports_add
        or not capabilities.supports_update
    ):
        return False
    if not capabilities.requires_evaluation_extractor:
        return True
    adapter_info = report.adapter_info
    return (
        _nonempty(capabilities.extractor_version)
        and _nonempty(adapter_info.extractor_id)
        and _nonempty(adapter_info.extractor_version)
        and capabilities.extractor_version == adapter_info.extractor_version
    )


def evaluate_candidate_admission(report: ExternalAdmissionReportV1) -> bool:
    return _evaluate_validated_candidate(_revalidate_report(report))


def _authorize_validated_fallback(
    mem0_report: ExternalAdmissionReportV1,
    current_manifest_hash: str,
    current_evaluation_configuration_hash: str,
) -> bool:
    if mem0_report.candidate_id is not ExternalCandidateId.MEM0_OSS:
        raise ValueError("fallback authorization requires the Mem0 candidate report")
    if (
        mem0_report.source_task_manifest_hash != current_manifest_hash
        or mem0_report.evaluation_configuration_hash
        != current_evaluation_configuration_hash
        or mem0_report.outcome is not GateStatus.FAIL
    ):
        return False
    statuses = _gate_statuses(mem0_report)
    eligible_statuses = tuple(statuses[name] for name in _FALLBACK_ELIGIBLE_GATES)
    return (
        all(statuses[name] is GateStatus.PASS for name in _FALLBACK_INVARIANT_GATES)
        and all(
            status in {GateStatus.PASS, GateStatus.FAIL}
            for status in eligible_statuses
        )
        and any(status is GateStatus.FAIL for status in eligible_statuses)
    )


def authorize_fallback(
    mem0_report: ExternalAdmissionReportV1,
    current_manifest_hash: str,
    current_evaluation_configuration_hash: str,
) -> bool:
    mem0_report = _revalidate_report(mem0_report)
    _require_sha256(current_manifest_hash, "current_manifest_hash")
    _require_sha256(
        current_evaluation_configuration_hash,
        "current_evaluation_configuration_hash",
    )
    return _authorize_validated_fallback(
        mem0_report,
        current_manifest_hash,
        current_evaluation_configuration_hash,
    )


def _candidate_sort_key(report: ExternalAdmissionReportV1) -> int:
    for candidate_id, index in _CANDIDATE_ORDER.items():
        if report.candidate_id == candidate_id:
            return index
    return len(_CANDIDATE_ORDER)


def select_single_admitted_candidate(
    reports: tuple[ExternalAdmissionReportV1, ...],
    *,
    current_manifest_hash: str,
    current_evaluation_configuration_hash: str,
) -> AdmissionDecisionV1:
    if len(reports) > 2:
        raise ValueError("selection accepts at most two candidate reports")
    _require_sha256(current_manifest_hash, "current_manifest_hash")
    _require_sha256(
        current_evaluation_configuration_hash,
        "current_evaluation_configuration_hash",
    )
    validated_inputs = tuple(_revalidate_report(report) for report in reports)
    candidate_ids = tuple(report.candidate_id for report in validated_inputs)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate IDs are forbidden")
    if any(
        report.source_task_manifest_hash != current_manifest_hash
        or report.evaluation_configuration_hash
        != current_evaluation_configuration_hash
        for report in validated_inputs
    ):
        raise ValueError(
            "all reports must match the authenticated current manifest/evaluation configuration"
        )

    validated_reports = tuple(sorted(validated_inputs, key=_candidate_sort_key))
    report_pairs: list[
        tuple[ExternalAdmissionReportV1, str, CandidateReportRefV1]
    ] = []
    for report in validated_reports:
        digest = sha256_model(report)
        report_pairs.append(
            (
                report,
                digest,
                CandidateReportRefV1(
                    candidate_id=report.candidate_id,
                    report_hash=digest,
                ),
            )
        )
    report_digests = tuple(digest for _, digest, _ in report_pairs)
    if len(report_digests) != len(set(report_digests)):
        raise ValueError("duplicate report hashes are forbidden")
    report_refs = tuple(report_ref for _, _, report_ref in report_pairs)

    mem0_pairs = [
        pair
        for pair in report_pairs
        if pair[0].candidate_id is ExternalCandidateId.MEM0_OSS
    ]
    langgraph_pairs = [
        pair
        for pair in report_pairs
        if pair[0].candidate_id
        is ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE
    ]
    if langgraph_pairs and (
        len(mem0_pairs) != 1
        or not _authorize_validated_fallback(
            mem0_pairs[0][0],
            current_manifest_hash,
            current_evaluation_configuration_hash,
        )
    ):
        return AdmissionDecisionV1(
            status=AdmissionDecisionStatus.RELEASE_STOPPED,
            source_task_manifest_hash=current_manifest_hash,
            evaluation_configuration_hash=(
                current_evaluation_configuration_hash
            ),
            reports=report_refs,
            reasons=("langgraph_fallback_not_authorized",),
        )

    admitted_pairs = [
        pair
        for pair in report_pairs
        if _evaluate_validated_candidate(pair[0])
    ]
    if len(admitted_pairs) > 1:
        raise ValueError("double admission is forbidden")
    if admitted_pairs:
        selected_report, _, selected_ref = admitted_pairs[0]
        reason = (
            "admitted_langgraph_fallback"
            if selected_report.candidate_id
            is ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE
            else "admitted_mem0_primary"
        )
        return AdmissionDecisionV1(
            status=AdmissionDecisionStatus.ADMITTED,
            source_task_manifest_hash=current_manifest_hash,
            evaluation_configuration_hash=(
                current_evaluation_configuration_hash
            ),
            reports=report_refs,
            admitted_report=selected_ref,
            reasons=(reason,),
        )
    return AdmissionDecisionV1(
        status=AdmissionDecisionStatus.RELEASE_STOPPED,
        source_task_manifest_hash=current_manifest_hash,
        evaluation_configuration_hash=current_evaluation_configuration_hash,
        reports=report_refs,
        reasons=("no_candidate_admitted",),
    )


__all__ = [
    "authorize_fallback",
    "evaluate_candidate_admission",
    "select_single_admitted_candidate",
]
