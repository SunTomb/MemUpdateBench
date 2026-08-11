from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ArtifactRef,
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
)
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.common import StrictIdentifier

EXTERNAL_ADMISSION_CONTRACT_VERSION = "1.0.0"
StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]


def _exact_field_snapshot(value: object, expected_type: type) -> dict[str, object]:
    if type(value) is not expected_type:
        raise ValueError(f"nested contract requires exact {expected_type.__name__}")
    try:
        return {
            field_name: value.__dict__[field_name]
            for field_name in expected_type.model_fields
        }
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"{expected_type.__name__} stored fields are incomplete"
        ) from exc


class ExternalCandidateId(str, Enum):
    MEM0_OSS = "mem0_oss"
    LANGGRAPH_STORE_EXTRACT_THEN_STORE = "langgraph_store_extract_then_store"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


class ExternalGateName(str, Enum):
    SOURCE_AUTHENTICATION = "source_authentication"
    OFFICIAL_PROVENANCE_LICENSE = "official_provenance_license"
    OFFLINE_MODEL_PREREQUISITE = "offline_model_prerequisite"
    CANDIDATE_ENVIRONMENT = "candidate_environment"
    VISIBLE_ONLY_FAIRNESS = "visible_only_fairness"
    NAMESPACE_RESET = "namespace_reset"
    CAPABILITY_TRUTHFULNESS = "capability_truthfulness"
    RAW_NORMALIZED_EXPORT = "raw_normalized_export"
    FIELD_PROVENANCE = "field_provenance"
    TERMINAL_COMPLETENESS = "terminal_completeness"
    RETRIEVAL_POLICY = "retrieval_policy"
    PRESENTATION_LEVEL = "presentation_level"
    SECURITY_REDACTION = "security_redaction"
    REPETITION_RULE = "repetition_rule"


ADMISSION_GATE_NAMES = tuple(gate.value for gate in ExternalGateName)


class AdmissionDecisionStatus(str, Enum):
    ADMITTED = "admitted"
    RELEASE_STOPPED = "release_stopped"


class GateResultV1(ImmutableContractModel):
    contract_version: Literal["1.0.0"] = EXTERNAL_ADMISSION_CONTRACT_VERSION
    name: ExternalGateName
    status: GateStatus
    evidence_artifacts: tuple[ArtifactRef, ...] = ()
    reasons: tuple[StrictIdentifier, ...] = ()

    @model_validator(mode="after")
    def _evidence_is_admissible(self) -> Self:
        from mub.vnext.external.registry import validate_artifact_provenance

        if self.status is GateStatus.PASS:
            if not self.evidence_artifacts:
                raise ValueError("PASS gates require evidence artifacts")
            if self.reasons:
                raise ValueError("PASS gates cannot carry reasons")
        elif self.status is GateStatus.FAIL:
            if not self.evidence_artifacts or not self.reasons:
                raise ValueError("FAIL gates require evidence artifacts and reasons")
        elif not self.reasons:
            raise ValueError("BLOCKED and NOT_RUN gates require reasons")
        validated_artifacts = tuple(
            validate_artifact_provenance(artifact)
            for artifact in self.evidence_artifacts
        )
        object.__setattr__(self, "evidence_artifacts", validated_artifacts)
        return self


class ExternalAdmissionReportV1(ImmutableContractModel):
    contract_version: Literal["1.0.0"] = EXTERNAL_ADMISSION_CONTRACT_VERSION
    candidate_id: ExternalCandidateId
    source_task_manifest_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    source_task_manifest_ref: ArtifactRef
    evaluation_configuration_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    evaluation_configuration_ref: ArtifactRef
    adapter_configuration_ref: ArtifactRef
    probe_ref: ArtifactRef
    canary_ref: ArtifactRef
    package_provenance_ref: ArtifactRef
    model_provenance_ref: ArtifactRef
    adapter_info: AdapterInfoV3
    adapter_capabilities: AdapterCapabilitiesV3
    state_transition_linkage_available: StrictBool
    gates: tuple[GateResultV1, ...]
    outcome: GateStatus
    reasons: tuple[StrictIdentifier, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        from mub.vnext.external.registry import validate_artifact_provenance

        validated_adapter_info = AdapterInfoV3.model_validate(
            _exact_field_snapshot(self.adapter_info, AdapterInfoV3),
            strict=True,
        )
        validated_adapter_capabilities = AdapterCapabilitiesV3.model_validate(
            _exact_field_snapshot(
                self.adapter_capabilities,
                AdapterCapabilitiesV3,
            ),
            strict=True,
        )
        object.__setattr__(self, "adapter_info", validated_adapter_info)
        object.__setattr__(
            self,
            "adapter_capabilities",
            validated_adapter_capabilities,
        )

        artifact_fields = (
            "source_task_manifest_ref",
            "evaluation_configuration_ref",
            "adapter_configuration_ref",
            "probe_ref",
            "canary_ref",
            "package_provenance_ref",
            "model_provenance_ref",
        )
        for field_name in artifact_fields:
            validated_artifact = validate_artifact_provenance(
                getattr(self, field_name)
            )
            object.__setattr__(self, field_name, validated_artifact)
        validated_gates = tuple(
            GateResultV1.model_validate(
                _exact_field_snapshot(gate, GateResultV1),
                strict=True,
            )
            for gate in self.gates
        )
        object.__setattr__(self, "gates", validated_gates)

        expected_identity = self.candidate_id.value
        if (
            self.adapter_info.adapter_id != expected_identity
            or self.adapter_info.system_name != expected_identity
        ):
            raise ValueError(
                "candidate identity must match adapter_id and system_name exactly"
            )
        if self.source_task_manifest_ref.sha256 != self.source_task_manifest_hash:
            raise ValueError(
                "source task manifest hash must match source_task_manifest_ref"
            )
        if (
            self.evaluation_configuration_ref.sha256
            != self.evaluation_configuration_hash
        ):
            raise ValueError(
                "evaluation configuration hash must match evaluation_configuration_ref"
            )
        if (
            self.adapter_configuration_ref.sha256
            != self.adapter_info.configuration_hash
        ):
            raise ValueError(
                "adapter configuration ref must match adapter_info.configuration_hash"
            )
        observed_names = tuple(gate.name.value for gate in validated_gates)
        if observed_names != ADMISSION_GATE_NAMES:
            raise ValueError(
                "gates must contain all fixed admission gates exactly once in canonical order"
            )
        statuses = {gate.status for gate in validated_gates}
        expected = GateStatus.PASS
        for status in (GateStatus.FAIL, GateStatus.BLOCKED, GateStatus.NOT_RUN):
            if status in statuses:
                expected = status
                break
        if self.outcome is not expected:
            raise ValueError("outcome must equal the aggregate fixed-gate status")
        expected_reasons = (
            ()
            if self.outcome is GateStatus.PASS
            else ("candidate_gate_failed",)
        )
        if self.reasons != expected_reasons:
            raise ValueError("report reasons must match the aggregate outcome")
        return self


class CandidateReportRefV1(ImmutableContractModel):
    contract_version: Literal["1.0.0"] = EXTERNAL_ADMISSION_CONTRACT_VERSION
    candidate_id: ExternalCandidateId
    report_hash: StrictSha256


class AdmissionDecisionV1(ImmutableContractModel):
    contract_version: Literal["1.0.0"] = EXTERNAL_ADMISSION_CONTRACT_VERSION
    status: AdmissionDecisionStatus
    source_task_manifest_hash: StrictSha256
    evaluation_configuration_hash: StrictSha256
    reports: tuple[CandidateReportRefV1, ...] = ()
    admitted_report: CandidateReportRefV1 | None = None
    reasons: tuple[StrictIdentifier, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        validated_reports = tuple(
            CandidateReportRefV1.model_validate(
                _exact_field_snapshot(report, CandidateReportRefV1),
                strict=True,
            )
            for report in self.reports
        )
        object.__setattr__(self, "reports", validated_reports)
        validated_admitted_report = None
        if self.admitted_report is not None:
            validated_admitted_report = CandidateReportRefV1.model_validate(
                _exact_field_snapshot(
                    self.admitted_report,
                    CandidateReportRefV1,
                ),
                strict=True,
            )
            object.__setattr__(
                self,
                "admitted_report",
                validated_admitted_report,
            )

        candidate_ids = tuple(
            report.candidate_id for report in validated_reports
        )
        report_hashes = tuple(report.report_hash for report in validated_reports)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("reports require unique candidate IDs")
        if len(report_hashes) != len(set(report_hashes)):
            raise ValueError("reports require unique report hashes")
        candidate_order = {
            candidate: index
            for index, candidate in enumerate(ExternalCandidateId)
        }
        if candidate_ids != tuple(
            sorted(candidate_ids, key=candidate_order.__getitem__)
        ):
            raise ValueError("reports must use canonical candidate order")
        if self.status is AdmissionDecisionStatus.ADMITTED:
            if (
                validated_admitted_report is None
                or validated_admitted_report not in validated_reports
            ):
                raise ValueError(
                    "admitted_report must identify one participating report"
                )
            if (
                validated_admitted_report.candidate_id
                is ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE
                and candidate_ids
                != (
                    ExternalCandidateId.MEM0_OSS,
                    ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
                )
            ):
                raise ValueError(
                    "admitted LangGraph decisions require both Mem0 and LangGraph reports"
                )
            if (
                validated_admitted_report.candidate_id
                is ExternalCandidateId.MEM0_OSS
                and candidate_ids != (ExternalCandidateId.MEM0_OSS,)
            ):
                raise ValueError(
                    "admitted Mem0 decisions cannot include a LangGraph report"
                )
            expected_reasons = (
                ("admitted_mem0_primary",)
                if validated_admitted_report.candidate_id
                is ExternalCandidateId.MEM0_OSS
                else ("admitted_langgraph_fallback",)
            )
            if self.reasons != expected_reasons:
                raise ValueError("admitted decision reasons are not canonical")
        else:
            if validated_admitted_report is not None:
                raise ValueError(
                    "release_stopped decisions cannot carry admitted_report"
                )
            if self.reasons not in {
                ("no_candidate_admitted",),
                ("langgraph_fallback_not_authorized",),
            }:
                raise ValueError("release-stopped decision reasons are not canonical")
            if (
                self.reasons == ("langgraph_fallback_not_authorized",)
                and ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE
                not in candidate_ids
            ):
                raise ValueError(
                    "langgraph_fallback_not_authorized requires a LangGraph report"
                )
        return self

    @property
    def admitted_candidate_id(self) -> ExternalCandidateId | None:
        return (
            None
            if self.admitted_report is None
            else self.admitted_report.candidate_id
        )

    @property
    def admitted_report_hash(self) -> str | None:
        return (
            None
            if self.admitted_report is None
            else self.admitted_report.report_hash
        )

    @property
    def report_hashes(self) -> tuple[str, ...]:
        return tuple(report.report_hash for report in self.reports)


__all__ = [
    "ADMISSION_GATE_NAMES",
    "EXTERNAL_ADMISSION_CONTRACT_VERSION",
    "AdmissionDecisionStatus",
    "AdmissionDecisionV1",
    "CandidateReportRefV1",
    "ExternalAdmissionReportV1",
    "ExternalCandidateId",
    "ExternalGateName",
    "GateResultV1",
    "GateStatus",
]
