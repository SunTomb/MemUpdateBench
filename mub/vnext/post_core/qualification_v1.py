from __future__ import annotations

from typing import Literal, Mapping

from pydantic import StrictStr

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.post_core.contracts_v1 import CandidateIdentityState, ModelCandidateV1
from mub.vnext.post_core.provenance_v1 import validate_secret_free


class QualificationGateV1(ImmutableContractModel):
    schema_version: str = "memupdatebench.post-core.qualification-gate.v1"
    registry_key: StrictStr
    gate_id: StrictStr
    status: Literal["PASS", "FAIL", "PENDING", "BLOCKED", "NOT_RUN"]
    reason: StrictStr


class QualificationReportV1(ImmutableContractModel):
    schema_version: str = "memupdatebench.post-core.qualification-report.v1"
    gates: tuple[QualificationGateV1, ...]


class CapabilityProbeReportV1(ImmutableContractModel):
    schema_version: str = "memupdatebench.post-core.capability-probe-report.v1"
    network_allowed: Literal[False] = False
    provider_calls: Literal[0] = 0
    model_loads: Literal[0] = 0
    rows: tuple[QualificationGateV1, ...]


def qualify_registry_offline_v1(registry: Mapping[str, ModelCandidateV1]) -> tuple[QualificationReportV1, CapabilityProbeReportV1]:
    rows = []
    for key, candidate in registry.items():
        if candidate.identity is None:
            status = "PENDING"
            reason = f"{candidate.state.value.lower()}: authenticated official identity/provenance is not yet available"
        elif candidate.state is CandidateIdentityState.BLOCKED:
            status = "BLOCKED"
            reason = "; ".join(candidate.blocked_reasons)
        else:
            status = "NOT_RUN"
            reason = "identity is present, but Phase 0 forbids model/provider execution"
        rows.append(QualificationGateV1(registry_key=key, gate_id="identity_and_execution_readiness", status=status, reason=reason))
    report = QualificationReportV1(gates=tuple(rows))
    probes = CapabilityProbeReportV1(rows=tuple(rows))
    validate_secret_free(report.model_dump(mode="json"))
    validate_secret_free(probes.model_dump(mode="json"))
    return report, probes


__all__ = ["CapabilityProbeReportV1", "QualificationGateV1", "QualificationReportV1", "qualify_registry_offline_v1"]
