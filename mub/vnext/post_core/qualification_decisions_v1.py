"""Derive current qualification decisions without elevating plans or preflights to smoke results.

Capability smoke can become READY only in a future results-bearing release that imports
validated CapabilityAttemptReceiptV1 evidence; this release deliberately accepts no such input.
"""

from __future__ import annotations

from collections.abc import Sequence

from mub.vnext.post_core.contracts_v1 import CandidateIdentityState
from mub.vnext.post_core.identity_v1 import IdentityEvidenceBundleV1
from mub.vnext.post_core.qualification_receipts_v1 import (
    DecisionScope,
    GateStatus,
    OpenRuntimeReceiptV1,
    ProviderCapabilityAttestationV1,
    QualificationDecisionV1,
    QualificationStatus,
)
from mub.vnext.post_core.qualification_validation_v1 import (
    validate_provider_attestations_v1,
    validate_runtime_receipts_v1,
)


_OPEN_KEYS = frozenset(
    {
        "qwen35_9b_bf16",
        "meta_muse_glimmer_30b_int4",
        "meta_muse_glimmer_30b_bf16",
    }
)
_CLOSED_KEYS = frozenset(
    {
        "claude_sonnet_4_6",
        "claude_opus_4_8",
        "gemini_3_6_flash",
        "grok_4_5",
        "gpt_5_5",
    }
)
_AUTHENTICATED_CLOSED_EVIDENCE_CLASSES = frozenset(
    {
        "official_pinned_api_id",
        "official_stable_api_id_with_response_version",
    }
)
_SCOPE_ORDER = (
    DecisionScope.STORAGE_INPUT,
    DecisionScope.SHORT_GENERATION_GATE,
    DecisionScope.CAPABILITY_SMOKE,
    DecisionScope.BENCHMARK_ADMISSION,
)


def _decision(
    registry_key: str,
    scope: DecisionScope,
    status: QualificationStatus,
    reasons: tuple[str, ...],
    evidence_binding_ids: tuple[str, ...],
) -> QualificationDecisionV1:
    return QualificationDecisionV1(
        registry_key=registry_key,
        scope=scope,
        status=status,
        reasons=reasons,
        evidence_binding_ids=evidence_binding_ids,
    )


def _gate_value(status: GateStatus) -> str:
    return status.value


def _open_runtime_decisions(
    identity_key: str,
    runtime: OpenRuntimeReceiptV1,
) -> tuple[QualificationDecisionV1, ...]:
    evidence_ids = runtime.source_binding_ids
    storage = _decision(
        identity_key,
        DecisionScope.STORAGE_INPUT,
        QualificationStatus.READY,
        ("exact snapshot revision and tree identity are validated",),
        evidence_ids,
    )

    gates = (
        runtime.load_status,
        runtime.generation_status,
        runtime.determinism_status,
        runtime.unload_status,
    )
    gate_values = tuple(_gate_value(status) for status in gates)
    if GateStatus.UNSUPPORTED.value in gate_values:
        short_status = QualificationStatus.UNSUPPORTED
        short_reason = "backend incompatibility is demonstrated by an UNSUPPORTED runtime gate"
    elif all(value == GateStatus.PASS.value for value in gate_values):
        short_status = QualificationStatus.READY
        short_reason = "load, generation, determinism, and unload gates are validated"
    else:
        short_status = QualificationStatus.BLOCKED
        blocked = runtime.blocked_reasons
        short_reason = blocked[0] if blocked else (
            "short generation gate is not PASS: "
            + ", ".join(
                f"{name}={value}"
                for name, value in zip(
                    ("load", "generation", "determinism", "unload"), gate_values
                )
                if value != GateStatus.PASS.value
            )
        )
    short = _decision(
        identity_key,
        DecisionScope.SHORT_GENERATION_GATE,
        short_status,
        (short_reason,),
        evidence_ids,
    )
    smoke_gate_reason = (
        "short generation gate passed"
        if short_status is QualificationStatus.READY
        else "short generation gate is not READY"
    )
    smoke = _decision(
        identity_key,
        DecisionScope.CAPABILITY_SMOKE,
        QualificationStatus.BLOCKED,
        (
            f"uniform capability smoke is NOT_RUN; {smoke_gate_reason}",
            "CapabilityAttemptReceiptV1 results are required in a future release",
        ),
        evidence_ids,
    )
    benchmark = _decision(
        identity_key,
        DecisionScope.BENCHMARK_ADMISSION,
        QualificationStatus.BLOCKED,
        ("benchmark execution remains NOT_RUN",),
        evidence_ids,
    )
    return storage, short, smoke, benchmark


def _closed_decisions(
    identity_key: str,
    identity_state: CandidateIdentityState,
    evidence_class: str,
    provider: ProviderCapabilityAttestationV1,
) -> tuple[QualificationDecisionV1, ...]:
    evidence_ids = provider.source_binding_ids
    authenticated_identity = (
        identity_state is CandidateIdentityState.READY_FOR_PROVIDER_PREFLIGHT
        and evidence_class in _AUTHENTICATED_CLOSED_EVIDENCE_CLASSES
    )
    if authenticated_identity:
        storage_status = QualificationStatus.READY
        storage_reason = "authenticated provider identity is available for storage input"
    else:
        storage_status = QualificationStatus.BLOCKED
        storage_reason = "identity caveat blocks authenticated storage input"
    storage = _decision(
        identity_key,
        DecisionScope.STORAGE_INPUT,
        storage_status,
        (storage_reason,),
        evidence_ids,
    )
    short = _decision(
        identity_key,
        DecisionScope.SHORT_GENERATION_GATE,
        QualificationStatus.BLOCKED,
        ("not applicable: closed provider short-generation gate is not applicable",),
        evidence_ids,
    )
    smoke_reasons = [
        "uniform capability smoke is NOT_RUN; provider connectivity/interface preflight passed",
        "CapabilityAttemptReceiptV1 results are required in a future release",
    ]
    if not authenticated_identity:
        smoke_reasons.append("identity caveat blocks capability smoke")
    smoke = _decision(
        identity_key,
        DecisionScope.CAPABILITY_SMOKE,
        QualificationStatus.BLOCKED,
        tuple(smoke_reasons),
        evidence_ids,
    )
    benchmark_reasons = ["benchmark execution remains NOT_RUN"]
    if not authenticated_identity:
        benchmark_reasons.insert(0, "identity caveat blocks benchmark admission")
    benchmark = _decision(
        identity_key,
        DecisionScope.BENCHMARK_ADMISSION,
        QualificationStatus.BLOCKED,
        tuple(benchmark_reasons),
        evidence_ids,
    )
    return storage, short, smoke, benchmark


def derive_qualification_decisions_v1(
    identity_bundle: IdentityEvidenceBundleV1,
    provider_rows: Sequence[ProviderCapabilityAttestationV1],
    runtime_rows: Sequence[OpenRuntimeReceiptV1],
) -> tuple[QualificationDecisionV1, ...]:
    """Derive immutable operational qualification decisions from validated evidence."""
    if not isinstance(identity_bundle, IdentityEvidenceBundleV1):
        raise ValueError("identity evidence must use IdentityEvidenceBundleV1")
    providers = validate_provider_attestations_v1(provider_rows)
    runtimes = validate_runtime_receipts_v1(runtime_rows)
    records = identity_bundle.records
    provider_by_key = {row.registry_key: row for row in providers}
    runtime_by_key = {row.registry_key: row for row in runtimes}
    if set(provider_by_key) != _CLOSED_KEYS or set(runtime_by_key) != _OPEN_KEYS:
        raise ValueError("qualification evidence keys do not match the frozen candidate roles")

    decisions: list[QualificationDecisionV1] = []
    for record in records:
        if record.registry_key in _OPEN_KEYS:
            runtime = runtime_by_key[record.registry_key]
            if (
                runtime.revision != record.revision
                or runtime.snapshot_tree_sha256 == ""
                or not runtime.source_binding_ids
            ):
                raise ValueError("runtime receipt does not bind exact identity evidence")
            decisions.extend(_open_runtime_decisions(record.registry_key, runtime))
        elif record.registry_key in _CLOSED_KEYS:
            decisions.extend(
                _closed_decisions(
                    record.registry_key,
                    record.state,
                    record.evidence_class,
                    provider_by_key[record.registry_key],
                )
            )
        else:
            raise ValueError("identity evidence contains an unknown registry key")

    if tuple(row.registry_key for row in decisions[:: len(_SCOPE_ORDER)]) != tuple(
        record.registry_key for record in records
    ):
        raise ValueError("qualification decisions do not preserve frozen identity order")
    return tuple(decisions)


__all__ = ["derive_qualification_decisions_v1"]
