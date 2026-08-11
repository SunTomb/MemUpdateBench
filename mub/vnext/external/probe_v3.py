from __future__ import annotations

from enum import Enum
import hashlib
from typing import Annotated, Protocol

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
)
from mub.vnext.contracts.v3.adapter import (
    AdapterCapabilitiesV3,
    AdapterInfoV3,
)
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.io import canonical_json_bytes, sha256_model


StrictSha256 = Annotated[str, Field(strict=True, pattern=SHA256_PATTERN)]


class ResetProbeBackend(Protocol):
    def reset_namespace(self, namespace: str) -> None: ...

    def write_sentinel(
        self,
        namespace: str,
        sentinel_id: str,
        sentinel_text: str,
    ) -> None: ...

    def sentinel_visible(self, namespace: str, sentinel_text: str) -> bool: ...


class _ResetProbeIdentityV1(ImmutableContractModel):
    candidate_id: StrictIdentifier
    run_prefix: StrictIdentifier


class NamespaceResetTrialV1(ImmutableContractModel):
    trial_index: int = Field(strict=True, ge=0, lt=20)
    target_namespace: StrictIdentifier
    control_namespace: StrictIdentifier
    target_visible_in_target: StrictBool
    target_visible_in_control: StrictBool
    control_visible_in_control: StrictBool
    control_visible_in_target: StrictBool
    target_empty_after_reset: StrictBool
    control_preserved_after_target_reset: StrictBool
    cleanup_succeeded: StrictBool
    error_code: StrictIdentifier | None = None
    passed: StrictBool

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        expected = (
            self.target_visible_in_target
            and not self.target_visible_in_control
            and self.control_visible_in_control
            and not self.control_visible_in_target
            and self.target_empty_after_reset
            and self.control_preserved_after_target_reset
            and self.cleanup_succeeded
            and self.error_code is None
        )
        if self.passed is not expected:
            raise ValueError("reset trial pass state is inconsistent")
        return self


class NamespaceResetProbeV1(ImmutableContractModel):
    candidate_id: StrictIdentifier
    run_prefix: StrictIdentifier
    trials: tuple[NamespaceResetTrialV1, ...]
    passed: StrictBool

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.trials) != 20:
            raise ValueError("namespace reset probe requires exactly 20 trials")
        if tuple(trial.trial_index for trial in self.trials) != tuple(
            range(20)
        ):
            raise ValueError("namespace reset trials must be contiguous")
        namespaces = tuple(
            namespace
            for trial in self.trials
            for namespace in (
                trial.target_namespace,
                trial.control_namespace,
            )
        )
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("namespace reset trials require unique namespaces")
        if self.passed is not all(trial.passed for trial in self.trials):
            raise ValueError("namespace reset aggregate is inconsistent")
        return self


class DeterminismStatus(str, Enum):
    DETERMINISTIC = "deterministic"
    NONDETERMINISTIC = "nondeterministic"
    INCONCLUSIVE = "inconclusive"


class NormalizedCandidateSnapshotV1(ImmutableContractModel):
    state_hash: StrictSha256
    retrieval_entry_ids: tuple[StrictIdentifier, ...]
    action_trace_hash: StrictSha256 | None = None

    @model_validator(mode="after")
    def _unique_retrieval_ids(self) -> Self:
        if len(self.retrieval_entry_ids) != len(
            set(self.retrieval_entry_ids)
        ):
            raise ValueError("retrieval entry IDs must be unique")
        return self


class CapabilityVerificationV1(ImmutableContractModel):
    adapter_info_hash: StrictSha256
    declared_capabilities_hash: StrictSha256
    observed_capabilities_hash: StrictSha256
    overclaimed_fields: tuple[StrictIdentifier, ...]
    underclaimed_fields: tuple[StrictIdentifier, ...]
    extractor_coherent: StrictBool
    state_transition_linkage_available: StrictBool
    presentation_level: int | None = Field(default=None, strict=True, ge=0, le=3)
    passed: StrictBool

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        for values, label in (
            (self.overclaimed_fields, "overclaimed fields"),
            (self.underclaimed_fields, "underclaimed fields"),
        ):
            if values != tuple(sorted(values)) or len(values) != len(
                set(values)
            ):
                raise ValueError(f"{label} must be sorted and unique")
        expected = (
            not self.overclaimed_fields
            and self.extractor_coherent
            and self.presentation_level in {2, 3}
        )
        if self.passed is not expected:
            raise ValueError("capability verification pass state is inconsistent")
        return self


def _sentinel(namespace: str, role: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{namespace}\x1f{role}".encode("utf-8")
    ).hexdigest()
    return f"sentinel-{digest[:24]}", f"MUB_RESET_SENTINEL_{digest}"


def _sentinel_visible(
    backend: ResetProbeBackend,
    namespace: str,
    sentinel_text: str,
) -> bool:
    visible = backend.sentinel_visible(namespace, sentinel_text)
    if type(visible) is not bool:
        raise ValueError("probe backend visibility must be an exact boolean")
    return visible


def run_namespace_reset_probe(
    backend: ResetProbeBackend,
    *,
    candidate_id: str,
    run_prefix: str,
) -> NamespaceResetProbeV1:
    identity = _ResetProbeIdentityV1(
        candidate_id=candidate_id,
        run_prefix=run_prefix,
    )
    candidate_id = identity.candidate_id
    run_prefix = identity.run_prefix
    trials: list[NamespaceResetTrialV1] = []
    for trial_index in range(20):
        base = f"{run_prefix}-{candidate_id}-reset-{trial_index:02d}"
        target_namespace = f"{base}-target"
        control_namespace = f"{base}-control"
        target_id, target_text = _sentinel(target_namespace, "target")
        control_id, control_text = _sentinel(control_namespace, "control")
        target_visible_in_target = False
        target_visible_in_control = False
        control_visible_in_control = False
        control_visible_in_target = False
        target_empty_after_reset = False
        control_preserved_after_target_reset = False
        cleanup_succeeded = False
        error_code: str | None = None
        target_reset_after_test = False
        try:
            backend.reset_namespace(target_namespace)
            backend.reset_namespace(control_namespace)
            backend.write_sentinel(
                target_namespace,
                target_id,
                target_text,
            )
            backend.write_sentinel(
                control_namespace,
                control_id,
                control_text,
            )
            target_visible_in_target = _sentinel_visible(
                backend,
                target_namespace,
                target_text,
            )
            target_visible_in_control = _sentinel_visible(
                backend,
                control_namespace,
                target_text,
            )
            control_visible_in_control = _sentinel_visible(
                backend,
                control_namespace,
                control_text,
            )
            control_visible_in_target = _sentinel_visible(
                backend,
                target_namespace,
                control_text,
            )
            backend.reset_namespace(target_namespace)
            target_reset_after_test = True
            target_empty_after_reset = not _sentinel_visible(
                backend,
                target_namespace,
                target_text,
            ) and not _sentinel_visible(
                backend,
                target_namespace,
                control_text,
            )
            control_preserved_after_target_reset = _sentinel_visible(
                backend,
                control_namespace,
                control_text,
            ) and not _sentinel_visible(
                backend,
                control_namespace,
                target_text,
            )
        except Exception:
            error_code = "probe_backend_error"
        finally:
            try:
                if not target_reset_after_test:
                    backend.reset_namespace(target_namespace)
                backend.reset_namespace(control_namespace)
                cleanup_succeeded = True
            except Exception:
                cleanup_succeeded = False
                if error_code is None:
                    error_code = "probe_cleanup_error"
        passed = (
            target_visible_in_target
            and not target_visible_in_control
            and control_visible_in_control
            and not control_visible_in_target
            and target_empty_after_reset
            and control_preserved_after_target_reset
            and cleanup_succeeded
            and error_code is None
        )
        trials.append(
            NamespaceResetTrialV1(
                trial_index=trial_index,
                target_namespace=target_namespace,
                control_namespace=control_namespace,
                target_visible_in_target=target_visible_in_target,
                target_visible_in_control=target_visible_in_control,
                control_visible_in_control=control_visible_in_control,
                control_visible_in_target=control_visible_in_target,
                target_empty_after_reset=target_empty_after_reset,
                control_preserved_after_target_reset=(
                    control_preserved_after_target_reset
                ),
                cleanup_succeeded=cleanup_succeeded,
                error_code=error_code,
                passed=passed,
            )
        )
    trial_tuple = tuple(trials)
    return NamespaceResetProbeV1(
        candidate_id=candidate_id,
        run_prefix=run_prefix,
        trials=trial_tuple,
        passed=all(trial.passed for trial in trial_tuple),
    )


def classify_determinism(
    snapshots: tuple[NormalizedCandidateSnapshotV1 | None, ...],
) -> DeterminismStatus:
    if type(snapshots) is not tuple or len(snapshots) != 3:
        raise ValueError("determinism classification requires exactly three snapshots")
    if any(snapshot is None for snapshot in snapshots):
        return DeterminismStatus.INCONCLUSIVE
    if any(
        type(snapshot) is not NormalizedCandidateSnapshotV1
        for snapshot in snapshots
    ):
        raise ValueError("determinism snapshots require exact trusted types")
    validated = tuple(
        _revalidate_exact(
            NormalizedCandidateSnapshotV1,
            snapshot,
            "determinism snapshot",
        )
        for snapshot in snapshots
    )
    serialized = tuple(
        canonical_json_bytes(snapshot) for snapshot in validated
    )
    return (
        DeterminismStatus.DETERMINISTIC
        if len(set(serialized)) == 1
        else DeterminismStatus.NONDETERMINISTIC
    )


def required_canary_repetitions(status: DeterminismStatus) -> int:
    if type(status) is not DeterminismStatus:
        raise ValueError("determinism status must use the exact enum")
    return 1 if status is DeterminismStatus.DETERMINISTIC else 3


def _revalidate_exact(model_type, value, label: str):
    if type(value) is not model_type:
        raise ValueError(f"{label} requires exact {model_type.__name__}")
    try:
        return model_type.model_validate(
            {
                field_name: value.__dict__[field_name]
                for field_name in model_type.model_fields
            },
            strict=True,
        )
    except Exception as exc:
        raise ValueError(f"{label} fails trust-boundary validation") from exc


def verify_capability_truthfulness(
    adapter_info: AdapterInfoV3,
    declared: AdapterCapabilitiesV3,
    observed: AdapterCapabilitiesV3,
    *,
    state_transition_linkage_available: bool,
) -> CapabilityVerificationV1:
    adapter_info = _revalidate_exact(
        AdapterInfoV3,
        adapter_info,
        "adapter info",
    )
    declared = _revalidate_exact(
        AdapterCapabilitiesV3,
        declared,
        "declared capabilities",
    )
    observed = _revalidate_exact(
        AdapterCapabilitiesV3,
        observed,
        "observed capabilities",
    )
    if type(state_transition_linkage_available) is not bool:
        raise ValueError(
            "state-transition linkage must be an exact boolean"
        )
    boolean_fields = tuple(
        field_name
        for field_name in AdapterCapabilitiesV3.model_fields
        if type(getattr(declared, field_name)) is bool
    )
    overclaimed = tuple(
        sorted(
            field_name
            for field_name in boolean_fields
            if getattr(declared, field_name)
            and not getattr(observed, field_name)
        )
    )
    underclaimed = tuple(
        sorted(
            field_name
            for field_name in boolean_fields
            if not getattr(declared, field_name)
            and getattr(observed, field_name)
        )
    )
    requires_extractor = (
        declared.requires_evaluation_extractor
        or observed.requires_evaluation_extractor
    )
    info_has_complete_extractor = (
        adapter_info.extractor_id is not None
        and adapter_info.extractor_version is not None
    )
    info_has_no_extractor = (
        adapter_info.extractor_id is None
        and adapter_info.extractor_version is None
    )
    if requires_extractor:
        extractor_coherent = (
            info_has_complete_extractor
            and declared.extractor_version is not None
            and observed.extractor_version is not None
            and declared.extractor_version
            == observed.extractor_version
            == adapter_info.extractor_version
        )
    else:
        extractor_coherent = (
            info_has_no_extractor
            and declared.extractor_version is None
            and observed.extractor_version is None
        ) or (
            info_has_complete_extractor
            and declared.extractor_version
            == observed.extractor_version
            == adapter_info.extractor_version
        )
    presentation_level = observed.presentation_level(
        state_transition_linkage_available
    )
    passed = (
        not overclaimed
        and extractor_coherent
        and presentation_level in {2, 3}
    )
    return CapabilityVerificationV1(
        adapter_info_hash=sha256_model(adapter_info),
        declared_capabilities_hash=sha256_model(declared),
        observed_capabilities_hash=sha256_model(observed),
        overclaimed_fields=overclaimed,
        underclaimed_fields=underclaimed,
        extractor_coherent=extractor_coherent,
        state_transition_linkage_available=(
            state_transition_linkage_available
        ),
        presentation_level=presentation_level,
        passed=passed,
    )


__all__ = [
    "CapabilityVerificationV1",
    "DeterminismStatus",
    "NamespaceResetProbeV1",
    "NamespaceResetTrialV1",
    "NormalizedCandidateSnapshotV1",
    "ResetProbeBackend",
    "classify_determinism",
    "required_canary_repetitions",
    "run_namespace_reset_probe",
    "verify_capability_truthfulness",
]
