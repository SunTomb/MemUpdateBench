from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mub.vnext.contracts.v3.adapter import (
    AdapterCapabilitiesV3,
    AdapterInfoV3,
)


@dataclass
class FakeResetBackend:
    leaky: bool = False
    fail_once: bool = False
    states: dict[str, set[str]] = field(default_factory=dict)
    reset_calls: list[str] = field(default_factory=list)

    def reset_namespace(self, namespace: str) -> None:
        self.reset_calls.append(namespace)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("provider secret must not enter evidence")
        if self.leaky:
            self.states.clear()
        else:
            self.states[namespace] = set()

    def write_sentinel(
        self,
        namespace: str,
        sentinel_id: str,
        sentinel_text: str,
    ) -> None:
        self.states.setdefault(namespace, set()).add(sentinel_text)
        if self.leaky:
            for state in self.states.values():
                state.add(sentinel_text)

    def sentinel_visible(self, namespace: str, sentinel_text: str) -> bool:
        if self.leaky:
            return any(
                sentinel_text in state for state in self.states.values()
            )
        return sentinel_text in self.states.get(namespace, set())


def test_twenty_reset_trials_pass_with_isolated_namespaces_and_cleanup():
    from mub.vnext.external.probe_v3 import run_namespace_reset_probe

    backend = FakeResetBackend()
    report = run_namespace_reset_probe(
        backend,
        candidate_id="mem0_oss",
        run_prefix="task10-probe",
    )

    assert report.passed is True
    assert len(report.trials) == 20
    assert tuple(trial.trial_index for trial in report.trials) == tuple(
        range(20)
    )
    assert all(trial.passed for trial in report.trials)
    assert all(trial.cleanup_succeeded for trial in report.trials)
    assert len(set(backend.reset_calls)) == 40
    assert len(backend.reset_calls) == 80
    assert all(
        namespace.startswith("task10-probe-mem0_oss-reset-")
        for namespace in backend.reset_calls
    )


def test_reset_probe_records_all_terminal_trials_without_leaking_errors():
    from mub.vnext.external.probe_v3 import run_namespace_reset_probe

    backend = FakeResetBackend(fail_once=True)
    report = run_namespace_reset_probe(
        backend,
        candidate_id="mem0_oss",
        run_prefix="task10-probe-failure",
    )

    assert report.passed is False
    assert len(report.trials) == 20
    assert report.trials[0].passed is False
    assert report.trials[0].error_code == "probe_backend_error"
    assert "secret" not in report.model_dump_json()
    assert all(trial.cleanup_succeeded for trial in report.trials[1:])


def test_reset_probe_converts_invalid_backend_types_to_terminal_failures():
    from mub.vnext.external.probe_v3 import run_namespace_reset_probe

    class InvalidVisibilityBackend(FakeResetBackend):
        def sentinel_visible(self, namespace: str, sentinel_text: str):
            return "yes"

    report = run_namespace_reset_probe(
        InvalidVisibilityBackend(),
        candidate_id="mem0_oss",
        run_prefix="task10-probe-invalid",
    )
    assert report.passed is False
    assert len(report.trials) == 20
    assert all(trial.error_code == "probe_backend_error" for trial in report.trials)


def test_reset_probe_detects_cross_namespace_leakage():
    from mub.vnext.external.probe_v3 import run_namespace_reset_probe

    report = run_namespace_reset_probe(
        FakeResetBackend(leaky=True),
        candidate_id="mem0_oss",
        run_prefix="task10-probe-leaky",
    )
    assert report.passed is False
    assert len(report.trials) == 20
    assert any(not trial.passed for trial in report.trials)


def _snapshot(index: int = 0):
    from mub.vnext.external.probe_v3 import NormalizedCandidateSnapshotV1

    return NormalizedCandidateSnapshotV1(
        state_hash=f"{index + 1:064x}",
        retrieval_entry_ids=("entry-1", "entry-2"),
        action_trace_hash="f" * 64,
    )


def test_determinism_classification_and_repetition_rule_are_fixed():
    from mub.vnext.external.probe_v3 import (
        DeterminismStatus,
        classify_determinism,
        required_canary_repetitions,
    )

    deterministic = classify_determinism(
        (_snapshot(), _snapshot(), _snapshot())
    )
    nondeterministic = classify_determinism(
        (_snapshot(), _snapshot(1), _snapshot())
    )
    inconclusive = classify_determinism((_snapshot(), None, _snapshot()))

    assert deterministic is DeterminismStatus.DETERMINISTIC
    assert nondeterministic is DeterminismStatus.NONDETERMINISTIC
    assert inconclusive is DeterminismStatus.INCONCLUSIVE
    assert required_canary_repetitions(deterministic) == 1
    assert required_canary_repetitions(nondeterministic) == 3
    assert required_canary_repetitions(inconclusive) == 3
    with pytest.raises(ValueError, match="exactly three"):
        classify_determinism((_snapshot(), _snapshot()))

    from mub.vnext.external.probe_v3 import NormalizedCandidateSnapshotV1

    forged = NormalizedCandidateSnapshotV1.model_construct(
        state_hash="not-a-hash",
        retrieval_entry_ids=("entry-1",),
        action_trace_hash=None,
    )
    with pytest.raises(ValueError, match="trust-boundary"):
        classify_determinism((forged, forged, forged))


def _adapter_info(*, extractor: bool = False) -> AdapterInfoV3:
    return AdapterInfoV3(
        adapter_id="mem0_oss",
        adapter_version="adapter-v1",
        system_name="mem0_oss",
        system_version="system-v1",
        sdk_version="sdk-v1",
        configuration_hash="a" * 64,
        extractor_id="extractor" if extractor else None,
        extractor_version="extractor-v1" if extractor else None,
    )


def _level_two_capabilities(**changes) -> AdapterCapabilitiesV3:
    values = {
        "supports_isolated_reset": True,
        "supports_event_ingest": True,
        "supports_add": True,
        "supports_update": True,
        "exports_entries": True,
        "exports_object_keys": True,
        "exports_values": True,
    }
    values.update(changes)
    return AdapterCapabilitiesV3(**values)


def test_capability_verification_recomputes_level_and_rejects_overclaims():
    from mub.vnext.external.probe_v3 import verify_capability_truthfulness

    observed = _level_two_capabilities()
    verified = verify_capability_truthfulness(
        _adapter_info(),
        observed,
        observed,
        state_transition_linkage_available=False,
    )
    assert verified.passed is True
    assert verified.presentation_level == 2
    assert verified.overclaimed_fields == ()

    declared = _level_two_capabilities(exports_action_trace=True)
    rejected = verify_capability_truthfulness(
        _adapter_info(),
        declared,
        observed,
        state_transition_linkage_available=True,
    )
    assert rejected.passed is False
    assert rejected.overclaimed_fields == ("exports_action_trace",)


def test_capability_verification_requires_coherent_frozen_extractor():
    from mub.vnext.external.probe_v3 import verify_capability_truthfulness

    capabilities = _level_two_capabilities(
        requires_evaluation_extractor=True,
        exports_object_keys=False,
        exports_values=False,
        extractor_version="extractor-v1",
    )
    missing = verify_capability_truthfulness(
        _adapter_info(extractor=False),
        capabilities,
        capabilities,
        state_transition_linkage_available=False,
    )
    assert missing.passed is False
    assert missing.extractor_coherent is False

    coherent = verify_capability_truthfulness(
        _adapter_info(extractor=True),
        capabilities,
        capabilities,
        state_transition_linkage_available=False,
    )
    assert coherent.passed is True
    assert coherent.extractor_coherent is True
    assert coherent.presentation_level == 2
