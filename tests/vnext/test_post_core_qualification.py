from __future__ import annotations

from mub.vnext.post_core.model_registry_v1 import build_initial_model_registry_v1
from mub.vnext.post_core.qualification_v1 import qualify_registry_offline_v1


def test_phase0_qualification_is_pending_and_no_network() -> None:
    report, probes = qualify_registry_offline_v1(build_initial_model_registry_v1())
    assert len(report.gates) == 8
    assert all(row.status == "PENDING" for row in report.gates)
    assert probes.network_allowed is False
    assert probes.provider_calls == 0
    assert probes.model_loads == 0
