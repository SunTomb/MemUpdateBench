from __future__ import annotations

from mub.vnext.post_core.model_registry_v1 import build_initial_model_registry_v1
from mub.vnext.post_core.planning_v1 import build_phase0_execution_plan_v1


def test_phase0_plan_accounts_future_calls_but_executes_none() -> None:
    plan = build_phase0_execution_plan_v1(build_initial_model_registry_v1())
    assert plan.network_allowed is False
    assert plan.executable_call_count == 0
    assert all(scope.executable_calls == 0 for scope in plan.scopes)
    qwen = next(scope for scope in plan.scopes if scope.registry_key == "qwen35_9b_bf16")
    assert qwen.requested_calls == 320
    assert "20 semantic cores" in qwen.call_formula
    gpt = next(scope for scope in plan.scopes if scope.registry_key == "gpt_5_5")
    assert gpt.requested_calls == 0
    assert gpt.price_status == "PENDING"
