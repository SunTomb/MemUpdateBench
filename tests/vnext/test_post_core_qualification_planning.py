from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal

import pytest

from mub.vnext.post_core.qualification_receipts_v1 import (
    CapabilityBudgetV1,
    CapabilityFixtureV1,
)


FROZEN_REGISTRY_KEYS = (
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5",
)


def _budget() -> CapabilityBudgetV1:
    return CapabilityBudgetV1(
        max_prompt_tokens=256,
        max_output_tokens=64,
        estimated_cost=Decimal("0"),
        hard_max_cost=Decimal("0"),
        price_version="offline-v1",
        timeout_seconds=30,
    )


def _fixtures() -> tuple[CapabilityFixtureV1, ...]:
    fixture_ids = ("exact_ok_1", "exact_ok_2", "parser_city_1", "parser_city_2")
    return tuple(
        CapabilityFixtureV1(
            fixture_id=fixture_ids[index],
            category="EXACT_OUTPUT" if index < 2 else "CHAT_TEMPLATE_PARSER",
            prompt_sha256=f"{index + 1:064x}",
            parser_sha256=f"{index + 11:064x}",
            max_prompt_tokens=128,
            max_output_tokens=32,
        )
        for index in range(4)
    )


def _config(**changes):
    from mub.vnext.post_core.qualification_planning_v1 import CapabilitySmokePlanConfigV1

    config = CapabilitySmokePlanConfigV1(
        release_id="release-v1",
        registry_keys=FROZEN_REGISTRY_KEYS,
        budget=_budget(),
    )
    return replace(config, **changes)


def test_builds_exact_two_phase_two_repetition_plan_per_role() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    plan = build_capability_smoke_plan_v1(_config(), _fixtures())

    assert len(plan.attempts) == 128
    assert plan.base_attempts_per_role == 8
    assert plan.escalation_attempts_per_role == 8
    for key in plan.registry_keys:
        rows = [row for row in plan.attempts if row.registry_key == key]
        assert len(rows) == 16
        assert [row.phase.value for row in rows].count("BASE") == 8
        assert [row.phase.value for row in rows].count("ESCALATION") == 8
        assert {(row.fixture_id, row.repetition) for row in rows} == {
            (fixture.fixture_id, repetition)
            for fixture in _fixtures()
            for repetition in (1, 2)
        }


def test_runtime_class_and_execution_boundary_are_derived() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    plan = build_capability_smoke_plan_v1(_config(), _fixtures())

    classes = {row.registry_key: row.runtime_or_endpoint_class for row in plan.attempts}
    assert classes["qwen35_9b_bf16"] == "transformers_offline"
    assert classes["claude_sonnet_4_6"] == "api_transfer_station"
    assert all(not row.authorized and not row.executable for row in plan.attempts)
    assert all(row.budget.max_calls == 1 and row.budget.max_retries == 0 for row in plan.attempts)
    assert {
        (row.budget.max_prompt_tokens, row.budget.max_output_tokens)
        for row in plan.attempts
    } == {(128, 32)}


def test_plan_is_canonical_and_does_not_accept_caller_call_ids() -> None:
    from mub.vnext.post_core.contracts_v1 import canonical_bytes
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    first = build_capability_smoke_plan_v1(_config(), _fixtures())
    second = build_capability_smoke_plan_v1(_config(), tuple(reversed(_fixtures())))
    assert canonical_bytes(first) == canonical_bytes(second)
    assert len({row.call_id for row in first.attempts}) == 128
    assert all("call_id" not in row.model_fields_set for row in first.attempts)


def test_preserves_exact_frozen_registry_order() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    plan = build_capability_smoke_plan_v1(_config(), _fixtures())

    assert plan.registry_keys == FROZEN_REGISTRY_KEYS
    assert tuple(row.registry_key for row in plan.attempts[::16]) == FROZEN_REGISTRY_KEYS


@pytest.mark.parametrize(
    "registry_keys",
    [
        FROZEN_REGISTRY_KEYS[:-1],
        (*FROZEN_REGISTRY_KEYS, "closed_extra"),
        tuple(reversed(FROZEN_REGISTRY_KEYS)),
    ],
)
def test_rejects_registry_subset_extra_or_reordered(registry_keys) -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    with pytest.raises(ValueError, match="exact frozen"):
        build_capability_smoke_plan_v1(_config(registry_keys=registry_keys), _fixtures())


@pytest.mark.parametrize(
    "changes",
    [
        {"registry_keys": ()},
        {"registry_keys": ("qwen35_9b_bf16", "qwen35_9b_bf16")},
        {"base_attempts_per_role": 7},
        {"escalation_attempts_per_role": 9},
        {"registry_keys": ("unknown_role",)},
    ],
)
def test_rejects_invalid_role_or_shape_config(changes) -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    with pytest.raises((TypeError, ValueError)):
        build_capability_smoke_plan_v1(_config(**changes), _fixtures())


@pytest.mark.parametrize(
    "fixtures",
    [
        lambda: _fixtures()[:3],
        lambda: (*_fixtures(), _fixtures()[0]),
        lambda: tuple(
            fixture.model_copy(update={"category": "CHAT_TEMPLATE_PARSER"})
            for fixture in _fixtures()
        ),
    ],
)
def test_rejects_wrong_fixture_count_duplicate_or_category_balance(fixtures) -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    with pytest.raises(ValueError):
        build_capability_smoke_plan_v1(_config(), fixtures())


def test_rejects_balanced_categories_with_substituted_fixture_id() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    fixtures = list(_fixtures())
    fixtures[0] = fixtures[0].model_copy(update={"fixture_id": "exact_ok_wrong"})
    with pytest.raises(ValueError, match="exact fixture"):
        build_capability_smoke_plan_v1(_config(), tuple(fixtures))


@pytest.mark.parametrize(
    "payload",
    [
        {"metrics": {"accuracy": 1}},
        {"task_payload": {"prompt": "do not run"}},
        {"call_id": "caller-supplied"},
        {"authorized": True},
        {"endpoint": "https://example.test"},
        {"model_output": "answer"},
    ],
)
def test_rejects_forbidden_mapping_payloads(payload) -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    with pytest.raises(ValueError):
        build_capability_smoke_plan_v1({**asdict(_config()), **payload}, _fixtures())


def test_contract_rejects_duplicate_attempt_coordinate() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilitySmokePlanV1

    plan = build_capability_smoke_plan_v1(_config(), _fixtures())
    with pytest.raises(ValueError, match="coordinates"):
        CapabilitySmokePlanV1(
            release_id=plan.release_id,
            registry_keys=plan.registry_keys,
            attempts=(*plan.attempts[:-1], plan.attempts[0]),
        )


def test_exports_only_planner_public_api() -> None:
    import mub.vnext.post_core.qualification_planning_v1 as planning

    assert planning.__all__ == [
        "CapabilitySmokePlanConfigV1",
        "build_capability_smoke_plan_v1",
    ]
