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
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_budget_v1

    return build_capability_budget_v1()


def _fixtures() -> tuple[CapabilityFixtureV1, ...]:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_fixtures_v1

    return build_capability_fixtures_v1()


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
    second = build_capability_smoke_plan_v1(_config(), _fixtures())
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


def test_rejects_swapped_fixture_categories() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    fixtures = list(_fixtures())
    fixtures[0] = fixtures[0].model_copy(update={"category": "CHAT_TEMPLATE_PARSER"})
    fixtures[2] = fixtures[2].model_copy(update={"category": "EXACT_OUTPUT"})
    with pytest.raises(ValueError, match="canonical fixture"):
        build_capability_smoke_plan_v1(_config(), tuple(fixtures))


def test_rejects_balanced_categories_with_substituted_fixture_id() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_smoke_plan_v1

    fixtures = list(_fixtures())
    fixtures[0] = fixtures[0].model_copy(update={"fixture_id": "exact_ok_wrong"})
    with pytest.raises(ValueError, match="canonical fixture"):
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


def test_canonical_planner_owned_fixtures_budget_and_hashes_are_deterministic() -> None:
    import mub.vnext.post_core.qualification_planning_v1 as planning
    from mub.vnext.post_core.contracts_v1 import canonical_bytes
    from mub.vnext.post_core.qualification_planning_v1 import (
        build_capability_budget_v1,
        build_capability_fixtures_v1,
    )

    fixtures = build_capability_fixtures_v1()
    assert fixtures == build_capability_fixtures_v1()
    assert len(fixtures) == 4
    assert all(fixture.prompt_sha256 != "0" * 64 for fixture in fixtures)
    assert all(fixture.parser_sha256 != "0" * 64 for fixture in fixtures)
    assert canonical_bytes(tuple(fixture.model_dump(mode="json") for fixture in fixtures)) == canonical_bytes(
        tuple(fixture.model_dump(mode="json") for fixture in build_capability_fixtures_v1())
    )
    assert all("task" not in payload and "gold" not in payload for payload in planning._CAPABILITY_PROMPT_PAYLOADS.values())
    budget = build_capability_budget_v1()
    assert (budget.max_calls, budget.max_retries, budget.timeout_seconds) == (1, 0, 60)
    assert budget.estimated_cost == budget.hard_max_cost == Decimal("0")


def test_planner_rejects_caller_substituted_fixture_hash_or_budget() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import (
        build_capability_budget_v1,
        build_capability_fixtures_v1,
        build_capability_smoke_plan_v1,
    )

    fixtures = build_capability_fixtures_v1()
    substituted = (fixtures[0].model_copy(update={"prompt_sha256": "0" * 64}), *fixtures[1:])
    config = _config(budget=build_capability_budget_v1())
    with pytest.raises(ValueError, match="canonical fixture"):
        build_capability_smoke_plan_v1(config, substituted)
    with pytest.raises(ValueError, match="canonical capability budget"):
        build_capability_smoke_plan_v1(
            _config(budget=build_capability_budget_v1().model_copy(update={"timeout_seconds": 59})),
            fixtures,
        )


def test_request_envelope_contains_hash_bound_prompt_and_parser_payloads() -> None:
    from mub.vnext.post_core.qualification_planning_v1 import (
        build_capability_fixtures_v1,
        build_capability_request_envelope_v1,
        build_capability_smoke_plan_v1,
    )

    attempt = build_capability_smoke_plan_v1(_config(), build_capability_fixtures_v1()).attempts[0]
    envelope = build_capability_request_envelope_v1(attempt)

    assert envelope["prompt_payload"]["messages"][0]["role"] == "user"
    assert envelope["prompt_payload"]["messages"][0]["content"] == "Reply with exactly READY."
    assert envelope["parser_contract"]["projection"] == "literal_text"


    import mub.vnext.post_core.qualification_planning_v1 as planning

    assert planning.__all__ == [
        "CapabilitySmokePlanConfigV1",
        "build_capability_budget_v1",
        "build_capability_fixtures_v1",
        "build_capability_request_envelope_v1",
        "build_capability_smoke_plan_v1",
    ]
