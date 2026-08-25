from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
from typing import Any, Mapping, Sequence


from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilityAttemptPlanV1,
    CapabilityBudgetV1,
    CapabilityFixtureV1,
    CapabilitySmokePlanV1,
)


_FORBIDDEN_KEYS = {
    "authorized",
    "call_id",
    "endpoint",
    "executable",
    "metrics",
    "model",
    "model_output",
    "output",
    "task",
    "task_payload",
}


@dataclass(frozen=True, slots=True)
class CapabilitySmokePlanConfigV1:
    release_id: str
    registry_keys: tuple[str, ...]
    budget: CapabilityBudgetV1
    base_attempts_per_role: int = 8
    escalation_attempts_per_role: int = 8
    max_retries: int = 0
    authorized: bool = False

    def __post_init__(self) -> None:
        if type(self.release_id) is not str or not self.release_id:
            raise ValueError("release_id must be a nonempty string")
        if type(self.registry_keys) is not tuple:
            raise TypeError("registry_keys must be a tuple")
        if type(self.budget) is not CapabilityBudgetV1:
            raise TypeError("budget must be CapabilityBudgetV1")
        if type(self.base_attempts_per_role) is not int:
            raise TypeError("base_attempts_per_role must be an int")
        if type(self.escalation_attempts_per_role) is not int:
            raise TypeError("escalation_attempts_per_role must be an int")
        if type(self.max_retries) is not int:
            raise TypeError("max_retries must be an int")
        if type(self.authorized) is not bool:
            raise TypeError("authorized must be a bool")


def _reject_forbidden_mapping(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"planner input cannot carry {normalized}")
            _reject_forbidden_mapping(nested)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for nested in value:
            _reject_forbidden_mapping(nested)


def _reject_forbidden_object_fields(value: object) -> None:
    fields = getattr(value, "__dict__", {})
    if isinstance(fields, Mapping):
        _reject_forbidden_mapping(fields)
    for name in _FORBIDDEN_KEYS:
        if hasattr(value, name):
            raise ValueError(f"planner input cannot carry {name}")


_EXPECTED_RELEASE_ID = "memupdatebench.post-core.qualification.v1"
_EXPECTED_REGISTRY_KEYS = (
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5",
)
_EXPECTED_FIXTURE_CATEGORIES = {
    "exact_ok_1": "EXACT_OUTPUT",
    "exact_ok_2": "EXACT_OUTPUT",
    "parser_city_1": "CHAT_TEMPLATE_PARSER",
    "parser_city_2": "CHAT_TEMPLATE_PARSER",
}
_EXACT_OUTPUT_PARSER_CONTRACT_VERSION = "memupdatebench.post-core.capability-parser.exact-output.v1"
_CHAT_TEMPLATE_PARSER_CONTRACT_VERSION = "memupdatebench.post-core.capability-parser.chat-template.v1"
_CAPABILITY_PROMPT_PAYLOADS = {
    "exact_ok_1": {"messages": ({"content": "Reply with exactly READY.", "role": "user"},), "version": "v1"},
    "exact_ok_2": {"messages": ({"content": "Reply with exactly ACK.", "role": "user"},), "version": "v1"},
    "parser_city_1": {"messages": ({"content": "Reply with one concise label.", "role": "user"},), "version": "v1"},
    "parser_city_2": {"messages": ({"content": "Reply with one concise identifier.", "role": "user"},), "version": "v1"},
}
_CAPABILITY_PARSER_CONTRACT = {
    "exact_output": {"version": _EXACT_OUTPUT_PARSER_CONTRACT_VERSION, "projection": "literal_text"},
    "chat_template_parser": {"version": _CHAT_TEMPLATE_PARSER_CONTRACT_VERSION, "projection": "single_assistant_text"},
}
_CAPABILITY_FIXTURE_LIMITS = {
    "exact_ok_1": (128, 32),
    "exact_ok_2": (128, 32),
    "parser_city_1": (128, 32),
    "parser_city_2": (128, 32),
}


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def build_capability_fixtures_v1() -> tuple[CapabilityFixtureV1, ...]:
    fixtures: list[CapabilityFixtureV1] = []
    for fixture_id, category in _EXPECTED_FIXTURE_CATEGORIES.items():
        parser_key = "exact_output" if category == "EXACT_OUTPUT" else "chat_template_parser"
        max_prompt_tokens, max_output_tokens = _CAPABILITY_FIXTURE_LIMITS[fixture_id]
        fixtures.append(
            CapabilityFixtureV1(
                fixture_id=fixture_id,
                category=category,
                prompt_sha256=_sha256(_CAPABILITY_PROMPT_PAYLOADS[fixture_id]),
                parser_sha256=_sha256(_CAPABILITY_PARSER_CONTRACT[parser_key]),
                max_prompt_tokens=max_prompt_tokens,
                max_output_tokens=max_output_tokens,
            )
        )
    return tuple(fixtures)


def build_capability_budget_v1() -> CapabilityBudgetV1:
    return CapabilityBudgetV1(
        max_calls=1,
        max_prompt_tokens=128,
        max_output_tokens=32,
        estimated_cost=Decimal("0"),
        hard_max_cost=Decimal("0"),
        price_version="LOCAL_UNPRICED/PENDING qualification",
        max_retries=0,
        timeout_seconds=60,
    )


def _fixture_bundle_bytes(fixtures: Sequence[CapabilityFixtureV1]) -> bytes:
    return canonical_bytes(tuple(fixture.model_dump(mode="json") for fixture in fixtures))


def _canonical_fixture_bundle_bytes() -> bytes:
    return _fixture_bundle_bytes(build_capability_fixtures_v1())


def _canonical_parser_contract_bytes() -> bytes:
    return canonical_bytes(_CAPABILITY_PARSER_CONTRACT)


_RUNTIME_CLASSES = {
    "qwen35_9b_bf16": "transformers_offline",
    "meta_muse_glimmer_30b_int4": "llama_cpp_offline",
    "meta_muse_glimmer_30b_bf16": "transformers_offline",
    "claude_sonnet_4_6": "api_transfer_station",
    "claude_opus_4_8": "api_transfer_station",
    "gemini_3_6_flash": "api_transfer_station",
    "grok_4_5": "api_transfer_station",
    "gpt_5_5": "api_transfer_station",
}


def _runtime_class(registry_key: str) -> str:
    try:
        return _RUNTIME_CLASSES[registry_key]
    except KeyError as exc:
        raise ValueError(f"registry key has no recognized capability role prefix: {registry_key}") from exc


def _fixture_category(fixture: CapabilityFixtureV1) -> str:
    if fixture.category not in {"EXACT_OUTPUT", "CHAT_TEMPLATE_PARSER"}:
        raise ValueError("fixture must declare an exact output or chat-template parser category")
    return fixture.category


def _fixture_values(fixture: CapabilityFixtureV1) -> tuple[str, str, str, int, int]:
    return (
        fixture.fixture_id,
        fixture.prompt_sha256,
        fixture.parser_sha256,
        fixture.max_prompt_tokens,
        fixture.max_output_tokens,
    )


def _validate_inputs(
    config: CapabilitySmokePlanConfigV1 | Mapping[str, Any],
    fixtures: Sequence[CapabilityFixtureV1],
) -> CapabilitySmokePlanConfigV1:
    if isinstance(config, Mapping):
        _reject_forbidden_mapping(config)
        raise TypeError("planner config must be CapabilitySmokePlanConfigV1")
    if type(config) is not CapabilitySmokePlanConfigV1:
        _reject_forbidden_object_fields(config)
        raise TypeError("planner config must be CapabilitySmokePlanConfigV1")
    if config.base_attempts_per_role != 8 or config.escalation_attempts_per_role != 8:
        raise ValueError("planner requires exactly eight attempts per phase and role")
    if config.max_retries != 0 or config.authorized is not False:
        raise ValueError("planner cannot authorize execution or retries")
    if config.registry_keys != _EXPECTED_REGISTRY_KEYS:
        raise ValueError("registry_keys must match the exact frozen registry tuple and order")
    for key in config.registry_keys:
        if type(key) is not str or not key:
            raise ValueError("registry keys must be nonempty strings")
        _runtime_class(key)
    canonical_budget = build_capability_budget_v1()
    if canonical_bytes(config.budget) != canonical_bytes(canonical_budget):
        raise ValueError("planner requires the canonical capability budget")
    canonical_fixtures = build_capability_fixtures_v1()
    if len(fixtures) != 4:
        raise ValueError("planner requires exactly four fixtures")
    _reject_forbidden_mapping(fixtures)
    seen_ids: set[str] = set()
    fixture_categories: dict[str, str] = {}
    for fixture in fixtures:
        if type(fixture) is not CapabilityFixtureV1:
            _reject_forbidden_mapping(fixture)
            raise TypeError("fixtures must be CapabilityFixtureV1 instances")
        fixture_id = fixture.fixture_id
        if fixture_id in seen_ids:
            raise ValueError("fixture IDs must be unique")
        seen_ids.add(fixture_id)
        category = _fixture_category(fixture)
        fixture_categories[fixture_id] = category
        _fixture_values(fixture)
    if _fixture_bundle_bytes(tuple(fixtures)) != _fixture_bundle_bytes(canonical_fixtures):
        raise ValueError("planner requires the canonical fixture bundle")
    if fixture_categories != _EXPECTED_FIXTURE_CATEGORIES:
        raise ValueError("fixtures must use the exact fixture category mapping")
    if len(fixture_categories) != 4:
        raise ValueError("fixtures must use the exact fixture IDs")
    if tuple(fixture_categories.values()).count("EXACT_OUTPUT") != 2 or tuple(fixture_categories.values()).count("CHAT_TEMPLATE_PARSER") != 2:
        raise ValueError("planner requires two fixtures of each category")
    return config


def build_capability_request_envelope_v1(
    attempt: CapabilityAttemptPlanV1,
) -> dict[str, Any]:
    if type(attempt) is not CapabilityAttemptPlanV1:
        raise TypeError("attempt must be CapabilityAttemptPlanV1")
    try:
        prompt_payload = _CAPABILITY_PROMPT_PAYLOADS[attempt.fixture_id]
        category = _EXPECTED_FIXTURE_CATEGORIES[attempt.fixture_id]
    except KeyError as exc:
        raise ValueError("attempt fixture is not part of the canonical capability bundle") from exc
    parser_key = "exact_output" if category == "EXACT_OUTPUT" else "chat_template_parser"
    parser_contract = _CAPABILITY_PARSER_CONTRACT[parser_key]
    if attempt.prompt_sha256 != _sha256(prompt_payload):
        raise ValueError("attempt prompt hash does not match the canonical prompt payload")
    if attempt.parser_sha256 != _sha256(parser_contract):
        raise ValueError("attempt parser hash does not match the canonical parser contract")
    return {
        **attempt.model_dump(mode="json"),
        "prompt_payload": prompt_payload,
        "parser_contract": parser_contract,
    }


def build_capability_smoke_plan_v1(
    config: CapabilitySmokePlanConfigV1 | Mapping[str, Any],
    fixtures: Sequence[CapabilityFixtureV1],
) -> CapabilitySmokePlanV1:
    config = _validate_inputs(config, fixtures)
    fixture_rows = tuple(sorted((_fixture_values(fixture) for fixture in fixtures), key=lambda row: row[0]))
    attempts: list[CapabilityAttemptPlanV1] = []
    for registry_key in config.registry_keys:
        runtime_class = _runtime_class(registry_key)
        for phase in (AttemptPhase.BASE, AttemptPhase.ESCALATION):
            for fixture_id, prompt_sha256, parser_sha256, max_prompt_tokens, max_output_tokens in fixture_rows:
                attempt_budget = config.budget.model_copy(
                    update={
                        "max_prompt_tokens": min(config.budget.max_prompt_tokens, max_prompt_tokens),
                        "max_output_tokens": min(config.budget.max_output_tokens, max_output_tokens),
                    }
                )
                for repetition in (1, 2):
                    attempts.append(
                        CapabilityAttemptPlanV1(
                            release_id=config.release_id,
                            registry_key=registry_key,
                            fixture_id=fixture_id,
                            phase=phase,
                            repetition=repetition,
                            prompt_sha256=prompt_sha256,
                            parser_sha256=parser_sha256,
                            runtime_or_endpoint_class=runtime_class,
                            budget=attempt_budget,
                            authorized=False,
                            executable=False,
                        )
                    )
    return CapabilitySmokePlanV1(
        release_id=config.release_id,
        registry_keys=config.registry_keys,
        base_attempts_per_role=8,
        escalation_attempts_per_role=8,
        max_retries=0,
        authorized=False,
        attempts=tuple(attempts),
    )


__all__ = [
    "CapabilitySmokePlanConfigV1",
    "build_capability_budget_v1",
    "build_capability_fixtures_v1",
    "build_capability_request_envelope_v1",
    "build_capability_smoke_plan_v1",
]
