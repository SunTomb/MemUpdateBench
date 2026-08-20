from __future__ import annotations

from decimal import Decimal
from typing import Literal, Mapping

from pydantic import Field, StrictInt, StrictStr

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.post_core.contracts_v1 import CandidateIdentityState, ModelCandidateV1


LiteralZero = Literal[0]


class PlannedScopeV1(ImmutableContractModel):
    schema_version: str = "memupdatebench.post-core.planned-scope.v1"
    registry_key: StrictStr
    scope: StrictStr
    requested_calls: StrictInt = Field(ge=0)
    executable_calls: LiteralZero = 0
    call_formula: StrictStr
    max_retries: StrictInt = Field(ge=0)
    prompt_token_cap: StrictInt = Field(ge=0)
    output_token_cap: StrictInt = Field(ge=0)
    estimated_cost: Decimal = Field(ge=Decimal("0"))
    hard_max_cost: Decimal = Field(ge=Decimal("0"))
    price_status: StrictStr


class ExecutionPlanV1(ImmutableContractModel):
    schema_version: str = "memupdatebench.post-core.execution-plan.v1"
    phase: LiteralZero = 0
    network_allowed: Literal[False] = False
    executable_call_count: LiteralZero = 0
    scopes: tuple[PlannedScopeV1, ...]


def build_phase0_execution_plan_v1(registry: Mapping[str, ModelCandidateV1]) -> ExecutionPlanV1:
    scopes = []
    for key, candidate in registry.items():
        planned_scope = candidate.scopes[0]
        if key == "qwen35_9b_bf16":
            requested, formula = 320, "20 semantic cores * 4 tasks/core * 2 conditions * 2 seeds"
        else:
            requested, formula = 0, "pending qualification; no Phase 0 executable calls"
        if candidate.state is CandidateIdentityState.QUALIFIED:
            raise ValueError("Phase 0 registry must not contain qualified/executable candidates")
        scopes.append(
            PlannedScopeV1(
                registry_key=key,
                scope=planned_scope,
                requested_calls=requested,
                call_formula=formula,
                max_retries=0,
                prompt_token_cap=0,
                output_token_cap=0,
                estimated_cost=Decimal("0"),
                hard_max_cost=Decimal("0"),
                price_status="PENDING" if candidate.role.startswith("closed") else "LOCAL_UNPRICED",
            )
        )
    return ExecutionPlanV1(scopes=tuple(scopes))

__all__ = ["ExecutionPlanV1", "PlannedScopeV1", "build_phase0_execution_plan_v1"]
