import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    DerivationStepV3,
    MemoryQueryV3,
    QueryGoldEvidenceV3,
)


def key() -> MemoryObjectKey:
    return MemoryObjectKey(object_type="slot", namespace="n", entity="e", attribute="a")


def test_selector_query_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="selector kind"):
        MemoryQueryV3(query_id="q", query_type="previous", text="?", selector=CurrentSelector(), target_object_keys=(key(),), answer_schema="string", evaluation_mode="state_direct")


def test_gold_evidence_rejects_cycles_and_disconnected_steps() -> None:
    base = {
        "query_id": "q",
        "answer": "x",
        "supporting_object_keys": (key(),),
        "supporting_event_ids": ("ev",),
        "final_derivation_step_id": "b",
    }
    with pytest.raises(ValidationError, match="cyclic"):
        QueryGoldEvidenceV3(**base, derivation_steps=(
            DerivationStepV3(step_id="a", operation="identity", input_step_ids=("b",)),
            DerivationStepV3(step_id="b", operation="answer", input_step_ids=("a",)),
        ))
    with pytest.raises(ValidationError, match="disconnected"):
        QueryGoldEvidenceV3(**base, derivation_steps=(
            DerivationStepV3(step_id="a", operation="read", supporting_event_ids=("ev",)),
            DerivationStepV3(step_id="b", operation="constant"),
        ))
