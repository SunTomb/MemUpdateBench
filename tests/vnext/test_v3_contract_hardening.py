import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.v3.adapter import AdapterActionResultV3
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, ParsedManagerActionV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    DerivationStepV3,
    GoldActionV3,
    QueryGoldEvidenceV3,
    VersionHistoryEntry,
    VersionHistoryLedger,
)


def key(entity="e", object_type="slot"):
    return MemoryObjectKey(object_type=object_type, namespace="n", entity=entity, attribute="a")


def entry(index=0, value="x", start="e0", end=None, source=None):
    return VersionHistoryEntry(version_index=index, status="present", value=value, valid_from_event_id=start, valid_until_event_id=end, source_event_ids=(source or start,))


def test_semantic_hash_only_ignores_key_classification_object_type() -> None:
    base = VersionHistoryLedger(object_key=key(object_type="slot"), entries=(entry(value={"object_type": "application-a"}),))
    reclassified = VersionHistoryLedger(object_key=key(object_type="profile"), entries=base.entries)
    changed_value = VersionHistoryLedger(object_key=key(), entries=(entry(value={"object_type": "application-b"}),))
    assert base.semantic_hash == reclassified.semantic_hash
    assert base.semantic_hash != changed_value.semantic_hash


def test_nested_v3_state_is_deeply_immutable() -> None:
    ledger = VersionHistoryLedger(object_key=key(), entries=(entry(value={"nested": [1]}),))
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        ledger.object_key.entity = "changed"
    with pytest.raises((TypeError, AttributeError)):
        ledger.entries[0].value["nested"].append(2)

    prediction = AnswerPredictionV3(query_id="q", raw_output="1", parsed_answer=1, format_valid=True, usage={"tokens": 1})
    with pytest.raises((TypeError, AttributeError)):
        prediction.usage["tokens"] = 2


def test_gold_actions_use_immutable_targets_and_shared_scope_rules() -> None:
    action = GoldActionV3(action_id="a", event_id="e0", operation="DELETE", scope="object", target_object_keys=(key(),))
    with pytest.raises((TypeError, AttributeError)):
        action.target_object_keys.append(key("other"))
    with pytest.raises(ValidationError):
        GoldActionV3(action_id="a", event_id="e0", operation="DELETE", scope="object", target_object_keys=(key(), key("other")))
    with pytest.raises(ValidationError):
        GoldActionV3(action_id="a", event_id="e0", operation="UPDATE", scope="entity", target_object_keys=(key(),), value="x")


def test_runtime_executed_actions_require_coherent_operation_scope_and_payload() -> None:
    common = dict(event_id="e0", format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok")
    with pytest.raises(ValidationError):
        ParsedManagerActionV3(**common)
    with pytest.raises(ValidationError):
        ParsedManagerActionV3(**common, operation="DELETE", observed_scope="namespace", target_object_keys=(key(), key("other")), value="forbidden")


def test_adapter_action_result_is_strict_frozen_and_coherent() -> None:
    result = AdapterActionResultV3(event_id="e0", requested_operation="DELETE", effective_operation="DELETE", observed_scope="object", target_object_keys=(key(),), affected_entry_ids=("id",), raw_result={"nested": [1]})
    with pytest.raises((TypeError, AttributeError)):
        result.raw_result["nested"].append(2)
    for changes in (
        {"event_id": ""},
        {"affected_entry_ids": ("",)},
        {"affected_entry_ids": ("id", "id")},
        {"target_object_keys": (key(), key(object_type="profile"))},
        {"observed_scope": "object", "target_object_keys": (key(), key("other"))},
    ):
        data = result.model_dump(mode="python")
        data.update(changes)
        with pytest.raises(ValidationError):
            AdapterActionResultV3.model_validate(data)


def test_nonfinite_values_fail_during_contract_construction() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            entry(value={"nested": [bad]})
        with pytest.raises(ValidationError):
            AnswerPredictionV3(query_id="q", raw_output="x", parsed_answer={"nested": bad}, format_valid=True)
        with pytest.raises(ValidationError):
            AdapterActionResultV3(event_id="e0", raw_result={"nested": bad})
        with pytest.raises(ValidationError):
            RetrievalTraceV3(query_id="q", scores=(bad,))


def test_retrieval_scores_require_exact_builtin_floats() -> None:
    class HostileFloat(float):
        pass

    item = MemoryEntryRecordV3(entry_id="id", content="x")
    for bad in (1, HostileFloat(1.0)):
        with pytest.raises(ValidationError):
            RetrievalTraceV3(query_id="q", retrieved_entries=(item,), scores=(bad,))
    assert RetrievalTraceV3(query_id="q", retrieved_entries=(item,), scores=(1.0,)).scores == (1.0,)


def test_identifiers_reject_whitespace_and_string_subclasses() -> None:
    class HostileString(str):
        pass

    for bad in (" ", HostileString("id")):
        with pytest.raises(ValidationError):
            GoldActionV3(action_id=bad, event_id="e0", operation="NOOP")
        with pytest.raises(ValidationError):
            ParsedManagerActionV3(event_id=bad, format_valid=False, execution_status="failed", fallback_used=False, raw_output="")
        with pytest.raises(ValidationError):
            AdapterActionResultV3(event_id=bad)
        with pytest.raises(ValidationError):
            AnswerPredictionV3(query_id=bad, raw_output="", disposition="unavailable", format_valid=False)


def test_derivation_steps_must_be_topologically_ordered() -> None:
    with pytest.raises(ValidationError, match="topological"):
        QueryGoldEvidenceV3(query_id="q", answer="x", supporting_object_keys=(key(),), supporting_event_ids=("e0",), derivation_steps=(
            DerivationStepV3(step_id="answer", operation="answer", input_step_ids=("read",)),
            DerivationStepV3(step_id="read", operation="read", supporting_event_ids=("e0",)),
        ), final_derivation_step_id="answer")
