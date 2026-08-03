import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    EventAnchorSelector,
    GoldActionV3,
    LedgerEntryStatus,
    MemoryQueryV3,
    VersionHistoryEntry,
    VersionHistoryLedger,
)


def key(object_type: str = "slot") -> MemoryObjectKey:
    return MemoryObjectKey(object_type=object_type, namespace="n", entity="e", attribute="a")


def test_v3_selectors_and_ledger_are_strict_and_frozen() -> None:
    selector = CurrentSelector()
    with pytest.raises(ValidationError):
        selector.kind = "previous"
    assert EventAnchorSelector(event_id="ev-1").kind == "event_anchor"
    with pytest.raises(ValidationError):
        EventAnchorSelector(event_id="")

    ledger = VersionHistoryLedger(
        object_key=key(),
        entries=(
            VersionHistoryEntry(version_index=0, status=LedgerEntryStatus.PRESENT, value="old", valid_from_event_id="ev-1", source_event_ids=("ev-1",)),
            VersionHistoryEntry(version_index=1, status=LedgerEntryStatus.PRESENT, value="new", valid_from_event_id="ev-2", source_event_ids=("ev-2",)),
        ),
    )
    assert [entry.version_index for entry in ledger.entries] == [0, 1]
    with pytest.raises(ValidationError):
        VersionHistoryLedger(object_key=key(), entries=(ledger.entries[0], ledger.entries[1].model_copy(update={"version_index": 2})))


def test_object_type_does_not_change_v3_ledger_semantic_identity() -> None:
    entry = VersionHistoryEntry(version_index=0, status=LedgerEntryStatus.PRESENT, value="x", valid_from_event_id="ev", source_event_ids=("ev",))
    left = VersionHistoryLedger(object_key=key("slot"), entries=(entry,))
    right = VersionHistoryLedger(object_key=key("profile"), entries=(entry,))
    assert left.semantic_identity == right.semantic_identity
    assert left.semantic_hash == right.semantic_hash


def test_ttl_delete_requires_effective_at_but_object_delete_does_not() -> None:
    action = {
        "action_id": "a",
        "event_id": "ev",
        "operation": "DELETE",
        "scope": "ttl",
        "target_object_keys": (key(),),
    }
    with pytest.raises(ValidationError, match="TTL.*effective_at"):
        GoldActionV3(**action)
    with pytest.raises(ValidationError, match="TTL.*effective_at"):
        GoldActionV3(**action, effective_at=None)

    scheduled = GoldActionV3(**action, effective_at="010")
    assert scheduled.effective_at == "010"
    assert GoldActionV3(**{**action, "scope": "object"}).effective_at is None


@pytest.mark.parametrize("answer_schema", ["string", "number", "boolean"])
def test_direct_multi_object_current_rejects_unshapeable_answer_schema(answer_schema: str) -> None:
    other = MemoryObjectKey(object_type="slot", namespace="n", entity="other", attribute="a")
    targets = (key(), other)
    with pytest.raises(ValidationError, match="multi-object.*list/object answer schema"):
        MemoryQueryV3(
            query_id="q",
            query_type="multi_object_current",
            text="?",
            selector={"kind": "multi_object_current", "object_keys": targets},
            target_object_keys=targets,
            answer_schema=answer_schema,
            evaluation_mode="state_direct",
        )


@pytest.mark.parametrize("answer_schema", ["list", "object"])
def test_direct_multi_object_current_accepts_shapeable_answer_schema(answer_schema: str) -> None:
    other = MemoryObjectKey(object_type="slot", namespace="n", entity="other", attribute="a")
    targets = (key(), other)
    query = MemoryQueryV3(
        query_id="q",
        query_type="multi_object_current",
        text="?",
        selector={"kind": "multi_object_current", "object_keys": targets},
        target_object_keys=targets,
        answer_schema=answer_schema,
        evaluation_mode="state_direct",
    )
    assert query.answer_schema.value == answer_schema


def test_multi_object_current_consistency_preserves_boolean_answer_schema() -> None:
    other = MemoryObjectKey(object_type="slot", namespace="n", entity="other", attribute="a")
    targets = (key(), other)
    query = MemoryQueryV3(
        query_id="q",
        query_type="multi_object_current_consistency",
        text="?",
        selector={"kind": "multi_object_current", "object_keys": targets},
        target_object_keys=targets,
        answer_schema="boolean",
        evaluation_mode="state_direct",
        synthesis={"kind": "multi_object_current_consistency", "minimum_objects": 2},
    )
    assert query.answer_schema.value == "boolean"
