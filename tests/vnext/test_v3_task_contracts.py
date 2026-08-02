import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    EventAnchorSelector,
    LedgerEntryStatus,
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
    left = VersionHistoryLedger(object_key=key("slot"), entries=())
    right = VersionHistoryLedger(object_key=key("profile"), entries=())
    assert left.semantic_identity == right.semantic_identity
    assert left.semantic_hash == right.semantic_hash
