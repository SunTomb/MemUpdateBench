from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from mub.vnext.contracts.v3.common import object_identity, typed_json_equal
from mub.vnext.contracts.v3.enums import LedgerEntryStatus
from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
from mub.vnext.contracts.v3.task import MemoryQueryV3
from mub.vnext.validation.replay_v3 import ReplayResultV3, ReplayVersionV3


ObjectIdentityV3 = tuple[str, str, str, str | None]


@dataclass(frozen=True, slots=True)
class EntryLifecycleStatusV3:
    obsolete: bool | None
    stale: bool | None
    forgotten: bool | None


@dataclass(frozen=True, slots=True)
class TargetLifecycleClassifierV3:
    target_identities: frozenset[ObjectIdentityV3]
    replay: ReplayResultV3

    @classmethod
    def for_query(
        cls, query: MemoryQueryV3, replay: ReplayResultV3
    ) -> TargetLifecycleClassifierV3:
        return cls(
            target_identities=frozenset(
                object_identity(key) for key in query.target_object_keys
            ),
            replay=replay,
        )

    def classify_entry(self, entry: MemoryEntryRecordV3) -> EntryLifecycleStatusV3:
        if entry.object_key_candidate is None:
            return _indeterminate()
        identity = object_identity(entry.object_key_candidate)
        if identity not in self.target_identities:
            return _unrelated()
        ledger = self.replay.ledger_by_identity.get(identity)
        if ledger is None:
            return _indeterminate()
        versions = self.replay.active_versions(ledger)
        if not versions:
            return _indeterminate()
        resolved = _resolve_entry_version(entry, versions)
        if resolved is None:
            return _indeterminate()
        matched_position, matched = resolved
        if matched_position == len(versions) - 1:
            return _unrelated()
        if matched.status == LedgerEntryStatus.TOMBSTONE:
            return EntryLifecycleStatusV3(
                obsolete=True, stale=False, forgotten=False
            )

        stale, forgotten = _lifecycle_flags(
            matched.value, versions, matched_position
        )
        return EntryLifecycleStatusV3(
            obsolete=True, stale=stale, forgotten=forgotten
        )

    def is_stale_value(self, value: Any) -> bool:
        return any(
            _value_status(value, versions, forgotten=False)
            for versions in self._target_active_versions()
        )

    def is_forgotten_value(self, value: Any) -> bool:
        return any(
            _value_status(value, versions, forgotten=True)
            for versions in self._target_active_versions()
        )

    def _target_active_versions(self) -> Iterator[tuple[ReplayVersionV3, ...]]:
        ledgers = self.replay.ledger_by_identity
        for identity in self.target_identities:
            ledger = ledgers.get(identity)
            if ledger is not None:
                yield self.replay.active_versions(ledger)


def _entry_value_status(
    entry: MemoryEntryRecordV3, version: ReplayVersionV3
) -> bool | None:
    if version.status == LedgerEntryStatus.TOMBSTONE:
        tombstone_marker = (
            entry.raw_metadata.get("is_tombstone") is True
            or entry.raw_metadata.get("status") == "tombstone"
        )
        if entry.value_candidate is not None:
            return False
        return True if tombstone_marker else None
    if entry.value_candidate is None:
        return None
    return typed_json_equal(entry.value_candidate, version.value)


def _resolve_entry_version(
    entry: MemoryEntryRecordV3,
    versions: tuple[ReplayVersionV3, ...],
) -> tuple[int, ReplayVersionV3] | None:
    resolved = None
    for position, version in enumerate(versions):
        if (
            entry.version_index is not None
            and version.version_index != entry.version_index
        ):
            continue
        if entry.source_event_ids and not all(
            event_id in version.source_event_ids for event_id in entry.source_event_ids
        ):
            continue
        if _entry_value_status(entry, version) is not True:
            continue
        if resolved is not None:
            return None
        resolved = (position, version)
    return resolved


def _stale_against_current(value: Any, current: ReplayVersionV3) -> bool:
    return current.status == LedgerEntryStatus.TOMBSTONE or (
        current.status == LedgerEntryStatus.PRESENT
        and not typed_json_equal(value, current.value)
    )


def _lifecycle_flags(
    value: Any,
    versions: tuple[ReplayVersionV3, ...],
    position: int,
) -> tuple[bool, bool]:
    stale = _stale_against_current(value, versions[-1])
    forgotten = stale and any(
        versions[index].status == LedgerEntryStatus.TOMBSTONE
        for index in range(position + 1, len(versions))
    )
    return stale, forgotten


def _value_status(
    value: Any,
    versions: tuple[ReplayVersionV3, ...],
    *,
    forgotten: bool,
) -> bool:
    for position in range(len(versions) - 1, -1, -1):
        version = versions[position]
        if version.status != LedgerEntryStatus.PRESENT or not typed_json_equal(
            value, version.value
        ):
            continue
        stale, forgotten_status = _lifecycle_flags(value, versions, position)
        return forgotten_status if forgotten else stale
    return False


def _unrelated() -> EntryLifecycleStatusV3:
    return EntryLifecycleStatusV3(obsolete=False, stale=False, forgotten=False)


def _indeterminate() -> EntryLifecycleStatusV3:
    return EntryLifecycleStatusV3(obsolete=None, stale=None, forgotten=None)


__all__ = ["EntryLifecycleStatusV3", "TargetLifecycleClassifierV3"]
