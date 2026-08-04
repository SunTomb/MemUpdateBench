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
        resolved = resolve_entry_version_v3(entry, versions)
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

    def is_stale_value(self, value: Any) -> bool | None:
        return _raw_value_status(
            value, self._target_active_versions(), forgotten=False
        )

    def is_forgotten_value(self, value: Any) -> bool | None:
        return _raw_value_status(
            value, self._target_active_versions(), forgotten=True
        )

    def has_forgotten_entry(self) -> bool:
        return any(
            version.status == LedgerEntryStatus.PRESENT
            and _lifecycle_flags(version.value, versions, position)[1]
            for versions in self._target_active_versions()
            for position, version in enumerate(versions)
        )

    def has_forgotten_value(self) -> bool:
        for versions in self._target_active_versions():
            seen_values: list[Any] = []
            for position in range(len(versions) - 1, -1, -1):
                version = versions[position]
                if version.status != LedgerEntryStatus.PRESENT or any(
                    typed_json_equal(version.value, seen)
                    for seen in seen_values
                ):
                    continue
                seen_values.append(version.value)
                if _lifecycle_flags(version.value, versions, position)[1]:
                    return True
        return False

    def _target_active_versions(self) -> Iterator[tuple[ReplayVersionV3, ...]]:
        ledgers = self.replay.ledger_by_identity
        for identity in self.target_identities:
            ledger = ledgers.get(identity)
            if ledger is not None:
                yield self.replay.active_versions(ledger)


@dataclass(frozen=True, slots=True)
class QueryLifecycleEvidenceV3:
    classifier: TargetLifecycleClassifierV3
    entry_statuses: tuple[EntryLifecycleStatusV3, ...]
    stale_exposed: bool | None
    forgotten_exposed: bool | None
    stale_copied: bool | None
    forgotten_leaked: bool | None
    has_forgotten_entry: bool
    has_forgotten_value: bool


def build_query_lifecycle_evidence_v3(task, replay, traces, predictions=None):
    predictions = {} if predictions is None else predictions
    gold_by_query = {item.query_id: item.answer for item in task.gold_evidence}
    evidence_by_query = {}
    for query in task.queries:
        classifier = TargetLifecycleClassifierV3.for_query(query, replay)
        trace = traces.get(query.query_id)
        statuses = (
            ()
            if trace is None
            else tuple(
                classifier.classify_entry(entry)
                for entry in trace.retrieved_entries
            )
        )
        stale_exposed = None
        forgotten_exposed = None
        if trace is not None:
            stale_exposed = (
                trace.stale_in_context
                if trace.stale_in_context is not None
                else _tri_state_any(status.stale for status in statuses)
            )
            forgotten_exposed = _tri_state_any(
                status.forgotten for status in statuses
            )
        prediction = predictions.get(query.query_id)
        stale_copied = None
        forgotten_leaked = None
        if prediction is not None:
            wrong = not typed_json_equal(
                prediction.parsed_answer, gold_by_query[query.query_id]
            )
            stale_copied = wrong and classifier.is_stale_value(
                prediction.parsed_answer
            )
            forgotten_leaked = wrong and classifier.is_forgotten_value(
                prediction.parsed_answer
            )
        evidence_by_query[query.query_id] = QueryLifecycleEvidenceV3(
            classifier=classifier,
            entry_statuses=statuses,
            stale_exposed=stale_exposed,
            forgotten_exposed=forgotten_exposed,
            stale_copied=stale_copied,
            forgotten_leaked=forgotten_leaked,
            has_forgotten_entry=classifier.has_forgotten_entry(),
            has_forgotten_value=classifier.has_forgotten_value(),
        )
    return evidence_by_query


def _tri_state_any(statuses) -> bool | None:
    statuses = tuple(statuses)
    if any(status is None for status in statuses):
        return None
    return any(status is True for status in statuses)


def _entry_value_status(
    entry: MemoryEntryRecordV3, version: ReplayVersionV3
) -> bool | None:
    tombstone_marker = (
        entry.raw_metadata.get("is_tombstone") is True
        or entry.raw_metadata.get("status") == "tombstone"
    )
    if version.status == LedgerEntryStatus.TOMBSTONE:
        if entry.value_candidate is not None:
            return False
        return True if tombstone_marker else None
    if tombstone_marker or entry.value_candidate is None:
        return None
    return typed_json_equal(entry.value_candidate, version.value)


def resolve_entry_version_v3(
    entry: MemoryEntryRecordV3,
    versions: tuple[ReplayVersionV3, ...],
) -> tuple[int, ReplayVersionV3] | None:
    return _resolve_entry_version(entry, versions)


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


def _raw_value_status(
    value: Any,
    target_versions: Iterator[tuple[ReplayVersionV3, ...]],
    *,
    forgotten: bool,
) -> bool | None:
    statuses = tuple(
        _value_status(value, versions, forgotten=forgotten)
        for versions in target_versions
    )
    matched_statuses = tuple(status for status in statuses if status is not None)
    if any(status is True for status in matched_statuses) and any(
        status is False for status in matched_statuses
    ):
        return None
    return any(status is True for status in matched_statuses)


def _value_status(
    value: Any,
    versions: tuple[ReplayVersionV3, ...],
    *,
    forgotten: bool,
) -> bool | None:
    for position in range(len(versions) - 1, -1, -1):
        version = versions[position]
        if version.status != LedgerEntryStatus.PRESENT or not typed_json_equal(
            value, version.value
        ):
            continue
        stale, forgotten_status = _lifecycle_flags(value, versions, position)
        return forgotten_status if forgotten else stale
    return None


def _unrelated() -> EntryLifecycleStatusV3:
    return EntryLifecycleStatusV3(obsolete=False, stale=False, forgotten=False)


def _indeterminate() -> EntryLifecycleStatusV3:
    return EntryLifecycleStatusV3(obsolete=None, stale=None, forgotten=None)


__all__ = [
    "EntryLifecycleStatusV3",
    "QueryLifecycleEvidenceV3",
    "TargetLifecycleClassifierV3",
    "build_query_lifecycle_evidence_v3",
    "resolve_entry_version_v3",
]
