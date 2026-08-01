from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from mub.vnext.contracts.common import ContractModel, MemoryObjectKey, thaw_json
from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.task import MemUpdateTask

ContextOrder = Literal["chronological", "reverse_chronological"]
ContextAnnotation = Literal["none", "latest_outdated_label"]
_SUPPORTED = {
    ("chronological", "none"),
    ("reverse_chronological", "none"),
    ("reverse_chronological", "latest_outdated_label"),
}


class ContextEntry(ContractModel):
    """One immutable-in-meaning memory write presented to the answer layer."""

    entry_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_index: int = Field(ge=0)
    version_index: int = Field(ge=0)
    object_key: MemoryObjectKey
    value: JsonValue

    @model_validator(mode="after")
    def _reject_blank_ids(self) -> ContextEntry:
        if not self.entry_id.strip() or not self.event_id.strip():
            raise ValueError("context entry IDs must not be blank")
        if self.value is None:
            raise ValueError("context entries require a non-null value")
        return self


class RenderedContext(ContractModel):
    """Canonical text and audit projections for one mechanism condition."""

    rendered_context: str
    entry_ids: list[str]
    entry_order: list[int]
    labels: dict[str, str] = Field(default_factory=dict)
    context_order: ContextOrder
    context_annotation: ContextAnnotation

    @property
    def text(self) -> str:
        return self.rendered_context

    @property
    def order(self) -> list[int]:
        return self.entry_order

    def __iter__(self):
        yield self.rendered_context
        yield self.entry_ids
        yield self.entry_order
        yield self.labels


def _coerce_entry(value: ContextEntry | Mapping[str, Any]) -> ContextEntry:
    if isinstance(value, ContextEntry):
        return value
    if isinstance(value, Mapping):
        return ContextEntry.model_validate(value)
    raise TypeError("context entries must be ContextEntry or mapping values")


def _entry_label_map(entries: Sequence[ContextEntry]) -> dict[str, str]:
    latest_by_key: dict[str, tuple[int, int, str]] = {}
    for entry in entries:
        key_id = entry.object_key.canonical_id
        candidate = (entry.event_index, entry.version_index, entry.entry_id)
        previous = latest_by_key.get(key_id)
        if previous is None or candidate[:2] > previous[:2]:
            latest_by_key[key_id] = candidate
    return {
        entry.entry_id: (
            "latest" if latest_by_key[entry.object_key.canonical_id][2] == entry.entry_id else "outdated"
        )
        for entry in entries
    }


def _render_value(value: JsonValue) -> str:
    return json.dumps(thaw_json(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def render_context(
    entries: Sequence[ContextEntry | Mapping[str, Any]],
    order: str,
    annotation: str,
) -> RenderedContext:
    """Render one of the three approved paired context cells.

    Ordering is based only on canonical event/version metadata.  Labels are
    derived from the four-part object identity and the latest event position;
    answer text is never an input to this function.
    """
    if (order, annotation) not in _SUPPORTED:
        raise ValueError(f"unsupported mechanism context condition: {(order, annotation)!r}")
    normalized = [_coerce_entry(entry) for entry in entries]
    if not normalized:
        raise ValueError("mechanism context requires at least one entry")
    if len({entry.entry_id for entry in normalized}) != len(normalized):
        raise ValueError("duplicate context entry IDs are not allowed")
    if len({entry.event_id for entry in normalized}) != len(normalized):
        # Multiple writes in one event are valid, so only reject exact duplicate
        # entry/event/key triples while preserving one event's multi-object writes.
        triples = [(entry.event_id, entry.object_key.canonical_id, entry.version_index) for entry in normalized]
        if len(set(triples)) != len(triples):
            raise ValueError("duplicate context entries are not allowed")
    if order == "chronological":
        ordered = sorted(normalized, key=lambda entry: (entry.event_index, entry.version_index, entry.entry_id))
    else:
        ordered = sorted(normalized, key=lambda entry: (entry.event_index, entry.version_index, entry.entry_id), reverse=True)
    labels = _entry_label_map(normalized) if annotation == "latest_outdated_label" else {}
    lines = []
    for entry in ordered:
        label = f" [{labels[entry.entry_id]}]" if labels else ""
        lines.append(f"{entry.object_key.canonical_id} = {_render_value(entry.value)}{label}")
    return RenderedContext(
        rendered_context="\n".join(lines),
        entry_ids=[entry.entry_id for entry in ordered],
        entry_order=[entry.event_index for entry in ordered],
        labels={entry_id: labels[entry_id] for entry_id in (entry.entry_id for entry in ordered)} if labels else {},
        context_order=order,
        context_annotation=annotation,
    )


def entries_from_task(task: MemUpdateTask) -> tuple[ContextEntry, ...]:
    """Project canonical Family A task writes into auditable context entries."""
    if not isinstance(task, MemUpdateTask):
        raise TypeError("task must be a MemUpdateTask")
    events = {event.event_id: event for event in task.events}
    action_by_id = {action.action_id: action for action in task.gold.actions}
    entries: list[ContextEntry] = []
    versions: defaultdict[str, int] = defaultdict(int)
    for action_id in task.gold.action_sequence:
        action = action_by_id[action_id]
        if action.operation not in {Operation.ADD, Operation.UPDATE}:
            continue
        event = events.get(action.event_id)
        if event is None:
            raise ValueError(f"action {action.action_id} references missing event")
        if action.value is None:
            raise ValueError(f"action {action.action_id} has no context value")
        for key in action.target_object_keys:
            key_id = key.canonical_id
            version_index = versions[key_id]
            versions[key_id] += 1
            entries.append(
                ContextEntry(
                    entry_id=f"{event.event_id}:{key_id}:{version_index}",
                    event_id=event.event_id,
                    event_index=event.sequence_index,
                    version_index=version_index,
                    object_key=key,
                    value=action.value,
                )
            )
    if not entries:
        raise ValueError(f"task {task.task_id} has no ADD/UPDATE context entries")
    return tuple(entries)


ContextEntryRecord = ContextEntry
ContextRender = RenderedContext

__all__ = [
    "ContextEntry",
    "ContextEntryRecord",
    "ContextRender",
    "RenderedContext",
    "entries_from_task",
    "render_context",
]
