from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import RootModel

from mub.vnext.io import canonical_json_bytes


_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*$")


class _CanonicalPayload(RootModel[Any]):
    pass


def _canonical_payload(payload: object) -> bytes:
    return canonical_json_bytes(_CanonicalPayload(root=payload))


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _require_nonnegative_index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_variant(value: object, name: str) -> str | int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a string or nonnegative integer")
    if isinstance(value, int):
        return _require_nonnegative_index(value, name)
    return _require_nonempty_string(value, name)


def stable_id(prefix: str, payload: object) -> str:
    """Return a deterministic short identifier for canonical semantic material."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if _PREFIX_PATTERN.fullmatch(prefix) is None:
        raise ValueError("prefix must be a nonempty lowercase snake-case identifier")
    digest = hashlib.sha256(_canonical_payload(payload)).hexdigest()[:16]
    return f"{prefix}_{digest}"


def core_id(family: str, semantic_payload: object) -> str:
    return stable_id(
        "core",
        {
            "family": _require_nonempty_string(family, "family"),
            "semantic_payload": semantic_payload,
        },
    )


def task_id(semantic_core_id: str, surface_variant_index: int) -> str:
    return stable_id(
        "task",
        {
            "semantic_core_id": _require_nonempty_string(
                semantic_core_id, "semantic_core_id"
            ),
            "surface_variant_index": _require_nonnegative_index(
                surface_variant_index, "surface_variant_index"
            ),
        },
    )


def event_id(task_identifier: str, event_index: int) -> str:
    return stable_id(
        "event",
        {
            "task_id": _require_nonempty_string(task_identifier, "task_id"),
            "event_index": _require_nonnegative_index(event_index, "event_index"),
        },
    )


def action_id(
    task_identifier: str,
    event_index: int,
    action_index: int,
) -> str:
    return stable_id(
        "action",
        {
            "task_id": _require_nonempty_string(task_identifier, "task_id"),
            "event_index": _require_nonnegative_index(event_index, "event_index"),
            "action_index": _require_nonnegative_index(action_index, "action_index"),
        },
    )


def query_id(task_identifier: str, query_index: int) -> str:
    return stable_id(
        "query",
        {
            "task_id": _require_nonempty_string(task_identifier, "task_id"),
            "query_index": _require_nonnegative_index(query_index, "query_index"),
        },
    )


def source_id(namespace: str, source_index: int, semantic_payload: object) -> str:
    return stable_id(
        "source",
        {
            "namespace": _require_nonempty_string(namespace, "namespace"),
            "source_index": _require_nonnegative_index(source_index, "source_index"),
            "semantic_payload": semantic_payload,
        },
    )


def trajectory_id(semantic_core_id: str, trajectory_variant: str | int) -> str:
    return stable_id(
        "trajectory",
        {
            "semantic_core_id": _require_nonempty_string(
                semantic_core_id, "semantic_core_id"
            ),
            "trajectory_variant": _require_variant(
                trajectory_variant, "trajectory_variant"
            ),
        },
    )


def paraphrase_group_id(
    semantic_core_id: str,
    paraphrase_variant: str | int,
) -> str:
    return stable_id(
        "paraphrase_group",
        {
            "semantic_core_id": _require_nonempty_string(
                semantic_core_id, "semantic_core_id"
            ),
            "paraphrase_variant": _require_variant(
                paraphrase_variant, "paraphrase_variant"
            ),
        },
    )


__all__ = [
    "action_id",
    "core_id",
    "event_id",
    "paraphrase_group_id",
    "query_id",
    "source_id",
    "stable_id",
    "task_id",
    "trajectory_id",
]
