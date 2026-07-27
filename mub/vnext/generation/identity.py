from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from pydantic import RootModel

from mub.vnext.io import canonical_json_bytes


_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*$")


class _CanonicalPayload(RootModel[Any]):
    pass


def _validate_strict_json(
    value: object,
    path: str = "$",
    active_containers: set[int] | None = None,
) -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"strict JSON numbers must be finite at {path}")
        return
    if value_type not in (dict, list):
        raise TypeError(
            f"strict JSON requires exact built-in scalar/container types at {path}; "
            f"got {value_type.__name__}"
        )

    if active_containers is None:
        active_containers = set()
    container_id = id(value)
    if container_id in active_containers:
        raise ValueError(f"strict JSON cannot contain container cycles at {path}")
    active_containers.add(container_id)
    try:
        if value_type is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        f"strict JSON mapping keys must be exact strings at {path}"
                    )
                _validate_strict_json(item, f"{path}.{key}", active_containers)
        else:
            for index, item in enumerate(value):
                _validate_strict_json(item, f"{path}[{index}]", active_containers)
    finally:
        active_containers.remove(container_id)


def _canonical_payload(payload: object) -> bytes:
    _validate_strict_json(payload)
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
