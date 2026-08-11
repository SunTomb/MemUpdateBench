from __future__ import annotations

from collections.abc import Mapping
import re
import unicodedata
from typing import Any

from pydantic import Field

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.v3.common import StrictIdentifier, StrictPositiveInt
from mub.vnext.contracts.v3.task import MemoryEventV3, MemoryQueryV3
from mub.vnext.io import canonical_json_bytes


_FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "action",
        "actions",
        "answer",
        "answers",
        "derivation",
        "evidence",
        "expected",
        "gold",
        "historical",
        "history",
        "selector",
        "stalealternative",
        "stratification",
        "targetobject",
        "versionhistory",
    }
)
_NONALNUM = re.compile(r"[^0-9a-z]+")


class ProviderEventInputV1(ImmutableContractModel):
    event_id: StrictIdentifier
    sequence_index: int = Field(strict=True, ge=0)
    logical_time: str | None = Field(default=None, strict=True)
    raw_text: str = Field(strict=True)
    runtime_namespace: StrictIdentifier


class ProviderQueryInputV1(ImmutableContractModel):
    query_id: StrictIdentifier
    query_text: str = Field(strict=True)
    k: StrictPositiveInt
    runtime_namespace: StrictIdentifier


def _compact_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _NONALNUM.sub("", normalized)


def _forbidden_key(value: str) -> bool:
    compact = _compact_key(value)
    return any(
        compact == token or compact.startswith(token)
        for token in _FORBIDDEN_KEY_TOKENS
    )


def validate_visible_payload(value: Any) -> Any:
    def visit(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if type(key) is not str:
                    raise ValueError(
                        "provider-visible payload keys must be exact strings"
                    )
                if _forbidden_key(key):
                    location = ".".join((*path, key))
                    raise ValueError(
                        "provider-visible payload contains a privileged key: "
                        f"{location}"
                    )
                visit(nested, (*path, key))
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, (*path, str(index)))

    visit(value, ())
    return value


def _revalidate_exact(model_type, value, label: str):
    if type(value) is not model_type:
        raise ValueError(f"{label} requires exact {model_type.__name__}")
    try:
        rebuilt = model_type.model_validate(
            {
                field_name: value.__dict__[field_name]
                for field_name in model_type.model_fields
            },
            strict=True,
        )
    except Exception as exc:
        raise ValueError(f"{label} fails trust-boundary validation") from exc
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(value):
        raise ValueError(f"{label} serialization is not stable")
    return rebuilt


def visible_event_input(
    event: MemoryEventV3,
    *,
    runtime_namespace: str,
) -> ProviderEventInputV1:
    event = _revalidate_exact(
        MemoryEventV3,
        event,
        "provider event input",
    )
    visible = ProviderEventInputV1(
        event_id=event.event_id,
        sequence_index=event.sequence_index,
        logical_time=event.timestamp,
        raw_text=event.raw_text,
        runtime_namespace=runtime_namespace,
    )
    validate_visible_payload(visible.model_dump(mode="python"))
    return visible


def visible_query_input(
    query: MemoryQueryV3,
    *,
    k: int,
    runtime_namespace: str,
) -> ProviderQueryInputV1:
    query = _revalidate_exact(
        MemoryQueryV3,
        query,
        "provider query input",
    )
    visible = ProviderQueryInputV1(
        query_id=query.query_id,
        query_text=query.text,
        k=k,
        runtime_namespace=runtime_namespace,
    )
    validate_visible_payload(visible.model_dump(mode="python"))
    return visible


__all__ = [
    "ProviderEventInputV1",
    "ProviderQueryInputV1",
    "validate_visible_payload",
    "visible_event_input",
    "visible_query_input",
]
