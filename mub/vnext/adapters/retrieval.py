from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from mub.vnext.contracts import MemoryEntryRecord, MemoryQuery, RetrievalResult


def _validate_inputs(entries: Iterable[MemoryEntryRecord], query: MemoryQuery, k: int) -> list[MemoryEntryRecord]:
    if not isinstance(query, MemoryQuery):
        raise TypeError("query must be a MemoryQuery")
    if type(k) is not int:
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be nonnegative")
    result = list(entries)
    if not all(isinstance(entry, MemoryEntryRecord) for entry in result):
        raise TypeError("entries must contain MemoryEntryRecord values")
    return result


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w%|:-]+", text.lower()) if token}


def _score(entry: MemoryEntryRecord, query: MemoryQuery) -> float:
    query_tokens = _tokens(query.text)
    content_tokens = _tokens(entry.content)
    overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
    target_ids = {key.canonical_id for key in query.target_object_keys}
    exact = 1.0 if entry.object_key_candidate and entry.object_key_candidate.canonical_id in target_ids else 0.0
    return exact * 2.0 + overlap


def _result(query: MemoryQuery, entries: list[MemoryEntryRecord], scores: list[float], policy: str, *, rewrite: bool, full_scan: bool) -> RetrievalResult:
    metadata = {
        "policy": policy,
        "retrieval_rewrite": rewrite,
        "not_original_topk_filter": rewrite,
        "full_store_scan": full_scan,
    }
    return RetrievalResult(
        query_id=query.query_id,
        entries=entries,
        scores=scores,
        raw_result={**metadata, "metadata": metadata},
    )


def normal_topk(entries: Iterable[MemoryEntryRecord], query: MemoryQuery, k: int) -> RetrievalResult:
    """Rank the store directly and return the original top-k filter result."""
    all_entries = _validate_inputs(entries, query, k)
    ranked = sorted(
        ((_score(entry, query), entry) for entry in all_entries),
        key=lambda item: (-item[0], item[1].entry_id),
    )[:k]
    return _result(
        query,
        [entry for _, entry in ranked],
        [float(score) for score, _ in ranked],
        "normal_topk",
        rewrite=False,
        full_scan=False,
    )


def _order_key(entry: MemoryEntryRecord) -> tuple[Any, ...]:
    metadata = entry.raw_metadata
    order = next(
        (
            metadata[name]
            for name in ("canonical_order", "order_index", "event_sequence_index", "sequence_index")
            if name in metadata
        ),
        -1,
    )
    if isinstance(order, bool):
        order = -1
    if isinstance(order, (int, float)):
        order_key = (0, float(order))
    else:
        order_key = (1, str(order))
    version = entry.version_index if entry.version_index is not None else -1
    return order_key, version, entry.updated_at or "", entry.created_at or "", entry.entry_id


def latest_per_object(entries: Iterable[MemoryEntryRecord], query: MemoryQuery, k: int) -> RetrievalResult:
    """Scan all entries, retain the canonical latest version per exact object key, then rank."""
    all_entries = _validate_inputs(entries, query, k)
    groups: dict[str, MemoryEntryRecord] = {}
    for entry in all_entries:
        if entry.object_key_candidate is None:
            # Entries without identity cannot be silently merged.
            groups[f"__entry__:{entry.entry_id}"] = entry
            continue
        object_id = entry.object_key_candidate.canonical_id
        previous = groups.get(object_id)
        if previous is None or _order_key(entry) > _order_key(previous):
            groups[object_id] = entry
    selected = list(groups.values())
    ranked = sorted(
        ((_score(entry, query), entry) for entry in selected),
        key=lambda item: (-item[0], item[1].entry_id),
    )[:k]
    return _result(
        query,
        [entry for _, entry in ranked],
        [float(score) for score, _ in ranked],
        "latest_per_object",
        rewrite=True,
        full_scan=True,
    )


POLICIES = {"normal_topk": normal_topk, "latest_per_object": latest_per_object}


def apply_retrieval_policy(policy: str, entries: Iterable[MemoryEntryRecord], query: MemoryQuery, k: int) -> RetrievalResult:
    try:
        fn = POLICIES[policy]
    except KeyError as exc:
        raise ValueError(f"unknown retrieval policy: {policy}") from exc
    return fn(entries, query, k)


__all__ = ["POLICIES", "apply_retrieval_policy", "latest_per_object", "normal_topk"]
