from __future__ import annotations

from typing import Any

from mub.vnext.adapters.exact_crud import ExactCrudAdapter
from mub.vnext.adapters.heuristic_crud import HeuristicCrudAdapter
from mub.vnext.adapters.raw_append import RawAppendAdapter
from mub.vnext.adapters.reference import BaseBuiltinAdapter, ReferenceAdapter

BUILTIN_ADAPTERS = {
    "reference": ReferenceAdapter,
    "raw_add": RawAppendAdapter,
    "exact_crud": ExactCrudAdapter,
    "heuristic_crud": HeuristicCrudAdapter,
}


def build_adapter(adapter_id: str, **kwargs: Any) -> BaseBuiltinAdapter:
    try:
        adapter_type = BUILTIN_ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ValueError(f"unknown builtin adapter: {adapter_id}") from exc
    return adapter_type(**kwargs)


get_adapter = build_adapter
ADAPTER_REGISTRY = BUILTIN_ADAPTERS
create_adapter = build_adapter

__all__ = [
    "BUILTIN_ADAPTERS",
    "ADAPTER_REGISTRY",
    "BaseBuiltinAdapter",
    "ExactCrudAdapter",
    "HeuristicCrudAdapter",
    "RawAppendAdapter",
    "ReferenceAdapter",
    "build_adapter",
    "create_adapter",
    "get_adapter",
]
