from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


LEGACY_NAMESPACES: Mapping[str, str] = MappingProxyType(
    {
        "p63": "legacy_p63",
        "p65": "legacy_p65",
        "p68_p70": "legacy_p68_p70",
        "p80_p82": "legacy_p80_p82",
        "p83": "legacy_p83",
        "p84": "legacy_p84",
        "p85_api_replacement": "legacy_p85_api_replacement",
    }
)

LEGACY_CAVEATS: Mapping[str, str] = MappingProxyType(
    {
        "p63_split_leakage": "P6.3 semantic cores overlap across historical train/dev/test splits; compatibility only.",
        "state_direct_oracle": "Legacy slot_direct is an oracle-like structured state readout and maps to state_direct only when exact slot state semantics match.",
        "retrieved_prompt_legacy": "Legacy slot_prompt is a prompted answer condition and maps to retrieved_prompt only when retrieved-context semantics match.",
        "latest_per_slot_rewrite": "latest_per_slot scans the full store and rewrites retrieval; it is not a pure deletion from original top-k.",
        "p83_order_metadata": "Stale same-slot conflict is order- and metadata-sensitive and is not universally the strongest distractor.",
        "p84_answer_layer_only": "P8.4 probes the answer layer and is not a full external-memory baseline.",
    }
)


def legacy_namespace(legacy_phase: str) -> str:
    """Resolve a documented phase without guessing from filenames."""
    normalized = legacy_phase.strip().lower().replace(".", "")
    try:
        return LEGACY_NAMESPACES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported legacy_phase {legacy_phase!r}; expected one of {sorted(LEGACY_NAMESPACES)}") from exc


__all__ = ["LEGACY_CAVEATS", "LEGACY_NAMESPACES", "legacy_namespace"]
