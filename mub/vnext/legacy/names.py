from __future__ import annotations

import re


_DOCUMENTED_RUN_NAMES = re.compile(
    r"^(?P<mode>raw_add|long25|oracle)_(?P<answer_mode>slot_prompt|slot_direct)_k(?P<update_depth>1|2|4|8|16)$"
)


def parse_legacy_run_name(
    name: str,
) -> dict[str, str | int | tuple[str, ...]] | None:
    """Parse only documented legacy fallback directory names.

    The result is legacy identity, not a canonical evaluation-mode mapping.
    """

    if type(name) is not str:
        raise TypeError("legacy run name must be an exact built-in string")
    match = _DOCUMENTED_RUN_NAMES.fullmatch(name)
    if match is None:
        return None
    mode = match.group("mode")
    answer_mode = match.group("answer_mode")
    if mode == "oracle" and answer_mode != "slot_direct":
        return None
    return {
        "mode": mode,
        "answer_mode": answer_mode,
        "update_depth": int(match.group("update_depth")),
        "warnings": ("legacy_directory_name_inference",),
    }


__all__ = ["parse_legacy_run_name"]
