from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final
import unicodedata

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.external.contracts import ExternalCandidateId

CANDIDATE_LABELS: Final[Mapping[ExternalCandidateId, str]] = MappingProxyType(
    {
        ExternalCandidateId.MEM0_OSS: "Mem0 OSS",
        ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE: (
            "LangGraph Store extract-then-store"
        ),
    }
)

DENIED_EXTERNAL_EVIDENCE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "memory_r1",
        "mem0_memory_r1",
        "baselines/memory_r1_agent.py",
        "scripts/eval_mem0_baseline.py",
    }
)
_DENIED_PATH_MARKERS = (
    "local_approximation",
    "memory_r1",
    "scripts/eval_mem0_baseline.py",
)
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
    | {f"com{index}" for index in "¹²³"}
    | {f"lpt{index}" for index in "¹²³"}
)


_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)


def _is_default_ignorable_code_point(character: str) -> bool:
    code_point = ord(character)
    return any(
        lower <= code_point <= upper
        for lower, upper in _DEFAULT_IGNORABLE_RANGES
    )


def _validate_portable_path(path: str) -> str:
    if type(path) is not str or not path:
        raise ValueError("artifact path must be a portable canonical relative path")
    if path != unicodedata.normalize("NFC", path):
        raise ValueError("artifact path must be a portable canonical relative path")
    if "\\" in path or path.startswith("/"):
        raise ValueError("artifact path must be a portable canonical relative path")
    components = path.split("/")
    if any(
        component in {"", ".", ".."}
        or component.endswith((".", " "))
        or component.strip() in {".", ".."}
        or component.split(".", 1)[0].rstrip(". ").casefold()
        in _WINDOWS_RESERVED_BASENAMES
        or ":" in component
        or "~" in component
        or any(
            ord(character) < 32
            or ord(character) == 127
            or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            or _is_default_ignorable_code_point(character)
            or 0xD800 <= ord(character) <= 0xDFFF
            or character in '<>"|?*'
            for character in component
        )
        for component in components
    ):
        raise ValueError("artifact path must be a portable canonical relative path")
    return path


def _canonical_component_token(component: str) -> str:
    normalized = unicodedata.normalize("NFKC", component).casefold()
    token: list[str] = []
    separator_pending = False
    for character in normalized:
        category = unicodedata.category(character)
        if (
            character == "_"
            or character.isspace()
            or category[0] in {"P", "Z"}
        ):
            separator_pending = True
            continue
        if separator_pending:
            token.append("_")
        token.append(character)
        separator_pending = False
    if separator_pending:
        token.append("_")
    return "".join(token).strip("_")


def _component_canonical_tokens(component: str) -> frozenset[str]:
    current = unicodedata.normalize("NFKC", component).casefold()
    tokens: set[str] = set()
    while True:
        tokens.add(_canonical_component_token(current))
        if "." not in current:
            break
        current = current.rsplit(".", 1)[0]
    return frozenset(tokens)


def _matches_denied_path_components(
    candidate_components: list[str],
    denied_path: str,
) -> bool:
    denied_components = (
        unicodedata.normalize("NFKC", denied_path).casefold().split("/")
    )
    if len(candidate_components) < len(denied_components):
        return False
    denied_final = _canonical_component_token(denied_components[-1])
    window_size = len(denied_components)
    for start in range(len(candidate_components) - window_size + 1):
        candidate_window = candidate_components[start : start + window_size]
        if any(
            _canonical_component_token(candidate)
            != _canonical_component_token(denied)
            for candidate, denied in zip(
                candidate_window[:-1],
                denied_components[:-1],
            )
        ):
            continue
        if denied_final in _component_canonical_tokens(candidate_window[-1]):
            return True
    return False


def _is_denied_evidence_path(label: str) -> bool:
    normalized_label = (
        unicodedata.normalize("NFKC", label).casefold().replace("\\", "/")
    )
    components = normalized_label.split("/")
    denied_labels = DENIED_EXTERNAL_EVIDENCE_LABELS | frozenset(
        _DENIED_PATH_MARKERS
    )
    for denied_label in denied_labels:
        if "/" in denied_label:
            if _matches_denied_path_components(components, denied_label):
                return True
            continue
        marker_token = _canonical_component_token(denied_label)
        if any(
            marker_token in _component_canonical_tokens(component)
            for component in components
        ):
            return True
    return False


def reject_denied_evidence(labels: Iterable[str]) -> tuple[str, ...]:
    if isinstance(labels, str):
        raise ValueError(
            "evidence labels must be supplied as a collection, not a scalar string"
        )
    try:
        checked = tuple(labels)
    except TypeError as exc:
        raise ValueError("evidence labels must be supplied as a collection") from exc
    for label in checked:
        if type(label) is not str or not label.strip():
            raise ValueError("evidence labels must be nonblank exact built-in strings")
        if label in DENIED_EXTERNAL_EVIDENCE_LABELS or _is_denied_evidence_path(label):
            raise ValueError(f"denied prior-system evidence: {label}")
        _validate_portable_path(label)
    return checked


def validate_artifact_provenance(ref: ArtifactRef) -> ArtifactRef:
    if type(ref) is not ArtifactRef:
        raise ValueError("artifact provenance requires an exact ArtifactRef")
    try:
        payload = {
            field_name: ref.__dict__[field_name]
            for field_name in ArtifactRef.model_fields
        }
    except (AttributeError, KeyError) as exc:
        raise ValueError("ArtifactRef stored fields are incomplete") from exc
    validated_ref = ArtifactRef.model_validate(payload, strict=True)
    path = _validate_portable_path(validated_ref.path)
    reject_denied_evidence((path,))
    return validated_ref


def resolve_candidate_id(label: str | ExternalCandidateId) -> ExternalCandidateId:
    if isinstance(label, ExternalCandidateId):
        return label
    if type(label) is not str:
        raise ValueError("external candidate labels must be exact built-in strings")
    if label in DENIED_EXTERNAL_EVIDENCE_LABELS or _is_denied_evidence_path(label):
        raise ValueError(f"denied prior-system evidence: {label}")
    try:
        return ExternalCandidateId(label)
    except ValueError as exc:
        raise ValueError(f"unknown external candidate: {label}") from exc


__all__ = [
    "CANDIDATE_LABELS",
    "DENIED_EXTERNAL_EVIDENCE_LABELS",
    "reject_denied_evidence",
    "resolve_candidate_id",
    "validate_artifact_provenance",
]
