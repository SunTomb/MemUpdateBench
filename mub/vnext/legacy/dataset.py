from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from mub.vnext.contracts.common import GeneratorProvenance, MemoryObjectKey, SourceRecord
from mub.vnext.contracts.enums import (
    ActionScope, AnswerSchema, Difficulty, EvaluationMode, EventRole, Operation,
    QueryType, SourceType, Split, TaskFamily,
)
from mub.vnext.contracts.task import (
    GoldAction, GoldRecord, LegacyProvenance, MemUpdateTask, MemoryEvent,
    MemoryQuery, SplitKey, TaskMetadata,
)
from mub.vnext.legacy.caveats import LEGACY_CAVEATS, legacy_namespace
from mub.vnext.validation.replay import validate_distractors, validate_gold_replay
from mub.vnext.validation.task import validate_task
from mub.vnext.version import COMPILER_VERSION, SCHEMA_VERSION

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NO_RANDOMNESS_SEED = 0  # Convention: zero denotes deterministic compilation with no RNG.
_COMPILER_CODE_REVISION = "legacy-compatibility-import"
_CONFIG = {
    "compiler": "legacy_p63_episode_compiler",
    "compiler_version": COMPILER_VERSION,
    "schema_version": SCHEMA_VERSION,
    "parser": "scripts.eval_evomemory.parse_event_slot",
    "semantic_identity_version": "semantic-core-v4",
    "canonical_id_version": "legacy-independent-v2",
    "source_normalization_version": "semantic-source-v1",
    "randomness": "none",
    "seed_convention": "zero_means_no_rng",
}
_CONFIG_HASH = hashlib.sha256(json.dumps(_CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
_REQUIRED = ("events", "question", "answer", "entity", "attribute", "latest_event_idx", "num_events", "num_target_updates", "num_updates")
_EVENT_FIELDS = frozenset({"text", "raw_text", "timestamp", "speaker", "source_anchor", "metadata"})
_RAW_ANCHOR_FIELDS = frozenset({"start_char", "end_char"})
_EVENT_METADATA_FIELDS = frozenset({"annotation", "condition"})
_NON_TARGET_ROLES = frozenset({
    EventRole.SAME_ENTITY_OTHER_ATTRIBUTE,
    EventRole.SAME_NAME_OTHER_ENTITY,
    EventRole.NOOP_NEAR_MISS,
    EventRole.NEUTRAL,
})
_MISSING = object()
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 10_000
_SUPPORTED_EPISODE_NAMESPACES = frozenset({"legacy_p63", "legacy_p65", "legacy_p68_p70"})


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _exact_unicode_string(
    value: Any,
    field: str,
    source_path: Path,
    example_index: int,
    *,
    nonblank: bool = True,
) -> str:
    if type(value) is not str:
        _fail(source_path, example_index, field, "must be an exact built-in string")
    if nonblank and not value.strip():
        _fail(source_path, example_index, field, "must be a non-blank string")
    if _contains_surrogate(value):
        _fail(source_path, example_index, field, "must contain only Unicode scalar values")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _json_bytes(value)).hexdigest()


def _context(source_path: Path, example_index: int, field: str) -> str:
    return f"{source_path} example_index={example_index} field={field}"


def _fail(source_path: Path, example_index: int, field: str, message: str) -> None:
    raise ValueError(f"{_context(source_path, example_index, field)}: {message}")


def _require(episode: Mapping[str, Any], field: str, source_path: Path, example_index: int) -> Any:
    if field not in episode:
        _fail(source_path, example_index, field, "required field is missing")
    value = episode[field]
    if value is None:
        _fail(source_path, example_index, field, "must not be null")
    return value


def _strict_int(value: Any, field: str, source_path: Path, example_index: int) -> int:
    if type(value) is not int or value < 0:
        _fail(source_path, example_index, field, "must be a non-negative integer")
    return value


def _strict_json_copy(value: Any, field: str, source_path: Path, example_index: int) -> Any:
    active: set[int] = set()
    seen_containers: set[int] = set()
    node_count = 0

    def visit(item: Any, path: str, depth: int) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_JSON_NODES:
            _fail(
                source_path,
                example_index,
                path,
                f"JSON node budget {_MAX_JSON_NODES} exceeded",
            )
        if depth > _MAX_JSON_DEPTH:
            _fail(source_path, example_index, path, f"JSON nesting exceeds maximum depth {_MAX_JSON_DEPTH}")
        if item is None or type(item) in {bool, int}:
            return item
        if type(item) is str:
            return _exact_unicode_string(
                item, path, source_path, example_index, nonblank=False
            )
        if type(item) is float:
            if not math.isfinite(item):
                _fail(source_path, example_index, path, "must contain only finite JSON numbers")
            return item
        if type(item) in {list, dict}:
            identity = id(item)
            if identity in active:
                _fail(source_path, example_index, path, "active JSON recursion cycle is not allowed")
            if identity in seen_containers:
                _fail(
                    source_path,
                    example_index,
                    path,
                    "repeated shared JSON container identity is not allowed",
                )
            seen_containers.add(identity)
            active.add(identity)
            try:
                if type(item) is list:
                    return [
                        visit(child, f"{path}[{index}]", depth + 1)
                        for index, child in enumerate(item)
                    ]
                result: dict[str, Any] = {}
                for key, child in item.items():
                    key = _exact_unicode_string(
                        key, path, source_path, example_index, nonblank=False
                    )
                    result[key] = visit(child, f"{path}.{key}", depth + 1)
                return result
            finally:
                active.remove(identity)
        _fail(source_path, example_index, path, f"unsupported JSON value type {type(item).__name__}")

    return visit(value, field, 0)


def _mapping_with_string_keys(value: Any, field: str, source_path: Path, example_index: int) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(source_path, example_index, field, "must be an exact built-in JSON object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        key = _exact_unicode_string(
            key, field, source_path, example_index, nonblank=False
        )
        result[key] = item
    return result


def _normalize_raw_anchor(value: Any, event_index: int, source_path: Path, example_index: int) -> dict[str, Any]:
    field = f"events[{event_index}].source_anchor"
    anchor = _mapping_with_string_keys(value, field, source_path, example_index)
    unknown = sorted(set(anchor) - _RAW_ANCHOR_FIELDS)
    if unknown:
        _fail(source_path, example_index, field, f"unsupported or compiler-owned fields: {unknown}")
    has_start, has_end = "start_char" in anchor, "end_char" in anchor
    if has_start != has_end:
        _fail(source_path, example_index, field, "start_char and end_char must be supplied together")
    normalized: dict[str, Any] = {}
    if has_start:
        start, end = anchor["start_char"], anchor["end_char"]
        if type(start) is not int or type(end) is not int or start < 0 or end < 0 or start > end:
            _fail(source_path, example_index, field, "character span must be ordered non-negative integers")
        normalized.update(start_char=start, end_char=end)
    return _strict_json_copy(normalized, field, source_path, example_index)


def _normalize_event_metadata(value: Any, event_index: int, source_path: Path, example_index: int) -> dict[str, Any]:
    field = f"events[{event_index}].metadata"
    metadata = _mapping_with_string_keys(value, field, source_path, example_index)
    unknown = sorted(set(metadata) - _EVENT_METADATA_FIELDS)
    if unknown:
        _fail(source_path, example_index, field, f"unsupported or sensitive fields: {unknown}")
    normalized = _strict_json_copy(metadata, field, source_path, example_index)
    for name, item in normalized.items():
        if item is not None and type(item) not in {str, bool, int, float}:
            _fail(source_path, example_index, f"{field}.{name}", "benign event metadata must be scalar")
    return normalized


def _event_parts(value: Any, index: int, source_path: Path, example_index: int) -> tuple[str, str | None, str | None, dict[str, Any], dict[str, Any]]:
    field = f"events[{index}]"
    if type(value) is str:
        raw = _exact_unicode_string(value, field, source_path, example_index)
        return raw, None, None, {}, {}
    record = _mapping_with_string_keys(value, field, source_path, example_index)
    unknown = sorted(set(record) - _EVENT_FIELDS)
    if unknown:
        _fail(source_path, example_index, field, f"unsupported fields: {unknown}")
    if "text" in record and "raw_text" in record and record["text"] != record["raw_text"]:
        _fail(source_path, example_index, field, "text and raw_text conflict")
    raw = _exact_unicode_string(
        record.get("raw_text", record.get("text")),
        f"{field}.raw_text",
        source_path,
        example_index,
    )
    timestamp, speaker = record.get("timestamp"), record.get("speaker")
    if timestamp is not None:
        timestamp = _exact_unicode_string(
            timestamp, f"{field}.timestamp", source_path, example_index
        )
    if speaker is not None:
        speaker = _exact_unicode_string(
            speaker, f"{field}.speaker", source_path, example_index
        )
    return raw, timestamp, speaker, _normalize_raw_anchor(record.get("source_anchor", {}), index, source_path, example_index), _normalize_event_metadata(record.get("metadata", {}), index, source_path, example_index)


def _validate_parsed_slot(parsed: Any, event_index: int, source_path: Path, example_index: int) -> dict[str, Any] | None:
    field = f"events[{event_index}].parsed_slot"
    if parsed is None:
        return None
    record = _mapping_with_string_keys(parsed, field, source_path, example_index)
    if set(record) != {"entity", "attribute", "value", "event_idx"}:
        _fail(source_path, example_index, field, "parser result must have exactly entity, attribute, value, event_idx")
    for name in ("entity", "attribute"):
        record[name] = _exact_unicode_string(
            record[name], f"{field}.{name}", source_path, example_index
        ).strip()
    if type(record["event_idx"]) is not int or record["event_idx"] != event_index:
        _fail(source_path, example_index, f"{field}.event_idx", "must equal the current event index")
    record["value"] = _strict_json_copy(record["value"], f"{field}.value", source_path, example_index)
    return record


def _parse_legacy_slot(parser: Any, resolver: Any, text: str, event_index: int, source_path: Path, example_index: int) -> dict[str, Any] | None:
    try:
        parsed = parser(text, event_index, resolver=resolver)
    except Exception as exc:
        raise ValueError(f"{_context(source_path, example_index, f'events[{event_index}].parsed_slot')}: legacy parser failed: {type(exc).__name__}: {exc}") from exc
    return _validate_parsed_slot(parsed, event_index, source_path, example_index)


def _validate_role_metadata(marker: Mapping[str, Any] | None, *, marker_name: str, event_index: int, parsed: dict[str, Any] | None, raw_text: str, target_entity: str, target_attribute: str, target_evidence_texts: tuple[str, ...], source_path: Path, example_index: int) -> None:
    if marker is None:
        return
    record = _mapping_with_string_keys(marker, marker_name, source_path, example_index)
    unknown = sorted(set(record) - {"event_idx", "entity", "attribute", "surface_name"})
    if unknown:
        _fail(source_path, example_index, marker_name, f"unsupported fields: {unknown}")
    for field in ("entity", "attribute", "surface_name"):
        if field in record:
            record[field] = _exact_unicode_string(
                record[field], f"{marker_name}.{field}", source_path, example_index
            )
    if type(record.get("event_idx")) is not int or record["event_idx"] != event_index:
        _fail(source_path, example_index, f"{marker_name}.event_idx", "does not match the marked event")
    marked_entity, marked_attribute = record.get("entity"), record.get("attribute")
    if marker_name == "semantic_near_miss" and (marked_entity != target_entity or marked_attribute != target_attribute):
        _fail(source_path, example_index, marker_name, "must identify the exact target slot")
    if parsed is not None:
        if marked_entity is not None and parsed["entity"] != marked_entity:
            _fail(source_path, example_index, marker_name, "entity metadata conflicts with parsed identity")
        if marked_attribute is not None and parsed["attribute"] != marked_attribute:
            _fail(source_path, example_index, marker_name, "attribute metadata conflicts with parsed identity")
    elif marker_name != "semantic_near_miss":
        _fail(source_path, example_index, marker_name, "event identity is unresolvable")
    if marker_name == "same_name_distractor" and "surface_name" not in record:
        _fail(
            source_path,
            example_index,
            marker_name,
            "surface_name is required for shared-name evidence",
        )
    surface_name = record.get("surface_name")
    if surface_name is not None and surface_name.casefold() not in raw_text.casefold():
        _fail(source_path, example_index, marker_name, "surface_name is absent from event text")
    if (
        marker_name == "same_name_distractor"
        and surface_name is not None
        and not any(_text_contains_scalar(text, surface_name) for text in target_evidence_texts)
    ):
        _fail(
            source_path,
            example_index,
            marker_name,
            "surface_name is not shared with target question or exact-target events",
        )


def _canonical_enum(value: Any, enum_type: Any, field: str, source_path: Path, example_index: int) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        _fail(source_path, example_index, field, f"unsupported value {value!r}: {exc}")


def _resolve_answer_mode(raw: Mapping[str, Any], source_path: Path, example_index: int) -> EvaluationMode:
    if "answer_mode" in raw:
        _fail(source_path, example_index, "answer_mode", "run-specific legacy answer_mode belongs to result import, not task compilation")
    if "evaluation_mode" not in raw:
        return EvaluationMode.STATE_DIRECT
    explicit = _exact_unicode_string(
        raw["evaluation_mode"], "evaluation_mode", source_path, example_index
    )
    if explicit != EvaluationMode.STATE_DIRECT.value:
        _fail(source_path, example_index, "evaluation_mode", "Task11 P6 tasks support only state_direct")
    return EvaluationMode.STATE_DIRECT


def _text_contains_scalar(text: str, value: Any) -> bool:
    if isinstance(value, bool):
        form = "true" if value else "false"
    else:
        form = str(value)
    if form and all(ord(character) < 128 for character in form) and any(character.isalnum() for character in form):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None
    return form.casefold() in text.casefold()


def _legacy_analysis_fields(
    raw: Mapping[str, Any], source_path: Path, example_index: int
) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    if "k_updates" in raw:
        analysis["k_updates"] = _strict_int(
            raw["k_updates"], "k_updates", source_path, example_index
        )
    for field in ("distractor_level", "noop_level"):
        if field in raw:
            analysis[field] = _exact_unicode_string(
                raw[field], field, source_path, example_index
            )
    if "explicit_zero" in raw:
        if type(raw["explicit_zero"]) is not int or raw["explicit_zero"] != 0:
            _fail(source_path, example_index, "explicit_zero", "must be exact integer 0")
        analysis["explicit_zero"] = 0
    if "explicit_false" in raw:
        if type(raw["explicit_false"]) is not bool or raw["explicit_false"] is not False:
            _fail(source_path, example_index, "explicit_false", "must be exact boolean false")
        analysis["explicit_false"] = False
    if "explicit_null" in raw:
        if raw["explicit_null"] is not None:
            _fail(source_path, example_index, "explicit_null", "must be exact null")
        analysis["explicit_null"] = None
    return analysis


def _semantic_object_projection(key: MemoryObjectKey) -> dict[str, Any]:
    return {"namespace": key.namespace, "entity": key.entity, "attribute": key.attribute, "subkey": key.subkey}


def _build_anchor(semantic_document_id: str, event_index: int, raw_anchor: Mapping[str, Any]) -> dict[str, Any]:
    anchor = {"document_id": semantic_document_id, "section_id": "events", "paragraph": event_index}
    anchor.update(raw_anchor)
    return anchor


def _report_failure(report: Any, label: str, source_path: Path, example_index: int) -> None:
    if report.valid:
        return
    details = "; ".join(f"{issue.code} at {issue.path}: {issue.message}" for issue in report.issues)
    _fail(source_path, example_index, label, details)


def compile_legacy_episode(episode: dict[str, Any], *, source_path: Path, source_sha256: str, split: Split, example_index: int, legacy_phase: str) -> MemUpdateTask:
    if not isinstance(source_path, Path):
        raise TypeError("source_path must be a pathlib.Path")
    if type(episode) is not dict:
        _fail(source_path, example_index, "episode", "must be an exact built-in JSON object")
    if type(example_index) is not int or example_index < 0:
        raise ValueError(f"{source_path} example_index={example_index}: example_index must be non-negative")
    if not isinstance(split, Split):
        _fail(source_path, example_index, "split", "unsupported split")
    legacy_phase = _exact_unicode_string(
        legacy_phase, "legacy_phase", source_path, example_index
    )
    try:
        namespace = legacy_namespace(legacy_phase)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(source_path, example_index, "legacy_phase", str(exc))
    if namespace not in _SUPPORTED_EPISODE_NAMESPACES:
        _fail(
            source_path,
            example_index,
            "legacy_phase",
            "phase is not compatible with the P6 episode dataset compiler",
        )
    if type(source_sha256) is not str or _SHA256_RE.fullmatch(source_sha256) is None:
        _fail(source_path, example_index, "source_sha256", "must be exact lowercase 64-hex SHA-256")
    raw = dict(episode)
    for field in _REQUIRED:
        _require(raw, field, source_path, example_index)
    if "source_type" in raw:
        _fail(source_path, example_index, "source_type", "legacy source_type is not accepted")
    episode_id = raw.get("episode_id")
    if "episode_id" in raw:
        episode_id = _exact_unicode_string(
            episode_id, "episode_id", source_path, example_index
        )
    record_anchor = episode_id if episode_id is not None else "record_" + _digest("legacy-record-anchor-v1", {"source_sha256": source_sha256, "example_index": example_index})
    entity = _exact_unicode_string(raw["entity"], "entity", source_path, example_index).strip()
    attribute = _exact_unicode_string(raw["attribute"], "attribute", source_path, example_index).strip()
    question = _exact_unicode_string(raw["question"], "question", source_path, example_index)
    category = (
        _exact_unicode_string(raw["category"], "category", source_path, example_index)
        if "category" in raw
        else "p63"
    )
    stress_type = (
        _exact_unicode_string(raw["stress_type"], "stress_type", source_path, example_index)
        if "stress_type" in raw
        else None
    )
    answer = _strict_json_copy(raw["answer"], "answer", source_path, example_index)
    if answer is None or type(answer) not in {str, bool, int, float}:
        _fail(source_path, example_index, "answer", "must be a finite scalar JSON answer")
    events_raw = raw["events"]
    if type(events_raw) is not list or not events_raw:
        _fail(source_path, example_index, "events", "must be a non-empty exact built-in JSON array")
    num_events = _strict_int(raw["num_events"], "num_events", source_path, example_index)
    num_target_updates = _strict_int(raw["num_target_updates"], "num_target_updates", source_path, example_index)
    num_updates = _strict_int(raw["num_updates"], "num_updates", source_path, example_index)
    if num_events != len(events_raw):
        _fail(source_path, example_index, "num_events", f"declared {num_events}, observed {len(events_raw)}")
    if num_updates != num_events:
        _fail(source_path, example_index, "num_updates", "P6.3 num_updates must equal num_events")
    if "k_updates" in raw:
        k_updates = _strict_int(raw["k_updates"], "k_updates", source_path, example_index)
        if k_updates != num_target_updates:
            _fail(source_path, example_index, "k_updates", "must equal num_target_updates")
    latest_index = _strict_int(raw["latest_event_idx"], "latest_event_idx", source_path, example_index)
    if latest_index >= num_events:
        _fail(source_path, example_index, "latest_event_idx", "must index an event")
    near_miss, distractor = raw.get("semantic_near_miss"), raw.get("same_name_distractor")
    near_miss_index = distractor_index = None
    if near_miss is not None:
        near_miss_index = _strict_int(_mapping_with_string_keys(near_miss, "semantic_near_miss", source_path, example_index).get("event_idx"), "semantic_near_miss.event_idx", source_path, example_index)
    if distractor is not None:
        distractor_index = _strict_int(_mapping_with_string_keys(distractor, "same_name_distractor", source_path, example_index).get("event_idx"), "same_name_distractor.event_idx", source_path, example_index)
    if near_miss_index is not None and distractor_index == near_miss_index:
        _fail(source_path, example_index, "role_metadata", "semantic_near_miss and same_name_distractor indices overlap")
    for field, index in (("semantic_near_miss.event_idx", near_miss_index), ("same_name_distractor.event_idx", distractor_index)):
        if index is not None and index >= num_events:
            _fail(source_path, example_index, field, "must index an event")
    try:
        from scripts.eval_evomemory import EpisodeEntityResolver, parse_event_slot
        resolver = EpisodeEntityResolver()
    except Exception as exc:
        raise ValueError(f"{_context(source_path, example_index, 'events')}: legacy parser dependency unavailable") from exc
    event_parts: list[tuple[str, str | None, str | None, dict[str, Any], dict[str, Any]]] = []
    parsed_rows: list[dict[str, Any] | None] = []
    for index, item in enumerate(events_raw):
        parts = _event_parts(item, index, source_path, example_index)
        event_parts.append(parts)
        parsed = _parse_legacy_slot(parse_event_slot, resolver, parts[0], index, source_path, example_index)
        parsed_rows.append(parsed)
    target_evidence_texts = tuple(
        [question]
        + [
            event_parts[index][0]
            for index, parsed in enumerate(parsed_rows)
            if parsed is not None
            and (parsed["entity"], parsed["attribute"]) == (entity, attribute)
        ]
    )
    if near_miss_index is not None:
        _validate_role_metadata(
            near_miss,
            marker_name="semantic_near_miss",
            event_index=near_miss_index,
            parsed=parsed_rows[near_miss_index],
            raw_text=event_parts[near_miss_index][0],
            target_entity=entity,
            target_attribute=attribute,
            target_evidence_texts=target_evidence_texts,
            source_path=source_path,
            example_index=example_index,
        )
    if distractor_index is not None:
        _validate_role_metadata(
            distractor,
            marker_name="same_name_distractor",
            event_index=distractor_index,
            parsed=parsed_rows[distractor_index],
            raw_text=event_parts[distractor_index][0],
            target_entity=entity,
            target_attribute=attribute,
            target_evidence_texts=target_evidence_texts,
            source_path=source_path,
            example_index=example_index,
        )
    target_indices = [index for index, parsed in enumerate(parsed_rows) if parsed is not None and (parsed["entity"], parsed["attribute"]) == (entity, attribute) and index not in {near_miss_index, distractor_index}]
    target_index_set = frozenset(target_indices)
    if not target_indices:
        _fail(source_path, example_index, "latest_event_idx", "no exact target events were resolved")
    if latest_index != max(target_indices):
        _fail(source_path, example_index, "latest_event_idx", f"must equal terminal target event index {max(target_indices)}")
    latest_parsed = parsed_rows[latest_index]
    if latest_parsed is None or latest_parsed["value"] != answer:
        _fail(source_path, example_index, "answer", "does not match the parsed latest target event")
    if len(target_indices) != num_target_updates:
        _fail(source_path, example_index, "num_target_updates", f"declared {num_target_updates}, observed {len(target_indices)}")
    roles: list[EventRole] = []
    target_values: list[Any] = []
    current_value: Any = _MISSING
    for index, parsed in enumerate(parsed_rows):
        if index == near_miss_index:
            if parsed is not None and (parsed["entity"], parsed["attribute"]) == (entity, attribute):
                _fail(source_path, example_index, f"events[{index}]", "semantic near-miss conflicts with exact target parse")
            role = EventRole.NOOP_NEAR_MISS
        elif index == distractor_index:
            if parsed is None:
                _fail(source_path, example_index, "same_name_distractor", "event identity is unresolvable")
            if parsed["entity"] == entity:
                _fail(source_path, example_index, "same_name_distractor", "must identify a different entity")
            role = EventRole.SAME_NAME_OTHER_ENTITY
        elif index in target_index_set:
            value = parsed["value"]
            if index == latest_index:
                role = EventRole.LATEST_GOLD
            elif current_value is not _MISSING and value == current_value:
                role = EventRole.DUPLICATE_CURRENT
            elif value == answer:
                role = EventRole.HISTORICAL_SUPPORT
            else:
                role = EventRole.STALE_SAME_SLOT
            current_value = deepcopy(value)
            target_values.append(value)
        elif parsed is not None and parsed["entity"] == entity:
            role = EventRole.SAME_ENTITY_OTHER_ATTRIBUTE
        else:
            role = EventRole.NEUTRAL
        roles.append(role)
    key = MemoryObjectKey(namespace="default", entity=entity, attribute=attribute, subkey=None, object_type="slot")
    evaluation_mode = _resolve_answer_mode(raw, source_path, example_index)
    if raw.get("query_type", QueryType.CURRENT_STATE.value) != QueryType.CURRENT_STATE.value:
        _fail(source_path, example_index, "query_type", "Task11 P6 tasks support only current_state")
    query_type = QueryType.CURRENT_STATE
    inferred_schema = AnswerSchema.STRING.value if isinstance(answer, str) else AnswerSchema.BOOLEAN.value if isinstance(answer, bool) else AnswerSchema.NUMBER.value
    answer_schema = _canonical_enum(raw.get("answer_schema", inferred_schema), AnswerSchema, "answer_schema", source_path, example_index)
    final_state, version_history = {key.canonical_id: answer}, {key.canonical_id: target_values}
    semantic_document_id = "document_" + _digest("semantic-document-v1", _semantic_object_projection(key))
    anchors = [_build_anchor(semantic_document_id, index, parts[3]) for index, parts in enumerate(event_parts)]
    action_specs: list[dict[str, Any]] = []
    mutation_seen = 0
    for index, (parsed, _, parts) in enumerate(zip(parsed_rows, roles, event_parts, strict=True)):
        if index in target_index_set:
            operation = Operation.ADD if mutation_seen == 0 else Operation.UPDATE
            value = parsed["value"]
            action_specs.append({"operation": operation.value, "scope": ActionScope.OBJECT.value, "targets": [_semantic_object_projection(key)], "value": value, "effective_at": parts[1], "effect": {"canonical_id": key.canonical_id, "value": value, "operation": operation.value}})
            mutation_seen += 1
        else:
            action_specs.append({"operation": Operation.NOOP.value, "scope": ActionScope.OBJECT.value, "targets": [], "value": None, "effective_at": None, "effect": {"operation": Operation.NOOP.value}})
    semantic_payload = {
        "objects": [_semantic_object_projection(key)],
        "events": [{"sequence_index": index, "timestamp": event_parts[index][1], "role": roles[index].value, "source_anchor": anchors[index]} for index in range(num_events)],
        "actions": action_specs,
        "queries": [{"query_type": query_type.value, "answer_schema": answer_schema.value, "evaluation_mode": evaluation_mode.value, "targets": [_semantic_object_projection(key)]}],
        "final_state": final_state,
        "version_history": version_history,
    }
    semantic_core_id = "semcore_" + _digest("semantic-core-v4", semantic_payload)
    normalized_source_hash = _digest("normalized-source-v1", semantic_payload)
    source_group_id = "sourcegroup_" + _digest("source-group-v2", {"record_anchor": record_anchor})
    trajectory_id = "trajectory_" + _digest("trajectory-v1", {"semantic_core_id": semantic_core_id})
    split_key = SplitKey(semantic_core_id=semantic_core_id, source_group_id=source_group_id, trajectory_id=trajectory_id, split_policy_version="vnext-phase0-legacy-v1")
    task_id = "task_" + _digest("task-v2", {"semantic_core_id": semantic_core_id, "split": split.value, "example_index": example_index})
    events: list[MemoryEvent] = []
    actions: list[GoldAction] = []
    target_event_ids: list[str] = []
    ambiguity_policy_used = False
    mutation_seen = 0
    for index, (raw_text, timestamp, speaker, _, safe_metadata) in enumerate(event_parts):
        parsed = parsed_rows[index]
        event_id = "event_" + _digest("event-v3", {"semantic_core_id": semantic_core_id, "index": index, "role": roles[index].value, "value": parsed["value"] if parsed is not None else None})
        if index in target_index_set:
            operation = Operation.ADD if mutation_seen == 0 else Operation.UPDATE
            value = parsed["value"]
            action_id = "action_" + _digest("action-v3", {"event_id": event_id, "operation": operation.value, "value": value})
            action = GoldAction(action_id=action_id, event_id=event_id, operation=operation, scope=ActionScope.OBJECT, target_object_keys=[key], value=value, effective_at=timestamp, expected_effect={"canonical_id": key.canonical_id, "value": value, "operation": operation.value})
            mutation_seen += 1
            target_event_ids.append(event_id)
        else:
            action_id = "action_" + _digest("action-v3", {"event_id": event_id, "operation": Operation.NOOP.value})
            action = GoldAction(action_id=action_id, event_id=event_id, operation=Operation.NOOP, scope=ActionScope.OBJECT, expected_effect={"operation": Operation.NOOP.value})
        actions.append(action)
        metadata = {"legacy_role": roles[index].value}
        if safe_metadata:
            metadata["legacy_event"] = safe_metadata
        if roles[index] in _NON_TARGET_ROLES and _text_contains_scalar(raw_text, answer):
            metadata["allow_accepted_answer_ambiguity"] = True
            metadata["compatibility_rule"] = "non_target_accepted_answer_text_overlap_v1"
            ambiguity_policy_used = True
        events.append(MemoryEvent(event_id=event_id, sequence_index=index, timestamp=timestamp, raw_text=raw_text, normalized_text=raw_text.strip(), speaker=speaker, gold_action_ids=[action_id], role=roles[index], source_anchor=anchors[index], metadata=metadata))
    query_id = "query_" + _digest("query-v3", {"semantic_core_id": semantic_core_id, "kind": query_type.value})
    query_metadata: dict[str, Any] = {}
    query = MemoryQuery(query_id=query_id, query_type=query_type, text=question.strip(), target_object_keys=[key], answer_schema=answer_schema, evaluation_mode=evaluation_mode, metadata=query_metadata)
    gold = GoldRecord(actions=actions, action_sequence=[action.action_id for action in actions], final_state=final_state, version_history=version_history, expected_present_objects=[key], expected_absent_objects=[], gold_source_event_ids=target_event_ids, gold_answers={query_id: answer}, acceptable_answers={query_id: answer})
    legacy_analysis = _legacy_analysis_fields(raw, source_path, example_index)
    extra = {
        "num_events": num_events,
        "num_target_updates": num_target_updates,
        "legacy_num_updates": num_updates,
        "legacy_analysis": legacy_analysis,
        "legacy_parser_dependency": "scripts.eval_evomemory.parse_event_slot",
        "compatibility_policies": (
            ["non_target_accepted_answer_text_overlap_v1"] if ambiguity_policy_used else []
        ),
    }
    legacy_trajectory_id = "legacy_trajectory_" + _digest(
        "legacy-trajectory-v1", {"namespace": namespace, "record_anchor": record_anchor}
    )
    provenance = LegacyProvenance(
        legacy_family_id="evomemory_update_frequency",
        legacy_phase=legacy_phase,
        legacy_dataset_id=category,
        legacy_split_id=split.value,
        legacy_metric_namespace=namespace,
        legacy_run_condition_id=stress_type,
        answer_mode=None,
        memory_trajectory_id=legacy_trajectory_id,
        source_artifact_path=str(source_path),
        source_artifact_hash=source_sha256,
        known_caveats=[LEGACY_CAVEATS["p63_split_leakage"]],
    )
    metadata = TaskMetadata(
        split=split,
        split_key=split_key,
        profile_name=Difficulty.HARD,
        resolved_profile={
            "task_family": TaskFamily.REPEATED_SAME_SLOT.value,
            "update_depth": num_target_updates,
        },
        generation_config_hash=_CONFIG_HASH,
        compiler_version=COMPILER_VERSION,
        tags=["compatibility", "repeated_same_slot"],
        legacy_provenance=provenance,
        extra=extra,
    )
    source = SourceRecord(
        source_id="source_" + _digest("source-v3", {"record_anchor": record_anchor}),
        source_type=SourceType.SYNTHETIC,
        source_uri=None,
        license_or_privacy="compatibility_only",
        raw_hash=source_sha256,
        normalized_hash=normalized_source_hash,
        normalization_version="semantic-source-v1",
        provenance={"normalization_version": "semantic-source-v1"},
        generator=GeneratorProvenance(
            generator_name=_CONFIG["compiler"],
            seed=_NO_RANDOMNESS_SEED,
            config_sha256=_CONFIG_HASH,
            code_revision=_COMPILER_CODE_REVISION,
            compiler_version=COMPILER_VERSION,
        ),
    )
    try:
        task = MemUpdateTask(task_id=task_id, schema_version=SCHEMA_VERSION, task_family=TaskFamily.REPEATED_SAME_SLOT.value, difficulty=Difficulty.HARD, source=source, events=events, target_objects=[key], queries=[query], gold=gold, metadata=metadata)
        _report_failure(validate_task(task), "task_validation", source_path, example_index)
        _report_failure(validate_gold_replay(task), "gold_replay", source_path, example_index)
        _report_failure(validate_distractors(task), "distractor_validation", source_path, example_index)
        _json_bytes(task.model_dump(mode="json"))
        return task
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{_context(source_path, example_index, 'compiled_task')}: {type(exc).__name__}: {exc}") from exc


__all__ = ["compile_legacy_episode"]
