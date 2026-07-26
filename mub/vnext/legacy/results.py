from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import unicodedata
from pathlib import Path
from typing import Any

from mub.vnext.contracts.adapter import AdapterCapabilities, AdapterInfo
from mub.vnext.contracts.common import ArtifactRef, MetricFieldSupport
from mub.vnext.contracts.enums import CompletionStatus, EvaluationMode, SupportReason
from mub.vnext.contracts.manifest import RunManifest, ScorerConfig
from mub.vnext.contracts.runtime import (
    AnswerPrediction,
    MemoryEntryRecord,
    MemorySnapshot,
    ParserExtractorProvenance,
    RetrievalTrace,
    TaskRunRecord,
)
from mub.vnext.contracts.score import (
    ActionScores,
    AnswerScores,
    AuditScores,
    METRIC_FIELD_PATHS,
    ProtocolScores,
    RetrievalScores,
    ScoreRecord,
    StateScores,
    StoreScores,
    SystemScores,
)
from mub.vnext.contracts.task import MemUpdateTask, MemoryQuery
from mub.vnext.legacy.caveats import legacy_namespace
from mub.vnext.io.canonical import canonical_json_bytes
from mub.vnext.legacy.names import parse_legacy_run_name
from mub.vnext.legacy.loaders import load_evomemory_results, parse_legacy_bool
from mub.vnext.scoring.scorer import score_task
from mub.vnext.scoring.registry import (
    LEGACY_ALIAS_TO_FIELD,
    METRIC_REGISTRY,
    metric_applies_to_family,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 1_000_000
_MAX_ROW_JSON_NODES = 100_000
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_JSON_KEY_LENGTH = 1_024
_MAX_PATH_COMPONENT_LENGTH = 64
_MAX_ERROR_PATH_LENGTH = 512
_MAX_STRUCTURED_SECURITY_RECORDS = 10_000
_MISSING_SAMPLE_LIMIT = 20
_SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "xapikey",
        "apitoken",
        "authtoken",
        "authenticationtoken",
        "accesstoken",
        "refreshtoken",
        "authorization",
        "clientsecret",
        "bearertoken",
        "idtoken",
        "privatekey",
    }
)
_AWS_SECRET_KEY_ALIASES = frozenset(
    {"awssecretaccesskey", "secretaccesskey", "awsaccesskeyid"}
)
_STRUCTURED_SECURITY_CONTAINERS = frozenset(
    {
        "headers",
        "httpheaders",
        "requestheaders",
        "config",
        "configuration",
        "env",
        "environment",
        "environmentvariables",
    }
)
_HEADER_SECURITY_CONTAINERS = frozenset(
    {"headers", "httpheaders", "requestheaders"}
)
_CONFIG_SECURITY_CONTAINERS = frozenset(
    {
        "config",
        "configuration",
        "env",
        "environment",
        "environmentvariables",
    }
)
_TOKENIZER_ANCESTRY_KEYS = frozenset({"tokenizerconfig", "tokenizersettings"})
_TOKENIZER_STRING_KEYS = frozenset(
    {
        "bostoken",
        "eostoken",
        "padtoken",
        "unktoken",
        "masktoken",
        "septoken",
        "clstoken",
    }
)
_TOKENIZER_CONTAINER_KEYS = frozenset({"specialtokensmap"})
_CONFUSABLE_ASCII = {
    "а": "a",
    "е": "e",
    "і": "i",
    "ј": "j",
    "к": "k",
    "м": "m",
    "о": "o",
    "р": "p",
    "с": "c",
    "ѕ": "s",
    "т": "t",
    "у": "y",
    "х": "x",
    "α": "a",
    "β": "b",
    "ε": "e",
    "ι": "i",
    "κ": "k",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "y",
    "ν": "v",
    "χ": "x",
    "ɑ": "a",
    "ı": "i",
    "ᴄ": "c",
    "ᴅ": "d",
    "ᴇ": "e",
    "ꜰ": "f",
    "ɡ": "g",
    "ʜ": "h",
    "ɪ": "i",
    "ᴊ": "j",
    "ᴋ": "k",
    "ʟ": "l",
    "ᴍ": "m",
    "ɴ": "n",
    "ᴏ": "o",
    "ᴘ": "p",
    "ʀ": "r",
    "ꜱ": "s",
    "ᴛ": "t",
    "ᴜ": "u",
    "ᴠ": "v",
    "ᴡ": "w",
    "ʏ": "y",
}
_BENIGN_TOKEN_SCALAR_COUNT_KEYS = frozenset(
    {
        "tokencount",
        "prompttokens",
        "completiontokens",
        "inputtokens",
        "outputtokens",
        "totaltokens",
        "cachedtokens",
        "reasoningtokens",
        "prompttokencount",
        "completiontokencount",
        "inputtokencount",
        "outputtokencount",
        "totaltokencount",
        "cachedtokencount",
        "reasoningtokencount",
    }
)
_BENIGN_TOKEN_AGGREGATE_KEYS = frozenset({"tokenusage", "tokencounts"})
_BENIGN_TOKEN_NONCREDENTIAL_KEYS = frozenset(
    {
        "tokenizer",
        "tokenizername",
        "tokenizerversion",
        "tokenizerconfig",
        "tokenizersettings",
        "tokenization",
        "tokenf1",
        "answertokenf1",
    }
)
_BENIGN_TOKEN_STRING_METADATA_KEYS = frozenset(
    {"tokenizer", "tokenizername", "tokenizerversion", "tokenization"}
)
_BENIGN_TOKEN_SCORE_KEYS = frozenset({"tokenf1", "answertokenf1"})
_SECURITY_ALIAS_TARGETS = frozenset(
    {
        "apikey",
        "password",
        "secret",
        "secrets",
        "privatekey",
        "secretaccesskey",
        "awssecretaccesskey",
        "accesstoken",
        "refreshtoken",
        "authorization",
        "clientsecret",
        "bearertoken",
        "idtoken",
        "awsaccesskeyid",
    }
)
_BENIGN_TOKEN_INTEGER_KEYS = frozenset(
    {
        "maxtokens",
        "maxnewtokens",
        "mintokens",
        "minnewtokens",
        "tokenbudget",
    }
)
_BENIGN_TOKEN_RATE_KEYS = frozenset({"tokenspersecond"})
_RUN_IDENTITY_ALIASES = {
    "mode": ("mode",),
    "answer_mode": ("answer_mode",),
    "retrieval_policy": ("retrieval_policy",),
    "context_order": ("context_order",),
    "context_annotation": ("context_annotation", "annotation"),
    "checkpoint_family": (
        "checkpoint_family", "checkpoint", "lora_checkpoint"
    ),
    "training_seed": ("training_seed", "seed"),
    "memory_trajectory_id": ("memory_trajectory_id",),
    "legacy_run_condition_id": ("legacy_run_condition_id", "run_condition_id"),
    "update_depth": ("update_depth",),
    "answer_topk": ("answer_topk",),
    "slot_prompt_variant": ("slot_prompt_variant",),
}
_TRACE_IDENTITY_ALIASES = {
    "answer_topk": ("answer_topk",),
    "retrieval_policy": ("retrieval_policy",),
    "context_order": ("context_order",),
    "context_annotation": ("context_annotation", "annotation"),
    "slot_prompt_variant": ("slot_prompt_variant",),
}
_RUN_IDENTITY_FIELDS = tuple(
    alias for aliases in _RUN_IDENTITY_ALIASES.values() for alias in aliases
)
_INTEGER_IDENTITY_FIELDS = frozenset(
    {"training_seed", "seed", "update_depth", "answer_topk"}
)
_STRING_IDENTITY_FIELDS = frozenset(_RUN_IDENTITY_FIELDS) - _INTEGER_IDENTITY_FIELDS
_LEGACY_METRIC_EXCLUDED_FIELDS = frozenset(
    {
        "example_id",
        "shard_local_example_id",
        "question",
        "gold_answer",
        "predicted",
        "answer_trace",
        *_RUN_IDENTITY_FIELDS,
    }
)
_LAYER_TYPES = {
    "protocol_scores": ProtocolScores,
    "action_scores": ActionScores,
    "state_scores": StateScores,
    "store_scores": StoreScores,
    "retrieval_scores": RetrievalScores,
    "answer_scores": AnswerScores,
    "system_scores": SystemScores,
    "audit_scores": AuditScores,
}


def _fail(source_path: Path, field: str, message: str) -> None:
    raise ValueError(f"{source_path} field={field}: {message}")


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _exact_string(value: Any, source_path: Path, field: str, *, nonblank: bool = False) -> str:
    if type(value) is not str:
        _fail(source_path, field, "must be an exact built-in string")
    if nonblank and not value.strip():
        _fail(source_path, field, "must be a non-blank string")
    if _contains_surrogate(value):
        _fail(source_path, field, "must contain only Unicode scalar values")
    return value


def _normalized_optional_summary_string(
    summary: dict[str, Any],
    aliases: tuple[str, ...],
    source_path: Path,
    canonical_field: str,
) -> str | None:
    present: list[tuple[str, str | None]] = []
    for alias in aliases:
        if alias not in summary:
            continue
        value = summary[alias]
        normalized = (
            None
            if value is None
            else _exact_string(
                value,
                source_path,
                f"summary.{alias}",
                nonblank=True,
            )
        )
        present.append((alias, normalized))
    if not present:
        return None
    first_alias, first_value = present[0]
    for alias, value in present[1:]:
        if type(value) is not type(first_value) or value != first_value:
            _fail(
                source_path,
                f"summary.{canonical_field}",
                f"alias conflict between {first_alias} and {alias}",
            )
    return first_value


def _json_key_forms(key: str) -> tuple[str, str, bool]:
    normalized_text = unicodedata.normalize("NFKC", key).casefold()
    normalized_key = re.sub(r"[^a-z0-9]", "", normalized_text)
    had_confusable = any(character in _CONFUSABLE_ASCII for character in normalized_text)
    skeleton_text = "".join(
        _CONFUSABLE_ASCII.get(character, character)
        for character in normalized_text
    )
    skeleton = re.sub(r"[^a-z0-9]", "", skeleton_text)
    return normalized_key, skeleton, had_confusable


def _normalized_json_key(key: str) -> str:
    return _json_key_forms(key)[0]


def _within_one_edit(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) <= 1
    mismatch_seen = False
    left_index = 0
    for right_character in right:
        if (
            left_index < len(left)
            and left[left_index] == right_character
        ):
            left_index += 1
            continue
        if mismatch_seen:
            return False
        mismatch_seen = True
    return True


def _bounded_confusable_near_alias(
    raw_key: str,
    normalized_key: str,
    skeleton: str,
    had_confusable: bool,
    targets: frozenset[str] | set[str],
) -> bool:
    normalized_text = unicodedata.normalize("NFKC", raw_key).casefold()
    has_ascii_letter = any(
        "a" <= character <= "z"
        for character in normalized_text
    )
    has_greek_or_cyrillic = any(
        unicodedata.name(character, "").startswith(("GREEK", "CYRILLIC"))
        for character in normalized_text
    )
    if not has_ascii_letter or not (
        had_confusable or has_greek_or_cyrillic
    ):
        return False
    for candidate in {normalized_key, skeleton}:
        if len(candidate) > 64:
            continue
        if any(
            _within_one_edit(candidate, target)
            for target in targets
        ):
            return True
    return False


def _is_security_confusable(
    raw_key: str,
    normalized_key: str,
    skeleton: str,
    had_confusable: bool,
) -> bool:
    if had_confusable and _is_sensitive_security_alias(skeleton):
        return True
    return _bounded_confusable_near_alias(
        raw_key,
        normalized_key,
        skeleton,
        had_confusable,
        _SECURITY_ALIAS_TARGETS,
    )


def _is_secret_key_alias(normalized_key: str) -> bool:
    return (
        normalized_key in {"secret", "secrets"}
        or normalized_key.endswith("secret")
        or normalized_key.endswith("secrets")
        or normalized_key.startswith("secretkey")
        or normalized_key.startswith("secretvalue")
        or normalized_key in _AWS_SECRET_KEY_ALIASES
    )


def _is_password_cookie_alias(normalized_key: str) -> bool:
    if normalized_key in {"password", "cookie", "cookies", "setcookie"}:
        return True
    if normalized_key.endswith("password"):
        return True
    if normalized_key.startswith("password") and normalized_key.endswith(
        ("hash", "value", "secret", "field")
    ):
        return True
    return normalized_key.endswith("cookie")


def _is_sensitive_security_alias(normalized_key: str) -> bool:
    return (
        "token" in normalized_key
        or _is_secret_key_alias(normalized_key)
        or _is_password_cookie_alias(normalized_key)
        or normalized_key in _AWS_SECRET_KEY_ALIASES
        or any(
            sensitive_key in normalized_key
            for sensitive_key in _SENSITIVE_KEYS
        )
    )


def _is_sensitive_container_alias(normalized_key: str) -> bool:
    if "credential" in normalized_key:
        return True
    if "authentication" in normalized_key and not normalized_key.endswith(
        ("method", "mode", "type")
    ):
        return True
    if normalized_key == "auth" or normalized_key.endswith("auth"):
        return True
    if (
        normalized_key.startswith("auth")
        and not normalized_key.startswith("author")
        and normalized_key.endswith(
            ("config", "data", "headers", "store", "payload", "container")
        )
    ):
        return True
    if _is_secret_key_alias(normalized_key):
        return True
    return normalized_key.startswith(
        (
            "secretstore",
            "secretconfig",
            "secretdata",
            "secretvault",
            "secretpayload",
            "secretcontainer",
        )
    )


def _bounded_path_component(key: str) -> str:
    if len(key) <= _MAX_PATH_COMPONENT_LENGTH:
        return key
    return key[:_MAX_PATH_COMPONENT_LENGTH] + "…"


def _render_json_path(path: tuple[str, ...]) -> str:
    rendered = "payload"
    for component in path:
        separator = "" if component.startswith("[") else "."
        addition = separator + component
        if len(rendered) + len(addition) > _MAX_ERROR_PATH_LENGTH:
            rendered += ".<truncated>"
            break
        rendered += addition
    return rendered


def _json_fail(
    source_path: Path,
    path: tuple[str, ...],
    message: str,
) -> None:
    _fail(source_path, _render_json_path(path), message)


def _validate_token_telemetry(
    normalized_key: str,
    value: Any,
    source_path: Path,
    path: tuple[str, ...],
) -> None:
    if normalized_key in _BENIGN_TOKEN_SCALAR_COUNT_KEYS:
        if type(value) is not int or value < 0:
            _json_fail(
                source_path,
                path,
                "scalar token telemetry requires an exact non-negative integer",
            )
        return
    if normalized_key not in _BENIGN_TOKEN_AGGREGATE_KEYS:
        return
    if type(value) is not dict:
        _json_fail(
            source_path,
            path,
            "aggregate token telemetry requires an exact built-in object",
        )
    if len(value) > _MAX_STRUCTURED_SECURITY_RECORDS:
        _json_fail(source_path, path, "aggregate token telemetry is too large")
    for raw_leaf_key, leaf_value in value.items():
        leaf_path = path + ("<telemetry-key>",)
        if type(raw_leaf_key) is not str or _contains_surrogate(raw_leaf_key):
            _json_fail(
                source_path,
                leaf_path,
                "telemetry keys require exact Unicode scalar strings",
            )
        if len(raw_leaf_key) > _MAX_JSON_KEY_LENGTH:
            _json_fail(source_path, leaf_path, "JSON key length bound exceeded")
        if _normalized_json_key(raw_leaf_key) not in _BENIGN_TOKEN_SCALAR_COUNT_KEYS:
            _json_fail(
                source_path,
                leaf_path,
                "aggregate token telemetry contains an unapproved count field",
            )
        if type(leaf_value) is not int or leaf_value < 0:
            _json_fail(
                source_path,
                path + ("<telemetry-value>",),
                "aggregate token telemetry leaves require exact non-negative integers",
            )


def _structured_field_alias(
    raw_key: str,
    normalized_container: str,
    source_path: Path,
    path: tuple[str, ...],
) -> str | None:
    allowed_aliases = {"name", "value"}
    if normalized_container in _HEADER_SECURITY_CONTAINERS:
        allowed_aliases.add("key")
    elif normalized_container in _CONFIG_SECURITY_CONTAINERS:
        allowed_aliases.update({"key", "variable"})
    normalized_key, skeleton, had_confusable = _json_key_forms(raw_key)
    if normalized_key in allowed_aliases:
        return normalized_key
    if (
        had_confusable
        and skeleton in allowed_aliases
    ) or _bounded_confusable_near_alias(
        raw_key,
        normalized_key,
        skeleton,
        had_confusable,
        allowed_aliases,
    ):
        _json_fail(
            source_path,
            path + ("<redacted-structural-key>",),
            "confusable structured record fields are not importable",
        )
    return None


def _validate_typed_benign_token_field(
    normalized_key: str,
    value: Any,
    source_path: Path,
    path: tuple[str, ...],
) -> bool:
    if normalized_key in _BENIGN_TOKEN_INTEGER_KEYS:
        if type(value) is not int or value < 0:
            _json_fail(
                source_path,
                path,
                "typed token limit fields require an exact non-negative integer",
            )
        return True
    if normalized_key in _BENIGN_TOKEN_RATE_KEYS:
        if (
            type(value) not in {int, float}
            or type(value) is bool
            or not math.isfinite(float(value))
            or value < 0
        ):
            _json_fail(
                source_path,
                path,
                "token throughput fields require an exact finite non-negative number",
            )
        return True
    if normalized_key in _BENIGN_TOKEN_STRING_METADATA_KEYS:
        if type(value) is not str:
            _json_fail(
                source_path,
                path,
                "tokenizer metadata fields require exact strings",
            )
        return True
    if normalized_key in _BENIGN_TOKEN_SCORE_KEYS:
        if (
            type(value) not in {int, float}
            or type(value) is bool
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            _json_fail(
                source_path,
                path,
                "token score fields require finite numbers in [0, 1]",
            )
        return True
    if (
        normalized_key in _BENIGN_TOKEN_SCALAR_COUNT_KEYS
        or normalized_key in _BENIGN_TOKEN_AGGREGATE_KEYS
    ):
        _validate_token_telemetry(normalized_key, value, source_path, path)
        return True
    return False


def _structured_record_items(
    record: Any,
    normalized_container: str,
    source_path: Path,
    path: tuple[str, ...],
) -> list[tuple[Any, Any]]:
    if type(record) is list and len(record) == 2:
        return [(record[0], record[1])]
    if type(record) is not dict:
        return []
    normalized_fields: dict[str, Any] = {}
    for raw_key, value in record.items():
        if (
            type(raw_key) is not str
            or len(raw_key) > _MAX_JSON_KEY_LENGTH
            or _contains_surrogate(raw_key)
        ):
            continue
        structural_field = _structured_field_alias(
            raw_key,
            normalized_container,
            source_path,
            path,
        )
        if structural_field is None:
            continue
        if structural_field in normalized_fields:
            _json_fail(
                source_path,
                path + ("<redacted-structural-key>",),
                "ambiguous structured record fields are not importable",
            )
        normalized_fields[structural_field] = value
    allowed_identifier_fields = {"name"}
    if "value" in normalized_fields:
        if normalized_container in _HEADER_SECURITY_CONTAINERS:
            allowed_identifier_fields.add("key")
        elif normalized_container in _CONFIG_SECURITY_CONTAINERS:
            allowed_identifier_fields.update({"key", "variable"})
    return [
        (
            normalized_fields[field],
            normalized_fields.get("value"),
        )
        for field in sorted(allowed_identifier_fields)
        if field in normalized_fields
    ]


def _screen_structured_credentials(
    normalized_container: str,
    value: Any,
    source_path: Path,
    path: tuple[str, ...],
) -> None:
    if normalized_container not in _STRUCTURED_SECURITY_CONTAINERS:
        return
    records: list[Any]
    if type(value) is list:
        if len(value) > _MAX_STRUCTURED_SECURITY_RECORDS:
            _json_fail(source_path, path, "structured security records are too large")
        records = value
    elif type(value) is dict:
        if len(value) > _MAX_STRUCTURED_SECURITY_RECORDS:
            _json_fail(source_path, path, "structured security records are too large")
        records = [value]
        if normalized_container in _HEADER_SECURITY_CONTAINERS:
            records.extend([[raw_key, child] for raw_key, child in value.items()])
    else:
        return
    for record in records:
        for name, record_value in _structured_record_items(
            record,
            normalized_container,
            source_path,
            path,
        ):
            if type(name) is not str:
                continue
            if len(name) > _MAX_JSON_KEY_LENGTH or _contains_surrogate(name):
                _json_fail(
                    source_path,
                    path + ("<redacted-sensitive-name>",),
                    "structured security name is invalid",
                )
            normalized_name, skeleton, had_confusable = _json_key_forms(name)
            if _validate_typed_benign_token_field(
                normalized_name,
                record_value,
                source_path,
                path + ("<redacted-sensitive-name>",),
            ):
                continue
            if (
                _is_sensitive_security_alias(normalized_name)
                or _is_sensitive_container_alias(normalized_name)
                or _is_sensitive_security_alias(skeleton)
                or _is_sensitive_container_alias(skeleton)
                or _is_security_confusable(
                    name,
                    normalized_name,
                    skeleton,
                    had_confusable,
                )
            ):
                _json_fail(
                    source_path,
                    path + ("<redacted-sensitive-name>",),
                    "structured credential records are not importable",
                )


def _strict_json_copy(value: Any, source_path: Path) -> Any:
    active: set[int] = set()
    seen: set[int] = set()
    node_count = 0

    def visit(
        item: Any,
        path: tuple[str, ...],
        depth: int,
        *,
        tokenizer_ancestry: bool = False,
    ) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_JSON_NODES:
            _json_fail(
                source_path,
                path,
                f"JSON node budget {_MAX_JSON_NODES} exceeded",
            )
        if depth > _MAX_JSON_DEPTH:
            _json_fail(
                source_path,
                path,
                f"JSON nesting exceeds maximum depth {_MAX_JSON_DEPTH}",
            )
        if item is None or type(item) in {bool, int}:
            return item
        if type(item) is str:
            if _contains_surrogate(item):
                _json_fail(
                    source_path,
                    path,
                    "must contain only Unicode scalar values",
                )
            return item
        if type(item) is float:
            if not math.isfinite(item):
                _json_fail(
                    source_path,
                    path,
                    "must contain only finite JSON numbers",
                )
            return item
        if type(item) not in {dict, list}:
            raise TypeError(
                f"{source_path} field={_render_json_path(path)}: "
                "must contain exact built-in JSON containers and scalars"
            )
        identity = id(item)
        if identity in active:
            _json_fail(
                source_path,
                path,
                "active JSON recursion cycle is not allowed",
            )
        if identity in seen:
            _json_fail(
                source_path,
                path,
                "repeated shared JSON container identity is not allowed",
            )
        seen.add(identity)
        active.add(identity)
        try:
            if type(item) is list:
                return [
                    visit(
                        child,
                        path + (f"[{index}]",),
                        depth + 1,
                        tokenizer_ancestry=tokenizer_ancestry,
                    )
                    for index, child in enumerate(item)
                ]
            copied: dict[str, Any] = {}
            for raw_key, child in item.items():
                if type(raw_key) is not str:
                    _json_fail(
                        source_path,
                        path + ("<invalid-key>",),
                        "JSON object keys require exact built-in strings",
                    )
                if len(raw_key) > _MAX_JSON_KEY_LENGTH:
                    _json_fail(
                        source_path,
                        path + ("<oversized-key>",),
                        f"JSON key length bound {_MAX_JSON_KEY_LENGTH} exceeded",
                    )
                if _contains_surrogate(raw_key):
                    _json_fail(
                        source_path,
                        path + ("<invalid-key>",),
                        "JSON keys must contain only Unicode scalar values",
                    )
                normalized_key, skeleton, had_confusable = _json_key_forms(raw_key)
                child_path = path + (_bounded_path_component(raw_key),)
                redacted_path = path + ("<redacted-sensitive-key>",)
                if (
                    _is_sensitive_container_alias(normalized_key)
                    or _is_sensitive_container_alias(skeleton)
                    or _is_security_confusable(
                        raw_key,
                        normalized_key,
                        skeleton,
                        had_confusable,
                    )
                ):
                    _json_fail(
                        source_path,
                        redacted_path,
                        "sensitive credential containers are not importable",
                    )
                _screen_structured_credentials(
                    normalized_key,
                    child,
                    source_path,
                    child_path,
                )
                typed_benign_token_key = _validate_typed_benign_token_field(
                    normalized_key,
                    child,
                    source_path,
                    child_path,
                )
                if normalized_key in _TOKENIZER_ANCESTRY_KEYS and type(child) is not dict:
                    _json_fail(
                        source_path,
                        child_path,
                        "tokenizer configuration fields require exact objects",
                    )
                tokenizer_benign = False
                if tokenizer_ancestry and normalized_key in _TOKENIZER_STRING_KEYS:
                    if type(child) is not str:
                        _json_fail(
                            source_path,
                            child_path,
                            "tokenizer token fields require exact strings",
                        )
                    tokenizer_benign = True
                elif tokenizer_ancestry and normalized_key in _TOKENIZER_CONTAINER_KEYS:
                    if type(child) is not dict:
                        _json_fail(
                            source_path,
                            child_path,
                            "tokenizer token maps require exact objects",
                        )
                    tokenizer_benign = True
                safe_token_key = (
                    typed_benign_token_key
                    or tokenizer_benign
                    or normalized_key in _BENIGN_TOKEN_SCALAR_COUNT_KEYS
                    or normalized_key in _BENIGN_TOKEN_AGGREGATE_KEYS
                    or normalized_key in _BENIGN_TOKEN_NONCREDENTIAL_KEYS
                )
                if not safe_token_key and _is_sensitive_security_alias(
                    normalized_key
                ):
                    _json_fail(
                        source_path,
                        redacted_path,
                        "sensitive credential fields are not importable",
                    )
                next_tokenizer_ancestry = (
                    tokenizer_ancestry
                    or normalized_key in _TOKENIZER_ANCESTRY_KEYS
                    or (
                        tokenizer_ancestry
                        and normalized_key in _TOKENIZER_CONTAINER_KEYS
                    )
                )
                copied[raw_key] = visit(
                    child,
                    child_path,
                    depth + 1,
                    tokenizer_ancestry=next_tokenizer_ancestry,
                )
            return copied
        finally:
            active.remove(identity)

    if type(value) is not dict:
        raise TypeError(
            f"{source_path}: payload must be an exact built-in JSON object"
        )
    return visit(value, (), 0)


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) is float and left == 0.0 and right == 0.0:
        return math.copysign(1.0, left) == math.copysign(1.0, right)
    return left == right


def _count_json_nodes(value: Any) -> int:
    if type(value) is dict:
        return 1 + sum(_count_json_nodes(item) for item in value.values())
    if type(value) is list:
        return 1 + sum(_count_json_nodes(item) for item in value)
    return 1


def _validate_row_budget(row: dict[str, Any], source_path: Path, index: int) -> None:
    count = _count_json_nodes(row)
    if count > _MAX_ROW_JSON_NODES:
        _fail(
            source_path,
            f"results[{index}]",
            f"per-row JSON node budget {_MAX_ROW_JSON_NODES} exceeded",
        )


def _row_identity_material(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"example_id", "shard_local_example_id"}
    }


def _digest(domain: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()


LEGACY_OBJECT_EXTRACTOR_UNAVAILABLE_HASH = _digest(
    "legacy-object-extractor-unavailable-v1", None
)
LEGACY_EVOMEMORY_ADAPTER_VERSION = "legacy-import-v1"
LEGACY_EVOMEMORY_SYSTEM_NAME = "legacy_evomemory"
LEGACY_EVOMEMORY_SYSTEM_VERSION = "legacy-unknown"


def is_legacy_evomemory_adapter_identity(manifest: RunManifest) -> bool:
    match = re.fullmatch(r"legacy_run_([0-9a-f]{64})", manifest.run_id)
    if match is None:
        return False
    info = manifest.adapter_info
    return (
        info.adapter_id == f"legacy_evomemory_{match.group(1)[:16]}"
        and info.adapter_version == LEGACY_EVOMEMORY_ADAPTER_VERSION
        and info.system_name == LEGACY_EVOMEMORY_SYSTEM_NAME
        and info.system_version == LEGACY_EVOMEMORY_SYSTEM_VERSION
        and info.sdk_version is None
        and info.extractor_id is None
        and info.extractor_version is None
    )


def _validate_source(source_path: Path, source_sha256: str, run_name: str | None) -> tuple[str, str | None]:
    if type(source_path) is not type(Path()):
        raise TypeError("source_path must be an exact concrete pathlib.Path")
    path_text = _exact_string(str(source_path), source_path, "source_path", nonblank=True)
    if type(source_sha256) is not str or _SHA256_RE.fullmatch(source_sha256) is None:
        _fail(source_path, "source_sha256", "must be an exact lowercase SHA-256 string")
    if run_name is not None and type(run_name) is not str:
        raise TypeError(f"{source_path} field=run_name: must be an exact built-in string or None")
    if run_name is not None:
        _exact_string(run_name, source_path, "run_name")
    return path_text, run_name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_signature(path: Path) -> tuple[int, int, int, int]:
    try:
        result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Legacy result source does not exist: {path}") from exc
    if not stat.S_ISREG(result.st_mode):
        raise IsADirectoryError(f"Legacy result source is not a regular file: {path}")
    if result.st_size > _MAX_SOURCE_BYTES:
        _fail(path, "source_path", f"source byte cap {_MAX_SOURCE_BYTES} exceeded")
    return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns


def _verify_source_payload(
    payload: dict[str, Any], source_path: Path, source_sha256: str
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    pre_signature = _source_signature(source_path)
    pre_hash = _sha256_file(source_path)
    if pre_hash != source_sha256:
        _fail(source_path, "source_sha256", "does not match the source artifact")
    loaded = load_evomemory_results(source_path)
    post_hash = _sha256_file(source_path)
    post_signature = _source_signature(source_path)
    if post_hash != pre_hash or post_signature != pre_signature:
        raise RuntimeError(f"Legacy result source changed during import: {source_path}")
    loaded_copy = _strict_json_copy(loaded, source_path)
    payload_copy = _strict_json_copy(payload, source_path)
    if not _exact_json_equal(loaded_copy, payload_copy):
        _fail(source_path, "payload", "supplied payload does not match strict-loaded source payload")
    return pre_signature, payload_copy


def _verify_final_source(
    source_path: Path,
    source_sha256: str,
    original_signature: tuple[int, int, int, int],
) -> None:
    try:
        signature_before = _source_signature(source_path)
        digest = _sha256_file(source_path)
        signature_after = _source_signature(source_path)
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError(f"Legacy result source changed after verification: {source_path}") from exc
    if (
        signature_before != original_signature
        or signature_after != original_signature
        or signature_before != signature_after
        or digest != source_sha256
    ):
        raise RuntimeError(f"Legacy result source changed after verification: {source_path}")


def _validate_shape(payload: dict[str, Any], source_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    summary = payload.get("summary")
    rows = payload.get("results")
    if type(summary) is not dict:
        _fail(source_path, "summary", "must be an exact built-in JSON object")
    if type(rows) is not list:
        _fail(source_path, "results", "must be an exact built-in JSON array")
    expected = summary.get("num_examples")
    if type(expected) is not int or expected <= 0:
        _fail(source_path, "summary.num_examples", "must be a positive integer")
    for row_number, row in enumerate(rows):
        if type(row) is not dict:
            _fail(source_path, f"results[{row_number}]", "must be an exact built-in JSON object")
    return summary, rows, expected


def _reject_incomplete_status(container: dict[str, Any], source_path: Path, field_prefix: str) -> None:
    for field in ("status", "run_status", "completion_status", "row_status"):
        if field not in container:
            continue
        value = container[field]
        if type(value) is not str or value not in {"completed", "complete", "success", "succeeded", "clean"}:
            _fail(source_path, f"{field_prefix}.{field}", "source status cannot become a completed canonical run")
    for field in (
        "failed", "pending", "partial", "incomplete", "unsupported",
        "not_supported", "capacity_failed", "capacity_exhausted",
    ):
        if field not in container or container[field] is None:
            continue
        try:
            indicator = parse_legacy_bool(container[field])
        except (TypeError, ValueError) as exc:
            _fail(source_path, f"{field_prefix}.{field}", "malformed completion indicator cannot become a completed canonical run")
        if indicator is True:
            _fail(source_path, f"{field_prefix}.{field}", f"{field} cannot become a completed canonical run")
    if "supported" in container and container["supported"] is not None:
        try:
            supported = parse_legacy_bool(container["supported"])
        except (TypeError, ValueError) as exc:
            _fail(source_path, f"{field_prefix}.supported", "malformed support flag cannot become a completed canonical run")
        if supported is not True:
            _fail(source_path, f"{field_prefix}.supported", "unsupported source row cannot become a completed canonical run")
    for field in ("error", "exception", "error_message"):
        if field in container and container[field] is not None:
            _fail(source_path, f"{field_prefix}.{field}", "non-null error evidence cannot become a completed canonical run")


def _validate_completion_counts(summary: dict[str, Any], expected: int, source_path: Path) -> None:
    expected_values = {
        "completed_task_count": expected,
        "failed_task_count": 0,
        "not_supported_task_count": 0,
        "pending_task_count": 0,
        "partial_task_count": 0,
        "incomplete_task_count": 0,
        "unsupported_task_count": 0,
        "capacity_failed_task_count": 0,
        "capacity_exhausted_task_count": 0,
    }
    for field, required in expected_values.items():
        if field not in summary:
            continue
        value = summary[field]
        if type(value) is not int or value < 0 or value != required:
            _fail(source_path, f"summary.{field}", f"task_count must equal {required} for a completed canonical run")


def _is_type_valid_substantive_field(field: str, value: Any) -> bool:
    if value is None:
        return False
    if field == "predicted":
        return type(value) is str
    if field in {"answer_trace", "state_direct_trace"}:
        return type(value) is dict and bool(value)
    if field in {"memory", "final_memory", "memory_state"}:
        return type(value) in {dict, list}
    if field in {"em", "f1"}:
        return (
            type(value) in {int, float}
            and type(value) is not bool
            and math.isfinite(float(value))
            and 0 <= value <= 1
        )
    if field in {
        "value_em", "answer_value_present", "state_value_em", "stale_value_copied"
    }:
        try:
            return parse_legacy_bool(value) is not None
        except (TypeError, ValueError):
            return False
    canonical_path = next(
        (
            path
            for alias, path in LEGACY_ALIAS_TO_FIELD.items()
            if alias.rsplit(".", 1)[-1] == field
        ),
        None,
    )
    if canonical_path is None:
        return False
    if canonical_path == "store_scores.stale_conflicting_value_count":
        return type(value) is int and value >= 0
    return (
        type(value) in {int, float}
        and type(value) is not bool
        and math.isfinite(float(value))
        and 0 <= value <= 1
    )


def _validate_present_runtime_fields(row: dict[str, Any], source_path: Path, index: int) -> None:
    if "answer_trace" in row and row["answer_trace"] is not None and type(row["answer_trace"]) is not dict:
        _fail(source_path, f"results[{index}].answer_trace", "present answer_trace must be an exact built-in JSON object or null")
    if "predicted" in row and row["predicted"] is None:
        _fail(source_path, f"results[{index}].predicted", "explicit null answer fields are not valid completed evidence")
    if "predicted" in row and type(row["predicted"]) is not str:
        _fail(source_path, f"results[{index}].predicted", "present prediction must be an exact built-in string")
    if "gold_answer" in row and row["gold_answer"] is None:
        _fail(source_path, f"results[{index}].gold_answer", "explicit null answer fields are not valid completed evidence")
    if "answer_parse_valid" in row and type(row["answer_parse_valid"]) is not bool:
        _fail(source_path, f"results[{index}].answer_parse_valid", "must be an exact built-in boolean")
    if not any(
        _is_type_valid_substantive_field(field, value)
        for field, value in row.items()
    ):
        _fail(source_path, f"results[{index}]", "row lacks non-null type-valid substantive result evidence")


def _verified_answer_parser_version(
    rows: dict[int, dict[str, Any]], source_path: Path
) -> str:
    parser_contract_declared = any(
        "answer_parse_valid" in row or "answer_parser_version" in row
        for row in rows.values()
    )
    if not parser_contract_declared:
        return "legacy-unavailable"

    versions: set[str] = set()
    for index, row in rows.items():
        if "answer_parse_valid" not in row or "answer_parser_version" not in row:
            _fail(
                source_path,
                f"results[{index}].answer_parse_valid/answer_parser_version",
                "every row must declare both parser fields when any row declares parser provenance",
            )
        if type(row["answer_parse_valid"]) is not bool:
            _fail(
                source_path,
                f"results[{index}].answer_parse_valid",
                "must be an exact built-in boolean",
            )
        version = _exact_string(
            row["answer_parser_version"],
            source_path,
            f"results[{index}].answer_parser_version",
            nonblank=True,
        )
        if row["answer_parse_valid"] is True and type(row.get("predicted")) is not str:
            _fail(source_path, f"results[{index}].predicted", "parse-valid output requires authenticated parsed prediction")
        versions.add(version)
    if len(versions) > 1:
        _fail(source_path, "results.*.answer_parser_version", "mixed verified answer_parser_version values cannot share one run")
    return next(iter(versions))


def _validate_identity_value(field: str, value: Any, source_path: Path, location: str) -> Any:
    if value is None:
        return None
    if field in _INTEGER_IDENTITY_FIELDS:
        if type(value) is not int or value < 0:
            _fail(source_path, location, "run identity integer must be an exact non-negative integer")
        return value
    if field in _STRING_IDENTITY_FIELDS:
        if type(value) is not str or not value.strip():
            _fail(source_path, location, "run identity string must be an exact non-blank built-in string")
        return value
    raise AssertionError(f"unregistered run identity field {field}")


def authenticate_legacy_result_selection(
    payload: dict[str, Any],
    *,
    full_task_count: int,
    source_path: Path,
) -> tuple[int, ...]:
    if type(full_task_count) is not int or full_task_count <= 0:
        raise ValueError("full_task_count must be a positive exact integer")
    summary = payload.get("summary")
    rows = payload.get("results")
    if type(summary) is not dict or type(rows) is not list or not rows:
        _fail(source_path, "selection", "requires summary and nonempty results")
    indices: list[int] = []
    for row_number, row in enumerate(rows):
        if type(row) is not dict or type(row.get("example_id")) is not int:
            _fail(source_path, f"results[{row_number}].example_id", "requires exact global index")
        indices.append(row["example_id"])
    if len(indices) != len(set(indices)):
        _fail(source_path, "results.*.example_id", "duplicate global indices")
    has_start, has_end = "start_idx" in summary, "end_idx" in summary
    if has_start != has_end:
        _fail(source_path, "summary.start_idx/end_idx", "must be supplied together")
    if has_start:
        start, end = summary["start_idx"], summary["end_idx"]
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            _fail(source_path, "summary.start_idx/end_idx", "invalid explicit shard range")
        if "total_examples" in summary and summary["total_examples"] != full_task_count:
            _fail(source_path, "summary.total_examples", "must authenticate full task count")
        expected = tuple(range(start, end))
    else:
        expected = tuple(range(full_task_count))
    if tuple(indices) != expected or summary.get("num_examples") != len(expected):
        _fail(source_path, "selection", "result coverage does not match authenticated declaration")
    return tuple(indices)


def _indexed_rows(
    rows: list[dict[str, Any]],
    expected: int,
    summary: dict[str, Any],
    source_path: Path,
) -> dict[int, dict[str, Any]]:
    if len(rows) != expected:
        _fail(
            source_path,
            "results",
            f"row count {len(rows)} does not match declared num_examples {expected}",
        )
    has_start = "start_idx" in summary
    has_end = "end_idx" in summary
    if has_start != has_end:
        _fail(source_path, "summary.start_idx/end_idx", "must be supplied together")
    start = end = None
    if has_start:
        start, end = summary["start_idx"], summary["end_idx"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
        ):
            _fail(source_path, "summary.start_idx/end_idx", "must be ordered non-negative exact integers")
        if end - start != expected:
            _fail(
                source_path,
                "summary.start_idx/end_idx",
                "shard width must exactly equal summary.num_examples",
            )
    total_examples: int | None = None
    if "total_examples" in summary:
        total_examples = summary["total_examples"]
        if type(total_examples) is not int or total_examples < 0:
            _fail(source_path, "summary.total_examples", "must be an exact non-negative integer")
        if end is not None and end > total_examples:
            _fail(source_path, "summary.total_examples", "shard end_idx exceeds total_examples")
    merged_shards = summary.get("merged_shards", 1)
    if type(merged_shards) is not int or merged_shards < 1:
        _fail(source_path, "summary.merged_shards", "must be an exact positive integer")

    indexed: dict[int, dict[str, Any]] = {}
    shard_indices: list[int] = []
    for row_number, row in enumerate(rows):
        if "example_id" not in row:
            _fail(source_path, f"results[{row_number}].example_id", "required global row index is missing")
        index = row["example_id"]
        if type(index) is not int or index < 0:
            _fail(source_path, f"results[{row_number}].example_id", "must be a non-negative global integer")
        if total_examples is not None and index >= total_examples:
            _fail(
                source_path,
                f"results[{row_number}].example_id",
                "global example_id is outside summary.total_examples domain",
            )
        if index in indexed:
            _fail(source_path, f"results[{row_number}].example_id", f"duplicate global row index {index}")
        if "shard_local_example_id" not in row:
            _fail(source_path, f"results[{row_number}].shard_local_example_id", "required shard provenance index is missing")
        shard_index = row["shard_local_example_id"]
        if type(shard_index) is not int or shard_index < 0:
            _fail(source_path, f"results[{row_number}].shard_local_example_id", "must be a non-negative provenance integer")
        indexed[index] = row
        shard_indices.append(shard_index)

    if start is not None and end is not None:
        _bounded_index_mismatch(
            source_path,
            "results.*.example_id start_idx/end_idx shard range",
            set(range(start, end)),
            set(indexed),
        )
    if merged_shards == 1:
        _bounded_index_mismatch(
            source_path,
            "results.*.shard_local_example_id contiguous single-shard range",
            set(range(expected)),
            set(shard_indices),
        )
    else:
        multiplicities: dict[int, int] = {}
        for shard_index in shard_indices:
            multiplicities[shard_index] = multiplicities.get(shard_index, 0) + 1
        if multiplicities.get(0, 0) != merged_shards:
            _fail(
                source_path,
                "results.*.shard_local_example_id",
                "merged shard sequences must contain exactly merged_shards zero indices",
            )
        ordered_local_indices = sorted(multiplicities)
        if ordered_local_indices != list(range(len(ordered_local_indices))):
            _fail(
                source_path,
                "results.*.shard_local_example_id",
                "merged shard-local indices must form a contiguous zero-based range",
            )
        counts = [multiplicities[index] for index in ordered_local_indices]
        if any(left < right for left, right in zip(counts, counts[1:])):
            _fail(
                source_path,
                "results.*.shard_local_example_id",
                "merged shard-local multiplicities must be non-increasing",
            )
    return {index: indexed[index] for index in sorted(indexed)}


def _bounded_index_mismatch(
    source_path: Path,
    field: str,
    expected: set[int],
    observed: set[int],
) -> None:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if not missing and not extra:
        return
    _fail(
        source_path,
        field,
        "index mismatch: "
        f"missing_count={len(missing)} missing_sample={missing[:_MISSING_SAMPLE_LIMIT]} "
        f"extra_count={len(extra)} extra_sample={extra[:_MISSING_SAMPLE_LIMIT]}",
    )


def _validated_tasks(
    task_by_legacy_index: dict[int, MemUpdateTask],
    expected_indices: tuple[int, ...],
    source_path: Path,
    *,
    namespace: str,
    legacy_phase: str,
    summary: dict[str, Any],
    run_identity: dict[str, Any],
) -> dict[int, MemUpdateTask]:
    if type(task_by_legacy_index) is not dict:
        raise TypeError(f"{source_path} field=task_by_legacy_index: must be an exact built-in dict")
    if len(task_by_legacy_index) != len(expected_indices):
        _fail(
            source_path,
            "task_by_legacy_index",
            (
                f"task-map count {len(task_by_legacy_index)} does not match row count "
                f"{len(expected_indices)}; missing or extra indices"
            ),
        )
    expected_set = set(expected_indices)
    observed_indices: set[int] = set()
    tasks: dict[int, MemUpdateTask] = {}
    task_ids: set[str] = set()
    lineage: dict[str, str] | None = None
    analysis = summary.get("legacy_analysis_metadata")
    if analysis is None:
        analysis = {}
    elif type(analysis) is not dict:
        _fail(source_path, "summary.legacy_analysis_metadata", "must be an exact built-in object")
    explicit_lineage: dict[str, Any] = {}
    for field in ("legacy_family_id", "legacy_dataset_id", "legacy_split_id"):
        declarations: list[tuple[str, Any]] = []
        if field in summary:
            declarations.append((f"summary.{field}", summary[field]))
        if field in analysis:
            declarations.append(
                (f"summary.legacy_analysis_metadata.{field}", analysis[field])
            )
        if not declarations:
            continue
        location, value = declarations[0]
        for other_location, other_value in declarations[1:]:
            if type(other_value) is not type(value) or other_value != value:
                _fail(source_path, location, f"conflicts with {other_location}")
        explicit_lineage[field] = value
    for index, task in task_by_legacy_index.items():
        if type(index) is not int:
            _fail(source_path, "task_by_legacy_index", "keys must be non-negative integer indices")
        if index < 0:
            _fail(source_path, f"task_by_legacy_index[{index}]", "task index must be non-negative")
        observed_indices.add(index)
        if type(task) is not MemUpdateTask:
            raise TypeError(f"{source_path} field=task_by_legacy_index[{index}]: value must be MemUpdateTask")
        if task.task_id in task_ids:
            _fail(source_path, f"task_by_legacy_index[{index}].task_id", f"duplicate task_id {task.task_id!r}")
        task_ids.add(task.task_id)
        provenance = task.metadata.legacy_provenance
        if provenance is None:
            _fail(source_path, f"task_by_legacy_index[{index}].metadata.legacy_provenance", "LegacyProvenance is required")
        for field in ("answer_mode", "checkpoint_family", "training_seed"):
            if getattr(provenance, field) is not None:
                _fail(source_path, f"task_by_legacy_index[{index}].legacy_provenance.{field}", "dataset task provenance must not embed result-run identity")
        explicit_condition = run_identity["legacy_run_condition_id"]
        if explicit_condition is not None and (
            type(provenance.legacy_run_condition_id) is not type(explicit_condition)
            or provenance.legacy_run_condition_id != explicit_condition
        ):
            _fail(source_path, f"task_by_legacy_index[{index}].legacy_provenance.legacy_run_condition_id", "does not match explicit run-level legacy_run_condition_id")
        current = {
            "legacy_family_id": provenance.legacy_family_id,
            "legacy_phase": provenance.legacy_phase,
            "legacy_dataset_id": provenance.legacy_dataset_id,
            "legacy_split_id": provenance.legacy_split_id,
            "legacy_metric_namespace": provenance.legacy_metric_namespace,
        }
        if provenance.legacy_metric_namespace != namespace:
            _fail(source_path, f"task_by_legacy_index[{index}].legacy_metric_namespace", "does not match source legacy metric namespace")
        try:
            task_namespace = legacy_namespace(provenance.legacy_phase)
        except ValueError as exc:
            _fail(source_path, f"task_by_legacy_index[{index}].legacy_phase", str(exc))
        if task_namespace != namespace:
            _fail(source_path, f"task_by_legacy_index[{index}].legacy_phase", f"does not match source legacy phase {legacy_phase!r}")
        if provenance.legacy_split_id != task.metadata.split.value:
            _fail(source_path, f"task_by_legacy_index[{index}].legacy_split_id", "does not match canonical task split")
        if lineage is None:
            lineage = current
        else:
            for field, value in current.items():
                if type(value) is not type(lineage[field]) or value != lineage[field]:
                    _fail(source_path, f"task_by_legacy_index[{index}].{field}", "mixed task lineage is not allowed")
        for field, expected_value in explicit_lineage.items():
            if current[field] != expected_value:
                _fail(source_path, f"task_by_legacy_index[{index}].{field}", "does not match explicit source lineage")
        tasks[index] = task
    _bounded_index_mismatch(
        source_path, "task_by_legacy_index", expected_set, observed_indices
    )
    return {index: tasks[index] for index in expected_indices}


def _resolve_query(
    task: MemUpdateTask,
    row: dict[str, Any],
    index: int,
    source_path: Path,
) -> MemoryQuery:
    if not task.queries:
        _fail(source_path, f"task_by_legacy_index[{index}].query_linkage", "linked task has zero queries")
    if "query_id" not in row:
        if len(task.queries) != 1:
            _fail(source_path, f"results[{index}].query_id", "multi-query task requires exact row query_id")
        return task.queries[0]
    query_id = row["query_id"]
    if type(query_id) is not str:
        _fail(source_path, f"results[{index}].query_id", "must be an exact built-in string")
    matches = [query for query in task.queries if query.query_id == query_id]
    if len(matches) != 1:
        _fail(source_path, f"results[{index}].query_id", "must resolve exactly one linked task query")
    return matches[0]


def _validate_row_gold_linkage(
    rows: dict[int, dict[str, Any]],
    tasks: dict[int, MemUpdateTask],
    queries: dict[int, MemoryQuery],
    source_path: Path,
) -> None:
    for index, row in rows.items():
        if "gold_answer" not in row:
            continue
        task = tasks[index]
        query_id = queries[index].query_id
        if query_id not in task.gold.gold_answers:
            _fail(source_path, f"task_by_legacy_index[{index}].gold.gold_answers", f"missing linked query {query_id!r}")
        expected = task.gold.gold_answers[query_id]
        if not _exact_json_equal(row["gold_answer"], expected):
            _fail(source_path, f"results[{index}].gold_answer", "does not exactly match linked canonical task gold")


def _derive_phase_namespace(
    summary: dict[str, Any],
    task_by_legacy_index: dict[int, MemUpdateTask],
    source_path: Path,
) -> tuple[str, str]:
    if type(task_by_legacy_index) is not dict or not task_by_legacy_index:
        _fail(source_path, "task_by_legacy_index", "non-empty exact task map is required to derive legacy phase")
    phases: set[str] = set()
    namespaces: set[str] = set()
    for index, task in task_by_legacy_index.items():
        if type(index) is not int or type(task) is not MemUpdateTask:
            _fail(source_path, "task_by_legacy_index", "exact integer-to-MemUpdateTask mapping is required")
        provenance = task.metadata.legacy_provenance
        if provenance is None:
            _fail(source_path, f"task_by_legacy_index[{index}]", "LegacyProvenance is required to derive source phase")
        phases.add(provenance.legacy_phase)
        namespaces.add(provenance.legacy_metric_namespace)
    if len(phases) != 1:
        _fail(source_path, "task_by_legacy_index.legacy_phase", "linked tasks must have one consensus legacy phase")
    if len(namespaces) != 1:
        _fail(source_path, "task_by_legacy_index.legacy_metric_namespace", "linked tasks must have one consensus legacy metric namespace")
    phase = next(iter(phases))
    namespace = next(iter(namespaces))
    if legacy_namespace(phase) != namespace:
        _fail(source_path, "task_by_legacy_index", "task legacy phase and metric namespace are inconsistent")

    metadata = summary.get("legacy_analysis_metadata")
    if metadata is None:
        metadata = {}
    elif type(metadata) is not dict:
        _fail(source_path, "summary.legacy_analysis_metadata", "must be an exact built-in JSON object")
    phase_declarations = [
        value
        for value in (summary.get("legacy_phase"), metadata.get("legacy_phase"))
        if value is not None
    ]
    namespace_declarations = [
        value
        for value in (
            summary.get("legacy_metric_namespace"),
            metadata.get("legacy_metric_namespace"),
        )
        if value is not None
    ]
    for value in phase_declarations:
        if type(value) is not str or value != phase:
            _fail(source_path, "summary.legacy_phase", "does not exactly match linked task provenance")
    for value in namespace_declarations:
        if type(value) is not str or value != namespace:
            _fail(source_path, "summary.legacy_metric_namespace", "does not exactly match linked task provenance")
    return phase, namespace


def _run_identity(
    summary: dict[str, Any],
    rows: dict[int, dict[str, Any]],
    run_name: str | None,
    source_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    identity: dict[str, Any] = {field: None for field in _RUN_IDENTITY_ALIASES}

    def values_from(
        container: dict[str, Any],
        location: str,
        aliases_by_axis: dict[str, tuple[str, ...]],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for canonical, aliases in aliases_by_axis.items():
            present = [
                (
                    alias,
                    _validate_identity_value(
                        alias, container[alias], source_path, f"{location}.{alias}"
                    ),
                )
                for alias in aliases
                if alias in container
            ]
            if not present:
                continue
            first_alias, first_value = present[0]
            for alias, value in present[1:]:
                if type(value) is not type(first_value) or value != first_value:
                    _fail(
                        source_path,
                        f"{location}.{canonical}",
                        f"alias conflict between {first_alias} and {alias}",
                    )
            values[canonical] = first_value
        return values

    summary_identity = values_from(
        summary, "summary", _RUN_IDENTITY_ALIASES
    )
    identity.update(summary_identity)
    seen_axes = set(summary_identity)

    row_values = {
        index: values_from(row, f"results[{index}]", _RUN_IDENTITY_ALIASES)
        for index, row in rows.items()
    }
    for field in _RUN_IDENTITY_ALIASES:
        declarations = [
            (index, values[field])
            for index, values in row_values.items()
            if field in values
        ]
        if field in seen_axes:
            for index, value in declarations:
                if type(identity[field]) is not type(value) or identity[field] != value:
                    _fail(source_path, f"results[{index}].{field}", "row run identity conflicts with authoritative summary")
            continue
        if not declarations:
            continue
        if len(declarations) != len(rows):
            _fail(source_path, f"results.*.{field}", "partial row-level run identity is not allowed")
        first_value = declarations[0][1]
        if any(
            type(value) is not type(first_value) or value != first_value
            for _, value in declarations[1:]
        ):
            _fail(source_path, f"results.*.{field}", "mixed row-level run identity is not allowed")
        identity[field] = first_value
        seen_axes.add(field)

    trace_rows = {
        index: row["answer_trace"]
        for index, row in rows.items()
        if type(row.get("answer_trace")) is dict
    }
    trace_values = {
        index: values_from(
            trace,
            f"results[{index}].answer_trace",
            _TRACE_IDENTITY_ALIASES,
        )
        for index, trace in trace_rows.items()
    }
    for field in _TRACE_IDENTITY_ALIASES:
        declarations = [
            (index, values[field])
            for index, values in trace_values.items()
            if field in values
        ]
        if field in seen_axes:
            for index, value in declarations:
                if type(identity[field]) is not type(value) or identity[field] != value:
                    _fail(source_path, f"results[{index}].answer_trace.{field}", "trace configuration conflicts with normalized run identity")
            continue
        if not declarations:
            continue
        if len(trace_rows) != len(rows):
            _fail(
                source_path,
                f"results.*.answer_trace.{field}",
                "trace-only run identity inference requires every indexed result row to supply a trace",
            )
        if len(declarations) != len(rows):
            _fail(source_path, f"results.*.answer_trace.{field}", "partial trace configuration is not allowed")
        first_value = declarations[0][1]
        if any(
            type(value) is not type(first_value) or value != first_value
            for _, value in declarations[1:]
        ):
            _fail(source_path, f"results.*.answer_trace.{field}", "mixed trace configuration is not allowed")
        identity[field] = first_value
        seen_axes.add(field)

    inferred = parse_legacy_run_name(run_name) if run_name is not None else None
    if inferred is not None:
        adopted = False
        for field in ("mode", "answer_mode", "update_depth"):
            value = inferred[field]
            if field in seen_axes:
                if type(identity[field]) is not type(value) or identity[field] != value:
                    warnings.append(f"legacy_directory_name_conflict:{field}")
            else:
                identity[field] = value
                seen_axes.add(field)
                adopted = True
        if adopted:
            warnings.extend(inferred["warnings"])
    return identity


def _canonical_evaluation_mode(
    summary: dict[str, Any],
    identity: dict[str, Any],
    rows: dict[int, dict[str, Any]],
    tasks: dict[int, MemUpdateTask],
    queries: dict[int, MemoryQuery],
    source_path: Path,
    namespace: str,
    warnings: list[str],
) -> tuple[
    str | None,
    dict[int, list[MemorySnapshot]],
    dict[int, list[RetrievalTrace]],
]:
    answer_mode = identity["answer_mode"]
    claim = summary.get("semantic_compatibility")
    if claim is None:
        if answer_mode in {"slot_direct", "slot_prompt"}:
            warnings.append("legacy_answer_mode_unverified")
        return None, {}, {}
    if type(claim) is not dict or set(claim) != {
        "legacy_answer_mode", "canonical_evaluation_mode"
    }:
        _fail(source_path, "summary.semantic_compatibility", "must contain exactly legacy_answer_mode and canonical_evaluation_mode")
    expected_mode = {
        "slot_direct": EvaluationMode.STATE_DIRECT.value,
        "slot_prompt": EvaluationMode.RETRIEVED_PROMPT.value,
    }.get(answer_mode)
    if (
        claim["legacy_answer_mode"] != answer_mode
        or claim["canonical_evaluation_mode"] != expected_mode
    ):
        _fail(source_path, "summary.semantic_compatibility", "claimed mode mapping contradicts legacy run identity")

    snapshots: dict[int, list[MemorySnapshot]] = {}
    retrievals: dict[int, list[RetrievalTrace]] = {}
    for index, row in rows.items():
        task = tasks[index]
        query = queries[index]
        if query.evaluation_mode.value != expected_mode:
            _fail(source_path, f"task_by_legacy_index[{index}].queries[0].evaluation_mode", f"must equal {expected_mode}")
        if len(query.target_object_keys) != 1:
            _fail(source_path, f"task_by_legacy_index[{index}].queries[0]", "compatibility verification requires exactly one target object")

        if expected_mode == EvaluationMode.STATE_DIRECT.value:
            trace = row.get("state_direct_trace")
            required = {
                "query_id", "after_event_id", "object_key", "value",
                "state_by_object", "store_size",
            }
            if type(trace) is not dict or set(trace) != required:
                _fail(source_path, f"results[{index}].state_direct_trace", "state_direct_trace must contain exact canonical snapshot evidence")
            key = query.target_object_keys[0].canonical_id
            terminal_event_id = max(
                task.events, key=lambda event: event.sequence_index
            ).event_id
            if (
                type(trace["query_id"]) is not str
                or trace["query_id"] != query.query_id
                or type(trace["after_event_id"]) is not str
                or trace["after_event_id"] != terminal_event_id
                or type(trace["object_key"]) is not str
                or trace["object_key"] != key
                or type(trace["state_by_object"]) is not dict
                or set(trace["state_by_object"]) != {key}
                or type(trace["value"]) is not type(row.get("predicted"))
                or trace["value"] != row.get("predicted")
                or type(trace["state_by_object"][key]) is not type(trace["value"])
                or trace["state_by_object"][key] != trace["value"]
                or type(trace["store_size"]) is not int
                or trace["store_size"] < 1
            ):
                _fail(source_path, f"results[{index}].state_direct_trace", "snapshot evidence must use terminal after_event_id and exactly match linked task, query, and value")
            snapshots[index] = [
                MemorySnapshot(
                    after_event_id=trace["after_event_id"],
                    entries=[],
                    state_by_object=trace["state_by_object"],
                    store_size=trace["store_size"],
                    raw_adapter_state={namespace: trace},
                    snapshot_hash=None,
                )
            ]
            continue

        if summary.get("save_answer_traces") is not True:
            _fail(source_path, "summary.save_answer_traces", "retrieved_prompt verification requires exact saved traces")
        trace = row.get("answer_trace")
        if type(trace) is not dict:
            _fail(source_path, f"results[{index}].answer_trace", "retrieved_prompt verification requires an exact trace object")
        if "query_id" in trace and (
            type(trace["query_id"]) is not str
            or trace["query_id"] != query.query_id
        ):
            _fail(source_path, f"results[{index}].answer_trace.query_id", "does not match linked query")
        entries = trace.get("retrieved_entries")
        source_event_ids = trace.get("source_event_ids")
        if type(entries) is not list or not entries:
            _fail(source_path, f"results[{index}].answer_trace.retrieved_entries", "must be a non-empty exact list")
        if type(source_event_ids) is not list or not source_event_ids:
            _fail(source_path, f"results[{index}].answer_trace.source_event_ids", "must be a non-empty exact list")
        if (
            type(trace.get("predicted_answer")) is not str
            or trace["predicted_answer"] != row.get("predicted")
            or type(trace.get("gold_answer")) is not str
            or trace["gold_answer"] != row.get("gold_answer")
        ):
            _fail(source_path, f"results[{index}].answer_trace", "trace answers must exactly match the legacy result row")
        valid_event_ids = {event.event_id for event in task.events}
        canonical_entries: list[MemoryEntryRecord] = []
        scores: list[float] = []
        ranks: list[int] = []
        entry_event_ids: list[str] = []
        seen_entry_ids: set[str] = set()
        for entry_index, entry in enumerate(entries):
            field = f"results[{index}].answer_trace.retrieved_entries[{entry_index}]"
            if type(entry) is not dict:
                _fail(source_path, field, "must be an exact built-in object")
            for required_field in ("id", "content", "rank", "score", "source_event_id"):
                if required_field not in entry:
                    _fail(source_path, f"{field}.{required_field}", "required field is missing")
            entry_id = _exact_string(entry["id"], source_path, f"{field}.id", nonblank=True)
            content = _exact_string(entry["content"], source_path, f"{field}.content")
            source_event_id = _exact_string(entry["source_event_id"], source_path, f"{field}.source_event_id", nonblank=True)
            rank = entry["rank"]
            score = entry["score"]
            if entry_id in seen_entry_ids:
                _fail(source_path, f"{field}.id", "duplicate retrieved entry id")
            if source_event_id not in valid_event_ids:
                _fail(source_path, f"{field}.source_event_id", "source_event_id is not present in the linked task")
            if type(rank) is not int or rank != entry_index + 1:
                _fail(source_path, f"{field}.rank", "ranks must be exact contiguous positive integers in context order")
            if type(score) not in {int, float} or type(score) is bool or not math.isfinite(float(score)):
                _fail(source_path, f"{field}.score", "score must be an exact finite number")
            seen_entry_ids.add(entry_id)
            entry_event_ids.append(source_event_id)
            ranks.append(rank)
            scores.append(float(score))
            canonical_entries.append(
                MemoryEntryRecord(
                    entry_id=entry_id,
                    content=content,
                    source_event_ids=[source_event_id],
                    raw_metadata={namespace: entry},
                )
            )
        if any(type(value) is not str for value in source_event_ids) or source_event_ids != entry_event_ids:
            _fail(source_path, f"results[{index}].answer_trace.source_event_ids", "must exactly match retrieved entry source_event_id order")
        retrievals[index] = [
            RetrievalTrace(
                query_id=query.query_id,
                retrieved_entries=canonical_entries,
                scores=scores,
                ranks=ranks,
                gold_in_context=None,
                stale_in_context=None,
                distractor_in_context=None,
                retrieval_policy=identity["retrieval_policy"],
                context_order=identity["context_order"],
                version_metadata={namespace: trace},
                prompt_hash=None,
            )
        ]
    return expected_mode, snapshots, retrievals


def _dialect(summary: dict[str, Any], rows: dict[int, dict[str, Any]]) -> str:
    if summary.get("save_answer_traces") is True or any("answer_trace" in row for row in rows.values()):
        return "traced"
    return "old"


def _legacy_metrics(
    row: dict[str, Any], namespace: str, promoted_source_fields: set[str]
) -> dict[str, Any]:
    return {
        namespace: {
            key: value
            for key, value in row.items()
            if key not in _LEGACY_METRIC_EXCLUDED_FIELDS
            and key not in promoted_source_fields
        }
    }


def _mapped_metrics(
    row: dict[str, Any],
    namespace: str,
    task_family: str,
    source_path: Path,
    index: int,
) -> tuple[dict[str, Any], set[str], set[str]]:
    mapped: dict[str, Any] = {}
    null_aliases: set[str] = set()
    promoted_source_fields: set[str] = set()
    for source_field, value in row.items():
        alias = f"{namespace}.{source_field}"
        canonical_path = LEGACY_ALIAS_TO_FIELD.get(alias)
        if canonical_path is None:
            continue
        definition = METRIC_REGISTRY[canonical_path]
        if not metric_applies_to_family(definition, task_family):
            continue
        if canonical_path in mapped or canonical_path in null_aliases:
            _fail(source_path, f"results[{index}].{source_field}", f"duplicate exact aliases target {canonical_path}")
        if value is None:
            null_aliases.add(canonical_path)
            continue
        if canonical_path == "store_scores.stale_conflicting_value_count":
            if type(value) is not int or value < 0:
                _fail(source_path, f"results[{index}].{source_field}", "exact count alias must be a non-negative integer")
        elif type(value) not in {int, float} or type(value) is bool or not math.isfinite(float(value)) or not 0 <= value <= 1:
            _fail(source_path, f"results[{index}].{source_field}", "exact rate alias must be a finite number in [0, 1]")
        mapped[canonical_path] = float(value) if canonical_path != "store_scores.stale_conflicting_value_count" else value
        promoted_source_fields.add(source_field)
    return mapped, null_aliases, promoted_source_fields


def _score_record(task: MemUpdateTask, row: dict[str, Any], namespace: str, run_id: str, adapter_id: str, source_path: Path, index: int) -> ScoreRecord:
    mapped, null_aliases, promoted_source_fields = _mapped_metrics(
        row, namespace, task.task_family, source_path, index
    )
    layer_values: dict[str, dict[str, Any]] = {name: {} for name in _LAYER_TYPES}
    for path, value in mapped.items():
        layer, field = path.split(".", 1)
        layer_values[layer][field] = value
    supports: dict[str, MetricFieldSupport] = {}
    for path in sorted(path for path in METRIC_FIELD_PATHS if path not in mapped):
        definition = METRIC_REGISTRY[path]
        if not metric_applies_to_family(definition, task.task_family):
            reason = SupportReason.NOT_APPLICABLE
            detail = f"Metric does not apply to task family {task.task_family}."
        elif path in null_aliases:
            reason = SupportReason.MISSING_ARTIFACT
            detail = f"Exact {namespace} alias was explicitly null."
        else:
            reason = SupportReason.NOT_SUPPORTED
            detail = f"No exact {namespace} registry alias was present for this field."
        supports[path] = MetricFieldSupport(
            reason=reason,
            null_policy="exclude_from_aggregation",
            detail=detail,
        )
    layers = {name: layer_type(**layer_values[name]) for name, layer_type in _LAYER_TYPES.items()}
    return ScoreRecord(
        task_id=task.task_id,
        run_id=run_id,
        adapter_id=adapter_id,
        task_family=task.task_family,
        difficulty=task.difficulty,
        completion_status=CompletionStatus.COMPLETED,
        supported_metric_fields=supports,
        failure_flags=(),
        primary_failure=None,
        legacy_metrics=_legacy_metrics(row, namespace, promoted_source_fields),
        **layers,
    )


def _runtime_record(
    task: MemUpdateTask,
    query: MemoryQuery,
    row: dict[str, Any],
    namespace: str,
    dialect: str,
    run_id: str,
    adapter_id: str,
    source_path_text: str,
    source_sha256: str,
    *,
    memory_snapshots: list[MemorySnapshot],
    retrieval_traces: list[RetrievalTrace],
) -> TaskRunRecord:
    trace = row.get("answer_trace")
    trace_mapping = trace if type(trace) is dict else None
    optional = {
        "answer_trace": trace_mapping,
        "answer_topk": trace_mapping.get("answer_topk") if trace_mapping is not None else None,
        "gold_value_in_retrieved": trace_mapping.get("gold_value_in_retrieved") if trace_mapping is not None else None,
        "gold_retrieved": row.get("gold_retrieved"),
    }
    predictions = []
    answer_parser_version = "legacy-unavailable"
    if "answer_parse_valid" in row:
        answer_parser_version = row["answer_parser_version"]
    predicted = row.get("predicted")
    raw_output = row.get("raw_output")
    if row.get("answer_parse_valid") is True and raw_output is not None:
        if type(raw_output) is not str:
            _fail(Path(source_path_text), "raw_output", "canonical prediction raw_output must be an exact built-in string")
        predictions.append(
            AnswerPrediction(
                query_id=query.query_id,
                raw_output=raw_output,
                parsed_answer=predicted,
                format_valid=True,
            )
        )
    provenance = ParserExtractorProvenance(
        action_parser_version="legacy-unavailable",
        answer_parser_version=answer_parser_version,
        memory_entry_extractor_version="legacy-unavailable",
        object_value_extractor_config_hash=None,
        redaction_policy_version="legacy-import-redaction-v1",
        raw_provider_artifact_path=source_path_text,
        raw_provider_artifact_hash=source_sha256,
        raw_adapter_state_path=None,
        raw_adapter_state_hash=None,
    )
    return TaskRunRecord(
        task_id=task.task_id,
        adapter_id=adapter_id,
        run_id=run_id,
        parsed_actions=[],
        memory_snapshots=memory_snapshots,
        retrieval_traces=retrieval_traces,
        answer_predictions=predictions,
        system_events=[
            {
                "type": "legacy_evomemory_result",
                "legacy_namespace": namespace,
                "dialect": dialect,
                "raw_row": row,
                "legacy_optional": optional,
            }
        ],
        parser_extractor_provenance=provenance,
        exceptions=[],
        completion_status=CompletionStatus.COMPLETED,
    )


def import_evomemory_results(
    payload: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    run_name: str | None,
    task_by_legacy_index: dict[int, MemUpdateTask],
) -> tuple[RunManifest, list[TaskRunRecord], list[ScoreRecord], list[str]]:
    """Import a complete legacy EvoMemory run without reconstructing task semantics."""

    source_path_text, run_name = _validate_source(source_path, source_sha256, run_name)
    source_signature, copied = _verify_source_payload(
        payload,
        source_path,
        source_sha256,
    )
    summary, raw_rows, expected = _validate_shape(copied, source_path)
    _reject_incomplete_status(summary, source_path, "summary")
    _validate_completion_counts(summary, expected, source_path)
    indexed = _indexed_rows(raw_rows, expected, summary, source_path)
    for index, row in indexed.items():
        _validate_row_budget(row, source_path, index)
        _reject_incomplete_status(row, source_path, f"results[{index}]")
        _validate_present_runtime_fields(row, source_path, index)
    answer_parser_version = _verified_answer_parser_version(indexed, source_path)
    legacy_phase, namespace = _derive_phase_namespace(
        summary, task_by_legacy_index, source_path
    )
    warnings: list[str] = []
    identity = _run_identity(summary, indexed, run_name, source_path, warnings)
    tasks = _validated_tasks(
        task_by_legacy_index,
        tuple(indexed),
        source_path,
        namespace=namespace,
        legacy_phase=legacy_phase,
        summary=summary,
        run_identity=identity,
    )
    queries = {
        index: _resolve_query(tasks[index], indexed[index], index, source_path)
        for index in indexed
    }
    for index, row in indexed.items():
        trace = row.get("answer_trace")
        if type(trace) is dict and "query_id" in trace and (
            type(trace["query_id"]) is not str
            or trace["query_id"] != queries[index].query_id
        ):
            _fail(source_path, f"results[{index}].answer_trace.query_id", "does not match resolved linked query")
    _validate_row_gold_linkage(indexed, tasks, queries, source_path)
    canonical_mode, snapshots_by_index, retrievals_by_index = _canonical_evaluation_mode(
        summary, identity, indexed, tasks, queries, source_path, namespace, warnings
    )
    dialect = _dialect(summary, indexed)
    ordered_indices = tuple(indexed)
    task_bindings = [
        {
            "legacy_index": index,
            "task_sha256": hashlib.sha256(
                canonical_json_bytes(tasks[index])
            ).hexdigest(),
        }
        for index in ordered_indices
    ]
    task_map_hash = _digest("legacy-task-map-v2", task_bindings)
    row_bindings = [
        {
            "legacy_index": index,
            "row_sha256": _digest(
                "legacy-result-row-v2",
                _row_identity_material(indexed[index]),
            ),
        }
        for index in ordered_indices
    ]
    model_name = _normalized_optional_summary_string(
        summary, ("model_name", "model"), source_path, "model_name"
    )
    provider = _normalized_optional_summary_string(
        summary, ("provider",), source_path, "provider"
    )
    model_revision = _normalized_optional_summary_string(
        summary, ("model_revision",), source_path, "model_revision"
    )
    generation_identity = {
        "model_name": model_name,
        "provider": provider,
        "model_revision": model_revision,
    }
    run_identity_material = {
        "task_map_hash": task_map_hash,
        "rows": row_bindings,
        "namespace": namespace,
        "run_identity": identity,
        "generation_identity": generation_identity,
        "canonical_evaluation_mode": canonical_mode,
    }
    run_hash = _digest("legacy-evomemory-run-v1", run_identity_material)
    run_id = f"legacy_run_{run_hash}"
    adapter_id = f"legacy_evomemory_{run_hash[:16]}"
    configuration_hash = _digest(
        "legacy-evomemory-adapter-v1",
        {
            "run_identity": identity,
            "generation_identity": generation_identity,
        },
    )
    metadata = {
        "namespace": namespace,
        "dialect": dialect,
        "source_path": source_path_text,
        "source_sha256": source_sha256,
        "raw_summary": summary,
        "run_identity": identity,
        "generation_identity": generation_identity,
        "canonical_evaluation_mode": canonical_mode,
    }
    adapter_capabilities = AdapterCapabilities()
    scorer_config = ScorerConfig(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
        requested_metric_fields=(),
        legacy_compatibility_mode=None,
        strict_capability_check=False,
    )
    manifest = RunManifest(
        run_id=run_id,
        timestamp=summary.get("timestamp") if type(summary.get("timestamp")) is str else "legacy-unknown",
        code_revision=summary.get("code_revision") if type(summary.get("code_revision")) is str else "legacy-unknown",
        dirty_state=True,
        task_manifest=ArtifactRef(
            path=f"legacy-task-map://{task_map_hash}",
            sha256=task_map_hash,
            media_type="application/vnd.memupdatebench.task-map+json",
            record_count=expected,
        ),
        adapter_info=AdapterInfo(
            adapter_id=adapter_id,
            adapter_version=LEGACY_EVOMEMORY_ADAPTER_VERSION,
            system_name=LEGACY_EVOMEMORY_SYSTEM_NAME,
            system_version=LEGACY_EVOMEMORY_SYSTEM_VERSION,
            sdk_version=None,
            configuration_hash=configuration_hash,
            extractor_id=None,
            extractor_version=None,
        ),
        adapter_capabilities=adapter_capabilities,
        capability_verification_artifact=None,
        model_name=model_name,
        provider=provider,
        model_revision=model_revision,
        prompt_config={"legacy_result_import": metadata},
        decoding_config={},
        seed_information=(
            {"training_seed": identity["training_seed"]}
            if identity["training_seed"] is not None
            else {}
        ),
        action_parser_version="legacy-unavailable",
        answer_parser_version=answer_parser_version,
        memory_entry_extractor_version="legacy-unavailable",
        object_value_extractor_config_hash=LEGACY_OBJECT_EXTRACTOR_UNAVAILABLE_HASH,
        redaction_policy_version="legacy-import-redaction-v1",
        environment_summary={"legacy_import": True},
        package_summary={"compiler": "mub.vnext.legacy.results"},
        expected_task_count=expected,
        completed_task_count=expected,
        failed_task_count=0,
        not_supported_task_count=0,
        raw_provider_response_artifacts=(
            ArtifactRef(
                path=source_path_text,
                sha256=source_sha256,
                media_type="application/json",
                record_count=expected,
            ),
        ),
        raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(),
        score_artifacts=(),
        native_vs_extracted_field_summary={
            "compatibility_only": True,
            "legacy_namespace": namespace,
            "canonical_evaluation_mode": canonical_mode,
            "scorer_config": scorer_config.model_dump(mode="json"),
        },
    )
    task_runs = [
        _runtime_record(
            tasks[index], queries[index], indexed[index], namespace, dialect, run_id,
            adapter_id, source_path_text, source_sha256,
            memory_snapshots=snapshots_by_index.get(index, []),
            retrieval_traces=retrievals_by_index.get(index, []),
        )
        for index in ordered_indices
    ]
    scores = []
    for index, run in zip(ordered_indices, task_runs, strict=True):
        canonical_score = score_task(
            tasks[index], run, adapter_capabilities, scorer_config
        )
        score_payload = canonical_score.model_dump(mode="python")
        legacy_score = _score_record(
            tasks[index], indexed[index], namespace, run_id, adapter_id, source_path, index
        )
        score_payload["legacy_metrics"] = legacy_score.model_dump(mode="json")[
            "legacy_metrics"
        ]
        scores.append(ScoreRecord.model_validate(score_payload))
    _verify_final_source(source_path, source_sha256, source_signature)
    return manifest, task_runs, scores, list(dict.fromkeys(warnings))


__all__ = [
    "LEGACY_OBJECT_EXTRACTOR_UNAVAILABLE_HASH",
    "authenticate_legacy_result_selection",
    "import_evomemory_results",
    "is_legacy_evomemory_adapter_identity",
]
