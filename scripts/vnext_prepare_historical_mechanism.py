"""Prepare the offline historical-serving mechanism packet; never execute providers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.enums import EvaluationMode
from mub.vnext.contracts.v3.task import (
    MemUpdateTaskV3,
    MemoryEventV3,
    MemoryQueryV3,
)
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.visibility import visible_event_input, visible_query_input
from mub.vnext.io import canonical_json_bytes
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3

CONFIG_PATH = ROOT / "configs/vnext/version_transfer_historical_mechanism_v1.json"
SOURCE_FILES = (
    "scripts/vnext_prepare_historical_mechanism.py",
    "configs/vnext/version_transfer_historical_mechanism_v1.json",
    "mub/vnext/contracts/v3/task.py",
    "mub/vnext/contracts/v3/common.py",
    "mub/vnext/validation/replay_v3.py",
    "mub/vnext/external/visibility.py",
)
_ALLOWED_RESULT_PREFIX = "version_transfer_historical_mechanism_preparation_"
_ROLES = (
    "target_only_control",
    "cross_entity_same_relation",
    "historical_serving",
)
_CONFIG_SCHEMA = "memupdatebench.version-transfer-historical-mechanism.v1"
_MANIFEST_SCHEMA = "memupdatebench.version-transfer-historical-mechanism-manifest.v1"
_INDEX_SCHEMA = "memupdatebench.version-transfer-historical-mechanism-index.v1"
_SOURCE_HASH_POLICY = "python_lf_normalized"
_PUBLIC_PAYLOAD_POLICY = "visible_event_input_and_visible_query_input_only"
_STUDY_ID = "version-transfer-historical-mechanism-v1"
_EVENT_PATTERN = "initial target -> replacement -> replacement"
_CROSS_ENTITY_PATTERN = "initial target -> different entity same relation -> different entity same relation"
_HISTORICAL_QUERY_PATTERN = "current + as-of query"
_ZERO_EXECUTION_COUNTERS = {
    "model_calls": 0,
    "provider_calls": 0,
    "database_calls": 0,
    "execution_attempts": 0,
}
_CONFIG_KEYS = frozenset(
    {
        "schema",
        "study_id",
        "case_roles",
        "event_pattern",
        "cross_entity_pattern",
        "historical_query_pattern",
        "retrieval_k",
        "namespace_template",
        "status",
        "scientific_release_allowed",
        "execution_counters",
    }
)
_INDEX_KEYS = frozenset({"schema", "artifacts"})
_ARTIFACT_KEYS = frozenset({"config.json", "cases.jsonl", "manifest.json"})
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "scientific_release_allowed",
        "code_revision",
        "study_id",
        "event_pattern",
        "cross_entity_pattern",
        "historical_query_pattern",
        "config_sha256",
        "config_artifact_sha256",
        "config_execution_counters",
        "case_count",
        "roles",
        "retrieval_k",
        "source_hash_policy",
        "source_hashes",
        "task_hashes",
        "semantic_hashes",
        "execution_counters",
        "model_calls",
        "provider_calls",
        "database_calls",
        "execution_attempts",
        "public_payload_policy",
        "protected_roots",
    }
)
_CASE_KEYS = frozenset(
    {
        "role",
        "runtime_namespace",
        "namespace_template",
        "task_hash",
        "semantic_hash",
        "source_code_revision",
        "status",
        "model_calls",
        "provider_calls",
        "database_calls",
        "execution_attempts",
        "public_events",
        "public_queries",
    }
)
_PUBLIC_EVENT_KEYS = frozenset(
    {"event_id", "sequence_index", "logical_time", "raw_text", "runtime_namespace"}
)
_PUBLIC_QUERY_KEYS = frozenset({"query_id", "query_text", "k", "runtime_namespace"})


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _time(position: int) -> str:
    return (
        datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=position)
    ).isoformat().replace("+00:00", "Z")


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} keys mismatch")


def _require_sha256(value: Any, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_zero_execution_counters(value: Any, label: str) -> dict[str, int]:
    if type(value) is not dict or set(value) != set(_ZERO_EXECUTION_COUNTERS):
        raise ValueError(f"{label} keys mismatch")
    if any(type(counter) is not int or counter != 0 for counter in value.values()):
        raise ValueError(f"{label} (execution counters) must contain exact integer zeros")
    return dict(value)


def _require_string(value: Any, label: str) -> None:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _require_positive_int(value: Any, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class PublicEvent:
    source_event: MemoryEventV3
    public_payload: dict[str, Any]


@dataclass(frozen=True)
class PublicQuery:
    source_query: MemoryQueryV3
    public_payload: dict[str, Any]


@dataclass(frozen=True)
class ConfigSnapshot:
    schema: str
    raw_bytes: bytes
    canonical_bytes: bytes
    raw_sha256: str
    canonical_sha256: str
    case_roles: tuple[str, ...]
    study_id: str
    event_pattern: str
    cross_entity_pattern: str
    historical_query_pattern: str
    namespace_template: str
    retrieval_k: int
    execution_counters: MappingProxyType
    status: str
    scientific_release_allowed: bool


@dataclass(frozen=True)
class MechanismCase:
    role: str
    runtime_namespace: str
    namespace_template: str
    task: MemUpdateTaskV3
    events: tuple[PublicEvent, ...]
    queries: tuple[PublicQuery, ...]
    model_calls: int = 0
    provider_calls: int = 0
    database_calls: int = 0
    execution_attempts: int = 0

    @property
    def task_hash(self) -> str:
        return _sha256(canonical_json_bytes(self.task))

    @property
    def semantic_hash(self) -> str:
        return self.task.semantic_hash

    @property
    def status(self) -> str:
        return "PREPARED_NOT_EXECUTED"

    @property
    def source_revision(self) -> str:
        return self.task.source.generator.code_revision


def _load_config(raw_bytes: bytes | None = None) -> ConfigSnapshot:
    raw = CONFIG_PATH.read_bytes() if raw_bytes is None else bytes(raw_bytes)
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("historical mechanism config is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("historical mechanism config must be an object")
    _require_exact_keys(data, _CONFIG_KEYS, "config")
    if data["schema"] != _CONFIG_SCHEMA:
        raise ValueError("historical mechanism config schema disagrees")
    for field in (
        "schema",
        "study_id",
        "event_pattern",
        "cross_entity_pattern",
        "historical_query_pattern",
        "namespace_template",
        "status",
    ):
        _require_string(data[field], f"config.{field}")
    if type(data["case_roles"]) is not list or any(
        type(role) is not str for role in data["case_roles"]
    ):
        raise ValueError("config.case_roles must be a string list")
    case_roles = tuple(data["case_roles"])
    status = data["status"]
    scientific_release_allowed = data["scientific_release_allowed"]
    namespace_template = data["namespace_template"]
    retrieval_k = data["retrieval_k"]
    execution_counters = data["execution_counters"]
    if data["study_id"] != _STUDY_ID:
        raise ValueError("config study_id protocol metadata mismatch")
    if data["event_pattern"] != _EVENT_PATTERN:
        raise ValueError("config event_pattern protocol metadata mismatch")
    if data["cross_entity_pattern"] != _CROSS_ENTITY_PATTERN:
        raise ValueError("config cross_entity_pattern protocol metadata mismatch")
    if data["historical_query_pattern"] != _HISTORICAL_QUERY_PATTERN:
        raise ValueError("config historical_query_pattern protocol metadata mismatch")
    if case_roles != _ROLES:
        raise ValueError("historical mechanism config case roles disagree")
    if status != "PREPARED_NOT_EXECUTED":
        raise ValueError("historical mechanism config status disagrees")
    if type(scientific_release_allowed) is not bool or scientific_release_allowed is not False:
        raise ValueError("historical mechanism config release policy disagrees")
    if type(namespace_template) is not str or "{role}" not in namespace_template:
        raise ValueError("historical mechanism config namespace_template is invalid")
    try:
        namespaces = tuple(namespace_template.format(role=role) for role in _ROLES)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("historical mechanism config namespace_template is invalid") from exc
    if any(type(namespace) is not str or not namespace.strip() for namespace in namespaces):
        raise ValueError("historical mechanism config namespace_template is invalid")
    _require_positive_int(retrieval_k, "config.retrieval_k")
    _require_zero_execution_counters(execution_counters, "config.execution_counters")
    canonical = _canonical(data)
    return ConfigSnapshot(
        schema=_CONFIG_SCHEMA,
        raw_bytes=raw,
        canonical_bytes=canonical,
        raw_sha256=_sha256(raw),
        canonical_sha256=_sha256(canonical),
        case_roles=case_roles,
        study_id=data["study_id"],
        event_pattern=data["event_pattern"],
        cross_entity_pattern=data["cross_entity_pattern"],
        historical_query_pattern=data["historical_query_pattern"],
        namespace_template=namespace_template,
        retrieval_k=retrieval_k,
        execution_counters=MappingProxyType(dict(execution_counters)),
        status=status,
        scientific_release_allowed=scientific_release_allowed,
    )




def _validate_index_shape(index: Any) -> None:
    _require_exact_keys(index, _INDEX_KEYS, "index")
    if index["schema"] != _INDEX_SCHEMA:
        raise ValueError("index schema mismatch")
    artifacts = index["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != _ARTIFACT_KEYS:
        raise ValueError("index artifacts keys mismatch")
    for name, reference in artifacts.items():
        _require_exact_keys(reference, frozenset({"sha256", "bytes"}), f"index.artifacts.{name}")
        _require_sha256(reference["sha256"], f"index.artifacts.{name}.sha256")
        _require_nonnegative_int(reference["bytes"], f"index.artifacts.{name}.bytes")


def _validate_manifest_shape(manifest: Any) -> None:
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["schema"] != _MANIFEST_SCHEMA:
        raise ValueError("manifest schema mismatch")
    for field in (
        "schema",
        "status",
        "code_revision",
        "study_id",
        "event_pattern",
        "cross_entity_pattern",
        "historical_query_pattern",
        "config_sha256",
        "config_artifact_sha256",
        "source_hash_policy",
        "public_payload_policy",
    ):
        _require_string(manifest[field], f"manifest.{field}")
    if manifest["study_id"] != _STUDY_ID or manifest["event_pattern"] != _EVENT_PATTERN:
        raise ValueError("manifest protocol metadata mismatch")
    if manifest["cross_entity_pattern"] != _CROSS_ENTITY_PATTERN or manifest["historical_query_pattern"] != _HISTORICAL_QUERY_PATTERN:
        raise ValueError("manifest protocol metadata mismatch")
    if manifest["source_hash_policy"] != _SOURCE_HASH_POLICY:
        raise ValueError("manifest source_hash_policy mismatch")
    if manifest["public_payload_policy"] != _PUBLIC_PAYLOAD_POLICY:
        raise ValueError("manifest public_payload_policy mismatch")
    if manifest["status"] != "PREPARED_NOT_EXECUTED":
        raise ValueError("packet status is not preparation-only")
    if type(manifest["scientific_release_allowed"]) is not bool or manifest["scientific_release_allowed"] is not False:
        raise ValueError("scientific release policy is not locked")
    _require_sha256(manifest["config_sha256"], "manifest.config_sha256")
    _require_sha256(manifest["config_artifact_sha256"], "manifest.config_artifact_sha256")
    _require_zero_execution_counters(manifest["config_execution_counters"], "manifest.config_execution_counters")
    _require_nonnegative_int(manifest["case_count"], "manifest.case_count")
    if manifest["case_count"] != len(_ROLES):
        raise ValueError("manifest case_count mismatch")
    if type(manifest["roles"]) is not list or manifest["roles"] != list(_ROLES):
        raise ValueError("manifest roles mismatch")
    _require_positive_int(manifest["retrieval_k"], "manifest.retrieval_k")
    source_hashes = manifest["source_hashes"]
    if type(source_hashes) is not dict or set(source_hashes) != set(SOURCE_FILES):
        raise ValueError("manifest.source_hashes keys mismatch")
    for name, value in source_hashes.items():
        _require_sha256(value, f"manifest.source_hashes.{name}")
    for field in ("task_hashes", "semantic_hashes"):
        hashes = manifest[field]
        if type(hashes) is not dict or set(hashes) != set(_ROLES):
            raise ValueError(f"manifest.{field} keys mismatch")
        for role, value in hashes.items():
            _require_sha256(value, f"manifest.{field}.{role}")
    _require_zero_execution_counters(manifest["execution_counters"], "manifest.execution_counters")
    for field in ("model_calls", "provider_calls", "database_calls", "execution_attempts"):
        if type(manifest[field]) is not int or manifest[field] != 0:
            raise ValueError("manifest execution counters must contain exact integer zeros")
    if manifest["roles"] != list(_ROLES):
        raise ValueError("manifest roles mismatch")
    if manifest["protected_roots"] != ["data/vnext/core", "data/vnext/pilot", "results/unrelated"]:
        raise ValueError("manifest protected_roots mismatch")
    if type(manifest["protected_roots"]) is not list or any(
        type(root) is not str for root in manifest["protected_roots"]
    ):
        raise ValueError("manifest.protected_roots must be a string list")


def _validate_case_record_shape(record: Any) -> None:
    _require_exact_keys(record, _CASE_KEYS, "case record")
    for field in (
        "role",
        "runtime_namespace",
        "namespace_template",
        "source_code_revision",
        "status",
    ):
        _require_string(record[field], f"case record.{field}")
    for field in ("task_hash", "semantic_hash"):
        _require_sha256(record[field], f"case record.{field}")
    if record["status"] != "PREPARED_NOT_EXECUTED":
        raise ValueError("case record status mismatch")
    for field in ("model_calls", "provider_calls", "database_calls", "execution_attempts"):
        if type(record[field]) is not int or record[field] != 0:
            raise ValueError("case record execution counters must contain exact integer zeros")
    if type(record["public_events"]) is not list or type(record["public_queries"]) is not list:
        raise ValueError("case record public payloads must be lists")
    for event in record["public_events"]:
        _require_exact_keys(event, _PUBLIC_EVENT_KEYS, "case public event")
        _require_string(event["event_id"], "case public event.event_id")
        _require_nonnegative_int(event["sequence_index"], "case public event.sequence_index")
        if event["logical_time"] is not None:
            _require_string(event["logical_time"], "case public event.logical_time")
        _require_string(event["raw_text"], "case public event.raw_text")
        _require_string(event["runtime_namespace"], "case public event.runtime_namespace")
    for query in record["public_queries"]:
        _require_exact_keys(query, _PUBLIC_QUERY_KEYS, "case public query")
        _require_string(query["query_id"], "case public query.query_id")
        _require_string(query["query_text"], "case public query.query_text")
        _require_positive_int(query["k"], "case public query.k")
        _require_string(query["runtime_namespace"], "case public query.runtime_namespace")


def _key(entity: str, attribute: str = "relation") -> dict[str, Any]:
    return {
        "object_type": "slot",
        "namespace": "historical-mechanism-v1",
        "entity": entity,
        "attribute": attribute,
        "subkey": None,
    }


def _build_task(role: str, code_revision: str, config: ConfigSnapshot) -> MemUpdateTaskV3:
    target = _key("Target-00")
    other = _key("Other-00")
    if role == "cross_entity_same_relation":
        event_keys = (target, other, other)
        values = ("target-alpha", "other-alpha", "other-beta")
    else:
        event_keys = (target, target, target)
        values = ("target-alpha", "target-beta", "target-gamma")
    task_id = f"historical-mechanism-{role}"
    events: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    ledgers: dict[str, list[dict[str, Any]]] = {}
    key_by_identity: dict[str, dict[str, Any]] = {}
    for position, (object_key, value) in enumerate(zip(event_keys, values)):
        event_id = f"{task_id}-e{position}"
        action_id = f"{task_id}-a{position}"
        timestamp = _time(position)
        entity = object_key["entity"]
        text = (
            f"At {timestamp}, {entity}'s relation is {value}. "
            "This statement replaces its earlier relation when present."
        )
        identity = object_key["entity"]
        history = ledgers.setdefault(identity, [])
        if history:
            history[-1]["valid_until_event_id"] = event_id
        history.append(
            {
                "version_index": len(history),
                "status": "present",
                "value": value,
                "valid_from_event_id": event_id,
                "logical_time": timestamp,
                "source_event_ids": [event_id],
            }
        )
        key_by_identity[identity] = object_key
        events.append(
            {
                "event_id": event_id,
                "sequence_index": position,
                "timestamp": timestamp,
                "raw_text": text,
                "normalized_text": text,
                "speaker": "user",
                "role": (
                    "neutral"
                    if position == 0
                    else "same_name_other_entity"
                    if role == "cross_entity_same_relation"
                    else "stale_same_slot"
                ),
                "gold_action_ids": [action_id],
            }
        )
        actions.append(
            {
                "action_id": action_id,
                "event_id": event_id,
                "operation": "ADD" if not history[:-1] else "UPDATE",
                "scope": "object",
                "target_object_keys": [object_key],
                "value": value,
                "effective_at": timestamp,
            }
        )
    event_models = tuple(MemoryEventV3.model_validate(item) for item in events)
    current_answer = "target-alpha" if role == "cross_entity_same_relation" else "target-gamma"
    query_specs: list[tuple[str, str, dict[str, Any], str, str]] = [
        (
            "current",
            "current",
            {"kind": "current"},
            f"As of {event_models[-1].timestamp}, what is Target-00's relation? Return the value only.",
            current_answer,
        )
    ]
    if role in {"target_only_control", "historical_serving"}:
        query_specs.append(
            (
                "as_of",
                "point_in_time",
                {"kind": "logical_time_anchor", "logical_time": _time(1)},
                f"As of {event_models[1].timestamp}, what is Target-00's relation? Return the value only.",
                "target-beta",
            )
        )
    queries: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for query_name, query_type, selector, text, answer in query_specs:
        query_id = f"{task_id}-{query_name}"
        selected_event = (
            event_models[0].event_id
            if role == "cross_entity_same_relation" and query_name == "current"
            else event_models[-1].event_id
            if query_name == "current"
            else event_models[1].event_id
        )
        queries.append(
            {
                "query_id": query_id,
                "query_type": query_type,
                "text": text,
                "selector": selector,
                "target_object_keys": [target],
                "answer_schema": "string",
                "evaluation_mode": EvaluationMode.NATIVE_SYSTEM.value,
            }
        )
        evidence.append(
            {
                "query_id": query_id,
                "answer": answer,
                "supporting_object_keys": [target],
                "supporting_event_ids": [selected_event],
                "derivation_steps": [
                    {
                        "step_id": "read",
                        "operation": "read",
                        "supporting_object_keys": [target],
                        "supporting_event_ids": [selected_event],
                    }
                ],
                "final_derivation_step_id": "read",
            }
        )
    raw_hash = _sha256(_canonical(events))
    config_hash = config.canonical_sha256
    task = MemUpdateTaskV3.model_validate(
        {
            "task_id": task_id,
            "task_family": "historical_serving_mechanism",
            "difficulty": "hard",
            "source": {
                "source_id": task_id,
                "source_type": "synthetic",
                "source_uri": None,
                "license_or_privacy": "synthetic controlled diagnostic, no personal data",
                "raw_hash": raw_hash,
                "normalized_hash": raw_hash,
                "normalization_version": "historical-mechanism-v1",
                "generator": {
                    "generator_name": "historical-serving-mechanism",
                    "seed": _ROLES.index(role),
                    "config_sha256": config_hash,
                    "code_revision": code_revision,
                    "compiler_version": "historical-mechanism-v1",
                },
            },
            "events": events,
            "target_objects": list(key_by_identity.values()),
            "actions": actions,
            "queries": queries,
            "version_history": [
                {"object_key": key_by_identity[identity], "entries": entries}
                for identity, entries in ledgers.items()
            ],
            "gold_evidence": evidence,
            "metadata": {
                "split": "evaluation_only",
                "split_key": {
                    "semantic_core_id": task_id,
                    "source_group_id": task_id,
                    "trajectory_id": task_id,
                    "split_policy_version": "historical-mechanism-v1",
                },
                "profile_name": "hard",
                "generation_config_hash": config_hash,
                "compiler_version": "historical-mechanism-v1",
                "tags": ("offline_preparation", role),
                "extra": {
                    "case_role": role,
                    "public_event_time_policy": "UTC_event_times_v1",
                    "execution_status": "PREPARED_NOT_EXECUTED",
                },
            },
        }
    )
    replay = replay_task_v3(task)
    if replay.issues:
        raise ValueError(f"canonical replay failed for {role}: {replay.issues}")
    for query, gold in zip(task.queries, task.gold_evidence):
        resolution = resolve_query_v3(query, replay, task.events)
        if resolution.issues or resolution.answer != gold.answer:
            raise ValueError(f"canonical query replay failed for {query.query_id}")
    return task


def _public_case(role: str, code_revision: str, config: ConfigSnapshot) -> MechanismCase:
    task = _build_task(role, code_revision, config)
    runtime_namespace = config.namespace_template.format(role=role)
    public_events = tuple(
        PublicEvent(
            source_event=event,
            public_payload=visible_event_input(
                event, runtime_namespace=runtime_namespace
            ).model_dump(mode="json"),
        )
        for event in task.events
    )
    public_queries = tuple(
        PublicQuery(
            source_query=query,
            public_payload=visible_query_input(
                query, k=config.retrieval_k, runtime_namespace=runtime_namespace
            ).model_dump(mode="json"),
        )
        for query in task.queries
    )
    return MechanismCase(
        role=role,
        runtime_namespace=runtime_namespace,
        namespace_template=config.namespace_template,
        task=task,
        events=public_events,
        queries=public_queries,
    )


def build_cases(
    *, code_revision: str, config: ConfigSnapshot | None = None
) -> tuple[MechanismCase, ...]:
    """Build the exactly three canonical, replay-checked mechanism cases."""
    snapshot = _load_config() if config is None else config
    return tuple(_public_case(role, code_revision, snapshot) for role in _ROLES)


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        for name in SOURCE_FILES
    }


@dataclass(frozen=True)
class PacketOwnership:
    root: Path
    root_identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]
    index: Path
    index_identity: tuple[int, int]
    index_signature: tuple[int, int, int, int, int]
    artifacts: dict[str, Path]
    artifact_identities: dict[str, tuple[int, int]]
    artifact_signatures: dict[str, tuple[int, int, int, int, int]]
    contents: dict[str, bytes]


@dataclass(frozen=True)
class OutputOwnership:
    output: Path
    output_identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]
    anchor: Path
    anchor_identity: tuple[int, int]


def _file_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _capture_packet_ownership(root: Path) -> PacketOwnership:
    assert_no_reparse_components(root)
    index = root / "index.json"
    assert_no_reparse_components(index)
    return PacketOwnership(
        root=root,
        root_identity=_path_identity(root),
        parent=root.parent,
        parent_identity=_path_identity(root.parent),
        index=index,
        index_identity=_path_identity(index),
        index_signature=_file_signature(index.stat(follow_symlinks=False)),
        artifacts={},
        artifact_identities={},
        artifact_signatures={},
        contents={},
    )


def _assert_packet_ownership(ownership: PacketOwnership) -> None:
    try:
        assert_no_reparse_components(ownership.root)
        assert_no_reparse_components(ownership.index)
        if _path_identity(ownership.parent) != ownership.parent_identity:
            raise ValueError("packet parent ownership mismatch")
        if _path_identity(ownership.root) != ownership.root_identity:
            raise ValueError("packet root ownership mismatch")
        if _path_identity(ownership.index) != ownership.index_identity:
            raise ValueError("packet index ownership mismatch")
        if _file_signature(ownership.index.stat(follow_symlinks=False)) != ownership.index_signature:
            raise ValueError("packet index ownership/content changed")
        for name, path in ownership.artifacts.items():
            assert_no_reparse_components(path)
            if _path_identity(path) != ownership.artifact_identities[name]:
                raise ValueError(f"packet artifact ownership mismatch: {name}")
            if _file_signature(path.stat(follow_symlinks=False)) != ownership.artifact_signatures[name]:
                raise ValueError(f"packet artifact ownership/content changed: {name}")
    except OSError as exc:
        raise ValueError("packet ownership changed during read") from exc


def _read_packet_file(ownership: PacketOwnership, name: str) -> bytes:
    path = ownership.index if name == "index.json" else ownership.root / name
    _assert_packet_ownership(ownership)
    if name in ownership.contents:
        return ownership.contents[name]
    assert_no_reparse_components(path)
    signature = _file_signature(path.stat(follow_symlinks=False))
    with path.open("rb") as handle:
        handle_signature = _file_signature(os.fstat(handle.fileno()))
        # Windows path stat and fstat expose different ctime semantics.
        if handle_signature[:4] != signature[:4]:
            raise ValueError(f"packet file ownership changed before read: {name}")
        raw = handle.read()
        if _file_signature(os.fstat(handle.fileno())) != handle_signature:
            raise ValueError(f"packet file ownership/content changed during read: {name}")
    if _file_signature(path.stat(follow_symlinks=False)) != signature:
        raise ValueError(f"packet file ownership changed during read: {name}")
    if name != "index.json":
        ownership.artifacts[name] = path
        ownership.artifact_identities[name] = signature[:2]
        ownership.artifact_signatures[name] = signature
    ownership.contents[name] = raw
    _assert_packet_ownership(ownership)
    return raw


def _existing_anchor(path: Path) -> Path:
    candidate = path.parent
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def _capture_output_ownership(output: Path, anchor: Path, anchor_identity: tuple[int, int]) -> OutputOwnership:
    return OutputOwnership(
        output=output,
        output_identity=_path_identity(output),
        parent=output.parent,
        parent_identity=_path_identity(output.parent),
        anchor=anchor,
        anchor_identity=anchor_identity,
    )


def _assert_output_ownership(ownership: OutputOwnership) -> None:
    assert_no_reparse_components(ownership.output)
    if _path_identity(ownership.anchor) != ownership.anchor_identity:
        raise RuntimeError("output parent anchor ownership changed")
    if _path_identity(ownership.parent) != ownership.parent_identity:
        raise RuntimeError("output parent ownership changed")
    if _path_identity(ownership.output) != ownership.output_identity:
        raise RuntimeError("output directory ownership changed")


def _path_identity(path: Path) -> tuple[int, int]:
    stat = path.stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino


def _guard_output(output: Path) -> Path:
    lexical = output.absolute()
    assert_no_reparse_components(lexical)
    output = lexical.resolve(strict=False)
    assert_no_reparse_components(output)
    core = ROOT / "data/vnext/core"
    pilot = ROOT / "data/vnext/pilot"
    if output == core or core in output.parents or output == pilot or pilot in output.parents:
        raise ValueError("protected Core/Pilot root cannot receive mechanism preparation")
    results = ROOT / "results"
    if output == results or results in output.parents:
        relative = output.relative_to(results)
        if len(relative.parts) >= 2 and relative.parts[0] == "vnext" and relative.parts[1] == "pilot":
            raise ValueError("protected Pilot result root cannot receive mechanism preparation")
        if (
            len(relative.parts) < 2
            or relative.parts[0] != "vnext"
            or not relative.parts[1].startswith(_ALLOWED_RESULT_PREFIX)
        ):
            raise ValueError("unrelated result roots cannot receive mechanism preparation")
    return output


def _case_record(case: MechanismCase) -> dict[str, Any]:
    return {
        "role": case.role,
        "runtime_namespace": case.runtime_namespace,
        "namespace_template": case.namespace_template,
        "task_hash": case.task_hash,
        "semantic_hash": case.semantic_hash,
        "source_code_revision": case.task.source.generator.code_revision,
        "status": case.status,
        "model_calls": case.model_calls,
        "provider_calls": case.provider_calls,
        "database_calls": case.database_calls,
        "execution_attempts": case.execution_attempts,
        "public_events": [event.public_payload for event in case.events],
        "public_queries": [query.public_payload for query in case.queries],
    }


def publish_packet(output_root: str | Path, *, code_revision: str) -> dict[str, Any]:
    output = _guard_output(Path(output_root))
    if output.exists():
        raise FileExistsError(output)
    config = _load_config()
    cases = build_cases(code_revision=code_revision, config=config)
    records_raw = b"".join(_canonical(_case_record(case)) + b"\n" for case in cases)
    manifest: dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "status": "PREPARED_NOT_EXECUTED",
        "scientific_release_allowed": False,
        "code_revision": code_revision,
        "study_id": config.study_id,
        "event_pattern": config.event_pattern,
        "cross_entity_pattern": config.cross_entity_pattern,
        "historical_query_pattern": config.historical_query_pattern,
        "config_sha256": config.canonical_sha256,
        "config_artifact_sha256": config.raw_sha256,
        "config_execution_counters": dict(config.execution_counters),
        "case_count": len(cases),
        "roles": list(_ROLES),
        "retrieval_k": config.retrieval_k,
        "source_hash_policy": _SOURCE_HASH_POLICY,
        "source_hashes": _source_hashes(),
        "task_hashes": {case.role: case.task_hash for case in cases},
        "semantic_hashes": {case.role: case.semantic_hash for case in cases},
        "execution_counters": dict(config.execution_counters),
        "model_calls": config.execution_counters["model_calls"],
        "provider_calls": config.execution_counters["provider_calls"],
        "database_calls": config.execution_counters["database_calls"],
        "execution_attempts": config.execution_counters["execution_attempts"],
        "public_payload_policy": _PUBLIC_PAYLOAD_POLICY,
        "protected_roots": ["data/vnext/core", "data/vnext/pilot", "results/unrelated"],
    }
    artifacts = {
        "config.json": config.raw_bytes,
        "cases.jsonl": records_raw,
        "manifest.json": _canonical(manifest),
    }
    index = {
        "schema": _INDEX_SCHEMA,
        "artifacts": {
            name: {"sha256": _sha256(raw), "bytes": len(raw)}
            for name, raw in artifacts.items()
        },
    }
    index_raw = _canonical(index)
    anchor = _existing_anchor(output)
    anchor_identity = _path_identity(anchor)
    ownership: OutputOwnership | None = None
    try:
        output.mkdir(parents=True, exist_ok=False)
        ownership = _capture_output_ownership(output, anchor, anchor_identity)
        _assert_output_ownership(ownership)
        publish_files_atomically(
            {output / name: raw for name, raw in {**artifacts, "index.json": index_raw}.items()},
            overwrite=False,
            pre_publish=lambda: _assert_output_ownership(ownership),
        )
    except BaseException:
        if ownership is not None and output.exists():
            try:
                _assert_output_ownership(ownership)
                if not any(output.iterdir()):
                    output.rmdir()
            except (OSError, RuntimeError, ValueError):
                pass
        raise
    return {
        "index_sha256": _sha256(index_raw),
        "status": "PREPARED_NOT_EXECUTED",
        "scientific_release_allowed": False,
        "case_count": len(cases),
        "model_calls": 0,
    }


def read_packet(root: str | Path, index_sha256: str) -> dict[str, Any]:
    root = _guard_output(Path(root))
    assert_no_reparse_components(root)
    ownership = _capture_packet_ownership(root)
    index_raw = _read_packet_file(ownership, "index.json")
    if _sha256(index_raw) != index_sha256:
        raise ValueError("index hash mismatch")
    index = json.loads(index_raw)
    _validate_index_shape(index)
    expected_artifacts = {"config.json", "cases.jsonl", "manifest.json"}
    if set(index.get("artifacts", {})) != expected_artifacts:
        raise ValueError("artifact membership mismatch")
    if {path.name for path in root.iterdir()} != expected_artifacts | {"index.json"}:
        raise ValueError("artifact membership mismatch")
    for name, reference in index["artifacts"].items():
        path = root / name
        assert_no_reparse_components(path)
        content = _read_packet_file(ownership, name)
        if len(content) != reference["bytes"] or _sha256(content) != reference["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
    manifest = json.loads(_read_packet_file(ownership, "manifest.json"))
    _validate_manifest_shape(manifest)
    current_config = _load_config()
    packet_config = _load_config(_read_packet_file(ownership, "config.json"))
    if (
        packet_config.canonical_sha256 != current_config.canonical_sha256
        or packet_config.raw_sha256 != current_config.raw_sha256
    ):
        raise ValueError("config binding mismatch")
    if manifest.get("config_sha256") != current_config.canonical_sha256:
        raise ValueError("config manifest binding mismatch")
    if manifest.get("config_artifact_sha256") != current_config.raw_sha256:
        raise ValueError("config artifact manifest binding mismatch")
    if manifest.get("config_execution_counters") != dict(current_config.execution_counters):
        raise ValueError("config execution counters manifest binding mismatch")
    if manifest.get("retrieval_k") != current_config.retrieval_k:
        raise ValueError("config retrieval_k manifest binding mismatch")
    if manifest.get("status") != "PREPARED_NOT_EXECUTED":
        raise ValueError("packet status is not preparation-only")
    if manifest.get("scientific_release_allowed") is not False:
        raise ValueError("scientific release policy is not locked")
    if manifest.get("execution_counters") != {
        "model_calls": 0,
        "provider_calls": 0,
        "database_calls": 0,
        "execution_attempts": 0,
    }:
        raise ValueError("packet has nonzero execution counters")
    if any(
        manifest.get(name) != 0
        for name in ("model_calls", "provider_calls", "database_calls", "execution_attempts")
    ):
        raise ValueError("packet has nonzero execution counters")
    if manifest.get("source_hashes") != _source_hashes():
        raise ValueError("source hash binding mismatch")
    cases = build_cases(code_revision=manifest["code_revision"], config=current_config)
    expected_task_hashes = {case.role: case.task_hash for case in cases}
    expected_semantic_hashes = {case.role: case.semantic_hash for case in cases}
    if manifest.get("case_count") != len(cases):
        raise ValueError("manifest case_count binding mismatch")
    if manifest.get("roles") != list(_ROLES):
        raise ValueError("manifest roles binding mismatch")
    if manifest.get("task_hashes") != expected_task_hashes:
        raise ValueError("manifest task_hashes binding mismatch")
    if manifest.get("semantic_hashes") != expected_semantic_hashes:
        raise ValueError("manifest semantic_hashes binding mismatch")
    records = [json.loads(line) for line in _read_packet_file(ownership, "cases.jsonl").splitlines()]
    if len(records) != 3 or [record.get("role") for record in records] != list(_ROLES):
        raise ValueError("case role or count mismatch")
    for case, record in zip(cases, records):
        _validate_case_record_shape(record)
        if record.get("status") != case.status:
            raise ValueError(f"case status binding mismatch: {case.role}")
        if record.get("runtime_namespace") != case.runtime_namespace:
            raise ValueError(f"case runtime_namespace binding mismatch: {case.role}")
        if record.get("namespace_template") != case.namespace_template:
            raise ValueError(f"case namespace_template binding mismatch: {case.role}")
        if record.get("source_code_revision") != case.source_revision:
            raise ValueError(f"case source_code_revision binding mismatch: {case.role}")
        if record.get("task_hash") != case.task_hash or record.get("semantic_hash") != case.semantic_hash:
            raise ValueError(f"case hash binding mismatch: {case.role}")
        if record.get("public_events") != [event.public_payload for event in case.events]:
            raise ValueError(f"public event binding mismatch: {case.role}")
        if record.get("public_queries") != [query.public_payload for query in case.queries]:
            raise ValueError(f"public query binding mismatch: {case.role}")
        if any(record.get(name) != 0 for name in ("model_calls", "provider_calls", "database_calls", "execution_attempts")):
            raise ValueError(f"case execution counters are nonzero: {case.role}")
    _assert_packet_ownership(ownership)
    manifest["index_sha256"] = index_sha256
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-revision")
    parser.add_argument("--verify-index-sha256")
    args = parser.parse_args(argv)
    if bool(args.code_revision) == bool(args.verify_index_sha256):
        parser.error("provide exactly one of --code-revision or --verify-index-sha256")
    if args.verify_index_sha256:
        result = read_packet(args.output_root, args.verify_index_sha256)
    else:
        result = publish_packet(args.output_root, code_revision=args.code_revision)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
