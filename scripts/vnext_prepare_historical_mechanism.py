"""Prepare the offline historical-serving mechanism packet; never execute providers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
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
    raw_bytes: bytes
    canonical_bytes: bytes
    raw_sha256: str
    canonical_sha256: str
    case_roles: tuple[str, ...]
    retrieval_k: int
    status: str
    scientific_release_allowed: bool


@dataclass(frozen=True)
class MechanismCase:
    role: str
    runtime_namespace: str
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
    case_roles = tuple(data.get("case_roles", ()))
    status = data.get("status")
    scientific_release_allowed = data.get("scientific_release_allowed")
    retrieval_k = data.get("retrieval_k")
    if case_roles != _ROLES:
        raise ValueError("historical mechanism config case roles disagree")
    if status != "PREPARED_NOT_EXECUTED":
        raise ValueError("historical mechanism config status disagrees")
    if scientific_release_allowed is not False:
        raise ValueError("historical mechanism config release policy disagrees")
    if type(retrieval_k) is not int or retrieval_k <= 0:
        raise ValueError("historical mechanism config retrieval_k is invalid")
    canonical = _canonical(data)
    return ConfigSnapshot(
        raw_bytes=raw,
        canonical_bytes=canonical,
        raw_sha256=_sha256(raw),
        canonical_sha256=_sha256(canonical),
        case_roles=case_roles,
        retrieval_k=retrieval_k,
        status=status,
        scientific_release_allowed=scientific_release_allowed,
    )


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
    target_values = {
        "target_only_control": ("target-gamma", _time(2)),
        "cross_entity_same_relation": ("target-alpha", _time(0)),
        "historical_serving": ("target-gamma", _time(2)),
    }
    current_answer, _ = target_values[role]
    query_specs: list[tuple[str, str, dict[str, Any], str, str]] = [
        (
            "current",
            "current",
            {"kind": "current"},
            f"What is Target-00's relation now? Return the value only.",
            current_answer,
        )
    ]
    if role == "historical_serving":
        query_specs.append(
            (
                "as_of",
                "point_in_time",
                {"kind": "logical_time_anchor", "logical_time": _time(1)},
                f"What was Target-00's relation at {_time(1)}? Return the value only.",
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
    runtime_namespace = f"historical-mechanism-v1-{role}"
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


def _guard_output(output: Path) -> Path:
    output = output.absolute()
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
        "schema": "memupdatebench.version-transfer-historical-mechanism-manifest.v1",
        "status": "PREPARED_NOT_EXECUTED",
        "scientific_release_allowed": False,
        "code_revision": code_revision,
        "config_sha256": config.canonical_sha256,
        "config_artifact_sha256": config.raw_sha256,
        "case_count": len(cases),
        "roles": list(_ROLES),
        "retrieval_k": config.retrieval_k,
        "source_hash_policy": "python_lf_normalized",
        "source_hashes": _source_hashes(),
        "task_hashes": {case.role: case.task_hash for case in cases},
        "semantic_hashes": {case.role: case.semantic_hash for case in cases},
        "execution_counters": {
            "model_calls": 0,
            "provider_calls": 0,
            "database_calls": 0,
            "execution_attempts": 0,
        },
        "model_calls": 0,
        "provider_calls": 0,
        "database_calls": 0,
        "execution_attempts": 0,
        "public_payload_policy": "visible_event_input_and_visible_query_input_only",
        "protected_roots": ["data/vnext/core", "data/vnext/pilot", "results/unrelated"],
    }
    artifacts = {
        "config.json": config.raw_bytes,
        "cases.jsonl": records_raw,
        "manifest.json": _canonical(manifest),
    }
    index = {
        "schema": "memupdatebench.version-transfer-historical-mechanism-index.v1",
        "artifacts": {
            name: {"sha256": _sha256(raw), "bytes": len(raw)}
            for name, raw in artifacts.items()
        },
    }
    index_raw = _canonical(index)
    created_output = not output.exists()
    try:
        output.mkdir(parents=True, exist_ok=False)
        publish_files_atomically(
            {output / name: raw for name, raw in {**artifacts, "index.json": index_raw}.items()},
            overwrite=False,
        )
    except BaseException:
        if created_output and output.exists():
            try:
                if not any(output.iterdir()):
                    output.rmdir()
            except OSError:
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
    index_raw = (root / "index.json").read_bytes()
    if _sha256(index_raw) != index_sha256:
        raise ValueError("index hash mismatch")
    index = json.loads(index_raw)
    expected_artifacts = {"config.json", "cases.jsonl", "manifest.json"}
    if set(index.get("artifacts", {})) != expected_artifacts:
        raise ValueError("artifact membership mismatch")
    if {path.name for path in root.iterdir()} != expected_artifacts | {"index.json"}:
        raise ValueError("artifact membership mismatch")
    for name, reference in index["artifacts"].items():
        path = root / name
        assert_no_reparse_components(path)
        content = path.read_bytes()
        if len(content) != reference["bytes"] or _sha256(content) != reference["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
    manifest = json.loads((root / "manifest.json").read_bytes())
    current_config = _load_config()
    packet_config = _load_config((root / "config.json").read_bytes())
    if (
        packet_config.canonical_sha256 != current_config.canonical_sha256
        or packet_config.raw_sha256 != current_config.raw_sha256
    ):
        raise ValueError("config binding mismatch")
    if manifest.get("config_sha256") != current_config.canonical_sha256:
        raise ValueError("config manifest binding mismatch")
    if manifest.get("config_artifact_sha256") != current_config.raw_sha256:
        raise ValueError("config artifact manifest binding mismatch")
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
    records = [json.loads(line) for line in (root / "cases.jsonl").read_bytes().splitlines()]
    if len(records) != 3 or [record.get("role") for record in records] != list(_ROLES):
        raise ValueError("case role or count mismatch")
    for case, record in zip(cases, records):
        if record.get("status") != case.status:
            raise ValueError(f"case status binding mismatch: {case.role}")
        if record.get("task_hash") != case.task_hash or record.get("semantic_hash") != case.semantic_hash:
            raise ValueError(f"case hash binding mismatch: {case.role}")
        if record.get("public_events") != [event.public_payload for event in case.events]:
            raise ValueError(f"public event binding mismatch: {case.role}")
        if record.get("public_queries") != [query.public_payload for query in case.queries]:
            raise ValueError(f"public query binding mismatch: {case.role}")
        if any(record.get(name) != 0 for name in ("model_calls", "provider_calls", "database_calls", "execution_attempts")):
            raise ValueError(f"case execution counters are nonzero: {case.role}")
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
