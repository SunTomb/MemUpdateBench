from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mub.vnext.contracts.enums import (
    ActionScope,
    AnswerDisposition,
    AnswerSchema,
    EvaluationMode,
    EventRole,
    Operation,
    ReferenceResolutionStatus,
    SourceType,
    Split,
)
from mub.vnext.contracts.v3.enums import LedgerEntryStatus, QueryTypeV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    DerivationStepV3,
    GeneratorProvenanceV3,
    GoldActionV3,
    MemoryEventV3,
    MemoryQueryV3,
    MemUpdateTaskV3,
    QueryGoldEvidenceV3,
    ReferenceCandidateV3,
    ReferenceResolutionSelector,
    SourceRecordV3,
    SplitKeyV3,
    SurfaceReferenceV3,
    TaskMetadataV3,
    VersionHistoryEntry,
    VersionHistoryLedger,
)
from mub.vnext.generation.identity import action_id, event_id, query_id, source_id, stable_id, task_id
from mub.vnext.generation.post_core_config import PostCoreDataConfig
from mub.vnext.generation.post_core_families import PostCoreSemanticCore
from mub.vnext.io import sha256_model


_TRANSLATION_CATALOG_VERSION = "vnext-post-core-data-surfaces-v1"
_NORMALIZATION_VERSION = "vnext-post-core-data-semantic-v1"
_SPLIT_POLICY_VERSION = "vnext-post-core-data-splits-v1"
_GENERATOR_NAME = "memupdatebench_vnext_post_core_renderer"

_LOCALE_LANGUAGE = {"en-US": "en", "es-ES": "es", "ja-JP": "ja"}


def _payload_hash(value: object) -> str:
    """Hash deterministic JSON payloads without making surface text semantic."""
    from pydantic import RootModel

    class _Payload(RootModel[Any]):
        pass

    return sha256_model(_Payload(root=value))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _copy_key(key: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_type": str(key["object_type"]),
        "namespace": str(key["namespace"]),
        "entity": str(key["entity"]),
        "attribute": str(key["attribute"]),
        "subkey": key.get("subkey"),
    }


def _object_key(core: PostCoreSemanticCore, index: int, *, namespace: str | None = None, entity: str | None = None) -> dict[str, Any]:
    return {
        "object_type": "fact",
        "namespace": namespace or "post_core",
        "entity": entity or f"{core.domain}_entity_{index}",
        "attribute": core.attribute,
        "subkey": None,
    }


def _trajectory_identity(core: PostCoreSemanticCore) -> str:
    return stable_id("trajectory", {"semantic_core_id": core.expansion_id})


def _source_group_identity(core: PostCoreSemanticCore) -> str:
    return stable_id("source_group", {"semantic_core_id": core.expansion_id})


def _version_group_identity(trajectory_id: str) -> str:
    return stable_id("version_group", {"trajectory_id": trajectory_id})


def _surface_language(locale: str) -> str:
    try:
        return _LOCALE_LANGUAGE[locale]
    except KeyError as exc:
        raise ValueError(f"unsupported post-core locale: {locale}") from exc


def _surface_labels(locale: str, surface_id: str) -> dict[str, str]:
    if locale == "es-ES":
        if surface_id == "explicit_canonical":
            return {"add": "AÑADIR", "update": "ACTUALIZAR", "noop": "SIN CAMBIO", "query": "¿Cuál es el valor actual de"}
        return {"add": "Añade", "update": "Actualiza", "noop": "Sin cambio", "query": "¿Qué valor tiene ahora"}
    if locale == "ja-JP":
        return {"add": "追加", "update": "更新", "noop": "変更なし", "query": "現在の値は"}
    if surface_id == "explicit_canonical":
        return {"add": "ADD", "update": "UPDATE", "noop": "NOOP", "query": "What is the current value of"}
    return {"add": "Set", "update": "Change", "noop": "No change", "query": "What is"}


def _render_mutation_text(operation: Operation, key: Mapping[str, Any], value: Any, *, locale: str, surface_id: str) -> str:
    labels = _surface_labels(locale, surface_id)
    object_text = f"{key['entity']}.{key['attribute']}"
    if locale == "ja-JP":
        if operation is Operation.ADD:
            return f"{object_text}を{_json(value)}に{labels['add']}。"
        return f"{object_text}を{_json(value)}に{labels['update']}。"
    if locale == "es-ES":
        verb = labels["add"] if operation is Operation.ADD else labels["update"]
        return f"{verb} {object_text} = {_json(value)}."
    if surface_id == "explicit_canonical":
        return f"{labels['add'] if operation is Operation.ADD else labels['update']} {object_text} = {_json(value)}"
    verb = labels["add"] if operation is Operation.ADD else labels["update"]
    return f"{verb} {object_text} to {_json(value)}."


def _render_noop_text(core: PostCoreSemanticCore, trap_type: str, index: int, *, locale: str, surface_id: str) -> str:
    labels = _surface_labels(locale, surface_id)
    if locale == "ja-JP":
        return f"{labels['noop']}（{trap_type}, {core.attribute}, {index}）。"
    if locale == "es-ES":
        return f"{labels['noop']}: {trap_type} sobre {core.attribute} ({index}); no se modifica ninguna memoria."
    if surface_id == "explicit_canonical":
        return f"NOOP: {trap_type} statement {index}; do not mutate memory."
    return f"{labels['noop']}: {trap_type} statement {index}; memory stays unchanged."


def _render_query_text(core: PostCoreSemanticCore, key: Mapping[str, Any], *, locale: str, surface_id: str) -> str:
    labels = _surface_labels(locale, surface_id)
    object_text = f"{key['entity']}.{key['attribute']}"
    if locale == "ja-JP":
        return f"{labels['query']} {object_text}？"
    if locale == "es-ES":
        return f"{labels['query']} {object_text}?"
    return f"{labels['query']} {object_text}?"


def _base_events_and_actions(
    core: PostCoreSemanticCore,
    rendered_task_id: str,
    *,
    locale: str,
    surface_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    family = core.family_id
    if family == "interleaved_multi_slot_update":
        return _family_b_events_and_actions(core, rendered_task_id, locale=locale, surface_id=surface_id)
    if family == "entity_attribute_grounding":
        return _family_c_events_and_actions(core, rendered_task_id, locale=locale, surface_id=surface_id)
    if family == "noop_write_discipline":
        return _family_d_events_and_actions(core, rendered_task_id, locale=locale, surface_id=surface_id)
    raise ValueError(f"unsupported post-core family: {family}")


def _family_b_events_and_actions(core: PostCoreSemanticCore, rendered_task_id: str, *, locale: str, surface_id: str):
    count = int(core.family_axes["active_object_count"])
    pattern = str(core.family_axes["interleaving_pattern"])
    keys = [_object_key(core, index) for index in range(count)]
    values = [f"{core.attribute}-v{index}-0" for index in range(count)]
    updates = [f"{core.attribute}-v{index}-1" for index in range(count)]
    semantic_rows: list[tuple[Operation, int]] = []
    if pattern == "round_robin":
        semantic_rows = [(Operation.ADD, index) for index in range(count)] + [(Operation.UPDATE, index) for index in range(count)]
    elif pattern == "burst":
        semantic_rows = [(operation, index) for index in range(count) for operation in (Operation.ADD, Operation.UPDATE)]
    elif pattern == "adversarial_adjacent":
        semantic_rows = [(Operation.ADD, index) for index in range(count)]
        semantic_rows += [(Operation.UPDATE, index) for index in reversed(range(count))]
    else:
        raise ValueError(f"unsupported Family B interleaving pattern: {pattern}")
    events: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for sequence_index, (operation, object_index) in enumerate(semantic_rows):
        value = values[object_index] if operation is Operation.ADD else updates[object_index]
        eid = event_id(rendered_task_id, sequence_index)
        aid = action_id(rendered_task_id, sequence_index, 0)
        key = keys[object_index]
        events.append({
            "event_id": eid,
            "sequence_index": sequence_index,
            "timestamp": None,
            "raw_text": _render_mutation_text(operation, key, value, locale=locale, surface_id=surface_id),
            "normalized_text": f"{operation.value.title()} {key['entity']}.{key['attribute']} = {_json(value)}.",
            "speaker": "User" if sequence_index % 2 else "Narrator",
            "gold_action_ids": [aid],
            "role": EventRole.LATEST_GOLD if operation is Operation.UPDATE else EventRole.HISTORICAL_SUPPORT,
            "source_anchor": {"event_index": sequence_index},
            "metadata": {"family": "B", "object_index": object_index, "interleaving_pattern": pattern},
        })
        actions.append({
            "action_id": aid,
            "event_id": eid,
            "operation": operation,
            "scope": ActionScope.OBJECT,
            "target_object_keys": [key],
            "value": value,
            "effective_at": None,
            "expected_effect": {"semantic_effect": operation.value.lower(), "object_index": object_index},
        })
    return events, actions, keys


def _family_c_events_and_actions(core: PostCoreSemanticCore, rendered_task_id: str, *, locale: str, surface_id: str):
    entity_condition = str(core.family_axes["entity_condition"])
    attribute_condition = str(core.family_axes["attribute_condition"])
    if entity_condition in {"same_name", "namespace_collision"}:
        keys = [_object_key(core, 0, namespace="post_core_a", entity="shared_entity"), _object_key(core, 1, namespace="post_core_b", entity="shared_entity")]
    elif entity_condition == "alias":
        keys = [_object_key(core, 0, entity="primary_entity"), _object_key(core, 1, entity="alternate_entity")]
    else:
        keys = [_object_key(core, 0)]
    events: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for sequence_index, key in enumerate(keys):
        value = f"{core.domain}-{core.attribute}-candidate-{sequence_index}"
        eid = event_id(rendered_task_id, sequence_index)
        aid = action_id(rendered_task_id, sequence_index, 0)
        events.append({
            "event_id": eid,
            "sequence_index": sequence_index,
            "timestamp": None,
            "raw_text": _render_mutation_text(Operation.ADD, key, value, locale=locale, surface_id=surface_id),
            "normalized_text": f"Add {key['entity']}.{key['attribute']} = {_json(value)}.",
            "speaker": "Records clerk",
            "gold_action_ids": [aid],
            "role": EventRole.LATEST_GOLD,
            "source_anchor": {"event_index": sequence_index},
            "metadata": {"family": "C", "candidate_index": sequence_index, "entity_condition": entity_condition, "attribute_condition": attribute_condition},
        })
        actions.append({
            "action_id": aid,
            "event_id": eid,
            "operation": Operation.ADD,
            "scope": ActionScope.OBJECT,
            "target_object_keys": [key],
            "value": value,
            "effective_at": None,
            "expected_effect": {"semantic_effect": "add", "candidate_index": sequence_index},
        })
    return events, actions, keys


def _family_d_events_and_actions(core: PostCoreSemanticCore, rendered_task_id: str, *, locale: str, surface_id: str):
    key = _object_key(core, 0)
    original_value = f"{core.domain}-{core.attribute}-stable"
    density = float(core.family_axes["noop_density"])
    trap_type = str(core.family_axes["trap_type"])
    noop_count = max(1, round(4 * density))
    rows: list[tuple[Operation, int]] = [(Operation.ADD, 0)] + [(Operation.NOOP, index) for index in range(noop_count)]
    events: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for sequence_index, (operation, trap_index) in enumerate(rows):
        eid = event_id(rendered_task_id, sequence_index)
        aid = action_id(rendered_task_id, sequence_index, 0)
        is_noop = operation is Operation.NOOP
        events.append({
            "event_id": eid,
            "sequence_index": sequence_index,
            "timestamp": None,
            "raw_text": _render_noop_text(core, trap_type, trap_index, locale=locale, surface_id=surface_id) if is_noop else _render_mutation_text(operation, key, original_value, locale=locale, surface_id=surface_id),
            "normalized_text": "No memory object changes." if is_noop else f"Add {key['entity']}.{key['attribute']} = {_json(original_value)}.",
            "speaker": "User" if is_noop else "Narrator",
            "gold_action_ids": [aid],
            "role": EventRole.NOOP_NEAR_MISS if is_noop else EventRole.LATEST_GOLD,
            "source_anchor": {"event_index": sequence_index},
            "metadata": {"family": "D", "trap_type": trap_type, "semantic_effect": "noop" if is_noop else "add", "mutation": not is_noop},
        })
        actions.append({
            "action_id": aid,
            "event_id": eid,
            "operation": operation,
            "scope": None if is_noop else ActionScope.OBJECT,
            "target_object_keys": [] if is_noop else [key],
            "value": None if is_noop else original_value,
            "effective_at": None,
            "expected_effect": {"semantic_effect": "noop" if is_noop else "add"},
        })
    return events, actions, [key]


def _build_version_history(keys: list[dict[str, Any]], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_identity = {tuple(key.get(part) for part in ("namespace", "entity", "attribute", "subkey")): [] for key in keys}
    for action in actions:
        if action["operation"] is Operation.NOOP:
            continue
        for key in action["target_object_keys"]:
            identity = tuple(key.get(part) for part in ("namespace", "entity", "attribute", "subkey"))
            entries = by_identity[identity]
            if entries:
                entries[-1]["valid_until_event_id"] = action["event_id"]
            entries.append({
                "version_index": len(entries),
                "status": LedgerEntryStatus.PRESENT,
                "value": action["value"],
                "valid_from_event_id": action["event_id"],
                "valid_until_event_id": None,
                "logical_time": None,
                "source_event_ids": [action["event_id"]],
            })
    if any(not entries for entries in by_identity.values()):
        raise ValueError("every post-core target must have a mutation history")
    return [{"object_key": key, "entries": entries} for key, entries in zip(keys, by_identity.values(), strict=True)]


def _build_current_evidence(query_id_value: str, key: dict[str, Any], event_id_value: str, answer: Any) -> dict[str, Any]:
    step_id = f"derive_{query_id_value}_current"
    return {
        "query_id": query_id_value,
        "answer": answer,
        "supporting_object_keys": [key],
        "supporting_event_ids": [event_id_value],
        "derivation_steps": [{
            "step_id": step_id,
            "operation": "read_current",
            "input_step_ids": [],
            "supporting_object_keys": [key],
            "supporting_event_ids": [event_id_value],
        }],
        "final_derivation_step_id": step_id,
        "stale_alternative": None,
    }


def _build_family_c_query_and_evidence(query_id_value: str, keys: list[dict[str, Any]], events: list[dict[str, Any]], core: PostCoreSemanticCore) -> tuple[dict[str, Any], dict[str, Any]]:
    entity_condition = str(core.family_axes["entity_condition"])
    attribute_condition = str(core.family_axes["attribute_condition"])
    candidates = [
        {
            "candidate_id": f"candidate_{index}",
            "object_key": key,
            "evidence": f"candidate {index} evidence",
            "source_anchors": [{"event_index": index}],
        }
        for index, key in enumerate(keys)
    ]
    if attribute_condition == "near_name":
        linked_ids: tuple[str, ...] = ()
        status = ReferenceResolutionStatus.NO_MATCH
        disposition = AnswerDisposition.ABSTAINED
        abstention_reason = "no candidate matches the requested attribute"
    elif entity_condition in {"same_name", "namespace_collision"}:
        linked_ids = tuple(candidate["candidate_id"] for candidate in candidates)
        status = ReferenceResolutionStatus.AMBIGUOUS
        disposition = AnswerDisposition.ABSTAINED
        abstention_reason = "multiple candidates remain after typed grounding"
    else:
        linked_ids = (candidates[0]["candidate_id"],)
        status = ReferenceResolutionStatus.UNIQUE
        disposition = AnswerDisposition.ANSWERED
        abstention_reason = None
    reference = {
        "reference_id": f"reference_{query_id_value}",
        "surface_text": f"{entity_condition}/{attribute_condition}",
        "normalized_text": "typed reference",
        "condition_kind": entity_condition,
        "evidence_kind": attribute_condition,
        "candidate_ids": list(linked_ids),
    }
    query = {
        "query_id": query_id_value,
        "query_type": QueryTypeV3.UNRESOLVED_REFERENCE,
        "text": f"Resolve the reference for {core.domain}.{core.attribute} and answer only when it is unique.",
        "selector": {
            "kind": "reference_resolution",
            "reference_candidates": candidates,
            "surface_references": [reference],
        },
        "target_object_keys": keys,
        "answer_schema": AnswerSchema.STRING,
        "evaluation_mode": EvaluationMode.RETRIEVED_PROMPT,
        "synthesis": None,
    }
    all_event_ids = [event["event_id"] for event in events]
    read_key = keys[0]
    read_event_id = all_event_ids[0]
    read_step_id = f"derive_{query_id_value}_read"
    steps: list[dict[str, Any]] = [{
        "step_id": read_step_id,
        "operation": "read_current",
        "input_step_ids": [],
        "supporting_object_keys": [read_key],
        "supporting_event_ids": [read_event_id],
    }]
    final_step_id = read_step_id
    if disposition is AnswerDisposition.ABSTAINED:
        final_step_id = f"derive_{query_id_value}_abstain"
        steps.append({
            "step_id": final_step_id,
            "operation": "abstain",
            "input_step_ids": [read_step_id],
            "supporting_object_keys": keys,
            "supporting_event_ids": all_event_ids,
        })
    selected_ids = list(linked_ids) if disposition is AnswerDisposition.ANSWERED else []
    answer = "" if disposition is AnswerDisposition.ANSWERED else None
    if disposition is AnswerDisposition.ANSWERED:
        answer = f"{core.domain}-{core.attribute}-candidate-0"
    evidence = {
        "query_id": query_id_value,
        "answer": answer,
        "disposition": disposition,
        "resolution_status": status,
        "selected_candidate_ids": selected_ids,
        "abstention_reason": abstention_reason,
        "supporting_object_keys": keys,
        "supporting_event_ids": all_event_ids,
        "derivation_steps": steps,
        "final_derivation_step_id": final_step_id,
        "stale_alternative": None,
    }
    return query, evidence


def _build_task(core: PostCoreSemanticCore, config: PostCoreDataConfig, *, split: Split, surface_variant: int, code_revision: str) -> MemUpdateTaskV3:
    if not isinstance(core, PostCoreSemanticCore):
        raise TypeError("core must be a PostCoreSemanticCore")
    if not isinstance(config, PostCoreDataConfig):
        raise TypeError("config must be a PostCoreDataConfig")
    if not isinstance(split, Split):
        raise TypeError("split must be a Split")
    if type(surface_variant) is not int or not 0 <= surface_variant < len(config.surfaces):
        raise ValueError(f"surface_variant must be one of {', '.join(str(index) for index in range(len(config.surfaces)))}")
    if type(code_revision) is not str or not code_revision.strip():
        raise ValueError("code_revision must be a nonblank string")

    surface = config.surfaces[surface_variant]
    locale = surface.locale
    surface_id = surface.surface_id
    rendered_task_id = task_id(core.expansion_id, surface_variant)
    trajectory_id = _trajectory_identity(core)
    source_group_id = _source_group_identity(core)
    version_group_id = _version_group_identity(trajectory_id)
    events_raw, actions_raw, keys_raw = _base_events_and_actions(core, rendered_task_id, locale=locale, surface_id=surface_id)
    query_id_value = query_id(rendered_task_id, 0)
    version_history_raw = _build_version_history(keys_raw, actions_raw)
    if core.family_id == "entity_attribute_grounding":
        query_raw, evidence_raw = _build_family_c_query_and_evidence(query_id_value, keys_raw, events_raw, core)
    else:
        query_key = keys_raw[0]
        query_identity = tuple(query_key.get(part) for part in ("namespace", "entity", "attribute", "subkey"))
        last_event = next(
            event
            for event, action in reversed(list(zip(events_raw, actions_raw)))
            if action["operation"] is not Operation.NOOP
            and tuple(action["target_object_keys"][0].get(part) for part in ("namespace", "entity", "attribute", "subkey")) == query_identity
        )
        query_raw = {
            "query_id": query_id_value,
            "query_type": QueryTypeV3.CURRENT,
            "text": _render_query_text(core, query_key, locale=locale, surface_id=surface_id),
            "selector": {"kind": "current"},
            "target_object_keys": [query_key],
            "answer_schema": AnswerSchema.STRING,
            "evaluation_mode": EvaluationMode.RETRIEVED_PROMPT,
            "synthesis": None,
        }
        answer = version_history_raw[0]["entries"][-1]["value"]
        evidence_raw = _build_current_evidence(query_id_value, query_key, last_event["event_id"], answer)

    event_index_by_id = {event["event_id"]: event["sequence_index"] for event in events_raw}
    normalized_history = [
        {
            "object_key": ledger["object_key"],
            "entries": [
                {
                    "version_index": entry["version_index"],
                    "status": entry["status"].value,
                    "value": entry["value"],
                    "valid_from_event_index": None if entry["valid_from_event_id"] is None else event_index_by_id[entry["valid_from_event_id"]],
                    "valid_until_event_index": None if entry["valid_until_event_id"] is None else event_index_by_id[entry["valid_until_event_id"]],
                    "logical_time": entry["logical_time"],
                    "source_event_indices": [event_index_by_id[event_id] for event_id in entry["source_event_ids"]],
                }
                for entry in ledger["entries"]
            ],
        }
        for ledger in version_history_raw
    ]
    if core.family_id == "entity_attribute_grounding":
        semantic_selector = {
            "kind": "reference_resolution",
            "reference_candidates": [
                {"object_key": candidate["object_key"]}
                for candidate in query_raw["selector"]["reference_candidates"]
            ],
            "surface_references": [
                {
                    "condition_kind": reference["condition_kind"],
                    "evidence_kind": reference["evidence_kind"],
                    "candidate_indices": [
                        query_raw["selector"]["reference_candidates"].index(
                            next(
                                candidate
                                for candidate in query_raw["selector"]["reference_candidates"]
                                if candidate["candidate_id"] == candidate_id
                            )
                        )
                        for candidate_id in reference["candidate_ids"]
                    ],
                }
                for reference in query_raw["selector"]["surface_references"]
            ],
        }
    else:
        semantic_selector = query_raw["selector"]
    semantic_source_payload = {
        "family_id": core.family_id,
        "difficulty": core.difficulty.value,
        "events": [
            {
                "operation": action["operation"].value,
                "target_object_keys": action["target_object_keys"],
                "value": action["value"],
                "role": event["role"].value,
                "metadata": {key: value for key, value in event["metadata"].items() if key not in {"surface_text", "raw_text"}},
            }
            for event, action in zip(events_raw, actions_raw, strict=True)
        ],
        "targets": keys_raw,
        "query": {"query_type": query_raw["query_type"].value, "selector": semantic_selector, "targets": query_raw["target_object_keys"], "answer_schema": query_raw["answer_schema"].value},
        "version_history": normalized_history,
        "evidence": {"answer": evidence_raw["answer"], "supporting_objects": evidence_raw["supporting_object_keys"], "supporting_event_indices": [event_index_by_id[event_id] for event_id in evidence_raw["supporting_event_ids"]], "disposition": None if evidence_raw.get("disposition") is None else evidence_raw["disposition"].value, "resolution_status": None if evidence_raw.get("resolution_status") is None else evidence_raw["resolution_status"].value},
    }
    normalized_hash = _payload_hash(semantic_source_payload)
    raw_hash = _payload_hash({"surface_key": surface.surface_key, "events": [event["raw_text"] for event in events_raw], "query": query_raw["text"]})
    source = SourceRecordV3(
        source_id=source_id("post_core", core.core_index, {"semantic_core_id": core.expansion_id, "surface_variant": surface_variant}),
        source_type=SourceType.SYNTHETIC,
        source_uri=f"memory://post-core/{rendered_task_id}",
        license_or_privacy="synthetic_redistributable",
        raw_hash=raw_hash,
        normalized_hash=normalized_hash,
        normalization_version=_NORMALIZATION_VERSION,
        provenance={
            "release_id": config.release_id,
            "semantic_core_id": core.expansion_id,
            "trajectory_id": trajectory_id,
            "source_group_id": source_group_id,
            "version_group_id": version_group_id,
            "surface_key": surface.surface_key,
            "translation_catalog_version": _TRANSLATION_CATALOG_VERSION,
        },
        generator=GeneratorProvenanceV3(
            generator_name=_GENERATOR_NAME,
            seed=config.seed,
            config_sha256=sha256_model(config),
            code_revision=code_revision,
            compiler_version=config.compiler_version,
        ),
    )
    metadata = TaskMetadataV3(
        split=split,
        split_key=SplitKeyV3(
            semantic_core_id=core.expansion_id,
            source_group_id=source_group_id,
            trajectory_id=trajectory_id,
            paraphrase_group_id=stable_id("paraphrase", {"semantic_core_id": core.expansion_id, "catalog": _TRANSLATION_CATALOG_VERSION}),
            source_document_id=stable_id("source_document", {"semantic_core_id": core.expansion_id}),
            version_group_id=version_group_id,
            split_policy_version=_SPLIT_POLICY_VERSION,
        ),
        profile_name=core.difficulty,
        resolved_profile=dict(core.profile),
        generation_config_hash=sha256_model(config),
        compiler_version=config.compiler_version,
        tags=("post_core", core.family_id, core.difficulty.value, surface.surface_key),
        extra={
            "domain": core.domain,
            "attribute": core.attribute,
            "locale": locale,
            "language": _surface_language(locale),
            "surface_id": surface_id,
            "surface_key": surface.surface_key,
            "surface_variant": surface_variant,
            "semantic_core_id": core.expansion_id,
            "trajectory_id": trajectory_id,
            "source_group_id": source_group_id,
            "version_group_id": version_group_id,
            "translation_catalog_version": _TRANSLATION_CATALOG_VERSION,
        },
    )
    return MemUpdateTaskV3(
        task_id=rendered_task_id,
        task_family=core.family_id,
        difficulty=core.difficulty,
        source=source,
        events=tuple(MemoryEventV3.model_validate(event) for event in events_raw),
        target_objects=tuple(_copy_key(key) for key in keys_raw),
        actions=tuple(GoldActionV3.model_validate(action) for action in actions_raw),
        queries=(MemoryQueryV3.model_validate(query_raw),),
        version_history=tuple(VersionHistoryLedger.model_validate(ledger) for ledger in version_history_raw),
        gold_evidence=(QueryGoldEvidenceV3.model_validate(evidence_raw),),
        metadata=metadata,
    )


def render_post_core_v3(core: PostCoreSemanticCore, *, config: PostCoreDataConfig, split: Split, surface_variant: int, code_revision: str) -> MemUpdateTaskV3:
    """Render one metadata-only post-Core semantic core into a strict-v3 task."""
    return _build_task(core, config, split=split, surface_variant=surface_variant, code_revision=code_revision)


def render_post_core_tasks_v3(core: PostCoreSemanticCore, *, config: PostCoreDataConfig, split: Split, code_revision: str) -> tuple[MemUpdateTaskV3, ...]:
    """Expand one metadata core into the four canonical post-Core surfaces."""
    return tuple(
        render_post_core_v3(core, config=config, split=split, surface_variant=index, code_revision=code_revision)
        for index in range(len(config.surfaces))
    )


__all__ = ["render_post_core_tasks_v3", "render_post_core_v3"]
