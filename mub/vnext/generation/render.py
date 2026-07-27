from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import RootModel

from mub.vnext.contracts.common import (
    GeneratorProvenance,
    MemoryObjectKey,
    SourceRecord,
    thaw_json,
)
from mub.vnext.contracts.enums import (
    ActionScope,
    AnswerSchema,
    EvaluationMode,
    Operation,
    QueryType,
    SourceType,
    Split,
)
from mub.vnext.contracts.task import (
    GoldAction,
    GoldRecord,
    MemUpdateTask,
    MemoryEvent,
    MemoryQuery,
    SplitKey,
    TaskMetadata,
)
from mub.vnext.generation.catalogs import SURFACE_TEMPLATE_SETS
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import (
    action_id,
    event_id,
    paraphrase_group_id,
    query_id,
    source_id,
    stable_id,
    task_id,
)
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.profiles import (
    REGISTERED_PROFILE_PARAMETER_KEYS,
    ProfileSpec,
    build_generic_profile,
    resolve_profile,
)
from mub.vnext.validation.replay import replay_actions, validate_gold_replay
from mub.vnext.validation.task import validate_task
from mub.vnext.version import COMPILER_VERSION


_GENERATOR_NAME = "memupdatebench_vnext_pilot"
_CODE_REVISION = "vnext-pilot-task-2c"
_NORMALIZATION_VERSION = "vnext-pilot-semantic-v1"
_SPLIT_POLICY_VERSION = "vnext-pilot-core-v1"
_SPEAKERS = ("Narrator", "User", "Records clerk")


class _CanonicalPayload(RootModel[Any]):
    pass


def _payload_sha256(payload: object) -> str:
    return sha256_model(_CanonicalPayload(root=payload))


def _copy_key(key: MemoryObjectKey) -> MemoryObjectKey:
    return MemoryObjectKey.model_validate(key.model_dump(mode="python"))


def _identity(key: MemoryObjectKey) -> str:
    return key.canonical_id


def _plain(value: Any) -> Any:
    return thaw_json(value)


def _display_entity(entity: str, surface_variant: int) -> str:
    normalized = entity.replace("_", " ")
    if ":" not in normalized:
        return normalized
    relation, name = normalized.split(":", 1)
    if surface_variant == 0:
        return f"{relation} {name}"
    if surface_variant == 1:
        return f"{name}, the {relation}"
    return f"my {relation} {name}"


def _event_template_values(
    event: CoreEvent,
    surface_variant: int,
) -> tuple[str, str, str]:
    if event.object_keys:
        entities = " and ".join(
            _display_entity(key.entity, surface_variant) for key in event.object_keys
        )
        attributes = " and ".join(
            key.attribute.replace("_", " ") for key in event.object_keys
        )
    else:
        entities = "the referenced record"
        attributes = "setting"

    if event.operation == Operation.NOOP:
        rendered_value = "unchanged"
    elif event.operation == Operation.DELETE:
        rendered_value = "absent"
    elif isinstance(event.value, str):
        rendered_value = event.value
    else:
        rendered_value = json.dumps(
            _plain(event.value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return entities, attributes, rendered_value


def _normalized_event_text(event: CoreEvent) -> str:
    if event.operation == Operation.NOOP:
        return "No memory object changes."
    targets = ", ".join(key.canonical_id for key in event.object_keys)
    if event.operation == Operation.DELETE:
        return f"Delete {targets}."
    value = json.dumps(
        _plain(event.value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{event.operation.value.title()} {targets} with value {value}."


def _answer_schema(value: Any) -> AnswerSchema:
    if isinstance(value, bool):
        return AnswerSchema.BOOLEAN
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return AnswerSchema.NUMBER
    if isinstance(value, str):
        return AnswerSchema.STRING
    if isinstance(value, Mapping):
        return AnswerSchema.OBJECT
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return AnswerSchema.LIST
    raise ValueError(
        "render_core could not assign an answer schema to the replayed query answer"
    )


def _same_json(left: Any, right: Any) -> bool:
    return canonical_json_bytes(_CanonicalPayload(root=_plain(left))) == (
        canonical_json_bytes(_CanonicalPayload(root=_plain(right)))
    )


def _query_semantics(
    core: SemanticCore,
    replay: Any,
) -> tuple[QueryType, Any, AnswerSchema]:
    target_ids = [key.canonical_id for key in core.query_targets]
    present = [target_id in replay.final_state for target_id in target_ids]
    expected = _plain(core.expected_answer)

    if all(present):
        values = [_plain(replay.final_state[target_id]) for target_id in target_ids]
        if len(values) == 1:
            query_type = QueryType.CURRENT_STATE
            answer = values[0]
        else:
            query_type = QueryType.MULTI_OBJECT
            if isinstance(expected, Mapping):
                answer = {
                    target_id: value for target_id, value in zip(target_ids, values)
                }
            else:
                answer = values
    elif not any(present):
        query_type = QueryType.DELETION_COMPLIANCE
        if core.expected_answer is None:
            answer = True
        elif isinstance(expected, bool):
            answer = all(not item for item in present)
        elif isinstance(expected, str):
            answer = "absent"
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            answer = sum(not item for item in present)
        elif isinstance(expected, Mapping):
            answer = {target_id: True for target_id in target_ids}
        elif isinstance(expected, Sequence) and not isinstance(
            expected, (str, bytes, bytearray)
        ):
            answer = [True for _ in target_ids]
        else:
            raise ValueError(
                "render_core could not derive an expected absence answer"
            )
    else:
        raise ValueError(
            "render_core query targets cannot mix present and absent replay states"
        )

    if core.expected_answer is not None and not _same_json(expected, answer):
        raise ValueError(
            "render_core core expected_answer does not equal the replayed query answer"
        )
    return query_type, answer, _answer_schema(answer)


def _query_template_values(
    core: SemanticCore,
    surface_variant: int,
) -> tuple[str, str]:
    entities = " and ".join(
        _display_entity(key.entity, surface_variant) for key in core.query_targets
    )
    attributes = " and ".join(
        key.attribute.replace("_", " ") for key in core.query_targets
    )
    return entities, attributes


def _gold_source_event_ids(
    core: SemanticCore,
    rendered_event_ids: list[str],
) -> list[str]:
    latest_by_target: dict[str, int] = {}
    query_identities = {_identity(key) for key in core.query_targets}
    for index, event in enumerate(core.events):
        if event.operation == Operation.NOOP:
            continue
        for key in event.object_keys:
            identity = _identity(key)
            if identity in query_identities:
                latest_by_target[identity] = index
    return [
        rendered_event_ids[index]
        for index in sorted(set(latest_by_target.values()))
    ]


def _target_objects(core: SemanticCore) -> list[MemoryObjectKey]:
    targets: list[MemoryObjectKey] = []
    seen: set[str] = set()
    for event in core.events:
        for key in event.object_keys:
            identity = _identity(key)
            if identity not in seen:
                targets.append(_copy_key(key))
                seen.add(identity)
    for key in core.query_targets:
        identity = _identity(key)
        if identity not in seen:
            targets.append(_copy_key(key))
            seen.add(identity)
    return targets


def _resolve_core_profile(core: SemanticCore) -> Mapping[str, Any]:
    base = build_generic_profile(core.difficulty, core.task_family.value)
    overrides = _plain(core.profile)
    added_keys = sorted(
        (set(overrides) - set(base.parameters))
        & REGISTERED_PROFILE_PARAMETER_KEYS
    )
    if not added_keys:
        return resolve_profile(base, overrides)

    parameters = {key: _plain(value) for key, value in base.parameters.items()}
    parameters.update({key: overrides[key] for key in added_keys})
    extended = ProfileSpec(
        name=base.name,
        version=base.version,
        task_family=base.task_family,
        difficulty=base.difficulty,
        parameters=parameters,
        allowed_overrides=tuple(sorted((*base.allowed_overrides, *added_keys))),
    )
    return resolve_profile(extended, overrides)


def _report_error(stage: str, report: Any) -> ValueError:
    detail = "; ".join(
        f"{issue.code}@{issue.path}: {issue.message}" for issue in report.issues
    )
    return ValueError(f"render_core {stage} validation failed: {detail}")


def render_core(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
) -> MemUpdateTask:
    if not isinstance(core, SemanticCore):
        raise TypeError("core must be a SemanticCore")
    if not isinstance(split, Split):
        raise TypeError("split must be a Split")
    if type(surface_variant) is not int:
        raise TypeError("surface_variant must be an integer")
    if surface_variant not in (0, 1, 2):
        raise ValueError("surface_variant must be one of 0, 1, 2")

    template_name, event_template, query_template = SURFACE_TEMPLATE_SETS[
        surface_variant
    ]
    rendered_task_id = task_id(core.core_id, surface_variant)
    rendered_event_ids = [
        event_id(rendered_task_id, index) for index in range(len(core.events))
    ]
    rendered_action_ids = [
        action_id(rendered_task_id, index, 0) for index in range(len(core.events))
    ]
    rendered_query_id = query_id(rendered_task_id, 0)
    rendered_source_id = source_id(
        "vnext_pilot",
        core.core_index,
        {
            "semantic_core_id": core.core_id,
            "surface_variant": surface_variant,
        },
    )

    source_group = stable_id(
        "source_group", {"semantic_core_id": core.core_id}
    )
    paraphrase_group = paraphrase_group_id(core.core_id, "surface_variants")
    source_document = stable_id(
        "source_document", {"semantic_core_id": core.core_id}
    )
    version_group = stable_id(
        "version_group", {"trajectory_id": core.trajectory_id}
    )

    actions: list[GoldAction] = []
    events: list[MemoryEvent] = []
    for index, core_event in enumerate(core.events):
        action = GoldAction(
            action_id=rendered_action_ids[index],
            event_id=rendered_event_ids[index],
            operation=core_event.operation,
            scope=(
                ActionScope.OBJECT
                if core_event.operation == Operation.NOOP
                else ActionScope.ATTRIBUTE
            ),
            target_object_keys=[_copy_key(key) for key in core_event.object_keys],
            value=_plain(core_event.value),
            effective_at=None,
            expected_effect={},
        )
        entity, attribute, value = _event_template_values(
            core_event, surface_variant
        )
        raw_text = event_template.format(
            entity=entity,
            attribute=attribute,
            value=value,
        )
        metadata = _plain(core_event.metadata)
        metadata.update(
            {
                "surface_template": template_name,
                "surface_variant": surface_variant,
            }
        )
        event = MemoryEvent(
            event_id=rendered_event_ids[index],
            sequence_index=index,
            timestamp=None,
            raw_text=raw_text,
            normalized_text=_normalized_event_text(core_event),
            speaker=_SPEAKERS[surface_variant],
            gold_action_ids=[rendered_action_ids[index]],
            role=core_event.role,
            source_anchor={
                "semantic_core_id": core.core_id,
                "source_document_id": source_document,
                "event_index": index,
            },
            metadata=metadata,
        )
        actions.append(action)
        events.append(event)

    try:
        replay = replay_actions(actions)
    except ValueError as exc:
        raise ValueError(f"render_core gold replay failed: {exc}") from exc

    query_type, answer, answer_schema = _query_semantics(core, replay)
    query_entity, query_attribute = _query_template_values(core, surface_variant)
    query_text = query_template.format(
        entity=query_entity,
        attribute=query_attribute,
    )
    query = MemoryQuery(
        query_id=rendered_query_id,
        query_type=query_type,
        text=query_text,
        target_object_keys=[_copy_key(key) for key in core.query_targets],
        answer_schema=answer_schema,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
        metadata={
            "surface_template": template_name,
            "surface_variant": surface_variant,
        },
    )

    targets = _target_objects(core)
    expected_present = [
        _copy_key(key) for key in targets if key.canonical_id in replay.final_state
    ]
    expected_absent = [
        _copy_key(key) for key in targets if key.canonical_id not in replay.final_state
    ]
    gold = GoldRecord(
        actions=actions,
        action_sequence=list(rendered_action_ids),
        final_state=_plain(replay.final_state),
        version_history=_plain(replay.version_history),
        expected_present_objects=expected_present,
        expected_absent_objects=expected_absent,
        gold_source_event_ids=_gold_source_event_ids(
            core, rendered_event_ids
        ),
        gold_answers={rendered_query_id: _plain(answer)},
        acceptable_answers={rendered_query_id: _plain(answer)},
    )

    resolved_profile = _resolve_core_profile(core)
    generation_config_hash = _payload_sha256(
        {
            "generator_name": _GENERATOR_NAME,
            "compiler_version": COMPILER_VERSION,
            "profile": _plain(core.profile),
            "surface_template_sets": SURFACE_TEMPLATE_SETS,
            "split_policy_version": _SPLIT_POLICY_VERSION,
        }
    )
    normalized_source_hash = _payload_sha256(
        {
            "semantic_core": core.model_dump(mode="json"),
            "source_document_id": source_document,
        }
    )
    raw_source_hash = _payload_sha256(
        {
            "events": [
                {
                    "raw_text": event.raw_text,
                    "speaker": event.speaker,
                }
                for event in events
            ],
            "query_text": query.text,
        }
    )

    source = SourceRecord(
        source_id=rendered_source_id,
        source_type=SourceType.SYNTHETIC,
        source_uri=f"memory://{rendered_source_id}",
        license_or_privacy="synthetic_redistributable",
        raw_hash=raw_source_hash,
        normalized_hash=normalized_source_hash,
        normalization_version=_NORMALIZATION_VERSION,
        provenance={
            "redistributable": True,
            "license": "synthetic_redistributable",
            "semantic_core_id": core.core_id,
            "trajectory_id": core.trajectory_id,
            "source_group_id": source_group,
            "paraphrase_group_id": paraphrase_group,
            "source_document_id": source_document,
            "version_group_id": version_group,
            "surface_variant": surface_variant,
            "surface_template": template_name,
        },
        generator=GeneratorProvenance(
            generator_name=_GENERATOR_NAME,
            seed=core.core_index,
            config_sha256=generation_config_hash,
            code_revision=_CODE_REVISION,
            compiler_version=COMPILER_VERSION,
        ),
    )

    task = MemUpdateTask(
        task_id=rendered_task_id,
        task_family=core.task_family.value,
        difficulty=core.difficulty,
        source=source,
        events=events,
        target_objects=targets,
        queries=[query],
        gold=gold,
        metadata=TaskMetadata(
            split=split,
            split_key=SplitKey(
                semantic_core_id=core.core_id,
                source_group_id=source_group,
                trajectory_id=core.trajectory_id,
                paraphrase_group_id=paraphrase_group,
                source_document_id=source_document,
                version_group_id=version_group,
                split_exception_id=None,
                split_policy_version=_SPLIT_POLICY_VERSION,
            ),
            profile_name=core.difficulty,
            resolved_profile=_plain(resolved_profile),
            generation_config_hash=generation_config_hash,
            compiler_version=COMPILER_VERSION,
            tags=[
                "vnext_pilot",
                core.task_family.value,
                core.difficulty.value,
                template_name,
            ],
            extra={
                "semantic_core_id": core.core_id,
                "core_index": core.core_index,
                "surface_variant": surface_variant,
                "surface_template": template_name,
                "stratification": _plain(core.stratification),
            },
        ),
    )

    structural_report = validate_task(task)
    if not structural_report.valid:
        raise _report_error("structural", structural_report)
    replay_report = validate_gold_replay(task)
    if not replay_report.valid:
        raise _report_error("gold replay", replay_report)
    return task


__all__ = ["render_core"]
