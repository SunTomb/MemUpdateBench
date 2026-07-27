from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from string import Template
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
    EventRole,
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
from mub.vnext.generation.core import CoreEvent, GenerationContext, SemanticCore
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


_NORMALIZATION_VERSION = "vnext-pilot-semantic-v1"
_SPLIT_POLICY_VERSION = "vnext-pilot-core-v1"
_SPEAKERS = ("Narrator", "User", "Records clerk")
_RENDERER_METADATA_KEY = "__surface_renderer__"
_NOOP_ROLE_FALLBACKS = {
    EventRole.LATEST_GOLD: "The latest-gold statement does not direct a memory change.",
    EventRole.STALE_SAME_SLOT: (
        "The stale same-slot statement does not direct a memory change."
    ),
    EventRole.DUPLICATE_CURRENT: (
        "The statement repeats current information without changing memory."
    ),
    EventRole.SAME_ENTITY_OTHER_ATTRIBUTE: (
        "The same-entity other-attribute statement does not direct a memory change."
    ),
    EventRole.SAME_NAME_OTHER_ENTITY: (
        "The same-name other-entity statement does not direct a memory change."
    ),
    EventRole.NOOP_NEAR_MISS: (
        "The related near-miss statement does not change any stored value."
    ),
    EventRole.NEUTRAL: "The statement is informational and does not change memory.",
    EventRole.DELETION: "The deletion-related statement does not direct a deletion.",
    EventRole.HISTORICAL_SUPPORT: (
        "The historical-support statement does not change current memory."
    ),
}


class _CanonicalPayload(RootModel[Any]):
    pass


def _payload_sha256(payload: object) -> str:
    return sha256_model(_CanonicalPayload(root=payload))


def _copy_key(key: MemoryObjectKey) -> MemoryObjectKey:
    return MemoryObjectKey.model_validate(key.model_dump(mode="python"))


def _identity(key: MemoryObjectKey) -> str:
    return key.canonical_id


def _semantic_object_identity(key: MemoryObjectKey) -> dict[str, str | None]:
    return {
        "namespace": key.namespace,
        "entity": key.entity,
        "attribute": key.attribute,
        "subkey": key.subkey,
    }


def _normalized_source_semantic_projection(core: SemanticCore) -> dict[str, Any]:
    return {
        "events": [
            {
                "operation": event.operation.value,
                "target_object_keys": [
                    _semantic_object_identity(key) for key in event.object_keys
                ],
                "value": _plain(event.value),
                "role": event.role.value,
                "metadata": _plain(event.metadata),
            }
            for event in core.events
        ]
    }


def _plain(value: Any) -> Any:
    return thaw_json(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_object_reference(key: MemoryObjectKey) -> str:
    return (
        f"object(namespace={_json_text(key.namespace)}, "
        f"entity={_json_text(key.entity)}, "
        f"attribute={_json_text(key.attribute)}, "
        f"subkey={_json_text(key.subkey)})"
    )


def _atomic_object_references(keys: Sequence[MemoryObjectKey]) -> str:
    return "; ".join(_atomic_object_reference(key) for key in keys)


def _render_event_text(
    event: CoreEvent,
    operation_templates: Mapping[Operation, str],
) -> str:
    template = Template(operation_templates[event.operation])
    if event.operation == Operation.NOOP:
        statement = event.metadata.get("surface_statement")
        if statement is None:
            statement = _NOOP_ROLE_FALLBACKS[event.role]
        return template.substitute(statement=statement)
    substitutions = {"targets": _atomic_object_references(event.object_keys)}
    if event.operation in {Operation.ADD, Operation.UPDATE}:
        substitutions["value"] = _json_text(event.value)
    return template.substitute(substitutions)


def _normalized_event_text(event: CoreEvent) -> str:
    if event.operation == Operation.NOOP:
        return "No memory object changes."
    targets = ", ".join(key.canonical_id for key in event.object_keys)
    if event.operation == Operation.DELETE:
        return f"Delete {targets}."
    value = _json_text(event.value)
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
        if core.expected_answer is None:
            raise ValueError(
                "render_core expected_answer is required when query targets are present"
            )
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
    else:
        query_type = QueryType.DELETION_COMPLIANCE
        if core.expected_answer is None:
            if any(present):
                raise ValueError(
                    "render_core expected_answer is required for mixed present and absent query targets"
                )
            answer = True
        elif isinstance(expected, bool):
            answer = all(not item for item in present)
        elif isinstance(expected, str):
            answer = "absent" if not any(present) else "mixed"
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            answer = sum(not item for item in present)
        elif isinstance(expected, Mapping):
            answer = {
                target_id: not is_present
                for target_id, is_present in zip(target_ids, present)
            }
        elif isinstance(expected, Sequence) and not isinstance(
            expected, (str, bytes, bytearray)
        ):
            answer = [not item for item in present]
        else:
            raise ValueError("render_core could not derive an expected absence answer")

    if core.expected_answer is not None and not _same_json(expected, answer):
        raise ValueError(
            "render_core core expected_answer does not equal the replayed query answer"
        )
    return query_type, answer, _answer_schema(answer)


def _select_query_template(
    query_type: QueryType,
    answer_schema: AnswerSchema,
    current_template: str,
    deletion_template: str,
) -> str:
    if query_type is QueryType.DELETION_COMPLIANCE:
        if answer_schema not in {
            AnswerSchema.BOOLEAN,
            AnswerSchema.NUMBER,
            AnswerSchema.STRING,
            AnswerSchema.LIST,
            AnswerSchema.OBJECT,
        }:
            raise ValueError("render_core deletion query has unsupported answer schema")
        return deletion_template
    return current_template


def _render_query_text(core: SemanticCore, query_template: str) -> str:
    return Template(query_template).substitute(
        targets=_atomic_object_references(core.query_targets)
    )


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


def _resolve_core_profile(
    core: SemanticCore,
    query_type: QueryType,
) -> Mapping[str, Any]:
    base = build_generic_profile(core.difficulty, core.task_family.value)
    overrides = _plain(core.profile)
    explicit_query_type = overrides.get("query_type")
    if explicit_query_type is not None and explicit_query_type != query_type.value:
        raise ValueError(
            "render_core profile query_type conflicts with derived "
            f"query_type {query_type.value!r}"
        )
    overrides["query_type"] = query_type.value
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
    context: GenerationContext,
) -> MemUpdateTask:
    if not isinstance(core, SemanticCore):
        raise TypeError("core must be a SemanticCore")
    if not isinstance(context, GenerationContext):
        raise TypeError("context must be a GenerationContext")
    if not isinstance(split, Split):
        raise TypeError("split must be a Split")
    if type(surface_variant) is not int:
        raise TypeError("surface_variant must be an integer")
    if surface_variant not in (0, 1, 2):
        raise ValueError("surface_variant must be one of 0, 1, 2")

    (
        template_name,
        add_template,
        update_template,
        delete_template,
        noop_template,
        current_query_template,
        deletion_query_template,
    ) = SURFACE_TEMPLATE_SETS[surface_variant]
    operation_templates = {
        Operation.ADD: add_template,
        Operation.UPDATE: update_template,
        Operation.DELETE: delete_template,
        Operation.NOOP: noop_template,
    }
    renderer_admin = {
        "surface_template": template_name,
        "surface_variant": surface_variant,
    }

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
        if _RENDERER_METADATA_KEY in core_event.metadata:
            raise ValueError(
                f"render_core event {index} uses reserved renderer metadata key "
                f"'{_RENDERER_METADATA_KEY}'"
            )
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
        raw_text = _render_event_text(core_event, operation_templates)
        metadata = _plain(core_event.metadata)
        metadata[_RENDERER_METADATA_KEY] = dict(renderer_admin)
        event = MemoryEvent(
            event_id=rendered_event_ids[index],
            sequence_index=index,
            timestamp=None,
            raw_text=raw_text,
            normalized_text=_normalized_event_text(core_event),
            speaker=_SPEAKERS[surface_variant],
            gold_action_ids=[rendered_action_ids[index]],
            role=core_event.role,
            source_anchor={"event_index": index},
            metadata=metadata,
        )
        actions.append(action)
        events.append(event)

    try:
        replay = replay_actions(actions)
    except ValueError as exc:
        raise ValueError(f"render_core gold replay failed: {exc}") from exc

    query_type, answer, answer_schema = _query_semantics(core, replay)
    query_template = _select_query_template(
        query_type,
        answer_schema,
        current_query_template,
        deletion_query_template,
    )
    query_text = _render_query_text(core, query_template)
    query = MemoryQuery(
        query_id=rendered_query_id,
        query_type=query_type,
        text=query_text,
        target_object_keys=[_copy_key(key) for key in core.query_targets],
        answer_schema=answer_schema,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
        metadata={_RENDERER_METADATA_KEY: dict(renderer_admin)},
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

    resolved_profile = _resolve_core_profile(core, query_type)
    normalized_source_hash = _payload_sha256(
        _normalized_source_semantic_projection(core)
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
            "release_id": context.release_id,
            "schema_version": context.schema_version,
            "profile_version": context.profile_version,
        },
        generator=GeneratorProvenance(
            generator_name=context.generator_name,
            seed=context.seed,
            config_sha256=context.config_sha256,
            code_revision=context.code_revision,
            compiler_version=context.compiler_version,
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
            generation_config_hash=context.config_sha256,
            compiler_version=context.compiler_version,
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
