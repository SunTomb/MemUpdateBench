from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    AnswerDisposition,
    AnswerSchema,
    EvaluationMode,
    EventRole,
    Operation,
    QueryType,
    SourceType,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.task import (
    CanonicalAnswer,
    GoldAction,
    GoldRecord,
    MemUpdateTask,
    MemoryEvent,
    MemoryQuery,
    ReferenceCandidate,
    SplitKey,
    SurfaceReference,
    TaskMetadata,
)
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.generation.catalogs import REFERENCE_QUERY_TEMPLATE_SETS, SURFACE_TEMPLATE_SETS
from mub.vnext.generation.core import CoreEvent, GenerationContext, SemanticCore
from mub.vnext.generation.surface_catalog import SurfaceCatalog
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


_NORMALIZATION_VERSION = "vnext-pilot-semantic-v2"
_SPLIT_POLICY_VERSION = "vnext-pilot-core-v2"
_SPEAKERS = ("Narrator", "User", "Records clerk")
PILOT_SURFACE_CATALOG = SurfaceCatalog(
    catalog_version="vnext-pilot-surfaces-v1",
    template_sets=SURFACE_TEMPLATE_SETS,
    reference_query_template_sets=REFERENCE_QUERY_TEMPLATE_SETS,
    speakers=_SPEAKERS,
    source_namespace="vnext_pilot",
    task_tag="vnext_pilot",
    normalization_version=_NORMALIZATION_VERSION,
    split_policy_version=_SPLIT_POLICY_VERSION,
)
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


@dataclass(frozen=True, slots=True)
class _RenderPlan:
    request_key: bytes
    task_bytes: bytes
    task_sha256: str


@dataclass(frozen=True, slots=True)
class _RenderedTask:
    task: MemUpdateTask
    plan: _RenderPlan


def _payload_sha256(payload: object) -> str:
    return sha256_model(_CanonicalPayload(root=payload))


def _copy_key(key: MemoryObjectKey) -> MemoryObjectKey:
    return MemoryObjectKey.model_validate(key.model_dump(mode="python"))


def _copy_reference_candidate(candidate: ReferenceCandidate) -> ReferenceCandidate:
    return ReferenceCandidate.model_validate(candidate.model_dump(mode="python"))


def _copy_surface_reference(reference: SurfaceReference) -> SurfaceReference:
    return SurfaceReference.model_validate(reference.model_dump(mode="python"))


def _copy_canonical_answer(answer: CanonicalAnswer) -> CanonicalAnswer:
    return CanonicalAnswer.model_validate(answer.model_dump(mode="python"))


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
    if core.query_type is QueryType.UNRESOLVED_REFERENCE:
        canonical = core.canonical_answer
        if canonical is None:
            raise ValueError(
                "render_core unresolved query requires a canonical_answer"
            )
        if canonical.disposition is AnswerDisposition.ABSTAINED:
            return QueryType.UNRESOLVED_REFERENCE, None, AnswerSchema.STRING
        selected_candidate_id = canonical.selected_candidate_ids[0]
        selected_candidate = next(
            candidate
            for candidate in core.reference_candidates
            if candidate.candidate_id == selected_candidate_id
        )
        selected_key = selected_candidate.object_key.canonical_id
        if selected_key not in replay.final_state:
            raise ValueError(
                "render_core unresolved UNIQUE selected candidate is absent "
                "after gold replay"
            )
        current_value = _plain(replay.final_state[selected_key])
        if not _same_json(canonical.value, current_value):
            raise ValueError(
                "render_core unresolved UNIQUE canonical answer value does not "
                "equal the selected candidate replayed current value"
            )
        return (
            QueryType.UNRESOLVED_REFERENCE,
            _plain(canonical.value),
            _answer_schema(canonical.value),
        )

    target_ids = [key.canonical_id for key in core.query_targets]
    present = [target_id in replay.final_state for target_id in target_ids]
    expected = _plain(core.expected_answer)

    if core.task_family is TaskFamily.CURRENT_HISTORICAL_QUERY:
        from mub.vnext.generation.family_f import resolve_family_f_core_selector

        resolution = resolve_family_f_core_selector(core)
        return (
            resolution.core_query_type,
            _plain(resolution.answer),
            resolution.answer_schema,
        )

    if core.task_family is TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS:
        if core.expected_answer is None:
            raise ValueError("render_core Family G requires a synthesis answer")
        return core.query_type, expected, _answer_schema(expected)

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
    deletion_templates: Mapping[AnswerSchema, str],
) -> str:
    if query_type is QueryType.DELETION_COMPLIANCE:
        try:
            return deletion_templates[answer_schema]
        except KeyError as exc:
            raise ValueError(
                "render_core deletion query has unsupported answer schema"
            ) from exc
    return current_template


def _render_query_text(core: SemanticCore, query_template: str) -> str:
    return Template(query_template).substitute(
        targets=_atomic_object_references(core.query_targets)
    )


def _render_reference_candidates(core: SemanticCore) -> str:
    rendered = []
    for index, candidate in enumerate(core.reference_candidates, start=1):
        rendered.append(f"{index}. {_atomic_object_reference(candidate.object_key)}")
    return " ".join(rendered)


def _render_surface_references(core: SemanticCore) -> str:
    rendered = [
        f"{index}. {_json_text(reference.surface_text)}"
        for index, reference in enumerate(core.surface_references, start=1)
    ]
    return " ".join(rendered)


def _render_unresolved_query_text(
    core: SemanticCore,
    surface_variant: int,
    surface_catalog: SurfaceCatalog = PILOT_SURFACE_CATALOG,
) -> str:
    (
        _,
        query_template,
        resolution_instruction,
        abstention_instruction,
    ) = surface_catalog.reference_query_template_sets[surface_variant]
    return Template(query_template).substitute(
        candidates=_render_reference_candidates(core),
        references=_render_surface_references(core),
        resolution_instruction=resolution_instruction,
        abstention_instruction=abstention_instruction,
    )


def _gold_source_event_ids(
    core: SemanticCore,
    rendered_event_ids: list[str],
) -> list[str]:
    if core.task_family is TaskFamily.CURRENT_HISTORICAL_QUERY:
        from mub.vnext.generation.family_f import bind_family_f_core_selector

        _, resolution, _, _ = bind_family_f_core_selector(
            core, rendered_event_ids
        )
        return [rendered_event_ids[index] for index in resolution.selected_indices]
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
    for candidate in core.reference_candidates:
        key = candidate.object_key
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


def _build_render_plan(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
    context: GenerationContext,
    surface_catalog: SurfaceCatalog = PILOT_SURFACE_CATALOG,
) -> _RenderPlan:
    if not isinstance(core, SemanticCore):
        raise TypeError("core must be a SemanticCore")
    if not isinstance(context, GenerationContext):
        raise TypeError("context must be a GenerationContext")
    if not isinstance(surface_catalog, SurfaceCatalog):
        raise TypeError("surface_catalog must be a SurfaceCatalog")
    if not isinstance(split, Split):
        raise TypeError("split must be a Split")
    if type(surface_variant) is not int:
        raise TypeError("surface_variant must be an integer")
    if surface_variant < 0 or surface_variant >= surface_catalog.surface_count:
        allowed_variants = ", ".join(
            str(index) for index in range(surface_catalog.surface_count)
        )
        raise ValueError(
            f"surface_variant must be one of {allowed_variants}"
        )

    (
        template_name,
        add_template,
        update_template,
        delete_template,
        noop_template,
        current_query_template,
        deletion_boolean_template,
        deletion_number_template,
        deletion_sequence_template,
        deletion_string_template,
    ) = surface_catalog.template_sets[surface_variant]
    operation_templates = {
        Operation.ADD: add_template,
        Operation.UPDATE: update_template,
        Operation.DELETE: delete_template,
        Operation.NOOP: noop_template,
    }
    deletion_templates = {
        AnswerSchema.BOOLEAN: deletion_boolean_template,
        AnswerSchema.NUMBER: deletion_number_template,
        AnswerSchema.LIST: deletion_sequence_template,
        AnswerSchema.OBJECT: deletion_sequence_template,
        AnswerSchema.STRING: deletion_string_template,
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
        surface_catalog.source_namespace,
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
            speaker=surface_catalog.speakers[surface_variant],
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
    if query_type is QueryType.UNRESOLVED_REFERENCE:
        query_text = _render_unresolved_query_text(
            core,
            surface_variant,
            surface_catalog,
        )
    else:
        query_template = _select_query_template(
            query_type,
            answer_schema,
            current_query_template,
            deletion_templates,
        )
        query_text = _render_query_text(core, query_template)
    query_metadata = {_RENDERER_METADATA_KEY: dict(renderer_admin)}
    if core.task_family is TaskFamily.CURRENT_HISTORICAL_QUERY:
        from mub.vnext.generation.family_f import (
            FAMILY_F_QUERY_TEMPLATE,
            bind_family_f_core_selector,
            family_f_query_tokens,
        )

        selector, family_f_resolution, selected, entries = (
            bind_family_f_core_selector(core, rendered_event_ids)
        )
        query_text = _render_query_text(core, FAMILY_F_QUERY_TEMPLATE)
        query_text += " [" + "; ".join(
            family_f_query_tokens(selector, selected, entries)
        ) + "]"
        if family_f_resolution.task_query_type is QueryTypeV3.ORDERED_HISTORY:
            query_metadata["history_start_version_index"] = (
                family_f_resolution.selected_indices[0]
            )
            query_metadata["history_end_version_index"] = (
                family_f_resolution.selected_indices[-1]
            )
        elif family_f_resolution.core_query_type is QueryType.HISTORICAL_STATE:
            query_metadata["version_index"] = family_f_resolution.selected_indices[-1]
        elif family_f_resolution.core_query_type is QueryType.TRANSITION:
            query_metadata["from_version_index"] = family_f_resolution.selected_indices[0]
            query_metadata["to_version_index"] = family_f_resolution.selected_indices[-1]
    query = MemoryQuery(
        query_id=rendered_query_id,
        query_type=query_type,
        text=query_text,
        target_object_keys=[_copy_key(key) for key in core.query_targets],
        reference_candidates=[
            _copy_reference_candidate(candidate)
            for candidate in core.reference_candidates
        ],
        surface_references=[
            _copy_surface_reference(reference)
            for reference in core.surface_references
        ],
        answer_schema=answer_schema,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
        metadata=query_metadata,
    )

    targets = _target_objects(core)
    expected_present = [
        _copy_key(key) for key in targets if key.canonical_id in replay.final_state
    ]
    expected_absent = [
        _copy_key(key) for key in targets if key.canonical_id not in replay.final_state
    ]
    if query_type is QueryType.UNRESOLVED_REFERENCE:
        if core.canonical_answer is None:
            raise ValueError(
                "render_core unresolved query requires a canonical_answer"
            )
        gold_answers: dict[str, Any] = {}
        acceptable_answers: dict[str, Any] = {}
        canonical_answers = {
            rendered_query_id: _copy_canonical_answer(core.canonical_answer)
        }
    else:
        gold_answers = {rendered_query_id: _plain(answer)}
        acceptable_answers = {rendered_query_id: _plain(answer)}
        canonical_answers = {}

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
        gold_answers=gold_answers,
        acceptable_answers=acceptable_answers,
        canonical_answers=canonical_answers,
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
        normalization_version=surface_catalog.normalization_version,
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

    metadata = TaskMetadata(
        split=split,
        split_key=SplitKey(
            semantic_core_id=core.core_id,
            source_group_id=source_group,
            trajectory_id=core.trajectory_id,
            paraphrase_group_id=paraphrase_group,
            source_document_id=source_document,
            version_group_id=version_group,
            split_exception_id=None,
            split_policy_version=surface_catalog.split_policy_version,
        ),
        profile_name=core.difficulty,
        resolved_profile=_plain(resolved_profile),
        generation_config_hash=context.config_sha256,
        compiler_version=context.compiler_version,
        tags=[
            surface_catalog.task_tag,
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
    )
    payload = {
        "task_id": rendered_task_id,
        "schema_version": context.schema_version,
        "task_family": core.task_family.value,
        "difficulty": core.difficulty.value,
        "source": source.model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in events],
        "target_objects": [target.model_dump(mode="json") for target in targets],
        "queries": [query.model_dump(mode="json")],
        "gold": gold.model_dump(mode="json"),
        "metadata": metadata.model_dump(mode="json"),
    }
    task_bytes = canonical_json_bytes(_CanonicalPayload(root=payload))
    request_key = canonical_json_bytes(
        _CanonicalPayload(
            root={
                "semantic_core_id": core.core_id,
                "task_family": core.task_family.value,
                "difficulty": core.difficulty.value,
                "split": split.value,
                "surface_variant": surface_variant,
                "surface_catalog_version": surface_catalog.catalog_version,
                "config_sha256": context.config_sha256,
                "code_revision": context.code_revision,
                "compiler_version": context.compiler_version,
                "generator_name": context.generator_name,
            }
        )
    )
    return _RenderPlan(
        request_key=request_key,
        task_bytes=task_bytes,
        task_sha256=hashlib.sha256(task_bytes).hexdigest(),
    )


def _construct_core_task(plan: _RenderPlan) -> MemUpdateTask:
    return MemUpdateTask.model_validate_json(plan.task_bytes)


def _render_envelope_issues(
    envelope: _RenderedTask,
    expected_plan: _RenderPlan,
) -> tuple[str, ...]:
    issues = []
    if not isinstance(envelope, _RenderedTask):
        return ("renderer did not return a _RenderedTask envelope",)
    if envelope.plan.request_key != expected_plan.request_key:
        issues.append("render plan request key mismatch")
    if envelope.plan.task_bytes != expected_plan.task_bytes:
        issues.append("render plan task bytes mismatch")
    if envelope.plan.task_sha256 != expected_plan.task_sha256:
        issues.append("render plan task SHA-256 mismatch")
    current_bytes = canonical_json_bytes(envelope.task)
    if current_bytes != expected_plan.task_bytes:
        issues.append("rendered task bytes mismatch")
    current_sha256 = hashlib.sha256(current_bytes).hexdigest()
    if current_sha256 != expected_plan.task_sha256:
        issues.append("rendered task SHA-256 mismatch")
    return tuple(issues)


def _render_core_unvalidated(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
    context: GenerationContext,
    plan: _RenderPlan | None = None,
    surface_catalog: SurfaceCatalog = PILOT_SURFACE_CATALOG,
) -> _RenderedTask:
    resolved_plan = plan or _build_render_plan(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        surface_catalog=surface_catalog,
    )
    task = _construct_core_task(resolved_plan)
    envelope = _RenderedTask(task=task, plan=resolved_plan)
    integrity_issues = _render_envelope_issues(envelope, resolved_plan)
    if integrity_issues:
        raise ValueError(
            "render construction integrity failed: "
            + "; ".join(integrity_issues)
        )
    return envelope


def _expected_render_plan(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
    context: GenerationContext,
    surface_catalog: SurfaceCatalog = PILOT_SURFACE_CATALOG,
) -> _RenderPlan:
    return _build_render_plan(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        surface_catalog=surface_catalog,
    )


def render_core_with_catalog(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
    context: GenerationContext,
    surface_catalog: SurfaceCatalog,
) -> MemUpdateTask:
    expected_plan = _expected_render_plan(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        surface_catalog=surface_catalog,
    )
    envelope = _render_core_unvalidated(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        plan=expected_plan,
        surface_catalog=surface_catalog,
    )
    integrity_issues = _render_envelope_issues(envelope, expected_plan)
    if integrity_issues:
        raise ValueError(
            "render_core integrity validation failed: "
            + "; ".join(integrity_issues)
        )
    task = envelope.task
    structural_report = validate_task(task)
    if not structural_report.valid:
        raise _report_error("structural", structural_report)
    replay_report = validate_gold_replay(task)
    if not replay_report.valid:
        raise _report_error("gold replay", replay_report)
    return task


def render_core(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
    context: GenerationContext,
) -> MemUpdateTask:
    expected_plan = _expected_render_plan(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
    )
    envelope = _render_core_unvalidated(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        plan=expected_plan,
    )
    integrity_issues = _render_envelope_issues(envelope, expected_plan)
    if integrity_issues:
        raise ValueError(
            "render_core integrity validation failed: "
            + "; ".join(integrity_issues)
        )
    task = envelope.task
    structural_report = validate_task(task)
    if not structural_report.valid:
        raise _report_error("structural", structural_report)
    replay_report = validate_gold_replay(task)
    if not replay_report.valid:
        raise _report_error("gold replay", replay_report)
    return task


__all__ = ["PILOT_SURFACE_CATALOG", "render_core", "render_core_with_catalog"]
