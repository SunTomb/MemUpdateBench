from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import Difficulty, EventRole, Operation, QueryType, Split, TaskFamily
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, MultiObjectCurrentSelector
from mub.vnext.generation.core import CoreEvent, GenerationContext, SemanticCore
from mub.vnext.generation.core_config import CoreConfig
from mub.vnext.generation.core_render_v3 import render_core_v3
from mub.vnext.generation.identity import core_id, stable_id, task_id
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3, resolve_query_v3


FAMILY_G_MICRO_PROFILE_ID = "family_g_selected_micro_v1"
FAMILY_G_SYNTHESIS_KINDS = (
    "update_sensitive_multi_hop",
    "multi_object_current_consistency",
)


@dataclass(frozen=True, slots=True)
class _MultiHopSpec:
    hop_count: int
    stale_sensitive_position: str
    stale_operand_index: int


@dataclass(frozen=True, slots=True)
class _ConsistencySpec:
    object_count: int
    answer_kind: str
    scenario: str


@dataclass(frozen=True, slots=True)
class CompiledFamilyGMicroPilot:
    profile_id: str
    cores: tuple[SemanticCore, ...]
    tasks: tuple[MemUpdateTaskV3, ...]


_MULTI_HOP_SPECS = (
    _MultiHopSpec(2, "early", 0),
    _MultiHopSpec(2, "early", 0),
    _MultiHopSpec(2, "final", 1),
    _MultiHopSpec(2, "final", 1),
    _MultiHopSpec(3, "early", 0),
    _MultiHopSpec(3, "middle", 1),
    _MultiHopSpec(3, "middle", 1),
    _MultiHopSpec(3, "final", 2),
    _MultiHopSpec(4, "early", 0),
    _MultiHopSpec(4, "middle", 2),
    _MultiHopSpec(4, "middle", 1),
    _MultiHopSpec(4, "final", 3),
)

_CONSISTENCY_SPECS = tuple(
    _ConsistencySpec(object_count, answer_kind, scenario)
    for object_count in (3, 5, 8)
    for answer_kind, scenario in (
        ("boolean_consistency", "currently_consistent"),
        ("boolean_consistency", "currently_inconsistent"),
        ("exact_inconsistent_object", "first_exact"),
        ("exact_inconsistent_object", "last_exact"),
    )
)


def _identity(key: Any) -> tuple[str, str, str, str | None]:
    if isinstance(key, dict):
        return key["namespace"], key["entity"], key["attribute"], key.get("subkey")
    return key.namespace, key.entity, key.attribute, key.subkey


def _trajectory_identifier(kind: str) -> str:
    return stable_id("trajectory", {"family_g_derivation_group": kind})


def _keys(kind: str, core_index: int, count: int) -> tuple[MemoryObjectKey, ...]:
    namespace = "family_g_multihop" if kind == "update_sensitive_multi_hop" else "family_g_consistency"
    return tuple(
        MemoryObjectKey(
            object_type="slot",
            namespace=namespace,
            entity=f"synthetic_case_{core_index}_object_{index}",
            attribute="operand" if kind == "update_sensitive_multi_hop" else "consistency_code",
            subkey=f"ordered_{index}",
        )
        for index in range(count)
    )


def _events(keys: tuple[MemoryObjectKey, ...], previous: tuple[int, ...], current: tuple[int, ...], core_index: int) -> list[CoreEvent]:
    events: list[CoreEvent] = []
    logical_time = core_index * 100
    for index, (key, old_value, new_value) in enumerate(zip(keys, previous, current)):
        events.append(
            CoreEvent(
                operation=Operation.ADD,
                object_keys=[key],
                value=old_value,
                role=EventRole.HISTORICAL_SUPPORT,
                metadata={"logical_time": f"{logical_time + index * 2:08d}"},
            )
        )
        events.append(
            CoreEvent(
                operation=Operation.UPDATE,
                object_keys=[key],
                value=new_value,
                role=EventRole.LATEST_GOLD,
                metadata={"logical_time": f"{logical_time + index * 2 + 1:08d}"},
            )
        )
    return events


def _profile(query_type: QueryType, object_count: int, reasoning_depth: int) -> dict[str, Any]:
    return {
        "update_depth": 2,
        "active_object_count": object_count,
        "entity_ambiguity": "none",
        "attribute_ambiguity": "none",
        "noop_density": 0.0,
        "cross_slot_interleaving": 1.0,
        "stale_count": object_count,
        "context_length": object_count * 2,
        "context_order": "chronological",
        "version_metadata": "logical_time",
        "source_naturalness": "synthetic",
        "query_type": query_type.value,
        "requested_version_distance": 0,
        "reasoning_depth": reasoning_depth,
    }


def _core_identifier(kind: str, index: int, payload: dict[str, Any]) -> str:
    canonical_payload = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }
    return core_id(
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value,
        {"family_g_kind": kind, "micro_index": index, **canonical_payload},
    )


def _build_multi_hop(index: int, spec: _MultiHopSpec) -> SemanticCore:
    keys = _keys("update_sensitive_multi_hop", index, spec.hop_count)
    current = tuple(30 + index * 7 + operand * 3 for operand in range(spec.hop_count))
    previous = tuple(
        value + (operand + 1) * 2
        for operand, value in enumerate(current)
    )
    answer = current[0]
    for value in current[1:]:
        answer -= value
    payload = {
        "hop_count": spec.hop_count,
        "stale_sensitive_position": spec.stale_sensitive_position,
        "stale_operand_index": spec.stale_operand_index,
        "current": current,
        "previous": previous,
    }
    return SemanticCore(
        core_id=_core_identifier("update_sensitive_multi_hop", index, payload),
        task_family=TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
        difficulty={2: Difficulty.EASY, 3: Difficulty.MEDIUM, 4: Difficulty.HARD}[spec.hop_count],
        core_index=index,
        trajectory_id=_trajectory_identifier("update_sensitive_multi_hop"),
        events=_events(keys, previous, current, index),
        query_targets=list(keys),
        query_type=QueryType.MULTI_OBJECT,
        query_selector=MultiObjectCurrentSelector(object_keys=keys),
        expected_answer=answer,
        profile=_profile(QueryType.MULTI_OBJECT, len(keys), spec.hop_count),
        stratification={
            "synthesis_kind": "update_sensitive_multi_hop",
            "hop_count": spec.hop_count,
            "stale_sensitive_position": spec.stale_sensitive_position,
            "stale_operand_index": spec.stale_operand_index,
        },
    )


def _consistency_values(index: int, spec: _ConsistencySpec) -> tuple[tuple[int, ...], tuple[int, ...], int | bool, int | bool, int, tuple[int, ...]]:
    count = spec.object_count
    current = [0] * count
    previous = [100 + index * 10 + operand for operand in range(count)]
    stale_indices: tuple[int, ...]
    if spec.scenario == "currently_consistent":
        previous[0] = 1
        answer: int | bool = True
        stale_answer: int | bool = False
        stale_indices = (0,)
    elif spec.scenario == "currently_inconsistent":
        current[1] = 1
        previous[1] = 0
        answer = False
        stale_answer = True
        stale_indices = (1,)
    elif spec.scenario == "first_exact":
        primary = min(2, count - 1)
        alternative = 0
        current[primary] = primary + 1
        previous[primary] = 0
        previous[alternative] = alternative + 1
        answer = primary + 1
        stale_answer = alternative + 1
        stale_indices = (primary, alternative)
    else:
        primary = count - 1
        alternative = 1
        current[primary] = primary + 1
        previous[primary] = 0
        previous[alternative] = alternative + 1
        answer = primary + 1
        stale_answer = alternative + 1
        stale_indices = (primary, alternative)
    return tuple(previous), tuple(current), answer, stale_answer, stale_indices[0], stale_indices


def _build_consistency(local_index: int, spec: _ConsistencySpec) -> SemanticCore:
    core_index = len(_MULTI_HOP_SPECS) + local_index
    keys = _keys("multi_object_current_consistency", core_index, spec.object_count)
    previous, current, answer, stale_answer, stale_object_index, stale_indices = _consistency_values(local_index, spec)
    payload = {
        "object_count": spec.object_count,
        "answer_kind": spec.answer_kind,
        "scenario": spec.scenario,
        "current": current,
        "previous": previous,
        "stale_indices": stale_indices,
    }
    return SemanticCore(
        core_id=_core_identifier("multi_object_current_consistency", local_index, payload),
        task_family=TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
        difficulty={3: Difficulty.EASY, 5: Difficulty.MEDIUM, 8: Difficulty.HARD}[spec.object_count],
        core_index=core_index,
        trajectory_id=_trajectory_identifier("multi_object_current_consistency"),
        events=_events(keys, previous, current, core_index),
        query_targets=list(keys),
        query_type=QueryType.MULTI_OBJECT,
        query_selector=MultiObjectCurrentSelector(object_keys=keys),
        expected_answer=answer,
        profile=_profile(QueryType.MULTI_OBJECT, len(keys), 2),
        stratification={
            "synthesis_kind": "multi_object_current_consistency",
            "object_count": spec.object_count,
            "answer_kind": spec.answer_kind,
            "scenario": spec.scenario,
            "stale_object_index": stale_object_index,
            "stale_indices": ",".join(str(value) for value in stale_indices),
            "stale_answer": stale_answer,
        },
    )


def validate_family_g_core(core: SemanticCore) -> None:
    if not isinstance(core, SemanticCore):
        raise TypeError("core must be a SemanticCore")
    if core.task_family is not TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS:
        raise ValueError("Family G validator requires long_horizon_memory_synthesis")
    kind = core.stratification.get("synthesis_kind")
    if kind not in FAMILY_G_SYNTHESIS_KINDS:
        raise ValueError("Family G synthesis kind is invalid")
    query_target_identities = tuple(_identity(key) for key in core.query_targets)
    if len(set(query_target_identities)) != len(query_target_identities):
        raise ValueError("Family G query targets must have unique four-part identities")
    event_identities = tuple(
        dict.fromkeys(_identity(event.object_keys[0]) for event in core.events)
    )
    histories: dict[tuple[str, str, str, str | None], list[CoreEvent]] = {
        identity: [] for identity in event_identities
    }
    for event in core.events:
        if event.operation not in {Operation.ADD, Operation.UPDATE} or len(event.object_keys) != 1:
            raise ValueError("Family G permits only exact-object ADD and UPDATE events")
        identity = _identity(event.object_keys[0])
        histories[identity].append(event)
    if any(
        tuple(event.operation for event in events) != (Operation.ADD, Operation.UPDATE)
        or events[0].value == events[1].value
        for events in histories.values()
    ):
        raise ValueError("Family G targets require one distinct stale and current version")
    logical_times = tuple(event.metadata.get("logical_time") for event in core.events)
    if (
        any(type(value) is not str or len(value) != 8 or not value.isdecimal() for value in logical_times)
        or logical_times != tuple(sorted(logical_times))
        or len(logical_times) != len(set(logical_times))
    ):
        raise ValueError("Family G requires unique increasing logical times")
    current_values = [events[-1].value for events in histories.values()]
    previous_values = [events[0].value for events in histories.values()]
    if any(
        type(value) not in {int, float}
        for value in (*current_values, *previous_values)
    ):
        raise ValueError("Family G derivation operands must be exact numeric values")
    if (
        not isinstance(core.query_selector, MultiObjectCurrentSelector)
        or core.query_type is not QueryType.MULTI_OBJECT
    ):
        raise ValueError("Family G cores require typed multi-object current selection")
    selector_identities = tuple(
        _identity(key) for key in core.query_selector.object_keys
    )
    if selector_identities != query_target_identities:
        raise ValueError("Family G selector order must equal exact query target order")
    if query_target_identities != event_identities:
        raise ValueError("Family G query targets must equal event operand order")
    if core.profile.get("query_type") != QueryType.MULTI_OBJECT.value:
        raise ValueError("Family G profile query type must be multi_object")
    if kind == "update_sensitive_multi_hop":
        if core.stratification.get("hop_count") != len(histories):
            raise ValueError("Family G multi-hop operand count must equal hop_count")
        stale_index = core.stratification.get("stale_operand_index")
        if type(stale_index) is not int or stale_index not in range(len(histories)):
            raise ValueError("Family G stale-sensitive operand index is invalid")
        expected_position = (
            "early"
            if stale_index == 0
            else "final"
            if stale_index == len(histories) - 1
            else "middle"
        )
        if core.stratification.get("stale_sensitive_position") != expected_position:
            raise ValueError("Family G stale-sensitive position is invalid")
        expected_answer = current_values[0] - sum(current_values[1:])
        stale_values = list(current_values)
        stale_values[stale_index] = previous_values[stale_index]
        stale_answer = stale_values[0] - sum(stale_values[1:])
        if core.expected_answer != expected_answer or stale_answer == expected_answer:
            raise ValueError("Family G multi-hop answer or stale derivation is invalid")
    else:
        if core.stratification.get("object_count") != len(core.query_targets):
            raise ValueError("Family G consistency target count must equal object_count")
        answer_kind = core.stratification.get("answer_kind")
        if answer_kind == "boolean_consistency":
            expected_answer = all(value == current_values[0] for value in current_values[1:])
        elif answer_kind == "exact_inconsistent_object":
            marked = [
                (index, value)
                for index, value in enumerate(current_values)
                if value != 0
            ]
            if len(marked) != 1 or marked[0][1] != marked[0][0] + 1:
                raise ValueError("Family G exact inconsistency must encode one exact object position")
            expected_answer = marked[0][1]
        else:
            raise ValueError("Family G consistency answer kind is invalid")
        try:
            stale_indices = {
                int(value)
                for value in core.stratification["stale_indices"].split(",")
            }
        except (AttributeError, KeyError, ValueError) as exc:
            raise ValueError("Family G stale consistency indices are invalid") from exc
        if not stale_indices or not stale_indices <= set(range(len(histories))):
            raise ValueError("Family G stale consistency indices are invalid")
        stale_values = [
            previous_values[index] if index in stale_indices else value
            for index, value in enumerate(current_values)
        ]
        stale_answer = (
            all(value == stale_values[0] for value in stale_values[1:])
            if answer_kind == "boolean_consistency"
            else sum(stale_values)
        )
        if answer_kind == "exact_inconsistent_object":
            stale_marked = [
                (index, value)
                for index, value in enumerate(stale_values)
                if value != 0
            ]
            if len(stale_marked) != 1 or stale_marked[0][1] != stale_marked[0][0] + 1:
                raise ValueError("Family G stale alternative must encode one exact object position")
        if (
            core.expected_answer != expected_answer
            or core.stratification.get("stale_answer") != stale_answer
            or stale_answer == expected_answer
        ):
            raise ValueError("Family G consistency answer or stale derivation is invalid")


def validate_family_g_micro_core(core: SemanticCore) -> None:
    validate_family_g_core(core)
    canonical = _canonical_cores()
    if core.core_index >= len(canonical):
        raise ValueError("Family G micro-pilot semantic core index is not canonical")
    expected = canonical[core.core_index]
    if core.model_dump(mode="python") != expected.model_dump(mode="python"):
        raise ValueError("Family G micro-pilot semantic core is not canonical")


def _canonical_cores() -> list[SemanticCore]:
    return [
        *(_build_multi_hop(index, spec) for index, spec in enumerate(_MULTI_HOP_SPECS)),
        *(_build_consistency(index, spec) for index, spec in enumerate(_CONSISTENCY_SPECS)),
    ]


def generate_core_family_g_cores(config: CoreConfig) -> list[SemanticCore]:
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    cores = _canonical_cores()
    for core in cores:
        validate_family_g_core(core)
    if len(cores) != 24 or Counter(core.stratification["synthesis_kind"] for core in cores) != Counter({
        "update_sensitive_multi_hop": 12,
        "multi_object_current_consistency": 12,
    }):
        raise ValueError("Family G micro-pilot requires exactly two twelve-core synthesis groups")
    return cores


def _read_support(evidence):
    read_steps = [
        step
        for step in evidence.derivation_steps
        if step.operation in {"read", "read_current", "read_version"}
    ]
    object_ids = {
        _identity(key)
        for step in read_steps
        for key in step.supporting_object_keys
    }
    event_ids = {
        event_id
        for step in read_steps
        for event_id in step.supporting_event_ids
    }
    return object_ids, event_ids


def validate_family_g_task(task: MemUpdateTaskV3) -> None:
    if not isinstance(task, MemUpdateTaskV3):
        raise TypeError("task must be a MemUpdateTaskV3")
    if task.task_family != TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value:
        raise ValueError("Family G task validator requires long_horizon_memory_synthesis")
    if len(task.queries) != 1 or len(task.gold_evidence) != 1:
        raise ValueError("Family G task requires one query and one evidence row")
    if any(action.operation not in {Operation.ADD, Operation.UPDATE} for action in task.actions):
        raise ValueError("Family G forbids deletion, history-only, and NOOP semantics")
    histories = {_identity(ledger.object_key): ledger for ledger in task.version_history}
    if len(histories) != len(task.target_objects):
        raise ValueError("Family G requires one exact ledger per canonical object")
    for identity, ledger in histories.items():
        if len(ledger.entries) != 2:
            raise ValueError("Family G requires exactly one stale and one current version")
        if any(entry.status.value != "present" for entry in ledger.entries):
            raise ValueError("Family G forbids deletion-history semantics")
        if ledger.entries[0].value == ledger.entries[1].value:
            raise ValueError("Family G stale and current values must be distinct")
        object_actions = [
            action
            for action in task.actions
            if action.target_object_keys
            and _identity(action.target_object_keys[0]) == identity
        ]
        if tuple(action.operation for action in object_actions) != (Operation.ADD, Operation.UPDATE):
            raise ValueError("Family G object history must be ADD followed by UPDATE")
    query = task.queries[0]
    evidence = task.gold_evidence[0]
    if query.query_type not in {
        QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
        QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
    }:
        raise ValueError("Family G task has an unsupported typed query")
    if query.synthesis is None or query.synthesis.kind != query.query_type.value:
        raise ValueError("Family G query requires its matching synthesis contract")
    visible_text = query.text.lower()
    if (
        "current" not in visible_text
        or "object_order=" not in query.text
        or any(key.canonical_id not in query.text for key in task.target_objects)
    ):
        raise ValueError("Family G visible query must state current ordered derivation intent")
    if "current_state" in visible_text:
        raise ValueError("Family G query cannot tunnel intent through CURRENT_STATE")
    stratification = task.metadata.extra.get("stratification", {})
    if query.query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP:
        if "subtract" not in visible_text:
            raise ValueError("Family G multi-hop query must visibly declare ordered subtraction")
    elif stratification.get("answer_kind") == "boolean_consistency":
        if "all codes are equal" not in visible_text:
            raise ValueError("Family G boolean query must visibly declare consistency semantics")
    elif (
        stratification.get("answer_kind") != "exact_inconsistent_object"
        or "add" not in visible_text
        or "1-based position" not in visible_text
    ):
        raise ValueError("Family G exact query must visibly declare inconsistency semantics")
    if evidence.stale_alternative is None or evidence.answer == evidence.stale_alternative.answer:
        raise ValueError("Family G requires a distinct authenticated stale alternative")
    declared_objects = {_identity(key) for key in task.target_objects}
    for item in (evidence, evidence.stale_alternative):
        read_objects, read_events = _read_support(item)
        if (
            read_objects != declared_objects
            or read_objects != {_identity(key) for key in item.supporting_object_keys}
        ):
            raise ValueError("Family G top-level object support must be exactly consumed by reads")
        if read_events != set(item.supporting_event_ids):
            raise ValueError("Family G top-level event support must be exactly consumed by reads")
    replay = replay_task_v3(task)
    if replay.issues:
        raise ValueError(f"Family G task replay failed: {replay.issues[0].code}")
    resolution = resolve_query_v3(query, replay, task.events)
    if resolution.issues:
        raise ValueError(f"Family G selector replay failed: {resolution.issues[0].code}")
    evaluated = evaluate_evidence_v3(
        evidence,
        replay,
        evidence.stale_alternative,
        query,
        task.events,
    )
    if evaluated.issues:
        raise ValueError(f"Family G evidence replay failed: {evaluated.issues[0].code}")
    if evaluated.answer != evidence.answer or evaluated.stale_alternative_answer != evidence.stale_alternative.answer:
        raise ValueError("Family G derivation answers are not replayable")


def validate_family_g_micro_task(
    task: MemUpdateTaskV3,
    core: SemanticCore | None = None,
) -> None:
    validate_family_g_task(task)
    if task.metadata.split is not Split.EVALUATION_ONLY:
        raise ValueError("Family G micro-pilot tasks must be evaluation_only")
    canonical_by_id = {item.core_id: item for item in _canonical_cores()}
    semantic_core_id = task.metadata.split_key.semantic_core_id
    if semantic_core_id not in canonical_by_id:
        raise ValueError("Family G micro task references a noncanonical semantic core")
    canonical = canonical_by_id[semantic_core_id]
    surface_variant = task.metadata.extra.get("surface_variant")
    if (
        type(surface_variant) is not int
        or surface_variant not in range(4)
        or task.task_id != task_id(canonical.core_id, surface_variant)
        or task.metadata.split_key.trajectory_id != canonical.trajectory_id
        or task.metadata.extra.get("core_index") != canonical.core_index
        or dict(task.metadata.extra.get("stratification", {})) != dict(canonical.stratification)
        or task.source.provenance.get("semantic_core_id") != canonical.core_id
        or task.source.provenance.get("trajectory_id") != canonical.trajectory_id
    ):
        raise ValueError("Family G micro task/core binding is not canonical")
    expected_version_group = stable_id(
        "version_group", {"trajectory_id": canonical.trajectory_id}
    )
    if task.metadata.split_key.version_group_id != expected_version_group:
        raise ValueError("Family G derivation group must share one version group")
    if core is not None:
        validate_family_g_micro_core(core)
        if (
            core.core_id != canonical.core_id
            or task.gold_evidence[0].answer != core.expected_answer
        ):
            raise ValueError("Family G compiler task/core semantics differ")


def _validate_compiled(tasks: list[MemUpdateTaskV3], cores: list[SemanticCore]) -> None:
    if len(cores) != 24 or len(tasks) != 96:
        raise ValueError("Family G micro-pilot requires exactly 24 cores and 96 tasks")
    core_by_id = {core.core_id: core for core in cores}
    if len(core_by_id) != 24:
        raise ValueError("Family G semantic core IDs must be unique")
    by_core: dict[str, list[MemUpdateTaskV3]] = defaultdict(list)
    by_group: dict[str, list[MemUpdateTaskV3]] = defaultdict(list)
    for task in tasks:
        semantic_core_id = task.metadata.split_key.semantic_core_id
        if semantic_core_id not in core_by_id:
            raise ValueError("Family G task references an unknown semantic core")
        validate_family_g_micro_task(task, core_by_id[semantic_core_id])
        by_core[semantic_core_id].append(task)
        by_group[core_by_id[semantic_core_id].stratification["synthesis_kind"]].append(task)
    for surfaces in by_core.values():
        if len(surfaces) != 4 or {task.metadata.extra["surface_variant"] for task in surfaces} != {0, 1, 2, 3}:
            raise ValueError("Family G semantic core requires four deterministic surfaces")
        if len({task.semantic_hash for task in surfaces}) != 1:
            raise ValueError("Family G surfaces must share one semantic hash")
        if len({task.task_id for task in surfaces}) != 4 or len({task.source.raw_hash for task in surfaces}) != 4:
            raise ValueError("Family G surfaces require distinct task and raw identities")
    if Counter(len(group) for group in by_group.values()) != Counter({48: 2}):
        raise ValueError("Family G derivation families require one 48-task evidence group each")
    for group in by_group.values():
        if len({task.metadata.split_key.version_group_id for task in group}) != 1:
            raise ValueError("Family G derivation family must share one version group")
        if {task.metadata.split for task in group} != {Split.EVALUATION_ONLY}:
            raise ValueError("Family G derivation family must share evaluation_only split")


def compile_family_g_micro_pilot(
    config: CoreConfig,
    *,
    code_revision: str,
) -> CompiledFamilyGMicroPilot:
    cores = generate_core_family_g_cores(config)
    context = GenerationContext(
        config=config,
        code_revision=code_revision,
        generator_name="memupdatebench_vnext_core_family_g_micro",
    )
    tasks = [
        render_core_v3(
            core,
            split=Split.EVALUATION_ONLY,
            surface_variant=surface_variant,
            context=context,
        )
        for core in cores
        for surface_variant in range(4)
    ]
    _validate_compiled(tasks, cores)
    return CompiledFamilyGMicroPilot(
        profile_id=FAMILY_G_MICRO_PROFILE_ID,
        cores=tuple(cores),
        tasks=tuple(tasks),
    )


__all__ = [
    "CompiledFamilyGMicroPilot",
    "FAMILY_G_MICRO_PROFILE_ID",
    "FAMILY_G_SYNTHESIS_KINDS",
    "compile_family_g_micro_pilot",
    "generate_core_family_g_cores",
    "validate_family_g_core",
    "validate_family_g_micro_core",
    "validate_family_g_micro_task",
    "validate_family_g_task",
]
