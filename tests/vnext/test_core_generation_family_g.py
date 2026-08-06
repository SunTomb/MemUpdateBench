from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

from mub.vnext.contracts.enums import AnswerSchema, QueryType, Split, SupportReason, TaskFamily
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import CurrentSelector, MemUpdateTaskV3, MultiObjectCurrentSelector
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3, resolve_query_v3


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG = ROOT / "configs" / "vnext" / "core.yaml"
FULL_CAPABILITIES = AdapterCapabilitiesV3(
    supports_isolated_reset=True,
    supports_event_ingest=True,
    supports_add=True,
    supports_update=True,
    supports_native_answer=True,
    supports_multi_object_query=True,
    exports_version_history=True,
    exports_entries=True,
    exports_raw_state=True,
    exports_source_event_ids=True,
    exports_timestamps_or_order=True,
    exports_object_keys=True,
    exports_values=True,
    exports_retrieval_ids=True,
    exports_retrieval_scores=True,
    exports_action_trace=True,
    exports_evidence_linkage=True,
)
ORACLE_INFO = AdapterInfoV3(
    adapter_id="family-g-oracle",
    adapter_version="1",
    system_name="reference",
    system_version="1",
    configuration_hash="a" * 64,
)
SYNTHESIS_PATHS = (
    "synthesis_scores.multi_hop_accuracy",
    "synthesis_scores.multi_object_accuracy",
    "synthesis_scores.evidence_precision",
    "synthesis_scores.evidence_recall",
    "synthesis_scores.evidence_f1",
    "synthesis_scores.reasoning_support_accuracy",
    "synthesis_scores.stale_propagation_rate",
)


def _config():
    return load_core_config(CORE_CONFIG)


def _api():
    from mub.vnext.generation.family_g import (
        compile_family_g_micro_pilot,
        generate_core_family_g_cores,
        validate_family_g_core,
        validate_family_g_micro_core,
        validate_family_g_micro_task,
        validate_family_g_task,
    )

    return (
        generate_core_family_g_cores,
        validate_family_g_core,
        validate_family_g_task,
        validate_family_g_micro_core,
        validate_family_g_micro_task,
        compile_family_g_micro_pilot,
    )


def test_family_g_public_generation_exports_are_available():
    import mub.vnext.generation as generation

    assert generation.FAMILY_G_MICRO_PROFILE_ID == "family_g_selected_micro_v1"
    assert generation.FAMILY_G_SYNTHESIS_KINDS == (
        "update_sensitive_multi_hop",
        "multi_object_current_consistency",
    )
    assert generation.compile_family_g_micro_pilot is not None
    assert generation.generate_core_family_g_cores is not None
    assert generation.validate_family_g_core is not None
    assert generation.validate_family_g_micro_core is not None
    assert generation.validate_family_g_task is not None
    assert generation.validate_family_g_micro_task is not None


def _mutate_core(core, mutate):
    from mub.vnext.generation.core import SemanticCore

    payload = core.model_dump(mode="python")
    mutate(payload)
    return SemanticCore.model_validate(payload)


def _identity(key):
    return key.namespace, key.entity, key.attribute, key.subkey


def _event_operand_order(core):
    return tuple(
        dict.fromkeys(
            _identity(key)
            for event in core.events
            for key in event.object_keys
        )
    )


def test_family_g_all_core_operand_orders_match_and_current_selector_laundering_rejects():
    generate, validate_core, _, _, _, _ = _api()
    cores = generate(_config())

    for core in cores:
        target_order = tuple(_identity(key) for key in core.query_targets)
        assert core.query_type is QueryType.MULTI_OBJECT
        assert isinstance(core.query_selector, MultiObjectCurrentSelector)
        assert core.profile["query_type"] == QueryType.MULTI_OBJECT.value
        assert tuple(_identity(key) for key in core.query_selector.object_keys) == target_order
        assert target_order == _event_operand_order(core)
        validate_core(core)

    source = next(
        core
        for core in cores
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    )

    def launder_as_one_target_current(payload):
        stale_index = payload["stratification"]["stale_operand_index"]
        payload["query_targets"] = [payload["query_targets"][stale_index]]
        payload["query_type"] = QueryType.CURRENT_STATE
        payload["query_selector"] = CurrentSelector().model_dump(mode="python")
        payload["profile"]["query_type"] = QueryType.CURRENT_STATE.value

    laundered = _mutate_core(source, launder_as_one_target_current)
    with pytest.raises(ValueError, match="multi-object|selector|target|operand|order"):
        validate_core(laundered)


def test_family_g_event_staging_core_is_honest_ordered_multi_object_current():
    from mub.vnext.generation.core_render_v3 import _family_g_event_staging_core

    generate, _, _, _, _, _ = _api()
    for core in generate(_config()):
        staging = _family_g_event_staging_core(core)
        operand_order = _event_operand_order(core)
        current_by_identity = {
            _identity(key): event.value
            for event in core.events
            if event.operation.value in {"ADD", "UPDATE"}
            for key in event.object_keys
        }
        expected_values = [current_by_identity[identity] for identity in operand_order]

        assert staging.query_type is QueryType.MULTI_OBJECT
        assert staging.profile["query_type"] == QueryType.MULTI_OBJECT.value
        assert isinstance(staging.query_selector, MultiObjectCurrentSelector)
        assert tuple(_identity(key) for key in staging.query_targets) == operand_order
        assert tuple(_identity(key) for key in staging.query_selector.object_keys) == operand_order
        assert list(staging.expected_answer) == expected_values
        assert staging.stratification == core.stratification
        assert QueryType.CURRENT_STATE.value not in {
            staging.query_type.value,
            staging.profile["query_type"],
        }


def test_family_g_generic_core_validation_is_reusable_but_micro_validation_is_exact():
    generate, validate_core, _, validate_micro_core, _, _ = _api()
    source = next(
        core
        for core in generate(_config())
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    )

    def make_valid_noncanonical(payload):
        payload["events"][1]["value"] += 10
        current = [event["value"] for event in payload["events"][1::2]]
        payload["expected_answer"] = current[0] - sum(current[1:])

    reusable = _mutate_core(source, make_valid_noncanonical)
    validate_core(reusable)
    with pytest.raises(ValueError, match="canonical"):
        validate_micro_core(reusable)

    wrong_answer = _mutate_core(
        source,
        lambda payload: payload.__setitem__(
            "expected_answer", payload["expected_answer"] + 1
        ),
    )
    with pytest.raises(ValueError, match="answer|derivation"):
        validate_core(wrong_answer)


def test_family_g_generic_task_validator_accepts_valid_nonmicro_projection_only():
    from mub.vnext.generation.core import GenerationContext
    from mub.vnext.generation.core_render_v3 import render_core_v3

    generate, validate_core, validate_task, _, validate_micro_task, _ = _api()
    source = next(
        core
        for core in generate(_config())
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    )

    def make_valid_noncanonical(payload):
        payload["events"][1]["value"] += 10
        current = [event["value"] for event in payload["events"][1::2]]
        payload["expected_answer"] = current[0] - sum(current[1:])

    reusable = _mutate_core(source, make_valid_noncanonical)
    validate_core(reusable)
    task = render_core_v3(
        reusable,
        split=Split.TEST,
        surface_variant=0,
        context=GenerationContext(config=_config(), code_revision="5423ef7"),
    )
    validate_task(task)
    with pytest.raises(ValueError, match="evaluation_only|canonical"):
        validate_micro_task(task)


def test_family_g_generator_has_exact_selected_micro_pilot_marginals():
    generate, validate_core, _, validate_micro_core, _, _ = _api()
    cores = generate(_config())

    assert len(cores) == 24
    assert cores == generate(_config())
    assert {core.task_family for core in cores} == {
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS
    }
    assert Counter(core.stratification["synthesis_kind"] for core in cores) == Counter(
        {"update_sensitive_multi_hop": 12, "multi_object_current_consistency": 12}
    )

    multi_hop = [
        core
        for core in cores
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    ]
    consistency = [
        core
        for core in cores
        if core.stratification["synthesis_kind"]
        == "multi_object_current_consistency"
    ]
    assert Counter(core.stratification["hop_count"] for core in multi_hop) == Counter(
        {2: 4, 3: 4, 4: 4}
    )
    assert Counter(
        core.stratification["stale_sensitive_position"] for core in multi_hop
    ) == Counter({"early": 4, "middle": 4, "final": 4})
    assert Counter(core.stratification["object_count"] for core in consistency) == Counter(
        {3: 4, 5: 4, 8: 4}
    )
    assert Counter(core.stratification["answer_kind"] for core in consistency) == Counter(
        {"boolean_consistency": 6, "exact_inconsistent_object": 6}
    )
    for object_count in (3, 5, 8):
        stratum = [
            core
            for core in consistency
            if core.stratification["object_count"] == object_count
        ]
        assert Counter(core.stratification["answer_kind"] for core in stratum) == Counter(
            {"boolean_consistency": 2, "exact_inconsistent_object": 2}
        )

    by_group = defaultdict(list)
    for core in cores:
        validate_core(core)
        validate_micro_core(core)
        by_group[core.stratification["synthesis_kind"]].append(core)
    assert all(len({core.trajectory_id for core in group}) == 1 for group in by_group.values())


def test_family_g_full_schedule_has_exact_synthesis_strata_and_evidence_groups():
    generate, _, _, _, _, _ = _api()

    cores = generate(_config(), profile="full")

    assert len(cores) == 300
    assert len({core.core_id for core in cores}) == 300
    assert len({core.trajectory_id for core in cores}) == 300
    assert Counter(core.stratification["synthesis_kind"] for core in cores) == {
        "update_sensitive_multi_hop": 180,
        "multi_object_current_consistency": 120,
    }

    multi_hop = [
        core
        for core in cores
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    ]
    assert Counter(core.stratification["hop_count"] for core in multi_hop) == {
        2: 60,
        3: 60,
        4: 60,
    }
    expected_positions = {
        2: {"early": 30, "final": 30},
        3: {"early": 20, "middle": 20, "final": 20},
        4: {"early": 20, "middle": 20, "final": 20},
    }
    for hop_count, expected in expected_positions.items():
        assert Counter(
            core.stratification["stale_sensitive_position"]
            for core in multi_hop
            if core.stratification["hop_count"] == hop_count
        ) == expected

    consistency = [
        core
        for core in cores
        if core.stratification["synthesis_kind"]
        == "multi_object_current_consistency"
    ]
    assert Counter(core.stratification["object_count"] for core in consistency) == {
        3: 40,
        5: 40,
        8: 40,
    }
    for object_count in (3, 5, 8):
        stratum = [
            core
            for core in consistency
            if core.stratification["object_count"] == object_count
        ]
        assert Counter(core.stratification["answer_kind"] for core in stratum) == {
            "boolean_consistency": 20,
            "exact_inconsistent_object": 20,
        }
        assert Counter(core.stratification["scenario"] for core in stratum) == {
            "currently_consistent": 10,
            "currently_inconsistent": 10,
            "first_exact": 10,
            "last_exact": 10,
        }

    fingerprints = {
        core.stratification["evidence_fingerprint"] for core in cores
    }
    evidence_groups = {core.stratification["evidence_group_id"] for core in cores}
    assert len(fingerprints) == 300
    assert len(evidence_groups) == 300



def test_family_g_full_validator_requires_semantic_evidence_group_binding():
    from mub.vnext.generation.family_g import validate_family_g_full_core

    generate, validate_core, _, _, _, _ = _api()
    cores = generate(_config(), profile="full")
    source = cores[0]
    other = cores[1]

    payload = source.model_dump(mode="python")
    payload["stratification"].pop("evidence_fingerprint")
    payload["stratification"].pop("evidence_group_id")
    missing = type(source).model_validate(payload)
    with pytest.raises(ValueError, match="evidence fingerprint|evidence group"):
        validate_family_g_full_core(missing)

    wrong_trajectory = source.model_copy(
        update={"trajectory_id": other.trajectory_id}
    )
    with pytest.raises(ValueError, match="evidence fingerprint|group binding"):
        validate_family_g_full_core(wrong_trajectory)

    reindexed = source.model_copy(update={"core_index": source.core_index + 1000})
    assert reindexed.stratification["evidence_fingerprint"] == source.stratification[
        "evidence_fingerprint"
    ]
    validate_core(reindexed)


def test_family_g_full_validator_rejects_swapped_canonical_core_id():
    from mub.vnext.generation.family_g import validate_family_g_full_core

    cores = _api()[0](_config(), profile="full")
    swapped = cores[0].model_copy(update={"core_id": cores[1].core_id})

    with pytest.raises(ValueError, match="core.*identifier|canonical"):
        validate_family_g_full_core(swapped)


def test_family_g_full_validator_rejects_drifted_canonical_core_index():
    from mub.vnext.generation.family_g import validate_family_g_full_core

    source = _api()[0](_config(), profile="full")[0]
    drifted = source.model_copy(update={"core_index": source.core_index + 1000})

    with pytest.raises(ValueError, match="core.*index|canonical"):
        validate_family_g_full_core(drifted)



def test_family_g_compiler_renders_exactly_four_replayable_v3_surfaces_per_core():
    _, _, validate_task, _, validate_micro_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")

    assert compiled.profile_id == "family_g_selected_micro_v1"
    assert compiled == compile_micro(_config(), code_revision="5423ef7")
    assert len(compiled.cores) == 24
    assert len(compiled.tasks) == 96
    by_core = defaultdict(list)
    for task in compiled.tasks:
        validate_task(task)
        validate_micro_task(task)
        assert task.metadata.split is Split.EVALUATION_ONLY
        assert len(task.queries) == len(task.gold_evidence) == 1
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        assert query.query_type in {
            QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
            QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
        }
        assert query.synthesis.kind == query.query_type.value
        assert evidence.stale_alternative is not None
        assert evidence.answer != evidence.stale_alternative.answer
        replay = replay_task_v3(task)
        assert replay.issues == ()
        assert resolve_query_v3(query, replay, task.events).issues == ()
        evaluated = evaluate_evidence_v3(
            evidence,
            replay,
            evidence.stale_alternative,
            query,
            task.events,
        )
        assert evaluated.issues == ()
        assert evaluated.answer == evidence.answer
        assert evaluated.stale_alternative_answer == evidence.stale_alternative.answer
        by_core[task.metadata.split_key.semantic_core_id].append(task)

    assert set(by_core) == {core.core_id for core in compiled.cores}
    by_derivation_family = defaultdict(list)
    for task in compiled.tasks:
        by_derivation_family[task.queries[0].query_type].append(task)
    assert Counter(len(group) for group in by_derivation_family.values()) == Counter({48: 2})
    assert all(
        len({task.metadata.split_key.version_group_id for task in group}) == 1
        and {task.metadata.split for task in group} == {Split.EVALUATION_ONLY}
        for group in by_derivation_family.values()
    )
    for surfaces in by_core.values():
        assert len(surfaces) == 4
        assert {task.metadata.extra["surface_variant"] for task in surfaces} == {0, 1, 2, 3}
        assert len({task.semantic_hash for task in surfaces}) == 1
        assert len({task.task_id for task in surfaces}) == 4
        assert len({task.source.raw_hash for task in surfaces}) == 4


def test_family_g_all_generated_query_resolutions_equal_gold_and_preserve_operand_order():
    _, _, _, _, _, compile_micro = _api()
    tasks = compile_micro(_config(), code_revision="5423ef7").tasks

    assert len(tasks) == 96
    for task in tasks:
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        replay = replay_task_v3(task)
        resolution = resolve_query_v3(query, replay, task.events)
        target_order = tuple(_identity(key) for key in query.target_object_keys)
        selector_order = tuple(_identity(key) for key in query.selector.object_keys)
        ledger_order = tuple(_identity(ledger.object_key) for ledger in task.version_history)
        evidence_order = tuple(_identity(key) for key in evidence.supporting_object_keys)
        read_order = tuple(
            _identity(step.supporting_object_keys[0])
            for step in evidence.derivation_steps
            if step.operation == "read_current"
        )

        assert resolution.issues == ()
        assert resolution.answer == evidence.answer
        assert target_order == selector_order == ledger_order == evidence_order == read_order
        assert tuple(_identity(key) for key in resolution.selected_object_keys) == target_order
        assert tuple(version.value for version in resolution.selected_versions) == tuple(
            ledger.entries[-1].value for ledger in task.version_history
        )


def test_family_g_replay_unsupported_answer_schema_returns_issue_with_selection_provenance():
    _, _, _, _, _, compile_micro = _api()
    tasks = compile_micro(_config(), code_revision="5423ef7").tasks
    representatives = {
        task.queries[0].query_type: task
        for task in tasks
        if task.metadata.extra["surface_variant"] == 0
    }

    for query_type in (
        QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
        QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
    ):
        task = representatives[query_type]
        replay = replay_task_v3(task)
        query = task.queries[0].model_copy(update={"answer_schema": AnswerSchema.STRING})
        resolution = resolve_query_v3(query, replay, task.events)

        assert tuple(issue.code for issue in resolution.issues) == (
            "unsupported_g_answer_schema",
        )
        assert resolution.answer is None
        assert tuple(_identity(key) for key in resolution.selected_object_keys) == tuple(
            _identity(key) for key in query.target_object_keys
        )
        assert tuple(version.value for version in resolution.selected_versions) == tuple(
            ledger.entries[-1].value for ledger in task.version_history
        )
        assert resolution.selected_event_ids


def test_family_g_validator_rejects_coordinated_add_derivation_that_disagrees_with_typed_query():
    _, _, validate_task, _, _, compile_micro = _api()
    source = next(
        task
        for task in compile_micro(_config(), code_revision="5423ef7").tasks
        if task.queries[0].query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP
        and task.metadata.extra["surface_variant"] == 0
    )
    payload = source.model_dump(mode="json")
    evidence = payload["gold_evidence"][0]
    current_values = [ledger["entries"][-1]["value"] for ledger in payload["version_history"]]
    stale_values = list(current_values)
    stale_index = payload["metadata"]["extra"]["stratification"]["stale_operand_index"]
    stale_values[stale_index] = payload["version_history"][stale_index]["entries"][0]["value"]

    for item, values in (
        (evidence, current_values),
        (evidence["stale_alternative"], stale_values),
    ):
        for step in item["derivation_steps"]:
            if step["operation"] == "subtract":
                step["operation"] = "add"
        item["answer"] = sum(values)

    mutated = MemUpdateTaskV3.model_validate(payload)
    replay = replay_task_v3(mutated)
    evaluated = evaluate_evidence_v3(
        mutated.gold_evidence[0],
        replay,
        mutated.gold_evidence[0].stale_alternative,
        mutated.queries[0],
        mutated.events,
    )
    assert replay.issues == ()
    assert evaluated.issues == ()
    assert evaluated.answer == mutated.gold_evidence[0].answer
    assert evaluated.stale_alternative_answer == mutated.gold_evidence[0].stale_alternative.answer
    with pytest.raises(ValueError, match="typed|selector|gold|answer"):
        validate_task(mutated)


def test_family_g_micro_validator_rejects_equivalent_noncanonical_derivation_graph():
    _, _, validate_task, _, validate_micro_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    core = next(
        item
        for item in compiled.cores
        if item.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
        and item.stratification["hop_count"] == 3
    )
    source = next(
        task
        for task in compiled.tasks
        if task.metadata.split_key.semantic_core_id == core.core_id
        and task.metadata.extra["surface_variant"] == 0
    )
    payload = source.model_dump(mode="json")
    evidence = payload["gold_evidence"][0]
    for item in (evidence, evidence["stale_alternative"]):
        reads = [
            step
            for step in item["derivation_steps"]
            if step["operation"] in {"read_current", "read_version"}
        ]
        derived = [
            step
            for step in item["derivation_steps"]
            if step["operation"] == "subtract"
        ]
        assert len(reads) == 3
        assert len(derived) == 2
        derived[0]["operation"] = "add"
        derived[0]["input_step_ids"] = [reads[1]["step_id"], reads[2]["step_id"]]
        derived[1]["input_step_ids"] = [reads[0]["step_id"], derived[0]["step_id"]]

    mutated = MemUpdateTaskV3.model_validate(payload)
    replay = replay_task_v3(mutated)
    evaluated = evaluate_evidence_v3(
        mutated.gold_evidence[0],
        replay,
        mutated.gold_evidence[0].stale_alternative,
        mutated.queries[0],
        mutated.events,
    )
    assert replay.issues == ()
    assert evaluated.issues == ()
    assert evaluated.answer == source.gold_evidence[0].answer
    assert evaluated.stale_alternative_answer == source.gold_evidence[0].stale_alternative.answer
    validate_task(mutated)
    with pytest.raises(ValueError, match="canonical|query|evidence|derivation"):
        validate_micro_task(mutated, core)


def test_family_g_micro_validator_rejects_noncanonical_evaluation_mode():
    _, _, validate_task, _, validate_micro_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    core = compiled.cores[0]
    source = next(
        task
        for task in compiled.tasks
        if task.metadata.split_key.semantic_core_id == core.core_id
        and task.metadata.extra["surface_variant"] == 0
    )
    assert source.queries[0].evaluation_mode.value == "retrieved_prompt"
    payload = source.model_dump(mode="json")
    payload["queries"][0]["evaluation_mode"] = "state_direct"

    mutated = MemUpdateTaskV3.model_validate(payload)
    validate_task(mutated)
    with pytest.raises(ValueError, match="canonical|evaluation|query"):
        validate_micro_task(mutated, core)


def test_family_g_micro_validator_rejects_coordinated_query_id_drift():
    from mub.vnext.generation.identity import query_id

    _, _, validate_task, _, validate_micro_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    core = compiled.cores[0]
    source = next(
        task
        for task in compiled.tasks
        if task.metadata.split_key.semantic_core_id == core.core_id
        and task.metadata.extra["surface_variant"] == 0
    )
    assert source.queries[0].query_id == query_id(source.task_id, 0)
    payload = source.model_dump(mode="json")
    original_query_id = payload["queries"][0]["query_id"]
    drifted_query_id = "query_drifted_under_original_task_id"
    payload["queries"][0]["query_id"] = drifted_query_id
    evidence = payload["gold_evidence"][0]
    evidence["query_id"] = drifted_query_id
    for item in (evidence, evidence["stale_alternative"]):
        step_id_map = {
            step["step_id"]: step["step_id"].replace(
                original_query_id, drifted_query_id
            )
            for step in item["derivation_steps"]
        }
        for step in item["derivation_steps"]:
            step["step_id"] = step_id_map[step["step_id"]]
            step["input_step_ids"] = [
                step_id_map[input_step_id]
                for input_step_id in step["input_step_ids"]
            ]
        item["final_derivation_step_id"] = step_id_map[
            item["final_derivation_step_id"]
        ]

    mutated = MemUpdateTaskV3.model_validate(payload)
    validate_task(mutated)
    with pytest.raises(ValueError, match="canonical|query.*id"):
        validate_micro_task(mutated, core)


def test_family_g_reversed_selector_controls_replay_order_hash_and_validation():
    _, _, validate_task, _, _, compile_micro = _api()
    source = next(
        task
        for task in compile_micro(_config(), code_revision="5423ef7").tasks
        if task.queries[0].query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP
        and task.metadata.extra["surface_variant"] == 0
    )
    payload = source.model_dump(mode="json")
    payload["queries"][0]["selector"]["object_keys"].reverse()
    reversed_task = MemUpdateTaskV3.model_validate(payload)
    replay = replay_task_v3(reversed_task)
    resolution = resolve_query_v3(reversed_task.queries[0], replay, reversed_task.events)
    selector_order = tuple(
        _identity(key) for key in reversed_task.queries[0].selector.object_keys
    )
    current_by_identity = {
        _identity(ledger.object_key): ledger.entries[-1].value
        for ledger in reversed_task.version_history
    }
    ordered_values = [current_by_identity[identity] for identity in selector_order]
    expected = ordered_values[0]
    for value in ordered_values[1:]:
        expected -= value

    assert reversed_task.semantic_hash != source.semantic_hash
    assert tuple(_identity(key) for key in resolution.selected_object_keys) == selector_order
    assert resolution.answer == expected
    with pytest.raises(ValueError, match="selector|target|ledger|operand|order"):
        validate_task(reversed_task)


def test_family_g_generic_validator_rejects_unowned_contradictory_event():
    _, _, validate_task, _, _, compile_micro = _api()
    source = compile_micro(_config(), code_revision="5423ef7").tasks[0]
    payload = source.model_dump(mode="json")
    payload["events"].append(
        {
            "event_id": "event_unowned_contradiction",
            "sequence_index": len(payload["events"]),
            "timestamp": "99999999",
            "raw_text": "Contradict every declared current operand.",
            "normalized_text": "contradict every declared current operand",
            "speaker": "system",
            "gold_action_ids": [],
            "role": "latest_gold",
            "source_anchor": {"event_index": len(payload["events"])},
            "metadata": {},
        }
    )
    mutated = MemUpdateTaskV3.model_validate(payload)
    with pytest.raises(ValueError, match="event|action|ownership|extra"):
        validate_task(mutated)


def test_family_g_micro_validator_rejects_coordinated_action_ledger_value_drift():
    _, validate_core, validate_task, _, validate_micro_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    core = compiled.cores[0]
    validate_core(core)
    source = next(
        task
        for task in compiled.tasks
        if task.metadata.split_key.semantic_core_id == core.core_id
        and task.metadata.extra["surface_variant"] == 0
    )
    payload = source.model_dump(mode="json")
    drift_ledger = payload["version_history"][1]
    drift_entry = drift_ledger["entries"][0]
    drift_event_id = drift_entry["source_event_ids"][0]
    drifted_value = drift_entry["value"] + 1000
    drift_entry["value"] = drifted_value
    drift_action = next(
        action for action in payload["actions"] if action["event_id"] == drift_event_id
    )
    drift_action["value"] = drifted_value

    mutated = MemUpdateTaskV3.model_validate(payload)
    validate_task(mutated)
    with pytest.raises(ValueError, match="canonical|core|event|action|value"):
        validate_micro_task(mutated, core)


def _derivation_depth(item):
    depths = {}
    for step in item.derivation_steps:
        depths[step.step_id] = 1 + max(
            (depths[parent] for parent in step.input_step_ids),
            default=0,
        )
    return depths[item.final_derivation_step_id]


def _assert_connected_topological_derivation(item):
    steps = {step.step_id: step for step in item.derivation_steps}
    positions = {step.step_id: index for index, step in enumerate(item.derivation_steps)}
    reached = set()

    def visit(step_id):
        if step_id in reached:
            return
        reached.add(step_id)
        for parent in steps[step_id].input_step_ids:
            assert positions[parent] < positions[step_id]
            visit(parent)

    visit(item.final_derivation_step_id)
    assert reached == set(steps)


def test_family_g_corrupted_evidence_controls_reject_removed_replaced_fabricated_and_inconsistent_derivations():
    _, _, validate_task, _, _, compile_micro = _api()
    tasks = compile_micro(_config(), code_revision="5423ef7").tasks
    multi_hop = next(
        task
        for task in tasks
        if task.queries[0].query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP
        and task.metadata.extra["surface_variant"] == 0
    )

    removed = multi_hop.model_dump(mode="json")
    removed["gold_evidence"][0]["derivation_steps"][-1]["input_step_ids"] = removed[
        "gold_evidence"
    ][0]["derivation_steps"][-1]["input_step_ids"][:1]
    with pytest.raises(ValueError, match="operand|derivation|subtract"):
        MemUpdateTaskV3.model_validate(removed)

    replaced = multi_hop.model_dump(mode="json")
    final_inputs = replaced["gold_evidence"][0]["derivation_steps"][-1][
        "input_step_ids"
    ]
    replaced["gold_evidence"][0]["derivation_steps"][-1]["input_step_ids"] = list(
        reversed(final_inputs)
    )
    replaced_task = MemUpdateTaskV3.model_validate(replaced)
    with pytest.raises(ValueError, match="answer|evidence replay"):
        validate_task(replaced_task)

    fabricated = multi_hop.model_dump(mode="json")
    primary = fabricated["gold_evidence"][0]
    unrelated = next(
        event["event_id"]
        for event in fabricated["events"]
        if event["event_id"] not in primary["supporting_event_ids"]
    )
    primary["supporting_event_ids"].append(unrelated)
    primary["derivation_steps"][0]["supporting_event_ids"].append(unrelated)
    with pytest.raises(ValueError, match="support|event|provenance"):
        MemUpdateTaskV3.model_validate(fabricated)

    consistency = next(
        task
        for task in tasks
        if task.queries[0].query_type
        is QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY
        and task.metadata.extra["stratification"]["answer_kind"]
        == "exact_inconsistent_object"
        and task.metadata.extra["surface_variant"] == 0
    )
    inconsistent = consistency.model_dump(mode="json")
    inconsistent["gold_evidence"][0]["answer"] += 1
    inconsistent_task = MemUpdateTaskV3.model_validate(inconsistent)
    with pytest.raises(ValueError, match="answer|evidence replay"):
        validate_task(inconsistent_task)

    tunneled = multi_hop.model_dump(mode="json")
    tunneled["queries"][0]["text"] = "CURRENT_STATE"
    tunneled_task = MemUpdateTaskV3.model_validate(tunneled)
    with pytest.raises(ValueError, match="visible|tunnel|intent"):
        validate_task(tunneled_task)


def _metric_value(score, path):
    layer, leaf = path.split(".", 1)
    return getattr(getattr(score, layer), leaf)


def _authenticated_score_context(task, run, caps):
    from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
    from mub.vnext.contracts.v3.score import ScorerConfigV3
    from mub.vnext.io import sha256_model
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3

    task_artifact = {
        "path": "family-g-tasks.jsonl",
        "sha256": "b" * 64,
        "media_type": "application/jsonl",
        "record_count": 1,
    }
    run_artifact = {
        "path": "family-g-runs.jsonl",
        "sha256": "c" * 64,
        "media_type": "application/jsonl",
        "record_count": 1,
    }
    task_manifest = TaskManifestV3(
        data_release_id="family-g-diagnostic",
        split_policy_version="diagnostic-no-release-split",
        compiler_versions={"core_family_g_micro": "3"},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts={"evaluation_only": 1},
        family_difficulty_counts={f"G.{task.difficulty.value}": 1},
        semantic_core_counts={task.metadata.split_key.semantic_core_id: 1},
        task_file_paths_and_hashes=(task_artifact,),
        task_record_hashes={task.task_id: sha256_model(task)},
        leakage_check_summary={},
        human_audit_artifacts=(),
        created_at="2026-08-06T00:00:00Z",
        code_revision="5423ef7",
    )
    task_manifest_hash = sha256_model(task_manifest)
    config = ScorerConfigV3()
    run_manifest = RunManifestV3(
        run_id=run.run_id,
        timestamp="2026-08-06T00:00:00Z",
        code_revision="5423ef7",
        dirty_state=False,
        task_manifest={
            "path": "task-manifest.json",
            "sha256": task_manifest_hash,
            "media_type": "application/json",
        },
        scorer_config=config,
        adapter_info=ORACLE_INFO,
        adapter_capabilities=caps,
        capability_verification_artifact={
            "path": "caps.json",
            "sha256": "d" * 64,
            "media_type": "application/json",
        },
        model_name=None,
        provider=None,
        model_revision=None,
        prompt_config={},
        decoding_config={},
        seed_information={},
        action_parser_version="1",
        answer_parser_version="1",
        memory_entry_extractor_version="1",
        object_value_extractor_config_hash="a" * 64,
        redaction_policy_version="1",
        environment_summary={},
        package_summary={},
        expected_task_count=1,
        completed_task_count=1,
        failed_task_count=0,
        not_supported_task_count=0,
        raw_provider_response_artifacts=(),
        raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(run_artifact,),
        run_record_hashes={run.task_id: sha256_model(run)},
        score_artifacts=(),
        native_vs_extracted_field_summary={},
    )
    return VerifiedScoringContextV3.from_authenticated_manifests(
        task=task,
        run=run,
        task_manifest=task_manifest,
        run_manifest=run_manifest,
        task_artifact=task_artifact,
        run_artifact=run_artifact,
        authenticated_task_manifest_sha256=task_manifest_hash,
        authenticated_run_manifest_sha256=sha256_model(run_manifest),
    )


def _oracle_run(task, *, stale_prediction=False):
    from mub.vnext.contracts.v3.runtime import (
        AnswerPredictionV3,
        MemoryEntryRecordV3,
        MemorySnapshotV3,
        ParsedManagerActionV3,
        ParserExtractorProvenanceV3,
        RetrievalTraceV3,
        TaskRunRecordV3,
    )

    replay = replay_task_v3(task)
    evidence = task.gold_evidence[0]
    query = task.queries[0]
    all_entries = tuple(
        MemoryEntryRecordV3(
            entry_id=f"{ledger_index}-v-{version.version_index}",
            content=str(version.value),
            object_key_candidate=version.object_key,
            value_candidate=version.value,
            version_index=version.version_index,
            source_event_ids=version.source_event_ids,
        )
        for ledger_index, ledger in enumerate(replay.ledgers)
        for version in ledger.versions
    )
    required_events = set(evidence.supporting_event_ids)
    retrieved = tuple(
        entry
        for entry in all_entries
        if required_events & set(entry.source_event_ids)
    )
    current_entries = tuple(
        entry
        for entry in all_entries
        if any(
            entry.object_key_candidate.canonical_id == version.object_key.canonical_id
            and entry.version_index == version.version_index
            for version in replay.current_state.values()
        )
    )
    answer = (
        evidence.stale_alternative.answer
        if stale_prediction
        else evidence.answer
    )
    parsed_actions = tuple(
        ParsedManagerActionV3(
            action_id=action.action_id,
            event_id=action.event_id,
            operation=action.operation,
            observed_scope=action.scope,
            target_object_keys=action.target_object_keys,
            value=action.value,
            format_valid=True,
            execution_status="executed",
            fallback_used=False,
            raw_output="oracle",
        )
        for action in task.actions
    )
    prediction = AnswerPredictionV3(
        query_id=query.query_id,
        raw_output="oracle",
        parsed_answer=answer,
        cited_event_ids=evidence.supporting_event_ids,
        cited_object_keys=evidence.supporting_object_keys,
        cited_derivation_step_ids=tuple(
            step.step_id for step in evidence.derivation_steps
        ),
        format_valid=True,
    )
    return TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id=ORACLE_INFO.adapter_id,
        run_id=f"run-{task.task_id}-stale-{stale_prediction}",
        parsed_actions=parsed_actions,
        memory_snapshots=(
            MemorySnapshotV3(
                after_event_id=task.events[-1].event_id,
                entries=current_entries,
                state_by_object={
                    key: version.value for key, version in replay.current_state.items()
                },
                store_size=len(current_entries),
            ),
        ),
        retrieval_traces=(
            RetrievalTraceV3(
                query_id=query.query_id,
                retrieved_entries=retrieved,
                ranks=tuple(range(1, len(retrieved) + 1)),
                gold_in_context=True,
                stale_in_context=False,
                distractor_in_context=False,
            ),
        ),
        answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            object_value_extractor_config_hash="a" * 64,
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )


def test_family_g_reference_oracle_has_perfect_applicable_answer_evidence_retrieval_and_stale_metrics():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    *_, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    for task in compiled.tasks:
        run = _oracle_run(task)
        score = score_task_v3(
            task,
            run,
            _authenticated_score_context(task, run, FULL_CAPABILITIES),
        )
        assert score.action_scores.full_action_exact_match == 1.0
        assert score.state_scores.final_state_accuracy == 1.0
        assert score.answer_scores.exact_match == 1.0
        expected = {
            "synthesis_scores.evidence_precision": 1.0,
            "synthesis_scores.evidence_recall": 1.0,
            "synthesis_scores.evidence_f1": 1.0,
            "synthesis_scores.reasoning_support_accuracy": 1.0,
            "synthesis_scores.stale_propagation_rate": 0.0,
        }
        if task.queries[0].query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP:
            expected["synthesis_scores.multi_hop_accuracy"] = 1.0
            not_applicable = "synthesis_scores.multi_object_accuracy"
        else:
            expected["synthesis_scores.multi_object_accuracy"] = 1.0
            not_applicable = "synthesis_scores.multi_hop_accuracy"
        for path, value in expected.items():
            assert _metric_value(score, path) == value, (task.task_id, path)
        assert _metric_value(score, not_applicable) is None
        assert score.supported_metric_fields[not_applicable].reason is SupportReason.NOT_APPLICABLE


def test_family_g_withheld_capabilities_are_typed_nulls_and_stale_predictions_propagate():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    *_, compile_micro = _api()
    representatives = {}
    for task in compile_micro(_config(), code_revision="5423ef7").tasks:
        representatives.setdefault(task.queries[0].query_type, task)
    assert set(representatives) == {
        QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
        QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
    }

    for query_type, task in representatives.items():
        run = _oracle_run(task)
        withheld = score_task_v3(
            task,
            run,
            _authenticated_score_context(task, run, AdapterCapabilitiesV3()),
        )
        applicable_accuracy = (
            "synthesis_scores.multi_hop_accuracy"
            if query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP
            else "synthesis_scores.multi_object_accuracy"
        )
        for path in SYNTHESIS_PATHS:
            assert _metric_value(withheld, path) is None
            expected_reason = (
                SupportReason.NOT_APPLICABLE
                if path
                in {
                    "synthesis_scores.multi_hop_accuracy",
                    "synthesis_scores.multi_object_accuracy",
                }
                and path != applicable_accuracy
                else SupportReason.NOT_SUPPORTED
            )
            assert withheld.supported_metric_fields[path].reason is expected_reason

        stale_run = _oracle_run(task, stale_prediction=True)
        stale_score = score_task_v3(
            task,
            stale_run,
            _authenticated_score_context(task, stale_run, FULL_CAPABILITIES),
        )
        assert _metric_value(stale_score, applicable_accuracy) == 0.0
        assert stale_score.synthesis_scores.stale_propagation_rate == 1.0
        assert "stale_propagation" in stale_score.failure_flags


def test_family_g_derivations_are_ordered_connected_and_stale_sensitive_at_declared_hop():
    *_, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    one_surface = {
        task.metadata.split_key.semantic_core_id: task
        for task in compiled.tasks
        if task.metadata.extra["surface_variant"] == 0
    }
    core_by_id = {core.core_id: core for core in compiled.cores}
    boolean_answers = []
    exact_answers = []

    for core_id, task in one_surface.items():
        core = core_by_id[core_id]
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        stale = evidence.stale_alternative
        for item in (evidence, stale):
            _assert_connected_topological_derivation(item)
            read_events = {
                event_id
                for step in item.derivation_steps
                if step.operation in {"read", "read_current", "read_version"}
                for event_id in step.supporting_event_ids
            }
            assert read_events == set(item.supporting_event_ids)
        assert all(key.canonical_id in query.text for key in task.target_objects)
        assert "CURRENT_STATE" not in query.text

        if query.query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP:
            hop_count = core.stratification["hop_count"]
            stale_index = core.stratification["stale_operand_index"]
            assert _derivation_depth(evidence) == hop_count
            assert _derivation_depth(stale) == hop_count
            primary_reads = evidence.derivation_steps[:hop_count]
            stale_reads = stale.derivation_steps[:hop_count]
            assert [step.operation for step in primary_reads] == ["read_current"] * hop_count
            assert [step.operation for step in stale_reads].count("read_version") == 1
            assert stale_reads[stale_index].operation == "read_version"
            assert all(
                step.input_step_ids[1] == primary_reads[index].step_id
                for index, step in enumerate(
                    evidence.derivation_steps[hop_count:],
                    start=1,
                )
            )
        elif core.stratification["answer_kind"] == "boolean_consistency":
            boolean_answers.append(evidence.answer)
            assert type(evidence.answer) is bool
            assert stale.answer is (not evidence.answer)
        else:
            exact_answers.append(evidence.answer)
            assert type(evidence.answer) is int and evidence.answer > 0
            assert type(stale.answer) is int and stale.answer > 0
            assert stale.answer != evidence.answer

    assert Counter(boolean_answers) == Counter({True: 3, False: 3})
    assert len(exact_answers) == 6
