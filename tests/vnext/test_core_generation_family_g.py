from collections import Counter, defaultdict
from pathlib import Path

from mub.vnext.contracts.enums import Split, TaskFamily
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3, resolve_query_v3


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG = ROOT / "configs" / "vnext" / "core.yaml"


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


def test_family_g_compiler_renders_exactly_four_replayable_v3_surfaces_per_core():
    _, _, validate_task, _, validate_micro_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")

    assert compiled.profile_id == "family_g_selected_micro_v1"
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
    for surfaces in by_core.values():
        assert len(surfaces) == 4
        assert {task.metadata.extra["surface_variant"] for task in surfaces} == {0, 1, 2, 3}
        assert len({task.semantic_hash for task in surfaces}) == 1
        assert len({task.task_id for task in surfaces}) == 4
        assert len({task.source.raw_hash for task in surfaces}) == 4
