from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

import mub.vnext.generation.core_build as core_build
from mub.vnext.contracts import Split, TaskFamily
from mub.vnext.generation import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import canonical_json_bytes, semantic_task_hash_v3


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG_PATH = ROOT / "configs" / "vnext" / "core.yaml"
FAMILY_GENERATORS = (
    "generate_core_family_a_cores",
    "generate_core_family_b_cores",
    "generate_core_family_c_cores",
    "generate_core_family_d_cores",
)


def test_compile_core_snapshot_sample_is_grouped_leak_free_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    first = compile_core_snapshot(config, cores_per_family=10)
    repeated = compile_core_snapshot(config, cores_per_family=10)

    assert len(first.assignments) == 40
    assert len(first.tasks) == 160
    assert dict(first.core_counts) == {"train": 28, "dev": 4, "test": 8}
    assert dict(first.task_counts) == {"train": 112, "dev": 16, "test": 32}
    assert Counter(
        (assignment.task_family.value, assignment.split.value)
        for assignment in first.assignments
    ) == {
        (family, split): count
        for family in (
            "repeated_same_slot_update",
            "interleaved_multi_slot_update",
            "entity_attribute_grounding",
            "noop_write_discipline",
        )
        for split, count in (("train", 7), ("dev", 1), ("test", 2))
    }

    by_core = defaultdict(list)
    for task in first.tasks:
        by_core[task.metadata.split_key.semantic_core_id].append(task)
    assert set(by_core) == {
        assignment.semantic_core_id for assignment in first.assignments
    }
    for tasks in by_core.values():
        assert len(tasks) == 4
        assert len({task.metadata.split for task in tasks}) == 1
        assert len({task.task_id for task in tasks}) == 4
        assert len({task.source.raw_hash for task in tasks}) == 4
        assert len({semantic_task_hash_v3(task) for task in tasks}) == 1

    for field in (
        "semantic_core_id",
        "source_group_id",
        "trajectory_id",
        "paraphrase_group_id",
        "version_group_id",
    ):
        values_by_split = {
            split: {
                getattr(task.metadata.split_key, field)
                for task in first.tasks
                if task.metadata.split is split
            }
            for split in (Split.TRAIN, Split.DEV, Split.TEST)
        }
        assert values_by_split[Split.TRAIN].isdisjoint(values_by_split[Split.DEV])
        assert values_by_split[Split.TRAIN].isdisjoint(values_by_split[Split.TEST])
        assert values_by_split[Split.DEV].isdisjoint(values_by_split[Split.TEST])
    hashes_by_split = {
        split: {
            semantic_task_hash_v3(task)
            for task in first.tasks
            if task.metadata.split is split
        }
        for split in (Split.TRAIN, Split.DEV, Split.TEST)
    }
    assert hashes_by_split[Split.TRAIN].isdisjoint(hashes_by_split[Split.DEV])
    assert hashes_by_split[Split.TRAIN].isdisjoint(hashes_by_split[Split.TEST])
    assert hashes_by_split[Split.DEV].isdisjoint(hashes_by_split[Split.TEST])

    assert canonical_json_bytes(first) == canonical_json_bytes(repeated)
    for name in FAMILY_GENERATORS:
        original = getattr(core_build, name)
        monkeypatch.setattr(
            core_build,
            name,
            lambda core_config, generator=original: tuple(
                reversed(generator(core_config))
            ),
        )
    reordered = compile_core_snapshot(config, cores_per_family=10)
    assert canonical_json_bytes(first) == canonical_json_bytes(reordered)

    with pytest.raises(ValidationError, match="frozen"):
        first.tasks = ()


def test_compile_core_snapshot_rejects_noncanonical_split_ratios() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    payload = config.model_dump(mode="json")
    payload["splits"] = {"train": 0.5, "dev": 0.25, "test": 0.25}

    with pytest.raises(ValidationError, match="Core split ratios"):
        type(config).model_validate(payload)


def test_compile_core_snapshot_rejects_nonintegral_selected_split_quotas() -> None:
    config = load_core_config(CORE_CONFIG_PATH)

    with pytest.raises(ValueError, match="whole number"):
        compile_core_snapshot(config, cores_per_family=4)


def _replace_stratification(core, **changes):
    stratification = dict(core.stratification)
    stratification.update(changes)
    return core.model_copy(update={"stratification": stratification})


def _replace_profile_and_stratification(core, **changes):
    profile = dict(core.profile)
    stratification = dict(core.stratification)
    profile.update(changes)
    stratification.update(changes)
    return core.model_copy(
        update={"profile": profile, "stratification": stratification}
    )


def test_core_snapshot_rejects_aggregate_preserving_family_a_cell_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    original = core_build.generate_core_family_a_cores

    def corrupted(core_config):
        cores = original(core_config)
        victim_index = next(
            index
            for index, core in enumerate(cores)
            if core.profile["update_depth"] == 1
            and core.stratification["condition"] == "stale_burden"
        )
        cores[victim_index] = _replace_stratification(
            cores[victim_index],
            condition="duplicate_current",
        )
        return cores

    monkeypatch.setattr(core_build, "generate_core_family_a_cores", corrupted)

    with pytest.raises(ValueError, match="Core Family A schedule"):
        core_build._generated_cores(config)


def test_core_snapshot_rejects_aggregate_preserving_family_b_depth_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    original = core_build.generate_core_family_b_cores

    def corrupted(core_config):
        cores = original(core_config)
        victim_index = next(
            index
            for index, core in enumerate(cores)
            if core.stratification["update_depth"] == 1
        )
        cores[victim_index] = _replace_profile_and_stratification(
            cores[victim_index],
            update_depth=4,
        )
        return cores

    monkeypatch.setattr(core_build, "generate_core_family_b_cores", corrupted)

    with pytest.raises(ValueError, match="Core Family B schedule"):
        core_build._generated_cores(config)


def test_core_snapshot_rejects_family_b_cell_imbalance_with_balanced_depth_marginal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    original = core_build.generate_core_family_b_cores

    def corrupted(core_config):
        cores = original(core_config)
        cells = Counter(
            (
                core.stratification["active_object_count"],
                core.stratification["update_depth"],
                core.stratification["interleaving_pattern"],
            )
            for core in cores
        )
        active_count = 2
        pattern = "round_robin"
        source_depth = next(
            depth
            for depth in (1, 4, 16)
            if cells[(active_count, depth, pattern)] == min(
                cells[(active_count, candidate, pattern)]
                for candidate in (1, 4, 16)
            )
        )
        target_depth = next(
            depth
            for depth in (1, 4, 16)
            if cells[(active_count, depth, pattern)] == max(
                cells[(active_count, candidate, pattern)]
                for candidate in (1, 4, 16)
            )
            and depth != source_depth
        )
        first = next(
            index
            for index, core in enumerate(cores)
            if core.stratification["active_object_count"] == active_count
            and core.stratification["update_depth"] == source_depth
            and core.stratification["interleaving_pattern"] == pattern
        )
        second = next(
            index
            for index, core in enumerate(cores)
            if core.stratification["active_object_count"] == 4
            and core.stratification["update_depth"] == target_depth
            and core.stratification["interleaving_pattern"] == pattern
        )
        cores[first] = _replace_profile_and_stratification(
            cores[first], update_depth=target_depth
        )
        cores[second] = _replace_profile_and_stratification(
            cores[second], update_depth=source_depth
        )
        return cores

    monkeypatch.setattr(core_build, "generate_core_family_b_cores", corrupted)

    with pytest.raises(ValueError, match="Core Family B schedule"):
        core_build._generated_cores(config)


def test_core_snapshot_rejects_aggregate_preserving_family_c_cell_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    original = core_build.generate_core_family_c_cores

    def corrupted(core_config):
        cores = original(core_config)
        victim_index = next(
            index
            for index, core in enumerate(cores)
            if core.stratification["entity_condition"] == "distinct"
            and core.stratification["attribute_condition"] == "exact"
        )
        cores[victim_index] = _replace_stratification(
            cores[victim_index],
            entity_condition="alias",
        )
        return cores

    monkeypatch.setattr(core_build, "generate_core_family_c_cores", corrupted)

    with pytest.raises(ValueError, match="Core Family C schedule"):
        core_build._generated_cores(config)


def test_core_snapshot_rejects_family_c_resolution_outcome_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    original = core_build.generate_core_family_c_cores

    def corrupted(core_config):
        cores = original(core_config)
        victim_index = next(
            index
            for index, core in enumerate(cores)
            if core.stratification["resolution_status"] == "unique"
        )
        cores[victim_index] = _replace_stratification(
            cores[victim_index],
            resolution_status="ambiguous",
        )
        return cores

    monkeypatch.setattr(core_build, "generate_core_family_c_cores", corrupted)

    with pytest.raises(ValueError, match="Core Family C schedule"):
        core_build._generated_cores(config)


def test_core_snapshot_rejects_aggregate_preserving_family_d_cell_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    original = core_build.generate_core_family_d_cores

    def corrupted(core_config):
        cores = original(core_config)
        victim_index = next(
            index
            for index, core in enumerate(cores)
            if core.stratification["configured_noop_density"] == 0.25
            and core.stratification["trap_type"] == "transient"
        )
        cores[victim_index] = _replace_stratification(
            cores[victim_index],
            configured_noop_density=0.5,
        )
        return cores

    monkeypatch.setattr(core_build, "generate_core_family_d_cores", corrupted)

    with pytest.raises(ValueError, match="Core Family D schedule"):
        core_build._generated_cores(config)


def _replace_task_stratification(task, **changes):
    extra = dict(task.metadata.extra)
    stratification = dict(extra["stratification"])
    stratification.update(changes)
    extra["stratification"] = stratification
    metadata = task.metadata.validated_replace(extra=extra)
    return task.validated_replace(metadata=metadata)


def test_snapshot_validation_rejects_aggregate_preserving_family_a_corruption() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    snapshot = compile_core_snapshot(config)
    representatives = [
        task
        for task in snapshot.tasks
        if task.task_family == TaskFamily.REPEATED_SAME_SLOT.value
        and task.metadata.extra["surface_variant"] == 0
        and task.metadata.extra["stratification"].get("condition")
        in {"stale_burden", "duplicate_current"}
        and task.metadata.extra["stratification"].get("stale_same_slot_count") == 1
    ]
    stale_task = next(
        task
        for task in representatives
        if task.metadata.extra["stratification"]["condition"] == "stale_burden"
    )
    corrupted_tasks = list(snapshot.tasks)
    by_id = {task.task_id: index for index, task in enumerate(corrupted_tasks)}
    corrupted_tasks[by_id[stale_task.task_id]] = _replace_task_stratification(
        stale_task,
        condition="duplicate_current",
    )
    corrupted = snapshot.validated_replace(tasks=tuple(corrupted_tasks))

    with pytest.raises(ValueError, match="Core Family A schedule"):
        core_build._validate_snapshot(corrupted, config)


def test_core_snapshot_rejects_aggregate_preserving_per_family_split_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    original = core_build._select_and_assign

    def corrupted(*args, **kwargs):
        assignments = list(original(*args, **kwargs))
        family_a_train = next(
            index
            for index, assignment in enumerate(assignments)
            if assignment.task_family is TaskFamily.REPEATED_SAME_SLOT
            and assignment.split is Split.TRAIN
        )
        family_c_dev = next(
            index
            for index, assignment in enumerate(assignments)
            if assignment.task_family is TaskFamily.ENTITY_ATTRIBUTE_GROUNDING
            and assignment.split is Split.DEV
        )
        assignments[family_a_train] = assignments[family_a_train].validated_replace(
            split=Split.DEV
        )
        assignments[family_c_dev] = assignments[family_c_dev].validated_replace(
            split=Split.TRAIN
        )
        return tuple(assignments)

    monkeypatch.setattr(core_build, "_select_and_assign", corrupted)

    with pytest.raises(ValueError, match="per-family split schedule"):
        compile_core_snapshot(config, cores_per_family=10)
