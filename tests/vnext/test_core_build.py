from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

import mub.vnext.generation.core_build as core_build
from mub.vnext.contracts import Split
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


def test_compile_core_snapshot_honors_alternate_integral_split_ratios() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    payload = config.model_dump(mode="json")
    payload["splits"] = {"train": 0.5, "dev": 0.25, "test": 0.25}
    alternate = type(config).model_validate(payload)

    snapshot = compile_core_snapshot(alternate, cores_per_family=4)

    assert dict(snapshot.core_counts) == {"train": 8, "dev": 4, "test": 4}
    assert Counter(
        (assignment.task_family.value, assignment.split.value)
        for assignment in snapshot.assignments
    ) == {
        (family, split): count
        for family in (
            "repeated_same_slot_update",
            "interleaved_multi_slot_update",
            "entity_attribute_grounding",
            "noop_write_discipline",
        )
        for split, count in (("train", 2), ("dev", 1), ("test", 1))
    }


def test_compile_core_snapshot_rejects_nonintegral_selected_split_quotas() -> None:
    config = load_core_config(CORE_CONFIG_PATH)

    with pytest.raises(ValueError, match="whole number"):
        compile_core_snapshot(config, cores_per_family=4)
