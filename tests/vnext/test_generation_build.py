from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

import mub.vnext.generation.build as build_module
from mub.vnext.contracts import MemUpdateTask, Split, TaskFamily
from mub.vnext.generation import (
    CompiledPilotTasks,
    compile_pilot_tasks,
    generate_family_a_cores,
    generate_family_b_cores,
    generate_family_c_cores,
    generate_family_d_cores,
    load_pilot_config,
)
from mub.vnext.io import canonical_json_bytes, semantic_task_hash
from mub.vnext.validation import validate_gold_replay, validate_task


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
REVISION = "generation-build-test-revision"
SPLIT_ORDER = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
FAMILY_ORDER = {
    TaskFamily.REPEATED_SAME_SLOT.value: 0,
    TaskFamily.INTERLEAVED_MULTI_SLOT.value: 1,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: 2,
    TaskFamily.NOOP_WRITE_DISCIPLINE.value: 3,
}


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def compiled(config) -> CompiledPilotTasks:
    return compile_pilot_tasks(config, code_revision=REVISION)


@pytest.fixture(scope="module")
def cores_by_family(config):
    return (
        generate_family_a_cores(config),
        generate_family_b_cores(config),
        generate_family_c_cores(config),
        generate_family_d_cores(config),
    )


def _sort_key(task: MemUpdateTask):
    return (
        SPLIT_ORDER[task.metadata.split],
        FAMILY_ORDER[task.task_family],
        task.metadata.split_key.semantic_core_id,
        task.metadata.extra["surface_variant"],
    )


def test_compile_pilot_tasks_has_exact_counts_order_and_clean_validation(
    compiled,
) -> None:
    assert isinstance(compiled.tasks, tuple)
    assert len(compiled.tasks) == 1440
    assert compiled.tasks == tuple(sorted(compiled.tasks, key=_sort_key))
    assert Counter(task.metadata.split for task in compiled.tasks) == {
        Split.TRAIN: 1008,
        Split.DEV: 144,
        Split.TEST: 288,
    }
    assert Counter(task.task_family for task in compiled.tasks) == {
        family: 360 for family in FAMILY_ORDER
    }
    core_variants = Counter(
        task.metadata.split_key.semantic_core_id for task in compiled.tasks
    )
    assert len(core_variants) == 480
    assert set(core_variants.values()) == {3}
    for task in compiled.tasks:
        assert validate_task(task).valid
        assert validate_gold_replay(task).valid

    rows = compiled.tasks_jsonl.splitlines(keepends=True)
    assert len(rows) == 1440
    for task, row in zip(compiled.tasks, rows, strict=True):
        assert row.endswith(b"\n") and not row.endswith(b"\r\n")
        reparsed = MemUpdateTask.model_validate_json(row[:-1])
        assert reparsed == task
        assert canonical_json_bytes(reparsed) + b"\n" == row


def test_same_inputs_are_byte_identical_config_unchanged_and_create_no_files(
    config,
    compiled,
    tmp_path,
    monkeypatch,
) -> None:
    before = canonical_json_bytes(config)
    monkeypatch.chdir(tmp_path)

    repeated = compile_pilot_tasks(config, code_revision=REVISION)

    assert repeated == compiled
    assert repeated.tasks_jsonl == compiled.tasks_jsonl
    assert canonical_json_bytes(config) == before
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(AttributeError):
        repeated.tasks = ()
    with pytest.raises(ValidationError):
        repeated.split_assignment.assignments = ()


def test_reversed_generator_inputs_still_produce_canonical_order(
    config,
    compiled,
    cores_by_family,
    monkeypatch,
) -> None:
    generator_names = (
        "generate_family_a_cores",
        "generate_family_b_cores",
        "generate_family_c_cores",
        "generate_family_d_cores",
    )
    for name, cores in zip(generator_names, cores_by_family, strict=True):
        monkeypatch.setattr(
            build_module,
            name,
            lambda _config, records=cores: list(reversed(records)),
        )
    original_assign = build_module.assign_splits
    calls = 0

    def counted_assign(cores, seed):
        nonlocal calls
        calls += 1
        return original_assign(cores, seed)

    monkeypatch.setattr(build_module, "assign_splits", counted_assign)
    reversed_result = compile_pilot_tasks(config, code_revision=REVISION)

    assert calls == 1
    assert reversed_result == compiled
    assert reversed_result.tasks_jsonl == compiled.tasks_jsonl


def test_revision_changes_only_surface_artifact_provenance(
    config,
    compiled,
) -> None:
    changed = compile_pilot_tasks(config, code_revision="different-revision")

    assert changed.tasks_jsonl != compiled.tasks_jsonl
    assert changed.config_sha256 == compiled.config_sha256
    assert changed.split_assignment == compiled.split_assignment
    assert all(
        task.source.generator is not None
        and task.source.generator.code_revision == "different-revision"
        for task in changed.tasks
    )
    assert [semantic_task_hash(task) for task in changed.tasks] == [
        semantic_task_hash(task) for task in compiled.tasks
    ]


def test_invalid_render_aborts_at_first_sorted_task(
    config,
    compiled,
    monkeypatch,
) -> None:
    original_render = build_module.render_core
    first_task_id = compiled.tasks[0].task_id

    def invalid_render(*args, **kwargs):
        task = original_render(*args, **kwargs)
        if task.task_id == first_task_id:
            object.__setattr__(task, "task_id", " ")
        return task

    monkeypatch.setattr(build_module, "render_core", invalid_render)
    with pytest.raises(
        ValueError,
        match=r"compiled task ' ' failed task validation",
    ):
        compile_pilot_tasks(config, code_revision=REVISION)


@pytest.mark.parametrize("revision", ["", " ", "\t\n"])
def test_code_revision_must_be_explicit_and_nonblank(config, revision) -> None:
    with pytest.raises(ValueError, match="code_revision must not be blank"):
        compile_pilot_tasks(config, code_revision=revision)
