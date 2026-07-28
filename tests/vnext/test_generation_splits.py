from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import Difficulty, Split, TaskFamily
from mub.vnext.generation import (
    GenerationContext,
    assign_splits,
    generate_family_a_cores,
    generate_family_b_cores,
    generate_family_c_cores,
    generate_family_d_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.validation.split import FAMILY_STRATIFICATION_AXES


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
PILOT_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
)
CORE_SPLIT_COUNTS = {
    Split.TRAIN: 84,
    Split.DEV: 12,
    Split.TEST: 24,
}


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cores(config):
    return [
        *generate_family_a_cores(config),
        *generate_family_b_cores(config),
        *generate_family_c_cores(config),
        *generate_family_d_cores(config),
    ]


@pytest.fixture(scope="module")
def assigned(config, cores):
    return assign_splits(cores, config.seed)


def _assignment_map(result):
    return {
        assignment.semantic_core_id: assignment.split
        for assignment in result.assignments
    }


def _render_all(config, cores, result):
    context = GenerationContext(config=config, code_revision="split-test-revision")
    split_by_core = _assignment_map(result)
    return [
        render_core(
            core,
            split=split_by_core[core.core_id],
            surface_variant=variant,
            context=context,
        )
        for core in cores
        for variant in range(config.surface_variants_per_core)
    ]


@pytest.fixture(scope="module")
def rendered_tasks(config, cores, assigned):
    return _render_all(config, cores, assigned)


def _assert_pairwise_disjoint(values_by_split):
    for left, right in combinations((Split.TRAIN, Split.DEV, Split.TEST), 2):
        assert values_by_split[left].isdisjoint(values_by_split[right])


def test_assign_splits_is_exported_as_the_public_split_api():
    from mub.vnext.generation.splits import assign_splits as direct_assign_splits

    assert assign_splits is direct_assign_splits


def test_assign_splits_allocates_exact_family_and_global_core_counts(assigned):
    assert len(assigned.assignments) == 480
    per_family_split = Counter(
        (assignment.task_family, assignment.split)
        for assignment in assigned.assignments
    )
    for family in PILOT_FAMILIES:
        for split, expected in CORE_SPLIT_COUNTS.items():
            assert per_family_split[family, split] == expected

    assert Counter(assignment.split for assignment in assigned.assignments) == {
        Split.TRAIN: 336,
        Split.DEV: 48,
        Split.TEST: 96,
    }
    assert dict(assigned.split_balance.core_counts) == {
        "train": 336,
        "dev": 48,
        "test": 96,
    }
    assert dict(assigned.split_balance.projected_task_counts) == {
        "train": 1008,
        "dev": 144,
        "test": 288,
    }


def test_rendered_variants_inherit_core_split_and_have_exact_task_counts(
    cores, assigned, rendered_tasks
):
    tasks = rendered_tasks
    assert len(tasks) == 1440
    assert Counter(task.metadata.split for task in tasks) == {
        Split.TRAIN: 1008,
        Split.DEV: 144,
        Split.TEST: 288,
    }

    variants_by_core = defaultdict(list)
    for task in tasks:
        variants_by_core[task.metadata.split_key.semantic_core_id].append(task)
    assert set(variants_by_core) == {core.core_id for core in cores}
    for core_id, variants in variants_by_core.items():
        assert len(variants) == 3
        assert {task.metadata.split for task in variants} == {
            _assignment_map(assigned)[core_id]
        }


def test_rendered_assignments_are_leakage_safe_across_all_group_keys_and_hashes(
    rendered_tasks,
):
    tasks = rendered_tasks
    group_fields = (
        "semantic_core_id",
        "trajectory_id",
        "paraphrase_group_id",
        "source_group_id",
        "source_document_id",
        "version_group_id",
    )
    for field in group_fields:
        values_by_split = {split: set() for split in CORE_SPLIT_COUNTS}
        for task in tasks:
            value = getattr(task.metadata.split_key, field)
            if value is not None:
                values_by_split[task.metadata.split].add(value)
        assert all(values_by_split.values()), field
        _assert_pairwise_disjoint(values_by_split)

    hashes_by_split = {split: set() for split in CORE_SPLIT_COUNTS}
    for task in tasks:
        hashes_by_split[task.metadata.split].add(sha256_model(task))
    assert sum(map(len, hashes_by_split.values())) == len(tasks)
    _assert_pairwise_disjoint(hashes_by_split)


def test_assignment_is_byte_stable_for_reordered_input_without_mutation(config, cores):
    config_before = canonical_json_bytes(config)
    cores_before = tuple(canonical_json_bytes(core) for core in cores)
    original_order = tuple(core.core_id for core in cores)

    first = assign_splits(cores, config.seed)
    reversed_result = assign_splits(list(reversed(cores)), config.seed)

    assert canonical_json_bytes(first) == canonical_json_bytes(reversed_result)
    assert tuple(core.core_id for core in cores) == original_order
    assert tuple(canonical_json_bytes(core) for core in cores) == cores_before
    assert canonical_json_bytes(config) == config_before


def test_changed_seed_is_deterministic_and_preserves_exact_counts(config, cores):
    changed_seed = config.seed + 1
    first = assign_splits(cores, changed_seed)
    repeated = assign_splits(tuple(cores), changed_seed)
    baseline = assign_splits(cores, config.seed)

    assert canonical_json_bytes(first) == canonical_json_bytes(repeated)
    assert _assignment_map(first) != _assignment_map(baseline)
    assert Counter(assignment.split for assignment in first.assignments) == {
        Split.TRAIN: 336,
        Split.DEV: 48,
        Split.TEST: 96,
    }


@pytest.mark.parametrize("bad_seed", [True, 1.5, "17", -1])
def test_assign_splits_rejects_non_strict_or_negative_seeds(cores, bad_seed):
    with pytest.raises((TypeError, ValueError), match="seed"):
        assign_splits(cores, bad_seed)


def test_assign_splits_rejects_duplicate_semantic_cores(config, cores):
    with pytest.raises(ValueError, match="duplicate semantic_core_id"):
        assign_splits([*cores, cores[0]], config.seed)


@pytest.mark.parametrize(
    "malformed",
    [
        lambda records: records[:-1],
        lambda records: records[:120],
        lambda records: [
            records[0].model_copy(
                update={"task_family": TaskFamily.INTERLEAVED_MULTI_SLOT}
            ),
            *records[1:],
        ],
    ],
)
def test_assign_splits_rejects_missing_or_malformed_family_counts(
    config, cores, malformed
):
    with pytest.raises(ValueError, match="exactly 120 unique cores per Pilot family"):
        assign_splits(malformed(cores), config.seed)


def test_assign_splits_rejects_non_semantic_core_records(config, cores):
    with pytest.raises(TypeError, match="SemanticCore"):
        assign_splits([*cores[:-1], {}], config.seed)


def test_balance_report_explicitly_covers_authoritative_family_strata(assigned):
    cells = assigned.split_balance.cells
    assert cells
    cells_by_joint_stratum = defaultdict(list)
    for cell in cells:
        expected_axes = FAMILY_STRATIFICATION_AXES[cell.task_family.value]
        assert tuple(cell.strata) == expected_axes
        assert cell.difficulty in {
            Difficulty.EASY,
            Difficulty.MEDIUM,
            Difficulty.HARD,
        }
        assert cell.expected == pytest.approx(
            cell.total * CORE_SPLIT_COUNTS[cell.split] / 120
        )
        assert cell.deviation == pytest.approx(cell.observed - cell.expected)
        assert abs(cell.deviation) <= 1.0
        key = (
            cell.task_family,
            cell.difficulty,
            tuple(cell.strata.items()),
        )
        cells_by_joint_stratum[key].append(cell)

    assert {family for family, _, _ in cells_by_joint_stratum} == set(PILOT_FAMILIES)
    for stratum_cells in cells_by_joint_stratum.values():
        assert [cell.split for cell in stratum_cells] == [
            Split.TRAIN,
            Split.DEV,
            Split.TEST,
        ]
        assert sum(cell.observed for cell in stratum_cells) == stratum_cells[0].total
        assert len({cell.total for cell in stratum_cells}) == 1

    family_totals = Counter()
    family_observed = Counter()
    for (family, _, _), stratum_cells in cells_by_joint_stratum.items():
        family_totals[family] += stratum_cells[0].total
        family_observed[family] += sum(cell.observed for cell in stratum_cells)
    assert family_totals == {family: 120 for family in PILOT_FAMILIES}
    assert family_observed == family_totals


def test_assignment_and_balance_records_are_immutable(assigned):
    with pytest.raises(ValidationError, match="frozen"):
        assigned.assignments[0].split = Split.TEST
    with pytest.raises(TypeError):
        assigned.split_balance.core_counts["train"] = 0
    with pytest.raises(TypeError):
        assigned.split_balance.cells[0].strata["new_axis"] = "bad"
