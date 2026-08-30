from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import Difficulty, Split
from mub.vnext.generation.post_core_config import load_post_core_data_config
from mub.vnext.generation.post_core_families import generate_post_core_cores
from mub.vnext.generation.post_core_splits import assign_post_core_splits


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "post_core_data.yaml"
SPLIT_QUOTAS = {Split.TRAIN: 630, Split.DEV: 90, Split.TEST: 180}
FAMILY_SPLIT_QUOTAS = {Split.TRAIN: 210, Split.DEV: 30, Split.TEST: 60}
DIFFICULTY_QUOTAS = {
    Difficulty.EASY: {Split.TRAIN: 105, Split.DEV: 15, Split.TEST: 30},
    Difficulty.MEDIUM: {Split.TRAIN: 63, Split.DEV: 9, Split.TEST: 18},
    Difficulty.HARD: {Split.TRAIN: 42, Split.DEV: 6, Split.TEST: 12},
}


@pytest.fixture(scope="module")
def config():
    return load_post_core_data_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cores(config):
    return generate_post_core_cores(config)


def test_post_core_splits_assign_exact_global_family_and_difficulty_counts(config, cores):
    result = assign_post_core_splits(config, cores)

    assert len(result.assignments) == 900
    assert len(result.split_by_expansion_id) == 900
    assert {assignment.expansion_id for assignment in result.assignments} == {
        core.expansion_id for core in cores
    }
    assert set(result.split_by_expansion_id) == {core.expansion_id for core in cores}
    assert result.split_balance.seed == config.seed
    assert Counter(result.split_by_expansion_id.values()) == SPLIT_QUOTAS
    assert dict(result.split_balance.core_counts) == {
        split.value: count for split, count in SPLIT_QUOTAS.items()
    }

    family_split_counts = defaultdict(Counter)
    difficulty_split_counts = defaultdict(Counter)
    for assignment in result.assignments:
        family_split_counts[assignment.family_id][assignment.split] += 1
        difficulty_split_counts[assignment.difficulty][assignment.split] += 1

    for family_id in config.family_ids:
        assert family_split_counts[family_id] == FAMILY_SPLIT_QUOTAS
        assert dict(result.split_balance.family_counts[family_id]) == {
            split.value: count for split, count in FAMILY_SPLIT_QUOTAS.items()
        }
    for difficulty, per_family_expected in DIFFICULTY_QUOTAS.items():
        expected = {
            split: count * len(config.family_ids)
            for split, count in per_family_expected.items()
        }
        assert difficulty_split_counts[difficulty] == expected
        assert dict(result.split_balance.difficulty_counts[difficulty.value]) == {
            split.value: count for split, count in expected.items()
        }


def test_post_core_split_balance_cells_report_ratio_expectations_and_deviation(
    config, cores
):
    result = assign_post_core_splits(config, cores)

    assert len(result.split_balance.cells) == len(config.family_ids) * 3 * 3
    for cell in result.split_balance.cells:
        assert cell.expected == pytest.approx(cell.total * {
            Split.TRAIN: config.splits.train,
            Split.DEV: config.splits.dev,
            Split.TEST: config.splits.test,
        }[cell.split])
        assert cell.deviation == pytest.approx(cell.observed - cell.expected)
        assert abs(cell.deviation) <= 1.0


def test_post_core_split_assignment_is_reorder_invariant_and_does_not_mutate_inputs(
    config, cores
):
    original_ids = tuple(core.expansion_id for core in cores)
    first = assign_post_core_splits(config, cores)
    reversed_result = assign_post_core_splits(config, reversed(cores))

    assert first == reversed_result
    assert tuple(core.expansion_id for core in cores) == original_ids


def test_post_core_split_allocation_key_is_deterministic_and_changes_assignment(
    config, cores
):
    first = assign_post_core_splits(config, cores, allocation_key=17)
    repeated = assign_post_core_splits(config, tuple(cores), allocation_key=17)
    changed = assign_post_core_splits(config, cores, allocation_key=18)

    assert first == repeated
    assert first.split_by_expansion_id != changed.split_by_expansion_id
    assert Counter(changed.split_by_expansion_id.values()) == SPLIT_QUOTAS


def test_post_core_split_result_and_nested_balance_records_are_immutable(config, cores):
    result = assign_post_core_splits(config, cores)

    with pytest.raises(ValidationError, match="frozen"):
        result.assignments[0].split = Split.TEST
    with pytest.raises(TypeError):
        result.split_by_expansion_id[cores[0].expansion_id] = Split.TEST
    with pytest.raises(TypeError):
        result.split_balance.family_counts[config.family_ids[0]]["train"] = 0
    with pytest.raises(ValidationError, match="frozen"):
        result.split_balance.cells[0].observed = 0


def test_post_core_split_rejects_duplicate_missing_and_non_core_inputs(config, cores):
    with pytest.raises(ValueError, match="exactly 900"):
        assign_post_core_splits(config, cores[:-1])
    with pytest.raises(ValueError, match="duplicate expansion_id"):
        assign_post_core_splits(config, [*cores, cores[0]])
    with pytest.raises(TypeError, match="PostCoreSemanticCore"):
        assign_post_core_splits(config, [*cores[:-1], {}])


@pytest.mark.parametrize("bad_key", [True, -1, 1.5, "17"])
def test_post_core_split_rejects_invalid_allocation_keys(config, cores, bad_key):
    with pytest.raises((TypeError, ValueError), match="allocation_key"):
        assign_post_core_splits(config, cores, allocation_key=bad_key)
