from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
import json
import os
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import ArtifactRef, Difficulty, Split, TaskFamily, TaskManifest
from mub.vnext.contracts.common import FrozenDict, freeze_json, thaw_json
from mub.vnext.contracts.task import MemUpdateTask, SplitKey, TaskMetadata
from mub.vnext.io.canonical import sha256_model
import mub.vnext.validation.split as split_validation_module
from mub.vnext.validation.split import (
    FAMILY_STRATIFICATION_AXES,
    SliceDefinition,
    SplitException,
    validate_splits,
)
from tests.vnext.factories import build_task


ALL_SPLITS = (Split.TRAIN, Split.DEV, Split.TEST, Split.EVALUATION_ONLY)
GENERIC_PROFILE = {
    "update_depth": 1,
    "update_depth_bucket": "1",
    "active_object_count": 1,
    "entity_ambiguity": "none",
    "attribute_ambiguity": "none",
    "noop_density": 0.0,
    "cross_slot_interleaving": 0.0,
    "stale_count": 0,
    "context_length": 4,
    "context_order": "chronological",
    "version_metadata": "latest_outdated",
    "query_type": "current_state",
    "source_naturalness": "synthetic_direct",
}

EXPECTED_AXES = {
    TaskFamily.REPEATED_SAME_SLOT.value: (
        "update_depth_bucket",
        "active_object_count",
        "cross_slot_interleaving",
    ),
    TaskFamily.INTERLEAVED_MULTI_SLOT.value: (
        "update_depth_bucket",
        "active_object_count",
        "cross_slot_interleaving",
    ),
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: (
        "entity_ambiguity",
        "attribute_ambiguity",
        "alias_namespace_condition",
    ),
    TaskFamily.NOOP_WRITE_DISCIPLINE.value: (
        "write_trap_type",
        "noop_density",
        "duplicate_current_condition",
    ),
    TaskFamily.DELETION_FORGETTING.value: ("deletion_scope", "relearning_condition"),
    TaskFamily.CURRENT_HISTORICAL_QUERY.value: ("query_type", "requested_version_distance"),
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value: ("reasoning_depth", "active_object_count"),
    TaskFamily.REALISTIC_SOURCE_UPDATE.value: ("source_type", "provenance_class"),
}

class HostileStr(str):
    __hash__ = None

    def __str__(self):
        raise RuntimeError("hostile string conversion must not be used")

    def strip(self, *args, **kwargs):
        raise RuntimeError("hostile strip must not be used")

    def __eq__(self, other):
        raise RuntimeError("hostile equality must not be used")

    def __ne__(self, other):
        raise RuntimeError("hostile inequality must not be used")


class HostileKeyStr(HostileStr):
    __hash__ = str.__hash__


class HostileValuesFrozenDict(FrozenDict):
    def values(self):
        raise RuntimeError("hostile values must not escape slice containment")


FAMILY_AXIS_VALUES = {
    "alias_namespace_condition": "qualified",
    "write_trap_type": "irrelevant",
    "duplicate_current_condition": "absent",
    "deletion_scope": "attribute",
    "relearning_condition": "none",
    "requested_version_distance": 1,
    "reasoning_depth": 2,
    "source_type": "synthetic",
    "provenance_class": "generated",
}


def _profile(task_family: str, difficulty: Difficulty = Difficulty.EASY, *, omit: str | None = None):
    profile = {
        **GENERIC_PROFILE,
        "task_family": task_family,
        "difficulty": difficulty.value,
        "profile_name": difficulty.value,
        "profile_version": "1.0.0",
    }
    for axis in FAMILY_STRATIFICATION_AXES.get(task_family, ()):
        if axis not in profile:
            profile[axis] = FAMILY_AXIS_VALUES[axis]
    if omit is not None:
        profile.pop(omit, None)
    return profile


def _task(
    task_id: str,
    split: Split,
    *,
    task_family: str = TaskFamily.REPEATED_SAME_SLOT.value,
    difficulty: Difficulty = Difficulty.EASY,
    groups: dict[str, str | None] | None = None,
    exception_id: str | None = None,
    evaluation_slice: str | None = None,
    omit_axis: str | None = None,
    surface_variant: bool = True,
) -> MemUpdateTask:
    data = build_task().model_dump(mode="python")
    data["task_id"] = task_id
    data["task_family"] = task_family
    data["difficulty"] = difficulty
    if surface_variant:
        data["events"][0]["raw_text"] = f"surface {task_id}"
        data["events"][0]["normalized_text"] = f"surface {task_id}"
    suffix = task_id.replace("|", "_")
    split_key = {
        "semantic_core_id": f"semantic:{suffix}",
        "source_group_id": f"source:{suffix}",
        "trajectory_id": f"trajectory:{suffix}",
        "paraphrase_group_id": f"paraphrase:{suffix}",
        "source_document_id": f"document:{suffix}",
        "version_group_id": f"version:{suffix}",
        "split_exception_id": exception_id,
        "split_policy_version": "1.0.0",
    }
    split_key.update(groups or {})
    data["metadata"]["split"] = split
    data["metadata"]["split_key"] = split_key
    data["metadata"]["profile_name"] = difficulty
    data["metadata"]["resolved_profile"] = _profile(task_family, difficulty, omit=omit_axis)
    data["metadata"]["extra"] = {} if evaluation_slice is None else {"evaluation_slice": evaluation_slice}
    return MemUpdateTask.model_validate(data)


def _artifact(path: str, char: str, record_count: int | None = 0) -> ArtifactRef:
    return ArtifactRef(path=path, sha256=char * 64, media_type="application/jsonl", record_count=record_count)


def _bucket(task: MemUpdateTask) -> str:
    return str(task.metadata.resolved_profile.get("update_depth_bucket", "1"))


def _manifest(
    tasks: Iterable[MemUpdateTask],
    *,
    required_strata: list[dict] | None = None,
    small_cell_deviations: list[dict] | None = None,
) -> TaskManifest:
    tasks = tuple(tasks)
    split_counts = {split.value: sum(task.metadata.split == split for task in tasks) for split in ALL_SPLITS}
    family_counts: dict[str, int] = {}
    semantic_counts = {split.value: 0 for split in ALL_SPLITS}
    for split in ALL_SPLITS:
        semantic_counts[split.value] = len(
            {
                task.metadata.split_key.semantic_core_id
                for task in tasks
                if task.metadata.split == split and task.metadata.split_key.semantic_core_id is not None
            }
        )
    for task in tasks:
        key = f"{task.task_family}|{task.difficulty.value}"
        family_counts[key] = family_counts.get(key, 0) + 1
    if required_strata is None:
        required_strata = [
            {"task_family": family, "difficulty": difficulty, "update_depth_bucket": bucket}
            for family, difficulty, bucket in sorted(
                {(task.task_family, task.difficulty.value, _bucket(task)) for task in tasks}
            )
        ]
    summary = {
        "task_hashes": {task.task_id: sha256_model(task) for task in sorted(tasks, key=lambda item: item.task_id)},
        "required_minimum_strata": required_strata,
        "small_cell_deviations": small_cell_deviations or [],
    }
    return TaskManifest(
        data_release_id="release-test",
        split_policy_version="1.0.0",
        compiler_versions={"fixture": "1.0.0"},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts=split_counts,
        family_difficulty_counts=family_counts,
        semantic_core_counts=semantic_counts,
        task_file_paths_and_hashes=(_artifact("tasks/all.jsonl", "a", len(tasks)),),
        leakage_check_summary=summary,
        human_audit_artifacts=(),
        created_at="2026-07-20T00:00:00Z",
        code_revision="fixture-revision",
    )


def _task_with_unsafe_event_metadata(task: MemUpdateTask, value) -> MemUpdateTask:
    first_event = task.events[0].model_construct(
        **{**task.events[0].__dict__, "metadata": {"unsafe": value}}
    )
    return MemUpdateTask.model_construct(
        **{**task.__dict__, "events": [first_event, *task.events[1:]]}
    )


def _codes(report):
    return [issue.code for issue in report.issues]


def test_split_exception_and_slice_definition_are_strictly_immutable():
    allowed = ["semantic_core_id:shared"]
    filters = {"tags": ["stress"]}
    task_ids = ["a", "b"]
    exception = SplitException(
        split_exception_id="exception-1",
        version="1.0.0",
        rationale="evaluation robustness pair",
        allowed_group_ids=allowed,
        reviewer="reviewer",
    )
    definition = SliceDefinition(name="stress", filters=filters, task_ids=task_ids)
    allowed.append("mutated")
    filters["tags"].append("mutated")
    task_ids.append("c")

    assert exception.allowed_group_ids == ("semantic_core_id:shared",)
    assert definition.filters["tags"] == ("stress",)
    assert definition.task_ids == ("a", "b")
    with pytest.raises(ValidationError):
        exception.version = "2"
    with pytest.raises(TypeError):
        definition.filters["x"] = 1
    assert isinstance(definition.model_dump(mode="json")["filters"], dict)
    assert isinstance(definition.model_dump(mode="json")["task_ids"], list)


def test_split_exception_uses_exact_approved_public_fields():
    assert tuple(SplitException.model_fields) == (
        "split_exception_id",
        "version",
        "rationale",
        "allowed_group_ids",
        "reviewer",
    )


@pytest.mark.parametrize("field", ["split_exception_id", "version", "rationale", "reviewer"])
def test_split_exception_rejects_blank_text(field):
    data = {
        "split_exception_id": "exception-1",
        "version": "1.0.0",
        "rationale": "evaluation robustness pair",
        "allowed_group_ids": ("shared",),
        "reviewer": "reviewer",
    }
    data[field] = "  "
    with pytest.raises(ValidationError, match=field):
        SplitException(**data)


@pytest.mark.parametrize("allowed", [(), ("shared", "shared"), ("",)])
def test_split_exception_requires_nonempty_unique_nonblank_allowed_ids(allowed):
    with pytest.raises(ValidationError, match="allowed_group_ids"):
        SplitException(
            split_exception_id="exception-1",
            version="1.0.0",
            rationale="evaluation robustness pair",
            allowed_group_ids=allowed,
            reviewer="reviewer",
        )


def test_slice_definition_rejects_blank_name_duplicate_or_unsorted_task_ids():
    with pytest.raises(ValidationError, match="name"):
        SliceDefinition(name=" ", filters={}, task_ids=())
    with pytest.raises(ValidationError, match="unique"):
        SliceDefinition(name="x", filters={}, task_ids=("a", "a"))
    with pytest.raises(ValidationError, match="sorted"):
        SliceDefinition(name="x", filters={}, task_ids=("b", "a"))


@pytest.mark.parametrize(
    "field",
    [
        "semantic_core_id",
        "trajectory_id",
        "paraphrase_group_id",
        "source_group_id",
        "source_document_id",
        "version_group_id",
    ],
)
def test_every_explicit_group_field_is_checked_for_cross_split_leakage(field):
    tasks = (
        _task("dev", Split.DEV, groups={field: "shared"}),
        _task("test", Split.TEST, groups={field: "shared"}),
    )
    report = validate_splits(tasks, task_manifest=_manifest(tasks))

    assert "group_leakage_missing_exception" in _codes(report)
    assert any(issue.path == f"groups.{field}.shared" for issue in report.issues)


def test_semantic_core_is_the_explicit_semantic_equivalence_authority():
    tasks = (
        _task("dev-paraphrase", Split.DEV, groups={"semantic_core_id": "same-semantic"}),
        _task("test-paraphrase", Split.TEST, groups={"semantic_core_id": "same-semantic"}),
    )
    report = validate_splits(tasks, task_manifest=_manifest(tasks))
    assert "group_leakage_missing_exception" in _codes(report)
    assert not any("task_id" in issue.message and "infer" in issue.message for issue in report.issues)


def test_training_overlap_is_never_permitted_by_exception():
    tasks = (
        _task("train", Split.TRAIN, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
        _task("test", Split.TEST, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
    )
    exception = SplitException(
        split_exception_id="exception-1",
        version="1",
        rationale="not sufficient for training",
        allowed_group_ids=("trajectory_id:shared",),
        reviewer="reviewer",
    )
    report = validate_splits(tasks, (exception,), _manifest(tasks))
    assert "group_leakage_training" in _codes(report)


def test_nontraining_overlap_accepts_valid_field_qualified_exception():
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
        _task("test", Split.TEST, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
    )
    exception = SplitException(
        split_exception_id="exception-1",
        version="1.0.0",
        rationale="versioned robustness comparison",
        allowed_group_ids=("trajectory_id:shared",),
        reviewer="reviewer",
    )
    assert validate_splits(tasks, (exception,), _manifest(tasks)).valid


def test_nontraining_overlap_accepts_valid_raw_group_id_exception():
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
        _task("eval", Split.EVALUATION_ONLY, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
    )
    exception = SplitException(
        split_exception_id="exception-1",
        version="1.0.0",
        rationale="versioned robustness comparison",
        allowed_group_ids=("shared",),
        reviewer="reviewer",
    )
    assert validate_splits(tasks, (exception,), _manifest(tasks)).valid


@pytest.mark.parametrize(
    ("exceptions", "task_exception_ids", "expected_code"),
    [
        ((), (None, None), "group_leakage_missing_exception"),
        ((), ("missing", "missing"), "undeclared_split_exception"),
        ((), ("one", "two"), "group_leakage_mismatched_exception"),
    ],
)
def test_invalid_nontraining_exception_references_are_rejected(exceptions, task_exception_ids, expected_code):
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}, exception_id=task_exception_ids[0]),
        _task("test", Split.TEST, groups={"trajectory_id": "shared"}, exception_id=task_exception_ids[1]),
    )
    report = validate_splits(tasks, exceptions, _manifest(tasks))
    assert expected_code in _codes(report)


def test_any_task_exception_reference_must_be_declared_even_without_overlap():
    task = _task("eval", Split.EVALUATION_ONLY, exception_id="missing")
    report = validate_splits((task,), task_manifest=_manifest((task,)))
    assert "undeclared_split_exception" in _codes(report)


def test_duplicate_exception_id_is_ambiguous_even_for_identical_no_overlap_declarations():
    task = _task("eval", Split.EVALUATION_ONLY, exception_id="duplicate")
    exception = SplitException(
        split_exception_id="duplicate",
        version="1",
        rationale="same payload",
        allowed_group_ids=("unused",),
        reviewer="reviewer",
    )
    report = validate_splits(
        (task,),
        (exception, exception),
        _manifest((task,)),
    )
    duplicate = next(issue for issue in report.issues if issue.code == "duplicate_split_exception_id")
    assert duplicate.path == "declared_exceptions.duplicate"
    assert not report.valid


def test_conflicting_duplicate_exceptions_are_order_independent_and_never_authorize_overlap():
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}, exception_id="duplicate"),
        _task("test", Split.TEST, groups={"trajectory_id": "shared"}, exception_id="duplicate"),
    )
    first = SplitException(
        split_exception_id="duplicate",
        version="1",
        rationale="first",
        allowed_group_ids=("trajectory_id:shared",),
        reviewer="reviewer-a",
    )
    second = SplitException(
        split_exception_id="duplicate",
        version="2",
        rationale="second",
        allowed_group_ids=("other",),
        reviewer="reviewer-b",
    )
    manifest = _manifest(tasks)

    forward = validate_splits(tasks, (first, second), manifest)
    reverse = validate_splits(tasks, (second, first), manifest)

    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert "duplicate_split_exception_id" in _codes(forward)
    assert "invalid_split_exception_usage" in _codes(forward)
    assert not forward.valid


def test_conflicting_duplicate_exception_without_overlap_is_still_order_independent_invalid():
    task = _task("eval", Split.EVALUATION_ONLY, exception_id="duplicate")
    first = SplitException(
        split_exception_id="duplicate",
        version="1",
        rationale="first",
        allowed_group_ids=("a",),
        reviewer="reviewer-a",
    )
    second = SplitException(
        split_exception_id="duplicate",
        version="2",
        rationale="second",
        allowed_group_ids=("b",),
        reviewer="reviewer-b",
    )
    manifest = _manifest((task,))
    forward = validate_splits((task,), (first, second), manifest)
    reverse = validate_splits((task,), (second, first), manifest)
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert "duplicate_split_exception_id" in _codes(forward)
    assert not forward.valid


def test_exception_must_allow_the_specific_overlapping_group():
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
        _task("test", Split.TEST, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
    )
    exception = SplitException(
        split_exception_id="exception-1",
        version="1",
        rationale="robustness",
        allowed_group_ids=("other",),
        reviewer="reviewer",
    )
    report = validate_splits(tasks, (exception,), _manifest(tasks))
    assert "disallowed_split_exception_group" in _codes(report)


@pytest.mark.parametrize("field", ["version", "rationale", "reviewer", "allowed_group_ids"])
def test_constructed_malformed_declared_exception_is_reported_not_raised(field):
    values = {
        "split_exception_id": "exception-1",
        "version": "1",
        "rationale": "robustness",
        "allowed_group_ids": ("shared",),
        "reviewer": "reviewer",
    }
    values[field] = () if field == "allowed_group_ids" else ""
    exception = SplitException.model_construct(**values)
    task = _task("eval", Split.EVALUATION_ONLY)
    report = validate_splits((task,), (exception,), _manifest((task,)))
    assert not report.valid
    assert "invalid_split_exception" in _codes(report)


def test_distinct_evaluation_only_slices_require_an_exception_for_shared_group():
    tasks = (
        _task("eval-a", Split.EVALUATION_ONLY, groups={"version_group_id": "shared"}, evaluation_slice="a"),
        _task("eval-b", Split.EVALUATION_ONLY, groups={"version_group_id": "shared"}, evaluation_slice="b"),
    )
    report = validate_splits(tasks, task_manifest=_manifest(tasks))
    assert "group_leakage_missing_exception" in _codes(report)


def test_isolated_evaluation_only_task_does_not_require_named_slice():
    task = _task("isolated-eval", Split.EVALUATION_ONLY)
    assert validate_splits((task,), task_manifest=_manifest((task,))).valid


def test_distinct_named_evaluation_slices_accept_valid_exception():
    tasks = (
        _task(
            "eval-a",
            Split.EVALUATION_ONLY,
            groups={"version_group_id": "shared"},
            exception_id="exception-1",
            evaluation_slice="a",
        ),
        _task(
            "eval-b",
            Split.EVALUATION_ONLY,
            groups={"version_group_id": "shared"},
            exception_id="exception-1",
            evaluation_slice="b",
        ),
    )
    exception = SplitException(
        split_exception_id="exception-1",
        version="1",
        rationale="named evaluation robustness pair",
        allowed_group_ids=("version_group_id:shared",),
        reviewer="reviewer",
    )
    assert validate_splits(tasks, (exception,), _manifest(tasks)).valid


def test_exact_surface_content_cannot_cross_splits_even_with_exception():
    tasks = (
        _task("dev-exact", Split.DEV, exception_id="exception-1", surface_variant=False),
        _task("test-exact", Split.TEST, exception_id="exception-1", surface_variant=False),
    )
    exception = SplitException(
        split_exception_id="exception-1",
        version="1",
        rationale="cannot waive exact duplicates",
        allowed_group_ids=("unused",),
        reviewer="reviewer",
    )
    report = validate_splits(tasks, (exception,), _manifest(tasks))
    assert "exact_task_content_leakage" in _codes(report)


def test_evaluation_slice_assignment_is_excluded_from_exact_content_hash():
    tasks = (
        _task("dev-exact", Split.DEV, surface_variant=False),
        _task(
            "eval-exact",
            Split.EVALUATION_ONLY,
            evaluation_slice="named-robustness",
            surface_variant=False,
        ),
    )
    report = validate_splits(tasks, task_manifest=_manifest(tasks))
    assert "exact_task_content_leakage" in _codes(report)


@pytest.mark.parametrize(
    "unsafe",
    [float("nan"), Decimal("NaN"), {"unordered", "set"}, frozenset({"unordered", "set"})],
)
def test_exact_content_hash_rejects_noncanonical_included_values_locally(unsafe):
    task = _task_with_unsafe_event_metadata(_task("test", Split.TEST), unsafe)
    report = validate_splits((task,), task_manifest=_manifest((_task("test", Split.TEST),)))
    assert any(
        issue.code == "noncanonical_task_content_hash"
        and issue.path == "tasks.test.content_hash"
        for issue in report.issues
    )


def test_noncanonical_nan_projection_is_not_lossily_collapsed_with_null_projection():
    nan_task = _task_with_unsafe_event_metadata(
        _task("dev", Split.DEV, surface_variant=False),
        float("nan"),
    )
    null_task = _task_with_unsafe_event_metadata(
        _task("test", Split.TEST, surface_variant=False),
        None,
    )
    report = validate_splits(
        (nan_task, null_task),
        task_manifest=_manifest((
            _task("dev", Split.DEV, surface_variant=False),
            _task("test", Split.TEST, surface_variant=False),
        )),
    )
    assert "noncanonical_task_content_hash" in _codes(report)
    assert "exact_task_content_leakage" not in _codes(report)


@pytest.mark.parametrize(
    ("unsafe", "legal"),
    [
        (Decimal("1.25"), "1.25"),
        (b"bytes", "bytes"),
        (1 + 2j, "(1+2j)"),
    ],
)
def test_nonjson_raw_hash_values_are_rejected_and_never_collide_with_strings(unsafe, legal):
    bad = _task_with_unsafe_event_metadata(
        _task("dev", Split.DEV, surface_variant=False), unsafe
    )
    good = _task_with_unsafe_event_metadata(
        _task("test", Split.TEST, surface_variant=False), legal
    )
    report = validate_splits(
        (bad, good),
        task_manifest=_manifest(
            (
                _task("dev", Split.DEV, surface_variant=False),
                _task("test", Split.TEST, surface_variant=False),
            )
        ),
    )
    assert any(
        issue.code == "noncanonical_task_content_hash"
        and issue.path == "tasks.dev.content_hash"
        for issue in report.issues
    )
    assert "exact_task_content_leakage" not in _codes(report)


@pytest.mark.parametrize(
    ("sibling", "bad_value"),
    [
        ("semantic_core_id", []),
        ("source_group_id", HostileStr("hostile")),
        ("trajectory_id", " padded "),
        ("split_policy_version", object()),
    ],
)
def test_training_exception_reference_survives_malformed_split_key_sibling(sibling, bad_value):
    task = _task("train", Split.TRAIN, exception_id="exception-1")
    values = dict(task.metadata.split_key.__dict__)
    values[sibling] = bad_value
    bad_key = SplitKey.model_construct(**values)
    metadata = TaskMetadata.model_construct(
        **{**task.metadata.__dict__, "split_key": bad_key}
    )
    malformed = MemUpdateTask.model_construct(**{**task.__dict__, "metadata": metadata})
    report = validate_splits((malformed,), task_manifest=_manifest((task,)))
    assert any(
        issue.code == "training_split_exception_reference"
        and issue.path == "tasks.train.metadata.split_key.split_exception_id"
        for issue in report.issues
    )
    assert not any(issue.code.startswith("internal_") for issue in report.issues)


def test_unordered_set_hash_rejection_is_identical_across_python_hash_seeds():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script = r'''
import json
from mub.vnext.contracts import Split
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.validation.split import validate_splits
from tests.vnext.test_split_validation import _manifest, _task, _task_with_unsafe_event_metadata
base = _task("test", Split.TEST)
bad = _task_with_unsafe_event_metadata(base, {"alpha", "beta", "gamma"})
report = validate_splits((bad,), task_manifest=_manifest((base,)))
print(json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
'''
    outputs = []
    for seed in ("1", "777"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = root
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1]
    assert "noncanonical_task_content_hash" in outputs[0]


def test_duplicate_blank_ids_and_malformed_split_key_are_defensively_reported():
    valid = _task("duplicate", Split.TEST)
    malformed_metadata = TaskMetadata.model_construct(
        **{
            **valid.metadata.__dict__,
            "split": "bogus",
            "split_key": SplitKey.model_construct(
                semantic_core_id="",
                source_group_id=None,
                trajectory_id="trajectory",
                paraphrase_group_id=None,
                source_document_id=None,
                version_group_id=None,
                split_exception_id=None,
                split_policy_version="",
            ),
        }
    )
    malformed = MemUpdateTask.model_construct(**{**valid.__dict__, "task_id": "duplicate", "metadata": malformed_metadata})
    report = validate_splits((valid, malformed), task_manifest=_manifest((valid,)))
    codes = _codes(report)
    assert "duplicate_task_id" in codes
    assert "invalid_split" in codes
    assert "malformed_split_key" in codes


@pytest.mark.parametrize("lookalike", [
    {"semantic_core_id": "s", "source_group_id": "g", "trajectory_id": "t", "split_policy_version": "1.0.0"},
    SimpleNamespace(semantic_core_id="s", source_group_id="g", trajectory_id="t", split_policy_version="1.0.0"),
])
def test_split_key_lookalikes_are_rejected_as_malformed_contracts(lookalike):
    task = _task("test", Split.TEST)
    metadata = TaskMetadata.model_construct(**{**task.metadata.__dict__, "split_key": lookalike})
    malformed = MemUpdateTask.model_construct(**{**task.__dict__, "metadata": metadata})

    report = validate_splits((malformed,), task_manifest=_manifest((task,)))

    assert any(
        issue.code == "malformed_split_key"
        and issue.path == "tasks.test.metadata.split_key"
        for issue in report.issues
    )


def test_constructed_malformed_real_split_key_is_revalidated_and_contained():
    task = _task("test", Split.TEST)
    bad_key = SplitKey.model_construct(
        semantic_core_id=[],
        source_group_id="source",
        trajectory_id="trajectory",
        paraphrase_group_id=None,
        source_document_id=None,
        version_group_id=None,
        split_exception_id=None,
        split_policy_version="1.0.0",
    )
    metadata = TaskMetadata.model_construct(**{**task.metadata.__dict__, "split_key": bad_key})
    malformed = MemUpdateTask.model_construct(**{**task.__dict__, "metadata": metadata})

    report = validate_splits((malformed,), task_manifest=_manifest((task,)))

    assert "malformed_split_key" in _codes(report)
    assert not report.valid


@pytest.mark.parametrize(
    "field",
    [
        "semantic_core_id",
        "source_group_id",
        "trajectory_id",
        "paraphrase_group_id",
        "source_document_id",
        "version_group_id",
        "split_exception_id",
        "split_policy_version",
    ],
)
def test_split_key_rejects_whitespace_padded_identifiers_with_field_path(field):
    groups = {field: " shared "}
    exception_id = groups.pop("split_exception_id", None)
    task = _task(
        "test",
        Split.TEST,
        groups=groups,
        exception_id=exception_id,
    )
    report = validate_splits((task,), task_manifest=_manifest((task,)))
    assert any(
        issue.code == "malformed_split_key_field"
        and issue.path == f"tasks.test.metadata.split_key.{field}"
        for issue in report.issues
    )
    assert "malformed_split_key" in _codes(report)


def test_padded_group_id_cannot_form_separate_group_from_canonical_id():
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}),
        _task("test", Split.TEST, groups={"trajectory_id": " shared "}),
    )
    report = validate_splits(tasks, task_manifest=_manifest(tasks))
    assert any(
        issue.code == "malformed_split_key_field"
        and issue.path == "tasks.test.metadata.split_key.trajectory_id"
        for issue in report.issues
    )


def test_training_task_cannot_reference_exception_without_overlap():
    task = _task("train", Split.TRAIN, exception_id="exception-1")
    exception = SplitException(
        split_exception_id="exception-1",
        version="1",
        rationale="cannot authorize training",
        allowed_group_ids=("unused",),
        reviewer="reviewer",
    )
    report = validate_splits((task,), (exception,), _manifest((task,)))
    assert any(
        issue.code == "training_split_exception_reference"
        and issue.path == "tasks.train.metadata.split_key.split_exception_id"
        for issue in report.issues
    )


def test_conflicting_split_for_same_task_id_is_reported():
    tasks = (_task("same", Split.DEV), _task("same", Split.TEST))
    report = validate_splits(tasks, task_manifest=_manifest(tasks))
    assert "conflicting_task_split" in _codes(report)


def test_manifest_lookalike_is_rejected_as_malformed_task_manifest():
    task = _task("test", Split.TEST)
    fake = SimpleNamespace(
        split_counts={"train": 0, "dev": 0, "test": 1, "evaluation_only": 0}
    )
    report = validate_splits((task,), task_manifest=fake)
    assert any(
        issue.code == "malformed_task_manifest" and issue.path == "task_manifest"
        for issue in report.issues
    )


@pytest.mark.parametrize("artifact_value", [
    {"path": "tasks.jsonl", "sha256": "a" * 64, "media_type": "application/jsonl", "record_count": 1},
    SimpleNamespace(path="tasks.jsonl", sha256="a" * 64, media_type="application/jsonl", record_count=1),
    ArtifactRef.model_construct(path="tasks.jsonl", sha256="bad", media_type="application/jsonl", record_count=1),
])
def test_manifest_nested_artifacts_require_real_revalidated_artifact_contracts(artifact_value):
    task = _task("test", Split.TEST)
    base = _manifest((task,))
    malformed = TaskManifest.model_construct(
        **{**base.__dict__, "task_file_paths_and_hashes": (artifact_value,)}
    )
    report = validate_splits((task,), task_manifest=malformed)
    assert any(
        issue.code == "malformed_manifest_artifact"
        and issue.path == "task_manifest.task_file_paths_and_hashes[0]"
        for issue in report.issues
    )
    assert not report.valid


def test_constructed_manifest_mutable_maps_are_reported_and_not_mutated():
    task = _task("test", Split.TEST)
    base = _manifest((task,))
    mutable_counts = dict(base.split_counts)
    malformed = TaskManifest.model_construct(
        **{**base.__dict__, "split_counts": mutable_counts}
    )
    before = dict(mutable_counts)
    report = validate_splits((task,), task_manifest=malformed)
    assert any(
        issue.code == "malformed_task_manifest"
        and issue.path == "task_manifest.split_counts"
        for issue in report.issues
    )
    assert mutable_counts == before


def test_manifest_count_maps_have_exact_stable_keys_and_values():
    tasks = (_task("train", Split.TRAIN), _task("test", Split.TEST))
    manifest = _manifest(tasks)
    assert dict(manifest.split_counts) == {"train": 1, "dev": 0, "test": 1, "evaluation_only": 0}
    assert dict(manifest.semantic_core_counts) == {"train": 1, "dev": 0, "test": 1, "evaluation_only": 0}
    assert validate_splits(tasks, task_manifest=manifest).valid


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        ("split_counts", {"train": 0, "dev": 0, "test": 0, "evaluation_only": 0}, "manifest_split_counts_mismatch"),
        ("family_difficulty_counts", {}, "manifest_family_difficulty_counts_mismatch"),
        ("semantic_core_counts", {"train": 0, "dev": 0, "test": 0, "evaluation_only": 0}, "manifest_semantic_core_counts_mismatch"),
    ],
)
def test_manifest_count_mismatches_are_reported(field, replacement, code):
    tasks = (_task("test", Split.TEST),)
    manifest = _manifest(tasks).validated_replace(**{field: replacement})
    report = validate_splits(tasks, task_manifest=manifest)
    assert code in _codes(report)


def test_manifest_reports_unique_task_count_mismatch_for_duplicate_ids():
    tasks = (_task("duplicate", Split.TEST), _task("duplicate", Split.TEST))
    report = validate_splits(tasks, task_manifest=_manifest(tasks))
    assert "manifest_unique_task_count_mismatch" in _codes(report)


def test_task_file_record_counts_are_required_and_must_sum_to_total_tasks():
    tasks = (_task("test", Split.TEST),)
    missing = _manifest(tasks).validated_replace(task_file_paths_and_hashes=(_artifact("tasks.jsonl", "a", None),))
    mismatch = _manifest(tasks).validated_replace(task_file_paths_and_hashes=(_artifact("tasks.jsonl", "a", 2),))
    missing_codes = _codes(validate_splits(tasks, task_manifest=missing))
    assert "missing_task_file_record_count" in missing_codes
    assert "task_file_record_count_mismatch" in missing_codes
    assert "task_file_record_count_mismatch" in _codes(validate_splits(tasks, task_manifest=mismatch))


def test_manifest_task_schema_and_split_policy_versions_must_match_tasks():
    task = _task("test", Split.TEST)
    bad_schema_task = MemUpdateTask.model_construct(**{**task.__dict__, "schema_version": "9.9.9"})
    bad_manifest = _manifest((task,)).validated_replace(task_schema_version="9.9.9", split_policy_version="9.9.9")
    report = validate_splits((bad_schema_task,), task_manifest=bad_manifest)
    codes = _codes(report)
    assert "task_schema_version_mismatch" not in codes
    assert "task_split_policy_version_mismatch" in codes


def test_manifest_task_schema_version_mismatch_is_reported():
    task = _task("test", Split.TEST)
    manifest = _manifest((task,)).validated_replace(task_schema_version="9.9.9")
    assert "task_schema_version_mismatch" in _codes(validate_splits((task,), task_manifest=manifest))


def test_manifest_task_hash_ledger_requires_exact_task_set_and_sha256_model_hashes():
    task = _task("test", Split.TEST)
    base = _manifest((task,))
    missing_summary = base.validated_replace(leakage_check_summary={"required_minimum_strata": [], "small_cell_deviations": []})
    malformed_summary = base.validated_replace(leakage_check_summary={"task_hashes": [], "required_minimum_strata": [], "small_cell_deviations": []})
    wrong_summary = base.validated_replace(
        leakage_check_summary={
            "task_hashes": {"test": "0" * 64, "extra": "1" * 64},
            "required_minimum_strata": [],
            "small_cell_deviations": [],
        }
    )
    assert "missing_manifest_task_hashes" in _codes(validate_splits((task,), task_manifest=missing_summary))
    assert "malformed_manifest_task_hashes" in _codes(validate_splits((task,), task_manifest=malformed_summary))
    wrong_codes = _codes(validate_splits((task,), task_manifest=wrong_summary))
    assert "manifest_task_hash_set_mismatch" in wrong_codes
    assert "manifest_task_hash_mismatch" in wrong_codes


def test_minimum_stratum_derives_bucket_from_strict_positive_update_depth():
    task = _task("test", Split.TEST)
    data = task.model_dump(mode="python")
    data["metadata"]["resolved_profile"].pop("update_depth_bucket")
    derived = MemUpdateTask.model_validate(data)
    assert validate_splits((derived,), task_manifest=_manifest((derived,))).valid


@pytest.mark.parametrize("depth", [0, True, 1.0])
def test_minimum_stratum_rejects_nonpositive_or_nonstrict_update_depth(depth):
    task = _task("test", Split.TEST)
    data = task.model_dump(mode="python")
    data["metadata"]["resolved_profile"].pop("update_depth_bucket")
    data["metadata"]["resolved_profile"]["update_depth"] = depth
    malformed = MemUpdateTask.model_validate(data)
    report = validate_splits((malformed,), task_manifest=_manifest((malformed,)))
    assert "missing_update_depth_bucket" in _codes(report)


def test_profile_labels_and_difficulty_must_agree():
    task = _task("test", Split.TEST)
    data = task.model_dump(mode="python")
    data["metadata"]["resolved_profile"]["difficulty"] = "hard"
    data["metadata"]["resolved_profile"]["profile_name"] = "hard"
    mismatched = MemUpdateTask.model_validate(data)
    report = validate_splits((mismatched,), task_manifest=_manifest((mismatched,)))
    assert "profile_difficulty_mismatch" in _codes(report)


def test_empty_required_strata_ledger_cannot_bypass_derived_strata():
    task = _task("test", Split.TEST)
    manifest = _manifest((task,), required_strata=[])
    report = validate_splits((task,), task_manifest=manifest)
    assert "required_minimum_strata_mismatch" in _codes(report)


def test_required_strata_ledger_rejects_omitted_and_extra_strata():
    tasks = (
        _task("a", Split.TEST, task_family="future-a"),
        _task("b", Split.TEST, task_family="future-b"),
    )
    omitted = _manifest(
        tasks,
        required_strata=[
            {"task_family": "future-a", "difficulty": "easy", "update_depth_bucket": "1"}
        ],
    )
    extra_records = [
        {"task_family": family, "difficulty": "easy", "update_depth_bucket": "1"}
        for family in ("future-a", "future-b", "future-extra")
    ]
    extra = _manifest(tasks, required_strata=extra_records)

    assert "required_minimum_strata_mismatch" in _codes(validate_splits(tasks, task_manifest=omitted))
    assert "required_minimum_strata_mismatch" in _codes(validate_splits(tasks, task_manifest=extra))


def test_malformed_required_stratum_is_reported_and_cannot_replace_derived_set():
    task = _task("test", Split.TEST)
    manifest = _manifest((task,), required_strata=[{"task_family": task.task_family}])
    codes = _codes(validate_splits((task,), task_manifest=manifest))
    assert "invalid_required_minimum_stratum" in codes
    assert "required_minimum_strata_mismatch" in codes


@pytest.mark.parametrize(
    "record",
    [
        {"task_family": [], "difficulty": "easy", "update_depth_bucket": "1"},
        {"task_family": "future", "difficulty": {}, "update_depth_bucket": "1"},
        {"task_family": "future", "difficulty": "easy", "update_depth_bucket": []},
        {"task_family": 7, "difficulty": "easy", "update_depth_bucket": "1"},
        {"task_family": "future", "difficulty": "", "update_depth_bucket": "1"},
    ],
)
def test_required_strata_reject_nonstring_unhashable_fields_without_internal_error(record):
    task = _task("test", Split.TEST, task_family="future")
    manifest = _manifest((task,), required_strata=[record])
    report = validate_splits((task,), task_manifest=manifest)
    codes = _codes(report)
    assert "invalid_required_minimum_stratum" in codes
    assert "required_minimum_strata_mismatch" in codes
    assert "internal_strata_validation_error" not in codes


@pytest.mark.parametrize(
    "record",
    [
        {
            "task_family": "future",
            "difficulty": "easy",
            "update_depth_bucket": "1",
            "extra": "drift",
        },
        {
            "task_family": "future",
            "difficulty": "easy",
            "update_depth_buket": "1",
        },
        {
            7: "future",
            "difficulty": "easy",
            "update_depth_bucket": "1",
        },
        {
            HostileKeyStr("task_family"): "future",
            "difficulty": "easy",
            "update_depth_bucket": "1",
        },
    ],
)
def test_required_stratum_requires_exact_builtin_key_schema(record):
    issues = []
    parsed = split_validation_module._parse_required_strata([record], issues)
    assert not parsed
    assert any(
        issue.code == "invalid_required_minimum_stratum"
        and issue.path
        == "task_manifest.leakage_check_summary.required_minimum_strata[0]"
        for issue in issues
    )


def test_hostile_str_subclass_task_family_is_rejected_without_invoking_overrides():
    base = _task("test", Split.TEST)
    hostile = MemUpdateTask.model_construct(
        **{**base.__dict__, "task_family": HostileStr(base.task_family)}
    )
    report = validate_splits((hostile,), task_manifest=_manifest((base,)))
    assert "invalid_task_family" in _codes(report)
    assert not any(issue.code.startswith("internal_") for issue in report.issues)


def test_hostile_str_subclass_required_ledger_fields_are_rejected_locally():
    task = _task("test", Split.TEST, task_family="future")
    base = _manifest((task,))
    summary = thaw_json(base.leakage_check_summary)
    summary["required_minimum_strata"] = [
        {
            "task_family": HostileStr("future"),
            "difficulty": HostileStr("easy"),
            "update_depth_bucket": HostileStr("1"),
        }
    ]
    manifest = TaskManifest.model_construct(
        **{**base.__dict__, "leakage_check_summary": freeze_json(summary)}
    )
    report = validate_splits((task,), task_manifest=manifest)
    assert "invalid_required_minimum_stratum" in _codes(report)
    assert not any(issue.code.startswith("internal_") for issue in report.issues)


def test_hostile_str_subclass_deviation_fields_are_rejected_before_membership():
    tasks = (
        _task("dev-b", Split.DEV, task_family="future-b"),
        _task("test-a", Split.TEST, task_family="future-a"),
    )
    base = _manifest(tasks)
    summary = thaw_json(base.leakage_check_summary)
    summary["small_cell_deviations"] = [
        {
            "task_family": HostileStr("future-a"),
            "difficulty": HostileStr("easy"),
            "update_depth_bucket": HostileStr("1"),
            "split": HostileStr("dev"),
            "observed_count": 0,
            "rationale": "approved",
        },
        {
            "task_family": HostileStr("future-b"),
            "difficulty": HostileStr("easy"),
            "update_depth_bucket": HostileStr("1"),
            "split": HostileStr("test"),
            "observed_count": 0,
            "rationale": "approved",
        },
    ]
    manifest = TaskManifest.model_construct(
        **{**base.__dict__, "leakage_check_summary": freeze_json(summary)}
    )
    report = validate_splits(tasks, task_manifest=manifest)
    assert "invalid_small_cell_deviation" in _codes(report)
    assert not any(issue.code.startswith("internal_") for issue in report.issues)


def test_hostile_small_cell_deviation_key_is_rejected_before_lookup_or_set_use():
    record = {
        HostileKeyStr("task_family"): "future",
        "difficulty": "easy",
        "update_depth_bucket": "1",
        "split": "dev",
        "observed_count": 0,
        "rationale": "approved",
    }
    issues = []
    parsed = split_validation_module._parse_small_cell_deviations(
        [record], (("future", "easy", "1"),), {}, {"dev": 1}, issues
    )
    assert not parsed
    assert [issue.code for issue in issues] == ["invalid_small_cell_deviation"]


def test_required_strata_duplicate_detection_scales_near_linearly_for_large_unique_ledgers():
    def measure(size: int) -> float:
        records = [
            {
                "task_family": f"future-{index:05d}",
                "difficulty": "easy",
                "update_depth_bucket": "1",
            }
            for index in range(size)
        ]
        durations = []
        for _ in range(3):
            issues = []
            started = time.perf_counter()
            parsed = split_validation_module._parse_required_strata(records, issues)
            durations.append(time.perf_counter() - started)
            assert len(parsed) == size
            assert not issues
        return statistics.median(durations)

    two_thousand = measure(2000)
    four_thousand = measure(4000)
    assert four_thousand < two_thousand * 3.4


def test_small_cell_deviation_membership_builds_required_set_once():
    class CountingRequired(list):
        contains_calls = 0

        def __contains__(self, item):
            self.contains_calls += 1
            return super().__contains__(item)

    size = 500
    required = CountingRequired(
        (f"future-{index}", "easy", "1") for index in range(size)
    )
    deviations = [
        {
            "task_family": f"future-{index}",
            "difficulty": "easy",
            "update_depth_bucket": "1",
            "split": "dev",
            "observed_count": 0,
            "rationale": "approved",
        }
        for index in range(size)
    ]
    issues = []
    valid = split_validation_module._parse_small_cell_deviations(
        deviations,
        required,
        {},
        {"dev": 1},
        issues,
    )
    assert len(valid) == size
    assert not issues
    assert required.contains_calls == 0


def test_derived_strata_not_manifest_claim_drive_cross_split_coverage():
    tasks = (
        _task("train-a", Split.TRAIN, task_family="future-a"),
        _task("test-b", Split.TEST, task_family="future-b"),
    )
    report = validate_splits(tasks, task_manifest=_manifest(tasks))
    missing = [issue for issue in report.issues if issue.code == "missing_required_minimum_stratum"]
    assert len(missing) == 2


def test_exact_small_cell_deviations_can_cover_each_derived_missing_cell():
    tasks = (
        _task("dev-b", Split.DEV, task_family="future-b"),
        _task("test-a", Split.TEST, task_family="future-a"),
    )
    deviations = [
        {
            "task_family": "future-a",
            "difficulty": "easy",
            "update_depth_bucket": "1",
            "split": "dev",
            "observed_count": 0,
            "rationale": "approved small cell",
        },
        {
            "task_family": "future-b",
            "difficulty": "easy",
            "update_depth_bucket": "1",
            "split": "test",
            "observed_count": 0,
            "rationale": "approved small cell",
        },
    ]
    report = validate_splits(
        tasks,
        task_manifest=_manifest(tasks, small_cell_deviations=deviations),
    )
    assert report.valid


def test_required_minimum_stratum_must_exist_in_each_declared_standard_split():
    tasks = (_task("train", Split.TRAIN), _task("test", Split.TEST))
    manifest = _manifest(tasks).validated_replace(
        split_counts={"train": 1, "dev": 1, "test": 1, "evaluation_only": 0}
    )
    report = validate_splits(tasks, task_manifest=manifest)
    assert "missing_required_minimum_stratum" in _codes(report)
    assert any(issue.path.endswith(".dev") for issue in report.issues if issue.code == "missing_required_minimum_stratum")


def test_valid_small_cell_deviation_explicitly_accounts_for_missing_required_cell():
    task = _task("test", Split.TEST)
    stratum = {
        "task_family": task.task_family,
        "difficulty": task.difficulty.value,
        "update_depth_bucket": "1",
    }
    manifest = _manifest(
        (task,),
        required_strata=[stratum],
        small_cell_deviations=[
            {**stratum, "split": "dev", "observed_count": 0, "rationale": "licensed source shortage"}
        ],
    ).validated_replace(split_counts={"train": 0, "dev": 1, "test": 1, "evaluation_only": 0})
    report = validate_splits((task,), task_manifest=manifest)
    assert "missing_required_minimum_stratum" not in _codes(report)
    assert "invalid_small_cell_deviation" not in _codes(report)


@pytest.mark.parametrize(
    "deviation",
    [
        {},
        {"task_family": "x", "difficulty": "easy", "update_depth_bucket": "1", "split": "dev", "observed_count": 0, "rationale": ""},
        {"task_family": TaskFamily.REPEATED_SAME_SLOT.value, "difficulty": "easy", "update_depth_bucket": "1", "split": "bogus", "observed_count": 0, "rationale": "reason"},
        {"task_family": TaskFamily.REPEATED_SAME_SLOT.value, "difficulty": "easy", "update_depth_bucket": "1", "split": "dev", "observed_count": 1, "rationale": "reason"},
    ],
)
def test_malformed_small_cell_deviations_are_reported(deviation):
    task = _task("test", Split.TEST)
    manifest = _manifest((task,), small_cell_deviations=[deviation])
    assert "invalid_small_cell_deviation" in _codes(validate_splits((task,), task_manifest=manifest))


def test_small_cell_deviation_rejects_unhashable_split_without_internal_error():
    task = _task("test", Split.TEST, task_family="future")
    deviation = {
        "task_family": "future",
        "difficulty": "easy",
        "update_depth_bucket": "1",
        "split": [],
        "observed_count": 0,
        "rationale": "invalid container",
    }
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,), small_cell_deviations=[deviation]),
    )
    assert "invalid_small_cell_deviation" in _codes(report)
    assert "internal_strata_validation_error" not in _codes(report)


@pytest.mark.parametrize("case", ["zero_count_split", "unrequired", "superfluous", "extra_field"])
def test_small_cell_deviation_must_describe_exact_positive_declared_missing_cell(case):
    task = _task("test", Split.TEST, task_family="future")
    deviation = {
        "task_family": "future",
        "difficulty": "easy",
        "update_depth_bucket": "1",
        "split": "dev",
        "observed_count": 0,
        "rationale": "claimed small cell",
    }
    if case == "unrequired":
        deviation.update(task_family="other", split="test")
    elif case == "superfluous":
        deviation.update(split="test", observed_count=1)
    elif case == "extra_field":
        deviation["unexpected"] = "not allowed"
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,), small_cell_deviations=[deviation]),
    )
    assert "invalid_small_cell_deviation" in _codes(report)


def test_family_stratification_axes_are_exact_and_immutable():
    assert dict(FAMILY_STRATIFICATION_AXES) == EXPECTED_AXES
    with pytest.raises(TypeError):
        FAMILY_STRATIFICATION_AXES["x"] = ("y",)


@pytest.mark.parametrize("task_family", list(EXPECTED_AXES))
def test_all_a_through_h_families_require_their_registered_axes(task_family):
    missing_axis = EXPECTED_AXES[task_family][-1]
    task = _task("test", Split.TEST, task_family=task_family, omit_axis=missing_axis)
    report = validate_splits((task,), task_manifest=_manifest((task,)))
    assert any(
        issue.code == "missing_stratification_axis" and issue.path.endswith(f".{missing_axis}")
        for issue in report.issues
    )


@pytest.mark.parametrize(
    ("task_family", "axis", "bad_value"),
    [
        (TaskFamily.REPEATED_SAME_SLOT.value, "update_depth_bucket", True),
        (TaskFamily.INTERLEAVED_MULTI_SLOT.value, "active_object_count", None),
        (TaskFamily.REPEATED_SAME_SLOT.value, "cross_slot_interleaving", True),
        (TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value, "entity_ambiguity", " "),
        (TaskFamily.NOOP_WRITE_DISCIPLINE.value, "noop_density", []),
        (TaskFamily.DELETION_FORGETTING.value, "deletion_scope", None),
        (TaskFamily.CURRENT_HISTORICAL_QUERY.value, "requested_version_distance", True),
        (TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value, "reasoning_depth", {}),
        (TaskFamily.REALISTIC_SOURCE_UPDATE.value, "provenance_class", ""),
    ],
)
def test_known_family_axes_require_nonnull_axis_aware_usable_values(task_family, axis, bad_value):
    task = _task("test", Split.TEST, task_family=task_family)
    data = task.model_dump(mode="python")
    data["metadata"]["resolved_profile"][axis] = bad_value
    malformed = MemUpdateTask.model_validate(data)

    report = validate_splits((malformed,), task_manifest=_manifest((malformed,)))

    assert any(
        issue.code == "invalid_stratification_axis"
        and issue.path.endswith(f".{axis}")
        for issue in report.issues
    )


@pytest.mark.parametrize("bad_family", [[], {}, 7, " "])
def test_split_validation_rejects_nonstring_or_blank_task_family_without_internal_error(bad_family):
    task = _task("test", Split.TEST)
    malformed = MemUpdateTask.model_construct(
        **{**task.__dict__, "task_family": bad_family}
    )
    report = validate_splits((malformed,), task_manifest=_manifest((task,)))
    assert any(
        issue.code == "invalid_task_family"
        and issue.path == "tasks.test.task_family"
        for issue in report.issues
    )
    assert "internal_strata_validation_error" not in _codes(report)
    assert not report.valid


def test_unknown_future_family_only_requires_minimum_stratum():
    task = _task("future", Split.TEST, task_family="future_family")
    assert validate_splits((task,), task_manifest=_manifest((task,))).valid


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"task_family": TaskFamily.REPEATED_SAME_SLOT.value}, ("dev", "test")),
        ({"difficulty": "easy"}, ("dev", "test")),
        ({"split": "dev"}, ("dev",)),
        ({"source_type": "synthetic"}, ("dev", "test")),
        ({"query_type": "current_state"}, ("dev", "test")),
        ({"tags": ["stress"]}, ("test",)),
        ({"resolved_profile.update_depth_bucket": "1"}, ("dev", "test")),
    ],
)
def test_deterministic_slice_filters_use_only_explicit_task_fields(filters, expected):
    dev = _task("dev", Split.DEV)
    test_data = _task("test", Split.TEST).model_dump(mode="python")
    test_data["metadata"]["tags"] = ["stress", "same-slot"]
    test = MemUpdateTask.model_validate(test_data)
    tasks = (test, dev)
    definition = SliceDefinition(name="slice", filters=filters, task_ids=expected)
    assert validate_splits(tasks, task_manifest=_manifest(tasks), slice_definitions=(definition,)).valid


def test_slice_filter_output_is_identical_for_forward_and_reversed_task_input():
    tasks = (_task("a", Split.DEV), _task("b", Split.TEST), _task("c", Split.EVALUATION_ONLY))
    definition = SliceDefinition(name="current", filters={"query_type": "current_state"}, task_ids=("a", "b", "c"))
    first = validate_splits(tasks, task_manifest=_manifest(tasks), slice_definitions=(definition,))
    second = validate_splits(reversed(tasks), task_manifest=_manifest(tasks), slice_definitions=(definition,))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.valid


def test_slice_mismatch_preserves_unrelated_tasks_and_reports_exact_output():
    tasks = (_task("a", Split.DEV), _task("b", Split.TEST), _task("unrelated", Split.EVALUATION_ONLY))
    definition = SliceDefinition(name="dev", filters={"split": "dev"}, task_ids=("b",))
    report = validate_splits(tasks, task_manifest=_manifest(tasks), slice_definitions=(definition,))
    issue = next(issue for issue in report.issues if issue.code == "slice_task_ids_mismatch")
    assert "['a']" in issue.message
    assert "unrelated" not in issue.message


@pytest.mark.parametrize(
    "filters",
    [
        {"task_family": " "},
        {"difficulty": "expert"},
        {"split": []},
        {"source_type": "private_wiki"},
        {"query_type": "latest_answer"},
        {"tags": []},
        {"tags": [""]},
        {"resolved_profile.update_depth": True},
        {"resolved_profile.context_order": {}},
    ],
)
def test_malformed_slice_filter_values_cannot_false_validate_empty_slices(filters):
    task = _task("test", Split.TEST)
    definition = SliceDefinition(name="bad-value", filters=filters, task_ids=())
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,)),
        slice_definitions=(definition,),
    )
    assert "malformed_slice_filter" in _codes(report)
    assert not report.valid


def test_constructed_mutable_exception_and_slice_contracts_are_reported_without_mutation():
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
        _task("test", Split.TEST, groups={"trajectory_id": "shared"}, exception_id="exception-1"),
    )
    mutable_allowed = ["trajectory_id:shared"]
    exception = SplitException.model_construct(
        split_exception_id="exception-1",
        version="1",
        rationale="mutable constructed input",
        allowed_group_ids=mutable_allowed,
        reviewer="reviewer",
    )
    mutable_filters = {"split": "dev"}
    mutable_ids = ["dev"]
    definition = SliceDefinition.model_construct(
        name="dev",
        filters=mutable_filters,
        task_ids=mutable_ids,
    )

    report = validate_splits(
        tasks,
        (exception,),
        _manifest(tasks),
        (definition,),
    )

    assert "invalid_split_exception" in _codes(report)
    assert "malformed_slice_definition" in _codes(report)
    assert mutable_allowed == ["trajectory_id:shared"]
    assert mutable_filters == {"split": "dev"}
    assert mutable_ids == ["dev"]


def test_constructed_unhashable_slice_sequence_is_locally_reported_as_malformed_contract():
    task = _task("test", Split.TEST)
    definition = SliceDefinition.model_construct(
        name="bad",
        filters={"split": "test"},
        task_ids=([],),
    )
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,)),
        slice_definitions=(definition,),
    )
    assert "malformed_slice_definition" in _codes(report)
    assert "internal_slice_validation_error" not in _codes(report)


@pytest.mark.parametrize("filters", [{"path": "train.jsonl"}, {"resolved_profile.unknown": 1}, {"task_id": "looks-like-dev"}])
def test_unknown_or_inferred_slice_filters_fail(filters):
    with pytest.raises(ValidationError, match="unsupported slice filter"):
        SliceDefinition(name="bad", filters=filters, task_ids=())


def test_duplicate_name_slice_reports_are_canonical_across_definition_and_filter_order():
    tasks = (_task("dev", Split.DEV), _task("test", Split.TEST))
    family = TaskFamily.REPEATED_SAME_SLOT.value
    dev_forward = SliceDefinition(
        name="duplicate",
        filters={"split": "dev", "task_family": family},
        task_ids=(),
    )
    dev_reverse = SliceDefinition(
        name="duplicate",
        filters={"task_family": family, "split": "dev"},
        task_ids=(),
    )
    test_forward = SliceDefinition(
        name="duplicate",
        filters={"split": "test", "task_family": family},
        task_ids=(),
    )
    test_reverse = SliceDefinition(
        name="duplicate",
        filters={"task_family": family, "split": "test"},
        task_ids=(),
    )
    manifest = _manifest(tasks)

    forward = validate_splits(
        tasks,
        task_manifest=manifest,
        slice_definitions=(dev_forward, test_forward),
    )
    reverse = validate_splits(
        reversed(tasks),
        task_manifest=manifest,
        slice_definitions=(test_reverse, dev_reverse),
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")


def test_noncanonical_slice_filter_is_rejected_before_sort_without_internal_fallback():
    task = _task("test", Split.TEST)
    definition = SliceDefinition.model_construct(
        name="unsafe",
        filters={"tags": {"unordered", "set"}},
        task_ids=(),
    )
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,)),
        slice_definitions=(definition,),
    )
    assert "malformed_slice_definition" in _codes(report)
    assert "internal_slice_validation_error" not in _codes(report)


@pytest.mark.parametrize(
    "task_ids",
    [
        (HostileStr("test"),),
        (b"test",),
        (bytearray(b"test"),),
        (object(),),
    ],
)
def test_constructed_slice_task_ids_require_exact_strings_before_set_or_sort(task_ids):
    task = _task("test", Split.TEST)
    definition = SliceDefinition.model_construct(
        name="hostile-ids", filters=FrozenDict({}), task_ids=task_ids
    )
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,)),
        slice_definitions=(definition,),
    )
    assert _codes(report).count("malformed_slice_definition") == 1
    assert "internal_slice_validation_error" not in _codes(report)


def test_hostile_frozen_filter_values_are_contained_per_slice_definition():
    task = _task("test", Split.TEST)
    definition = SliceDefinition.model_construct(
        name="hostile-filter",
        filters=HostileValuesFrozenDict({"split": "test"}),
        task_ids=("test",),
    )
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,)),
        slice_definitions=(definition,),
    )
    assert _codes(report).count("malformed_slice_definition") == 1
    assert "internal_slice_validation_error" not in _codes(report)


def test_cyclic_constructed_slice_filters_are_local_and_order_deterministic():
    task = _task("test", Split.TEST)
    first_cycle = {}
    first_cycle["cycle"] = first_cycle
    second_cycle = {}
    second_cycle["cycle"] = second_cycle
    first = SliceDefinition.model_construct(
        name="cycle-a", filters=first_cycle, task_ids=()
    )
    second = SliceDefinition.model_construct(
        name="cycle-b", filters=second_cycle, task_ids=()
    )
    manifest = _manifest((task,))
    forward = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(first, second)
    )
    reverse = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(second, first)
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert _codes(forward).count("malformed_slice_definition") == 2
    assert "internal_slice_validation_error" not in _codes(forward)


def test_same_name_cyclic_slices_preserve_task_id_sort_distinction_and_reverse_determinism():
    task = _task("test", Split.TEST)
    first_cycle = {}
    first_cycle["cycle"] = first_cycle
    second_cycle = {}
    second_cycle["cycle"] = second_cycle
    safe_ids = SliceDefinition.model_construct(
        name="same", filters=first_cycle, task_ids=("test",)
    )
    hostile_ids = SliceDefinition.model_construct(
        name="same", filters=second_cycle, task_ids=(HostileStr("test"),)
    )
    assert split_validation_module._slice_sort_key(
        safe_ids
    ) != split_validation_module._slice_sort_key(hostile_ids)

    manifest = _manifest((task,))
    forward = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(safe_ids, hostile_ids)
    )
    reverse = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(hostile_ids, safe_ids)
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert _codes(forward).count("malformed_slice_definition") == 2
    assert "internal_slice_validation_error" not in _codes(forward)


def test_surrogate_slice_name_is_locally_malformed_and_reverse_deterministic():
    task = _task("test", Split.TEST)
    surrogate = chr(0xD800)
    cycle = {}
    cycle["cycle"] = cycle
    first = SliceDefinition.model_construct(
        name=surrogate, filters=FrozenDict({}), task_ids=("test",)
    )
    second = SliceDefinition.model_construct(
        name=surrogate, filters=cycle, task_ids=("test",)
    )
    manifest = _manifest((task,))
    forward = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(first, second)
    )
    reverse = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(second, first)
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert _codes(forward).count("malformed_slice_definition") == 2
    assert "internal_slice_validation_error" not in _codes(forward)


def test_surrogate_slice_task_id_is_locally_malformed_and_reverse_deterministic():
    task = _task("test", Split.TEST)
    surrogate = chr(0xD800)
    cycle = {}
    cycle["cycle"] = cycle
    first = SliceDefinition.model_construct(
        name="same", filters=FrozenDict({}), task_ids=(surrogate,)
    )
    second = SliceDefinition.model_construct(
        name="same", filters=cycle, task_ids=("test",)
    )
    manifest = _manifest((task,))
    forward = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(first, second)
    )
    reverse = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(second, first)
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert _codes(forward).count("malformed_slice_definition") == 2
    assert "internal_slice_validation_error" not in _codes(forward)


def test_failed_filter_canonicalization_does_not_preserve_caller_issue_order():
    task = _task("test", Split.TEST)
    nan_filter = SliceDefinition.model_construct(
        name="same",
        filters=FrozenDict({"resolved_profile.noop_density": float("nan")}),
        task_ids=(),
    )
    set_filter = SliceDefinition.model_construct(
        name="same",
        filters=FrozenDict({"tags": {"x"}}),
        task_ids=(),
    )
    manifest = _manifest((task,))
    forward = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(nan_filter, set_filter)
    )
    reverse = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(set_filter, nan_filter)
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert "internal_slice_validation_error" not in _codes(forward)


def test_duplicate_same_name_malformed_slice_diagnostics_are_order_neutral():
    task = _task("test", Split.TEST)
    first = SliceDefinition(
        name="duplicate", filters={"difficulty": "expert"}, task_ids=()
    )
    second = SliceDefinition(
        name="duplicate",
        filters={"resolved_profile.noop_density": float("nan")},
        task_ids=(),
    )
    manifest = _manifest((task,))
    forward = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(first, second)
    )
    reverse = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(second, first)
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert _codes(forward).count("duplicate_slice_name") == 1
    assert "internal_slice_validation_error" not in _codes(forward)


def test_mixed_valid_and_malformed_slice_diagnostics_have_deterministic_order():
    task = _task("test", Split.TEST)
    valid = SliceDefinition(name="mixed", filters={}, task_ids=("test",))
    malformed = SliceDefinition.model_construct(
        name="mixed", filters=FrozenDict({"tags": {"x"}}), task_ids=("test",)
    )
    manifest = _manifest((task,))
    forward = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(valid, malformed)
    )
    reverse = validate_splits(
        (task,), task_manifest=manifest, slice_definitions=(malformed, valid)
    )
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert _codes(forward).count("malformed_slice_definition") == 1
    assert "internal_slice_validation_error" not in _codes(forward)


def test_slice_sorting_never_depends_on_object_repr(monkeypatch):
    task = _task("test", Split.TEST)
    definition = SliceDefinition(name="all", filters={}, task_ids=("test",))

    def hostile_repr(self):
        raise AssertionError("repr must not participate in slice sorting")

    monkeypatch.setattr(SliceDefinition, "__repr__", hostile_repr)
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,)),
        slice_definitions=(definition,),
    )
    assert report.valid
    assert "internal_slice_validation_error" not in _codes(report)


def test_duplicate_slice_names_and_constructed_duplicate_ids_are_reported():
    task = _task("a", Split.TEST)
    first = SliceDefinition(name="same", filters={}, task_ids=("a",))
    second = SliceDefinition(name="same", filters={}, task_ids=("a",))
    malformed = SliceDefinition.model_construct(name="other", filters={}, task_ids=("a", "a"))
    report = validate_splits(
        (task,),
        task_manifest=_manifest((task,)),
        slice_definitions=(first, second, malformed),
    )
    assert "duplicate_slice_name" in _codes(report)
    assert "duplicate_slice_task_id" in _codes(report)


def test_group_issues_precede_manifest_strata_and_slice_issues():
    tasks = (
        _task("dev", Split.DEV, groups={"trajectory_id": "shared"}, omit_axis="active_object_count"),
        _task("test", Split.TEST, groups={"trajectory_id": "shared"}),
    )
    manifest = _manifest(tasks).validated_replace(family_difficulty_counts={})
    definition = SliceDefinition(name="wrong", filters={"split": "dev"}, task_ids=("test",))
    codes = _codes(validate_splits(tasks, task_manifest=manifest, slice_definitions=(definition,)))
    assert codes.index("group_leakage_missing_exception") < codes.index("manifest_family_difficulty_counts_mismatch")
    assert codes.index("manifest_family_difficulty_counts_mismatch") < codes.index("missing_stratification_axis")
    assert codes.index("missing_stratification_axis") < codes.index("slice_task_ids_mismatch")


def test_split_validation_does_not_mutate_tasks_manifest_exceptions_or_slices():
    task = _task("eval", Split.EVALUATION_ONLY)
    manifest = _manifest((task,))
    exception = SplitException(
        split_exception_id="unused",
        version="1",
        rationale="declared immutable fixture",
        allowed_group_ids=("unused",),
        reviewer="reviewer",
    )
    definition = SliceDefinition(name="all", filters={}, task_ids=("eval",))
    before = tuple(
        value.model_dump(mode="json")
        for value in (task, manifest, exception, definition)
    )

    validate_splits((task,), (exception,), manifest, (definition,))

    after = tuple(
        value.model_dump(mode="json")
        for value in (task, manifest, exception, definition)
    )
    assert after == before


def test_input_order_does_not_change_issue_order_and_iterable_is_consumed_once():
    tasks = (
        _task("b", Split.TEST, groups={"trajectory_id": "shared"}),
        _task("a", Split.DEV, groups={"trajectory_id": "shared"}),
    )
    manifest = _manifest(tasks)

    class Once:
        def __init__(self, values):
            self.values = values
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("tasks iterable consumed twice")
            return iter(self.values)

    once = Once(tasks)
    first = validate_splits(once, task_manifest=manifest)
    second = validate_splits(reversed(tasks), task_manifest=manifest)
    assert once.iterations == 1
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_malformed_constructed_profile_values_never_escape_validation():
    task = _task("test", Split.TEST)
    malformed_metadata = TaskMetadata.model_construct(
        **{
            **task.metadata.__dict__,
            "resolved_profile": {
                **task.metadata.resolved_profile,
                "update_depth_bucket": [],
            },
        }
    )
    malformed = MemUpdateTask.model_construct(
        **{**task.__dict__, "task_family": [], "metadata": malformed_metadata}
    )

    report = validate_splits((malformed,), task_manifest=_manifest((task,)))

    assert not report.valid
    assert "invalid_task_family" in _codes(report)
    assert "internal_strata_validation_error" not in _codes(report)


def test_malformed_constructed_manifest_and_slice_never_raise_or_return_false_valid():
    task = _task("test", Split.TEST)
    manifest = TaskManifest.model_construct(
        data_release_id="bad",
        split_policy_version=None,
        task_schema_version=None,
        split_counts=None,
        family_difficulty_counts=None,
        semantic_core_counts=None,
        task_file_paths_and_hashes=None,
        leakage_check_summary=None,
    )
    malformed_slice = SliceDefinition.model_construct(name="", filters=None, task_ids=None)
    report = validate_splits((task,), task_manifest=manifest, slice_definitions=(malformed_slice,))
    assert not report.valid
    assert "malformed_task_manifest" in _codes(report)
    assert "malformed_slice_definition" in _codes(report)
