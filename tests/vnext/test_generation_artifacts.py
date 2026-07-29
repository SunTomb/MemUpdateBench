from __future__ import annotations

import builtins
import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import mub.vnext.generation.build as build_module
from mub.vnext.contracts import MemUpdateTask, Split
from mub.vnext.generation import (
    CompiledPilotTasks,
    InMemoryPilotArtifact,
    PilotArtifactBundle,
    SplitBalanceReport,
    build_pilot_artifact_bundle,
    compile_pilot_tasks,
    load_pilot_config,
)
from mub.vnext.io import canonical_json_bytes
from mub.vnext.validation import ValidationIssue, build_report, validate_splits


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
REVISION = "generation-artifacts-test-revision"
ARTIFACT_NAMES = (
    "tasks.jsonl",
    "generation_config.json",
    "split_balance.json",
    "task_manifest.json",
    "validation_report.json",
)


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def compiled(config) -> CompiledPilotTasks:
    return compile_pilot_tasks(config, code_revision=REVISION)


@pytest.fixture(scope="module")
def bundle(compiled, config) -> PilotArtifactBundle:
    return build_pilot_artifact_bundle(compiled, config)


def _clone_compiled(compiled: CompiledPilotTasks) -> CompiledPilotTasks:
    return replace(compiled)


def _replace_artifact(bundle, index, artifact, **changes):
    artifacts = list(bundle.artifacts)
    artifacts[index] = artifact
    return replace(bundle, artifacts=tuple(artifacts), **changes)


def _invalid_report():
    return build_report(
        (
            ValidationIssue(
                code="injected_split_failure",
                message="injected failure",
                path="tasks",
                severity="error",
            ),
        )
    )


def _parsed_bundle_tasks(bundle):
    return [MemUpdateTask.model_validate_json(row) for row in bundle.tasks_jsonl.splitlines()]


def _coordinated_bundle_from_tasks(
    bundle,
    tasks,
    *,
    split_balance_report=None,
):
    import mub.vnext.generation.artifacts as artifacts_module

    tasks = tuple(tasks)
    tasks_bytes = b"".join(canonical_json_bytes(task) + b"\n" for task in tasks)
    tasks_artifact = InMemoryPilotArtifact(
        path="tasks.jsonl",
        content=tasks_bytes,
        media_type="application/x-ndjson",
        record_count=len(tasks),
    )
    config_artifact = bundle.artifacts[1]
    split_balance_report = split_balance_report or bundle.split_balance_report
    split_artifact = replace(
        bundle.artifacts[2],
        content=canonical_json_bytes(split_balance_report),
    )
    generator = tasks[0].source.generator
    manifest = artifacts_module._build_manifest(
        tasks=tasks,
        compiler_versions={generator.generator_name: generator.compiler_version},
        code_revision=generator.code_revision,
        config=bundle.resolved_config,
        config_ref=config_artifact.ref,
        tasks_ref=tasks_artifact.ref,
    )
    manifest_artifact = replace(
        bundle.artifacts[3],
        content=canonical_json_bytes(manifest),
    )
    validation_report = validate_splits(tasks, task_manifest=manifest)
    assert validation_report.valid and not validation_report.issues
    validation_artifact = replace(
        bundle.artifacts[4],
        content=canonical_json_bytes(validation_report),
    )
    return PilotArtifactBundle(
        resolved_config=bundle.resolved_config,
        split_balance_report=split_balance_report,
        task_manifest=manifest,
        validation_report=validation_report,
        artifacts=(
            tasks_artifact,
            config_artifact,
            split_artifact,
            manifest_artifact,
            validation_artifact,
        ),
    )


def _moved_core_split_report(report, exemplar):
    payload = report.model_dump(mode="python")
    payload["core_counts"]["train"] -= 1
    payload["core_counts"]["dev"] += 1
    payload["projected_task_counts"]["train"] -= 3
    payload["projected_task_counts"]["dev"] += 3
    matched = 0
    for cell in payload["cells"]:
        same_stratum = (
            cell["task_family"].value == exemplar.task_family
            and cell["difficulty"] == exemplar.difficulty
            and all(
                exemplar.metadata.resolved_profile[key] == value
                for key, value in cell["strata"].items()
            )
        )
        if not same_stratum or cell["split"] not in {Split.TRAIN, Split.DEV}:
            continue
        delta = -1 if cell["split"] is Split.TRAIN else 1
        cell["observed"] += delta
        cell["deviation"] += float(delta)
        matched += 1
    assert matched == 2
    return SplitBalanceReport.model_validate(payload)


def test_bundle_contains_exact_canonical_counts_hashes_and_artifacts(
    bundle, compiled, config
) -> None:
    tasks = compiled.tasks
    manifest = bundle.task_manifest

    assert bundle.tasks_jsonl is compiled.tasks_jsonl
    assert len(tasks) == 1440
    assert dict(manifest.split_counts) == {
        "train": 1008,
        "dev": 144,
        "test": 288,
        "evaluation_only": 0,
    }
    assert dict(manifest.semantic_core_counts) == {
        "train": 336,
        "dev": 48,
        "test": 96,
        "evaluation_only": 0,
    }
    expected_family_difficulty = Counter(
        f"{task.task_family}|{task.difficulty.value}" for task in tasks
    )
    assert len(manifest.family_difficulty_counts) == 12
    assert dict(manifest.family_difficulty_counts) == {
        key: expected_family_difficulty[key]
        for key in sorted(expected_family_difficulty)
    }
    assert manifest.data_release_id == config.release_id
    assert manifest.code_revision == REVISION
    assert manifest.task_schema_version == config.schema_version
    assert manifest.compiler_versions == {
        compiled.generator_name: compiled.compiler_version
    }
    assert manifest.source_manifest_paths_and_hashes == ()
    assert manifest.human_audit_artifacts == ()

    task_ref = manifest.task_file_paths_and_hashes[0]
    assert task_ref.path == "tasks.jsonl"
    assert task_ref.media_type == "application/x-ndjson"
    assert task_ref.record_count == 1440
    assert task_ref.sha256 == hashlib.sha256(compiled.tasks_jsonl).hexdigest()

    config_ref = manifest.generation_configs_and_hashes[0]
    assert config_ref.path == "generation_config.json"
    assert config_ref.media_type == "application/json"
    assert config_ref.record_count == 1
    assert config_ref.sha256 == compiled.config_sha256
    assert bundle.config_sha256 == compiled.config_sha256
    assert hashlib.sha256(bundle.resolved_config_bytes).hexdigest() == compiled.config_sha256
    assert canonical_json_bytes(bundle.resolved_config) == bundle.resolved_config_bytes

    assert canonical_json_bytes(manifest) == bundle.task_manifest_bytes
    assert canonical_json_bytes(bundle.split_balance_report) == bundle.split_balance_bytes
    assert canonical_json_bytes(bundle.validation_report) == bundle.validation_report_bytes
    assert bundle.split_balance_report == compiled.split_assignment.split_balance

    assert tuple(artifact.ref.path for artifact in bundle.artifacts) == ARTIFACT_NAMES
    for artifact in bundle.artifacts:
        assert artifact.ref.sha256 == hashlib.sha256(artifact.content).hexdigest()
        assert artifact.ref.media_type in {
            "application/json",
            "application/x-ndjson",
        }
    assert bundle.artifacts[0].content is compiled.tasks_jsonl

    declared_hashes = manifest.leakage_check_summary["task_hashes"]
    assert len(declared_hashes) == 1440
    assert declared_hashes == {
        task.task_id: hashlib.sha256(row).hexdigest()
        for task, row in zip(tasks, compiled.tasks_jsonl.splitlines(), strict=True)
    }


def test_repeat_construction_is_byte_identical_and_fresh(compiled, config, bundle) -> None:
    repeated = build_pilot_artifact_bundle(compiled, config)

    assert repeated is not bundle
    assert repeated.resolved_config is not bundle.resolved_config
    assert repeated.task_manifest is not bundle.task_manifest
    assert repeated.split_balance_report is not bundle.split_balance_report
    assert repeated.validation_report is not bundle.validation_report
    assert tuple(artifact.content for artifact in repeated.artifacts) == tuple(
        artifact.content for artifact in bundle.artifacts
    )
    assert repeated.task_manifest_bytes == bundle.task_manifest_bytes
    assert repeated.validation_report_bytes == bundle.validation_report_bytes


def test_factory_does_not_repeat_full_task_set_validation(
    monkeypatch,
    compiled,
    config,
) -> None:
    calls = 0
    original = build_module._validation_issues

    def counted_validation(tasks):
        nonlocal calls
        calls += 1
        return original(tasks)

    monkeypatch.setattr(build_module, "_validation_issues", counted_validation)

    result = build_pilot_artifact_bundle(compiled, config)

    assert result.validation_report.valid is True
    assert calls == 0


def test_bundle_direct_construction_and_replace_are_factory_only(bundle) -> None:
    fields = {
        "resolved_config": bundle.resolved_config,
        "split_balance_report": bundle.split_balance_report,
        "task_manifest": bundle.task_manifest,
        "validation_report": bundle.validation_report,
        "artifacts": bundle.artifacts,
    }

    with pytest.raises(TypeError, match="factory"):
        PilotArtifactBundle(**fields)
    with pytest.raises(TypeError, match="factory"):
        replace(bundle)


def test_bundle_manifest_passes_split_validation_without_issues(bundle, compiled) -> None:
    direct_report = validate_splits(compiled.tasks, task_manifest=bundle.task_manifest)

    assert bundle.validation_report == direct_report
    assert direct_report.valid is True
    assert direct_report.issues == ()


def test_required_strata_are_complete_and_deviations_are_honest(bundle, compiled) -> None:
    required = bundle.task_manifest.leakage_check_summary[
        "required_minimum_strata"
    ]
    deviations = bundle.task_manifest.leakage_check_summary[
        "small_cell_deviations"
    ]
    required_keys = {
        (
            record["task_family"],
            record["difficulty"],
            record["update_depth_bucket"],
        )
        for record in required
    }
    observed = Counter(
        (
            task.task_family,
            task.difficulty.value,
            task.metadata.resolved_profile["update_depth_bucket"],
            task.metadata.split.value,
        )
        for task in compiled.tasks
    )

    assert tuple(
        (
            record["task_family"],
            record["difficulty"],
            record["update_depth_bucket"],
        )
        for record in required
    ) == tuple(sorted(required_keys))
    assert required_keys
    assert all(
        observed[(*key, split.value)] > 0
        for key in required_keys
        for split in (Split.TRAIN, Split.DEV, Split.TEST)
    )
    assert deviations == ()


def test_manifest_maps_have_deterministic_key_order(bundle) -> None:
    manifest = bundle.task_manifest

    assert tuple(manifest.split_counts) == (
        "train",
        "dev",
        "test",
        "evaluation_only",
    )
    assert tuple(manifest.semantic_core_counts) == (
        "train",
        "dev",
        "test",
        "evaluation_only",
    )
    assert tuple(manifest.family_difficulty_counts) == tuple(
        sorted(manifest.family_difficulty_counts)
    )
    task_hashes = manifest.leakage_check_summary["task_hashes"]
    assert tuple(task_hashes) == tuple(sorted(task_hashes))


def test_bundle_is_deeply_immutable_and_detached_from_input(compiled, config) -> None:
    mutable_config = type(config).model_validate(config.model_dump(mode="python"))
    snapshot = build_pilot_artifact_bundle(compiled, mutable_config)
    original_bytes = snapshot.resolved_config_bytes

    mutable_config.families.repeated_same_slot_update.update_depths.append(999)

    assert snapshot.resolved_config_bytes == original_bytes
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.resolved_config.release_id = "changed"
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.resolved_config.families.repeated_same_slot_update.update_depths.append(
            999
        )
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.task_manifest.split_counts["train"] = 0
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.validation_report.issues += ()


def test_config_mismatch_is_rejected(compiled, config) -> None:
    mismatched = config.model_copy(update={"release_id": "different-release"})

    with pytest.raises(ValueError, match="config.*compiled snapshot"):
        build_pilot_artifact_bundle(compiled, mismatched)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_sha256", "0" * 64, "compiled snapshot"),
        ("code_revision", "wrong-revision", "compiled snapshot"),
        ("compiler_version", "0.0.0", "compiler version"),
        ("generator_name", "wrong-generator", "generator"),
    ],
)
def test_modified_compiled_metadata_is_rejected(
    compiled, config, field, value, message
) -> None:
    tampered = _clone_compiled(compiled)
    object.__setattr__(tampered, field, value)

    with pytest.raises(ValueError, match="authenticated|seal|" + message):
        build_pilot_artifact_bundle(tampered, config)


def test_tampered_compiled_bytes_are_rejected(compiled, config) -> None:
    tampered = _clone_compiled(compiled)
    lines = compiled.tasks_jsonl.splitlines()
    payload = MemUpdateTask.model_validate_json(lines[0]).model_dump(mode="python")
    payload["metadata"]["extra"]["surface_variant"] = 9
    lines[0] = canonical_json_bytes(MemUpdateTask.model_validate(payload))
    object.__setattr__(tampered, "tasks_jsonl", b"\n".join(lines) + b"\n")

    with pytest.raises(ValueError, match="authenticated|seal"):
        build_pilot_artifact_bundle(tampered, config)


def test_noncanonical_compiled_snapshot_is_rejected(compiled, config) -> None:
    tampered = _clone_compiled(compiled)
    first_line, rest = compiled.tasks_jsonl.split(b"\n", 1)
    object.__setattr__(tampered, "tasks_jsonl", b" " + first_line + b"\n" + rest)

    with pytest.raises(ValueError, match="authenticated|seal"):
        build_pilot_artifact_bundle(tampered, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code_revision", None),
        ("compiler_version", []),
        ("generator_name", {}),
        ("config_sha256", None),
    ],
)
def test_factory_bounds_malformed_authenticated_provenance(
    compiled,
    config,
    field,
    value,
) -> None:
    original = getattr(compiled, field)
    object.__setattr__(compiled, field, value)
    try:
        with pytest.raises(ValueError, match="authenticated compiler output"):
            build_pilot_artifact_bundle(compiled, config)
    finally:
        object.__setattr__(compiled, field, original)


def test_unsealed_compiled_direct_and_replace_copies_are_rejected(
    compiled,
    config,
) -> None:
    direct = CompiledPilotTasks(
        split_assignment=compiled.split_assignment,
        config_sha256=compiled.config_sha256,
        code_revision=compiled.code_revision,
        compiler_version=compiled.compiler_version,
        generator_name=compiled.generator_name,
        tasks_jsonl=compiled.tasks_jsonl,
    )
    replaced = replace(compiled)

    for unsealed in (direct, replaced):
        with pytest.raises(ValueError, match="authenticated compiler output"):
            build_pilot_artifact_bundle(unsealed, config)

    resealed = compiled.authenticated_clone()
    assert resealed is not compiled
    assert resealed.verify_authenticated_snapshot() is None
    assert build_pilot_artifact_bundle(resealed, config).tasks_jsonl == compiled.tasks_jsonl


def test_factory_rejects_object_tampered_nonsemantic_task_metadata(
    compiled,
    config,
) -> None:
    tasks = list(compiled.tasks)
    core_id = tasks[0].metadata.split_key.semantic_core_id
    for index, task in enumerate(tasks):
        if task.metadata.split_key.semantic_core_id != core_id:
            continue
        payload = task.model_dump(mode="python")
        payload["metadata"]["tags"].append("hostile-nonsemantic-tag")
        tasks[index] = MemUpdateTask.model_validate(payload)
    hostile_jsonl = b"".join(
        canonical_json_bytes(task) + b"\n" for task in tasks
    )
    original_jsonl = compiled.tasks_jsonl
    object.__setattr__(compiled, "tasks_jsonl", hostile_jsonl)
    try:
        with pytest.raises(ValueError, match="authenticated|seal"):
            build_pilot_artifact_bundle(compiled, config)
    finally:
        object.__setattr__(compiled, "tasks_jsonl", original_jsonl)


def test_direct_task_set_helper_bounds_unhashable_surface_variant(compiled) -> None:
    rows = compiled.tasks_jsonl.splitlines()
    payload = MemUpdateTask.model_validate_json(rows[0]).model_dump(mode="python")
    payload["metadata"]["extra"]["surface_variant"] = []
    rows[0] = canonical_json_bytes(MemUpdateTask.model_validate(payload))
    hostile_jsonl = b"\n".join(rows) + b"\n"

    with pytest.raises(ValueError, match="surface_variant|task set|compiled Pilot"):
        CompiledPilotTasks.validated_task_set(
            hostile_jsonl,
            config_sha256=compiled.config_sha256,
            code_revision=compiled.code_revision,
            compiler_version=compiled.compiler_version,
            generator_name=compiled.generator_name,
            seed=compiled.split_assignment.split_balance.seed,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code_revision", None),
        ("compiler_version", []),
        ("generator_name", {}),
        ("config_sha256", None),
    ],
)
def test_factory_bounds_malformed_authenticated_provenance(
    compiled,
    config,
    field,
    value,
) -> None:
    original = getattr(compiled, field)
    object.__setattr__(compiled, field, value)
    try:
        with pytest.raises(ValueError, match="authenticated compiler output"):
            build_pilot_artifact_bundle(compiled, config)
    finally:
        object.__setattr__(compiled, field, original)


def test_unsealed_compiled_direct_and_replace_copies_are_rejected(
    compiled,
    config,
) -> None:
    direct = CompiledPilotTasks(
        split_assignment=compiled.split_assignment,
        config_sha256=compiled.config_sha256,
        code_revision=compiled.code_revision,
        compiler_version=compiled.compiler_version,
        generator_name=compiled.generator_name,
        tasks_jsonl=compiled.tasks_jsonl,
    )
    replaced = replace(compiled)

    for unsealed in (direct, replaced):
        with pytest.raises(ValueError, match="authenticated compiler output"):
            build_pilot_artifact_bundle(unsealed, config)

    resealed = compiled.authenticated_clone()
    assert resealed is not compiled
    assert resealed.verify_authenticated_snapshot() is None
    assert build_pilot_artifact_bundle(resealed, config).tasks_jsonl == compiled.tasks_jsonl


def test_factory_rejects_object_tampered_nonsemantic_task_metadata(
    compiled,
    config,
) -> None:
    tasks = list(compiled.tasks)
    core_id = tasks[0].metadata.split_key.semantic_core_id
    for index, task in enumerate(tasks):
        if task.metadata.split_key.semantic_core_id != core_id:
            continue
        payload = task.model_dump(mode="python")
        payload["metadata"]["tags"].append("hostile-nonsemantic-tag")
        tasks[index] = MemUpdateTask.model_validate(payload)
    hostile_jsonl = b"".join(
        canonical_json_bytes(task) + b"\n" for task in tasks
    )
    original_jsonl = compiled.tasks_jsonl
    object.__setattr__(compiled, "tasks_jsonl", hostile_jsonl)
    try:
        with pytest.raises(ValueError, match="authenticated|seal"):
            build_pilot_artifact_bundle(compiled, config)
    finally:
        object.__setattr__(compiled, "tasks_jsonl", original_jsonl)


def test_direct_task_set_helper_bounds_unhashable_surface_variant(compiled) -> None:
    rows = compiled.tasks_jsonl.splitlines()
    payload = MemUpdateTask.model_validate_json(rows[0]).model_dump(mode="python")
    payload["metadata"]["extra"]["surface_variant"] = []
    rows[0] = canonical_json_bytes(MemUpdateTask.model_validate(payload))
    hostile_jsonl = b"\n".join(rows) + b"\n"

    with pytest.raises(ValueError, match="surface_variant|task set|compiled Pilot"):
        CompiledPilotTasks.validated_task_set(
            hostile_jsonl,
            config_sha256=compiled.config_sha256,
            code_revision=compiled.code_revision,
            compiler_version=compiled.compiler_version,
            generator_name=compiled.generator_name,
            seed=compiled.split_assignment.split_balance.seed,
        )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("record_count", 1439),
        ("media_type", "application/json"),
    ],
)
def test_public_bundle_rejects_tampered_task_artifact_metadata(
    bundle, change, value
) -> None:
    hostile = replace(bundle.artifacts[0], **{change: value})

    with pytest.raises(TypeError, match="factory"):
        _replace_artifact(bundle, 0, hostile)


def test_public_bundle_rejects_tampered_task_artifact_hash(bundle) -> None:
    hostile = replace(bundle.artifacts[0])
    object.__setattr__(hostile, "_sha256", "0" * 64)

    with pytest.raises(TypeError, match="factory"):
        _replace_artifact(bundle, 0, hostile)


def test_public_bundle_rejects_tampered_canonical_task_bytes(bundle) -> None:
    rows = bundle.tasks_jsonl.splitlines()
    payload = MemUpdateTask.model_validate_json(rows[0]).model_dump(mode="python")
    payload["metadata"]["extra"]["surface_variant"] = 9
    rows[0] = canonical_json_bytes(MemUpdateTask.model_validate(payload))
    hostile = replace(bundle.artifacts[0], content=b"\n".join(rows) + b"\n")

    with pytest.raises(TypeError, match="factory"):
        PilotArtifactBundle(
            resolved_config=bundle.resolved_config,
            split_balance_report=bundle.split_balance_report,
            task_manifest=bundle.task_manifest,
            validation_report=bundle.validation_report,
            artifacts=(hostile, *bundle.artifacts[1:]),
        )


def test_public_bundle_rejects_coordinated_config_and_artifact_replacement(
    bundle,
) -> None:
    hostile_config = bundle.resolved_config.model_copy(
        update={"release_id": "hostile-release"},
        deep=True,
    )
    hostile_artifact = replace(
        bundle.artifacts[1],
        content=canonical_json_bytes(hostile_config),
    )

    with pytest.raises((TypeError, ValueError), match="factory|config|manifest|task set"):
        _replace_artifact(
            bundle,
            1,
            hostile_artifact,
            resolved_config=hostile_config,
        )


def test_public_bundle_rejects_coordinated_manifest_and_artifact_replacement(
    bundle,
) -> None:
    hostile_manifest = bundle.task_manifest.validated_replace(created_at="hostile")
    hostile_artifact = replace(
        bundle.artifacts[3],
        content=canonical_json_bytes(hostile_manifest),
    )

    with pytest.raises((TypeError, ValueError), match="factory|manifest"):
        _replace_artifact(
            bundle,
            3,
            hostile_artifact,
            task_manifest=hostile_manifest,
        )


def test_public_bundle_rejects_coordinated_split_balance_and_artifact_replacement(
    bundle,
) -> None:
    hostile_report = bundle.split_balance_report.validated_replace(
        seed=bundle.split_balance_report.seed + 1
    )
    hostile_artifact = replace(
        bundle.artifacts[2],
        content=canonical_json_bytes(hostile_report),
    )

    with pytest.raises((TypeError, ValueError), match="factory|split balance"):
        _replace_artifact(
            bundle,
            2,
            hostile_artifact,
            split_balance_report=hostile_report,
        )


def test_public_bundle_rejects_coordinated_invalid_validation_report(bundle) -> None:
    hostile_report = _invalid_report()
    hostile_artifact = replace(
        bundle.artifacts[4],
        content=canonical_json_bytes(hostile_report),
    )

    with pytest.raises((TypeError, ValueError), match="factory|validation report|split validation"):
        _replace_artifact(
            bundle,
            4,
            hostile_artifact,
            validation_report=hostile_report,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resolved_config", {}),
        ("split_balance_report", {}),
        ("task_manifest", {}),
        ("validation_report", {}),
    ],
)
def test_public_bundle_rejects_noncontract_typed_fields(bundle, field, value) -> None:
    with pytest.raises(TypeError, match="factory"):
        replace(bundle, **{field: value})


def test_public_bundle_rejects_coordinated_generator_rewrite(bundle) -> None:
    hostile_tasks = []
    for task in _parsed_bundle_tasks(bundle):
        payload = task.model_dump(mode="python")
        payload["source"]["generator"]["generator_name"] = "hostile_generator"
        hostile_tasks.append(MemUpdateTask.model_validate(payload))

    with pytest.raises((TypeError, ValueError), match="factory|generator|task set"):
        _coordinated_bundle_from_tasks(bundle, hostile_tasks)


def test_bundle_rejects_coordinated_nonsemantic_metadata_rewrite(bundle) -> None:
    tasks = _parsed_bundle_tasks(bundle)
    core_id = tasks[0].metadata.split_key.semantic_core_id
    hostile_tasks = []
    for task in tasks:
        if task.metadata.split_key.semantic_core_id != core_id:
            hostile_tasks.append(task)
            continue
        payload = task.model_dump(mode="python")
        payload["metadata"]["tags"].append("hostile-nonsemantic-tag")
        hostile_tasks.append(MemUpdateTask.model_validate(payload))

    with pytest.raises((TypeError, ValueError), match="factory|authenticated"):
        _coordinated_bundle_from_tasks(bundle, hostile_tasks)


def test_public_bundle_rejects_coordinated_core_split_reassignment(bundle) -> None:
    tasks = _parsed_bundle_tasks(bundle)
    exemplar = next(task for task in tasks if task.metadata.split is Split.TRAIN)
    core_id = exemplar.metadata.split_key.semantic_core_id
    hostile_tasks = []
    for task in tasks:
        if task.metadata.split_key.semantic_core_id != core_id:
            hostile_tasks.append(task)
            continue
        payload = task.model_dump(mode="python")
        payload["metadata"]["split"] = Split.DEV
        hostile_tasks.append(MemUpdateTask.model_validate(payload))
    hostile_report = _moved_core_split_report(bundle.split_balance_report, exemplar)

    with pytest.raises((TypeError, ValueError), match="factory|quota|task set|split"):
        _coordinated_bundle_from_tasks(
            bundle,
            hostile_tasks,
            split_balance_report=hostile_report,
        )


def test_public_bundle_rejects_quota_preserving_core_split_swap(bundle) -> None:
    tasks = _parsed_bundle_tasks(bundle)
    representatives = {}
    for task in tasks:
        representatives.setdefault(task.metadata.split_key.semantic_core_id, task)
    groups = {}
    for core_id, task in representatives.items():
        key = (
            task.task_family,
            task.difficulty,
            tuple(task.metadata.resolved_profile.items()),
        )
        groups.setdefault(key, {})[task.metadata.split] = core_id
    pair = next(
        split_map
        for split_map in groups.values()
        if Split.TRAIN in split_map and Split.DEV in split_map
    )
    train_core = pair[Split.TRAIN]
    dev_core = pair[Split.DEV]
    hostile_tasks = []
    for task in tasks:
        core_id = task.metadata.split_key.semantic_core_id
        if core_id not in {train_core, dev_core}:
            hostile_tasks.append(task)
            continue
        payload = task.model_dump(mode="python")
        payload["metadata"]["split"] = (
            Split.DEV if core_id == train_core else Split.TRAIN
        )
        hostile_tasks.append(MemUpdateTask.model_validate(payload))

    hostile_tasks.sort(
        key=lambda task: (
            {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}[task.metadata.split],
            {
                "repeated_same_slot_update": 0,
                "interleaved_multi_slot_update": 1,
                "entity_attribute_grounding": 2,
                "noop_write_discipline": 3,
            }[task.task_family],
            task.metadata.split_key.semantic_core_id,
            task.metadata.extra["surface_variant"],
        )
    )

    with pytest.raises((TypeError, ValueError), match="factory|assignment|ranking|task set|split"):
        _coordinated_bundle_from_tasks(bundle, hostile_tasks)


def test_public_bundle_rejects_coordinated_task_reordering(bundle) -> None:
    hostile_tasks = _parsed_bundle_tasks(bundle)
    hostile_tasks[0], hostile_tasks[1] = hostile_tasks[1], hostile_tasks[0]

    with pytest.raises((TypeError, ValueError), match="factory|canonical order|task set"):
        _coordinated_bundle_from_tasks(bundle, hostile_tasks)


def test_public_bundle_rejects_missing_and_duplicate_surface_variant(bundle) -> None:
    hostile_tasks = _parsed_bundle_tasks(bundle)
    payload = hostile_tasks[2].model_dump(mode="python")
    assert payload["metadata"]["extra"]["surface_variant"] == 2
    payload["metadata"]["extra"]["surface_variant"] = 1
    hostile_tasks[2] = MemUpdateTask.model_validate(payload)

    with pytest.raises((TypeError, ValueError), match="factory|surface variant|task set"):
        _coordinated_bundle_from_tasks(bundle, hostile_tasks)


def test_public_bundle_rejects_rehashed_invalid_task_semantics(bundle) -> None:
    hostile_tasks = _parsed_bundle_tasks(bundle)
    payload = hostile_tasks[0].model_dump(mode="python")
    query_id = next(iter(payload["gold"]["gold_answers"]))
    payload["gold"]["gold_answers"][query_id] = "hostile-answer"
    hostile_tasks[0] = MemUpdateTask.model_validate(payload)

    with pytest.raises((TypeError, ValueError), match="factory|validation|task set|gold"):
        _coordinated_bundle_from_tasks(bundle, hostile_tasks)


def test_invalid_split_validation_is_rejected(monkeypatch, compiled, config) -> None:
    import mub.vnext.generation.artifacts as artifacts_module

    invalid = _invalid_report()
    monkeypatch.setattr(artifacts_module, "validate_splits", lambda *args, **kwargs: invalid)

    with pytest.raises(ValueError, match="split validation"):
        build_pilot_artifact_bundle(compiled, config)


def test_bundle_construction_performs_no_disk_io(monkeypatch, compiled, config) -> None:
    def reject_io(*args, **kwargs):
        raise AssertionError("disk I/O is forbidden")

    monkeypatch.setattr(builtins, "open", reject_io)
    monkeypatch.setattr(Path, "open", reject_io)
    monkeypatch.setattr(Path, "write_bytes", reject_io)
    monkeypatch.setattr(Path, "write_text", reject_io)

    result = build_pilot_artifact_bundle(compiled, config)

    assert result.tasks_jsonl == compiled.tasks_jsonl
