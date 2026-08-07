from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from mub.vnext.contracts import Split, TaskFamily
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.generation.core_artifacts import (
    CoreSplitBalance,
    CoreValidationReport,
    _VALIDATION_CHECKS,
    _manifest,
)
from mub.vnext.generation.core_build import (
    CompiledCoreSnapshot,
    _generated_cores,
    _select_and_assign,
    _validate_snapshot,
)
from mub.vnext.generation.core_config import CoreConfig
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.core_render_v3 import render_core_v3
from mub.vnext.generation.core_hard_suite import (
    CoreHardSuiteManifest,
    build_core_hard_suite,
)
from mub.vnext.io import canonical_json_bytes, read_models, semantic_task_hash_v3, sha256_model
from mub.vnext.io.canonical import _canonical_payload_bytes
from mub.vnext.validation.replay_v3 import replay_task_v3

_ARTIFACTS = (
    "tasks.jsonl",
    "semantic_cores.jsonl",
    "generation_config.json",
    "split_balance.json",
    "task_manifest.json",
    "core-hard-v1.json",
    "validation_report.json",
)
_SPLITS = (Split.TRAIN, Split.DEV, Split.TEST)
_FULL_FAMILY_COUNTS = {
    "repeated_same_slot_update": 480,
    "interleaved_multi_slot_update": 480,
    "entity_attribute_grounding": 420,
    "noop_write_discipline": 420,
    "deletion_forgetting": 480,
    "current_historical_query": 420,
    "long_horizon_memory_synthesis": 300,
}


def _canonical_json(path: Path, model_type):
    raw = path.read_bytes()
    model = model_type.model_validate_json(raw)
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"{path.name} is not canonical JSON")
    return model, raw


def _disjoint(values):
    for index, left in enumerate(_SPLITS):
        for right in _SPLITS[index + 1 :]:
            if not values[left].isdisjoint(values[right]):
                return False
    return True


def _read_semantic_cores(path: Path) -> tuple[dict, ...]:
    cores = []
    seen = set()
    for line_number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            raise ValueError(f"semantic_cores.jsonl line {line_number} is blank")
        payload = json.loads(raw.decode("utf-8"))
        if type(payload) is not dict or type(payload.get("core_id")) is not str:
            raise ValueError("semantic_cores.jsonl contains an invalid core record")
        if _canonical_payload_bytes(payload) != raw:
            raise ValueError("semantic_cores.jsonl contains a noncanonical row")
        if payload["core_id"] in seen:
            raise ValueError("semantic_cores.jsonl contains duplicate core IDs")
        seen.add(payload["core_id"])
        cores.append(payload)
    return tuple(cores)


def validate_core_release(
    release_dir: str | Path,
    *,
    expected_full: bool = True,
) -> CoreValidationReport:
    root = Path(release_dir)
    if not root.is_dir():
        raise ValueError("Core candidate directory does not exist")
    names = {path.name for path in root.iterdir() if path.is_file()}
    if names != set(_ARTIFACTS):
        raise ValueError(f"Core candidate must contain exactly {_ARTIFACTS}")

    config, config_bytes = _canonical_json(root / "generation_config.json", CoreConfig)
    split_balance, _ = _canonical_json(root / "split_balance.json", CoreSplitBalance)
    manifest, manifest_bytes = _canonical_json(root / "task_manifest.json", TaskManifestV3)
    hard_suite, _ = _canonical_json(root / "core-hard-v1.json", CoreHardSuiteManifest)
    stored_report, _ = _canonical_json(root / "validation_report.json", CoreValidationReport)
    tasks = tuple(read_models(root / "tasks.jsonl", MemUpdateTaskV3, id_field="task_id"))
    cores = _read_semantic_cores(root / "semantic_cores.jsonl")
    task_bytes = (root / "tasks.jsonl").read_bytes()
    core_bytes = (root / "semantic_cores.jsonl").read_bytes()
    if b"".join(canonical_json_bytes(task) + b"\n" for task in tasks) != task_bytes:
        raise ValueError("tasks.jsonl is not canonical")
    if b"".join(_canonical_payload_bytes(core) + b"\n" for core in cores) != core_bytes:
        raise ValueError("semantic_cores.jsonl is not canonical")
    if [task.task_id for task in tasks] != sorted(task.task_id for task in tasks):
        raise ValueError("tasks.jsonl must be sorted by task_id")
    if [core["core_id"] for core in cores] != sorted(core["core_id"] for core in cores):
        raise ValueError("semantic_cores.jsonl must be sorted by core_id")

    task_ref = manifest.task_file_paths_and_hashes
    config_ref = manifest.generation_configs_and_hashes
    core_ref = manifest.source_manifest_paths_and_hashes
    if len(task_ref) != 1 or task_ref[0].path != "tasks.jsonl" or task_ref[0].sha256 != hashlib.sha256(task_bytes).hexdigest() or task_ref[0].record_count != len(tasks):
        raise ValueError("task manifest does not authenticate tasks.jsonl")
    if len(config_ref) != 1 or config_ref[0].path != "generation_config.json" or config_ref[0].sha256 != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("task manifest does not authenticate generation_config.json")
    if len(core_ref) != 1 or core_ref[0].path != "semantic_cores.jsonl" or core_ref[0].sha256 != hashlib.sha256(core_bytes).hexdigest() or core_ref[0].record_count != len(cores):
        raise ValueError("task manifest does not authenticate semantic_cores.jsonl")
    observed_hashes = {task.task_id: sha256_model(task) for task in tasks}
    if dict(manifest.task_record_hashes) != observed_hashes:
        raise ValueError("task record hashes are invalid")

    core_ids = {core["core_id"] for core in cores}
    parsed_family_counts = Counter(core["task_family"] for core in cores)
    canonical_cores = _generated_cores(config)
    if len(cores) == config.total_semantic_cores:
        selection_limit = None
    else:
        partial_counts = set(parsed_family_counts.values())
        if len(partial_counts) != 1:
            raise ValueError("bounded candidate families must share one selection quota")
        selection_limit = next(iter(partial_counts))
    canonical_assignments = _select_and_assign(
        canonical_cores,
        seed=config.seed,
        splits=config.splits,
        cores_per_family=selection_limit,
    )
    canonical_by_id = {core.core_id: core for core in canonical_cores}
    expected_core_payloads = sorted(
        (
            canonical_by_id[assignment.semantic_core_id].model_dump(mode="json")
            for assignment in canonical_assignments
        ),
        key=lambda core: core["core_id"],
    )
    if list(cores) != expected_core_payloads:
        raise ValueError("semantic_cores.jsonl does not contain canonical selected cores")
    revisions = {task.source.generator.code_revision for task in tasks}
    generators = {task.source.generator.generator_name for task in tasks}
    if len(revisions) != 1 or len(generators) != 1:
        raise ValueError("candidate tasks must share one generator and code revision")
    context = GenerationContext(
        config=config,
        code_revision=next(iter(revisions)),
        generator_name=next(iter(generators)),
    )
    expected_tasks = tuple(sorted(
        (
            render_core_v3(
                canonical_by_id[assignment.semantic_core_id],
                split=assignment.split,
                surface_variant=surface_variant,
                context=context,
            )
            for assignment in canonical_assignments
            for surface_variant in range(4)
        ),
        key=lambda task: task.task_id,
    ))
    if tasks != expected_tasks:
        raise ValueError("tasks.jsonl does not match canonical Core rendering")
    tasks_by_core = defaultdict(list)
    for task in tasks:
        replay = replay_task_v3(task)
        if replay.issues:
            raise ValueError(f"task {task.task_id} fails v3 replay")
        tasks_by_core[task.metadata.split_key.semantic_core_id].append(task)
    if set(tasks_by_core) != core_ids:
        raise ValueError("semantic core and task coverage differs")
    for core_id, surfaces in tasks_by_core.items():
        if len(surfaces) != 4 or {task.metadata.extra["surface_variant"] for task in surfaces} != {0, 1, 2, 3}:
            raise ValueError(f"core {core_id} does not have four canonical surfaces")
        if len({task.metadata.split for task in surfaces}) != 1:
            raise ValueError(f"core {core_id} crosses splits")
        if len({semantic_task_hash_v3(task) for task in surfaces}) != 1:
            raise ValueError(f"core {core_id} surfaces are not semantically equivalent")

    for field in (
        "semantic_core_id",
        "source_group_id",
        "source_document_id",
        "trajectory_id",
        "paraphrase_group_id",
        "version_group_id",
    ):
        values = {
            split: {
                getattr(task.metadata.split_key, field)
                for task in tasks
                if task.metadata.split is split
            }
            for split in _SPLITS
        }
        if not _disjoint(values):
            raise ValueError(f"cross-split {field} leakage")
    normalized = {
        split: {task.source.normalized_hash for task in tasks if task.metadata.split is split}
        for split in _SPLITS
    }
    if not _disjoint(normalized):
        raise ValueError("cross-split normalized source leakage")
    fingerprints = {
        split: {
            task.metadata.extra["stratification"].get("evidence_fingerprint")
            for task in tasks
            if task.metadata.split is split
            and task.task_family == TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value
        }
        for split in _SPLITS
    }
    if not _disjoint(fingerprints):
        raise ValueError("cross-split Family G evidence leakage")

    family_core_counts = Counter(core["task_family"] for core in cores)
    split_core_counts = Counter(
        surfaces[0].metadata.split.value for surfaces in tasks_by_core.values()
    )
    split_task_counts = Counter(task.metadata.split.value for task in tasks)
    reconstructed = CompiledCoreSnapshot(
        config_sha256=sha256_model(config),
        assignments=canonical_assignments,
        semantic_cores=tuple(
            canonical_by_id[assignment.semantic_core_id]
            for assignment in canonical_assignments
        ),
        tasks=tasks,
        family_core_counts=dict(family_core_counts),
        core_counts={key: split_core_counts[key] for key in ("train", "dev", "test")},
        task_counts={key: split_task_counts[key] for key in ("train", "dev", "test")},
    )
    _validate_snapshot(reconstructed, config, canonical_cores)
    expected_split_balance = CoreSplitBalance(
        family_core_counts=dict(family_core_counts),
        split_core_counts={key: split_core_counts[key] for key in ("train", "dev", "test")},
        split_task_counts={key: split_task_counts[key] for key in ("train", "dev", "test")},
        total_semantic_cores=len(cores),
        total_tasks=len(tasks),
    )
    if split_balance != expected_split_balance:
        raise ValueError("split_balance.json does not match candidate records")
    expected_manifest = _manifest(
        reconstructed,
        config,
        task_ref=task_ref[0],
        core_ref=core_ref[0],
        config_ref=config_ref[0],
    )
    if manifest != expected_manifest:
        raise ValueError("task_manifest.json does not match candidate provenance")
    if expected_full:
        if len(cores) != 3000 or len(tasks) != 12000:
            raise ValueError("full Core candidate must contain 3,000 cores and 12,000 tasks")
        if dict(family_core_counts) != _FULL_FAMILY_COUNTS:
            raise ValueError("full Core family counts are invalid")
        if {key: split_core_counts[key] for key in ("train", "dev", "test")} != {"train": 2100, "dev": 300, "test": 600}:
            raise ValueError("full Core split core counts are invalid")
        if {key: split_task_counts[key] for key in ("train", "dev", "test")} != {"train": 8400, "dev": 1200, "test": 2400}:
            raise ValueError("full Core split task counts are invalid")
        family_split = Counter(
            (surfaces[0].task_family, surfaces[0].metadata.split.value)
            for surfaces in tasks_by_core.values()
        )
        expected_family_split = {
            family: quota
            for families, quota in (
                (("repeated_same_slot_update", "interleaved_multi_slot_update", "deletion_forgetting"), (336, 48, 96)),
                (("entity_attribute_grounding", "noop_write_discipline", "current_historical_query"), (294, 42, 84)),
                (("long_horizon_memory_synthesis",), (210, 30, 60)),
            )
            for family in families
        }
        for family, quotas in expected_family_split.items():
            observed = tuple(family_split[(family, split)] for split in ("train", "dev", "test"))
            if observed != quotas:
                raise ValueError(f"full Core per-family split quota is invalid for {family}")

    if hard_suite.source_task_manifest_hash != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("hard suite source manifest binding is invalid")
    expected_hard_suite = build_core_hard_suite(
        reconstructed,
        source_task_manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        per_family=hard_suite.per_family_core_count,
    )
    if hard_suite != expected_hard_suite:
        raise ValueError("hard suite does not match deterministic selection policy")
    task_by_id = {task.task_id: task for task in tasks}
    if any(task_id not in task_by_id for task_id in hard_suite.task_ids):
        raise ValueError("hard suite references an unknown task")
    hard_tasks = [task_by_id[task_id] for task_id in hard_suite.task_ids]
    if any(task.metadata.split is not Split.TEST for task in hard_tasks):
        raise ValueError("hard suite must be test-only")
    if {task.metadata.split_key.semantic_core_id for task in hard_tasks} != set(hard_suite.semantic_core_ids):
        raise ValueError("hard suite core and task coverage differs")
    observed_hard_family_counts = Counter(task.task_family for task in hard_tasks)
    if dict(hard_suite.family_task_counts) != {
        family.value: observed_hard_family_counts[family.value]
        for family in (
            TaskFamily.REPEATED_SAME_SLOT,
            TaskFamily.INTERLEAVED_MULTI_SLOT,
            TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
            TaskFamily.NOOP_WRITE_DISCIPLINE,
            TaskFamily.DELETION_FORGETTING,
            TaskFamily.CURRENT_HISTORICAL_QUERY,
            TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
        )
    }:
        raise ValueError("hard suite family task counts are invalid")
    if any(
        core_id not in hard_suite.semantic_core_ids
        for family_coverage in hard_suite.condition_coverage.values()
        for core_ids in family_coverage.values()
        for core_id in core_ids
    ):
        raise ValueError("hard suite condition coverage references an unselected core")
    if expected_full:
        if len(hard_suite.semantic_core_ids) != 140 or len(hard_suite.task_ids) != 560:
            raise ValueError("core-hard-v1 must contain 140 cores and 560 tasks")
        if any(count != 80 for count in hard_suite.family_task_counts.values()):
            raise ValueError("core-hard-v1 must contain 80 tasks per family")
        if sum(hard_suite.family_task_counts[name] for name in (
            TaskFamily.REPEATED_SAME_SLOT.value,
            TaskFamily.CURRENT_HISTORICAL_QUERY.value,
            TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value,
        )) != 240:
            raise ValueError("core-hard-v1 A/F/G total must be 240 tasks")

    report = CoreValidationReport(
        valid=True,
        semantic_core_count=len(cores),
        task_count=len(tasks),
        split_core_counts={key: split_core_counts[key] for key in ("train", "dev", "test")},
        split_task_counts={key: split_task_counts[key] for key in ("train", "dev", "test")},
        family_core_counts=dict(family_core_counts),
        checks=_VALIDATION_CHECKS,
    )
    if report != stored_report:
        raise ValueError("validation_report.json does not match candidate bytes")
    return report


__all__ = ["validate_core_release"]
