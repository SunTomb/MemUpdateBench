from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from mub.vnext.contracts import ArtifactRef, Split, TaskFamily
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.generation.core_artifacts import (
    CoreSplitBalance,
    CoreValidationReport,
    _VALIDATION_CHECKS,
    _manifest,
    _validate_core_artifact_tree,
)
from mub.vnext.generation.core_build import (
    CompiledCoreSnapshot,
    _generated_cores,
    _select_and_assign,
    _validate_snapshot,
)
from mub.vnext.generation.core_config import CoreConfig, load_core_config
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.core_render_v3 import render_core_v3
from mub.vnext.generation.core_hard_suite import (
    CoreHardSuiteManifest,
    build_core_hard_suite,
)
from mub.vnext.io import canonical_json_bytes, read_models, semantic_task_hash_v3, sha256_model
from mub.vnext.io.canonical import _canonical_payload_bytes
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3
from mub.vnext.version import COMPILER_VERSION

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_APPROVED_CONFIG_PATH = _PROJECT_ROOT / "configs" / "vnext" / "core.yaml"
_TRUSTED_GENERATOR_NAME = "memupdatebench_vnext_core"
_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
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
_TRACKED_CORE_SOURCE_PATHS = (
    "configs/vnext/core.yaml",
    "mub/vnext/version.py",
    "mub/vnext/generation/core.py",
    "mub/vnext/generation/core_artifacts.py",
    "mub/vnext/generation/core_build.py",
    "mub/vnext/generation/core_catalogs.py",
    "mub/vnext/generation/core_config.py",
    "mub/vnext/generation/core_hard_suite.py",
    "mub/vnext/generation/core_orchestrate.py",
    "mub/vnext/generation/core_render_v3.py",
    "mub/vnext/generation/family_a.py",
    "mub/vnext/generation/family_b.py",
    "mub/vnext/generation/family_c.py",
    "mub/vnext/generation/family_d.py",
    "mub/vnext/generation/family_e.py",
    "mub/vnext/generation/family_f.py",
    "mub/vnext/generation/family_g.py",
    "mub/vnext/generation/identity.py",
    "mub/vnext/generation/render.py",
    "mub/vnext/generation/splits.py",
    "mub/vnext/validation/core_release.py",
    "mub/vnext/validation/replay_v3.py",
)


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


def _anchored_git_directories() -> tuple[Path, Path]:
    marker = _PROJECT_ROOT / ".git"
    if marker.is_dir():
        git_dir = marker.resolve(strict=True)
    elif marker.is_file():
        declaration = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not declaration.startswith(prefix):
            raise ValueError("anchored repository .git file is malformed")
        declared = Path(declaration[len(prefix) :])
        git_dir = (
            declared if declared.is_absolute() else marker.parent / declared
        ).resolve(strict=True)
    else:
        raise ValueError("anchored repository has no .git metadata")
    common_marker = git_dir / "commondir"
    if common_marker.is_file():
        declared_common = Path(
            common_marker.read_text(encoding="utf-8").strip()
        )
        common_dir = (
            declared_common
            if declared_common.is_absolute()
            else git_dir / declared_common
        ).resolve(strict=True)
    else:
        common_dir = git_dir
    return git_dir, common_dir


def _packed_ref(common_dir: Path, ref_name: str) -> str | None:
    packed_refs = common_dir / "packed-refs"
    if not packed_refs.is_file():
        return None
    for line in packed_refs.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        revision, separator, candidate_ref = line.partition(" ")
        if separator and candidate_ref == ref_name:
            return revision
    return None


def _trusted_code_revision() -> str:
    git_dir, common_dir = _anchored_git_directories()
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        ref_name = head[5:]
        ref_path = Path(ref_name)
        if (
            not ref_name.startswith("refs/")
            or ref_path.is_absolute()
            or ".." in ref_path.parts
        ):
            raise ValueError("anchored repository HEAD reference is malformed")
        revision = None
        for base in (git_dir, common_dir):
            loose_ref = base / ref_path
            if loose_ref.is_file():
                revision = loose_ref.read_text(encoding="ascii").strip()
                break
        if revision is None:
            revision = _packed_ref(common_dir, ref_name)
        if revision is None:
            raise ValueError("anchored repository HEAD reference is unresolved")
    else:
        revision = head
    if type(revision) is not str or not _GIT_REVISION_PATTERN.fullmatch(revision):
        raise ValueError("trusted source revision must be a lowercase 40-character Git commit")
    return revision


def _trusted_git_executable() -> Path:
    candidates = (
        Path("C:/Program Files/Git/mingw64/bin/git.exe"),
        Path("C:/Program Files/Git/cmd/git.exe"),
        Path("/usr/bin/git"),
        Path("/usr/local/bin/git"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise ValueError("trusted Git executable is unavailable")


def _assert_tracked_core_sources_clean() -> None:
    git_dir, _ = _anchored_git_directories()
    revision = _trusted_code_revision()
    environment = os.environ.copy()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(name, None)
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    result = subprocess.run(
        (
            str(_trusted_git_executable()),
            f"--git-dir={git_dir}",
            f"--work-tree={_PROJECT_ROOT}",
            "-c",
            "core.autocrlf=true",
            "--no-pager",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            revision,
            "--",
            *_TRACKED_CORE_SOURCE_PATHS,
        ),
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=False,
        check=False,
    )
    if result.returncode == 1:
        raise ValueError(
            "tracked Core source differs from the anchored HEAD revision"
        )
    if result.returncode != 0:
        raise ValueError("tracked Core source cleanliness check failed")


def validate_core_release(
    release_dir: str | Path,
    *,
    expected_full: bool = True,
) -> CoreValidationReport:
    root = Path(release_dir)
    _assert_tracked_core_sources_clean()
    if not root.is_dir():
        raise ValueError("Core candidate directory does not exist")
    _validate_core_artifact_tree(root)

    trusted_config = load_core_config(_APPROVED_CONFIG_PATH)
    trusted_config_bytes = canonical_json_bytes(trusted_config)
    trusted_revision = _trusted_code_revision()

    candidate_config, config_bytes = _canonical_json(
        root / "generation_config.json", CoreConfig
    )
    if config_bytes != trusted_config_bytes or candidate_config != trusted_config:
        raise ValueError(
            "generation_config.json does not match the trusted approved config"
        )
    config = trusted_config
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

    task_ref = ArtifactRef(
        path="tasks.jsonl",
        sha256=hashlib.sha256(task_bytes).hexdigest(),
        media_type="application/x-ndjson",
        record_count=len(tasks),
    )
    config_ref = ArtifactRef(
        path="generation_config.json",
        sha256=hashlib.sha256(config_bytes).hexdigest(),
        media_type="application/json",
        record_count=1,
    )
    core_ref = ArtifactRef(
        path="semantic_cores.jsonl",
        sha256=hashlib.sha256(core_bytes).hexdigest(),
        media_type="application/x-ndjson",
        record_count=len(cores),
    )
    if tuple(manifest.task_file_paths_and_hashes) != (task_ref,):
        raise ValueError("task manifest does not authenticate tasks.jsonl")
    if tuple(manifest.generation_configs_and_hashes) != (config_ref,):
        raise ValueError(
            "task manifest does not authenticate generation_config.json"
        )
    if tuple(manifest.source_manifest_paths_and_hashes) != (core_ref,):
        raise ValueError(
            "task manifest does not authenticate semantic_cores.jsonl"
        )
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
    generator_versions = {
        (
            task.source.generator.generator_name,
            task.source.generator.compiler_version,
        )
        for task in tasks
    }
    if revisions != {trusted_revision} or manifest.code_revision != trusted_revision:
        raise ValueError("candidate does not match the trusted source revision")
    expected_generator_versions = {
        (_TRUSTED_GENERATOR_NAME, COMPILER_VERSION)
    }
    if generator_versions != expected_generator_versions or dict(
        manifest.compiler_versions
    ) != {_TRUSTED_GENERATOR_NAME: COMPILER_VERSION}:
        raise ValueError(
            "candidate does not match the trusted generator/compiler provenance"
        )
    context = GenerationContext(
        config=config,
        code_revision=trusted_revision,
        generator_name=_TRUSTED_GENERATOR_NAME,
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
        query_by_id = {query.query_id: query for query in task.queries}
        for evidence in task.gold_evidence:
            evaluation = evaluate_evidence_v3(
                evidence,
                replay,
                evidence.stale_alternative,
                query_by_id[evidence.query_id],
                task.events,
            )
            if evaluation.issues:
                issue_codes = ", ".join(
                    issue.code for issue in evaluation.issues
                )
                raise ValueError(
                    f"task {task.task_id} fails normative evidence evaluation: "
                    f"{issue_codes}"
                )
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
        task_ref=task_ref,
        core_ref=core_ref,
        config_ref=config_ref,
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
