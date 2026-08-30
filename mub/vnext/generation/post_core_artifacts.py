from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from pydantic import RootModel
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from mub.vnext.contracts import ArtifactRef, Split
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.post_core_config import PostCoreDataConfig
from mub.vnext.generation.post_core_families import (
    PostCoreSemanticCore,
    generate_post_core_cores,
)
from mub.vnext.generation.post_core_render import render_post_core_tasks_v3
from mub.vnext.generation.post_core_splits import (
    PostCoreSplitBalanceReport,
    assign_post_core_splits,
)
from mub.vnext.io import canonical_json_bytes, read_models, semantic_task_hash_v3, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3


POST_CORE_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "generation_config.json",
    "catalog_manifest.json",
    "semantic_cores.jsonl",
    "tasks.jsonl",
    "split_balance.json",
    "task_manifest.json",
    "validation_report.json",
    "release_index.json",
)
_JSON = "application/json"
_JSONL = "application/x-ndjson"
_GENERATOR_NAME = "memupdatebench_vnext_post_core_renderer"
_SPLIT_POLICY_VERSION = "vnext-post-core-data-splits-v1"
_FROZEN_ROOT_NAMES = ("core", "pilot")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class InMemoryPostCoreArtifact:
    path: str
    content: bytes
    media_type: str
    record_count: int
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise TypeError("content must be exact bytes")
        digest = hashlib.sha256(self.content).hexdigest()
        object.__setattr__(self, "_sha256", digest)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            path=self.path,
            sha256=self._sha256,
            media_type=self.media_type,
            record_count=self.record_count,
        )


@dataclass(frozen=True, slots=True)
class PostCoreArtifactBundle:
    config: PostCoreDataConfig
    cores: tuple[PostCoreSemanticCore, ...]
    tasks: tuple[MemUpdateTaskV3, ...]
    task_manifest: TaskManifestV3
    validation_report: dict[str, Any]
    artifacts: tuple[InMemoryPostCoreArtifact, ...]

    @property
    def semantic_core_count(self) -> int:
        return len(self.cores)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def split_core_counts(self) -> dict[str, int]:
        return dict(self.validation_report["split_core_counts"])

    @property
    def split_task_counts(self) -> dict[str, int]:
        return dict(self.validation_report["split_task_counts"])

    def artifact(self, path: str) -> InMemoryPostCoreArtifact:
        return next(item for item in self.artifacts if item.path == path)


@dataclass(frozen=True, slots=True)
class PublishedPostCoreArtifacts:
    output_dir: Path
    artifact_paths: tuple[Path, ...]
    artifact_refs: tuple[ArtifactRef, ...]


def _artifact(path: str, content: bytes, media_type: str, record_count: int) -> InMemoryPostCoreArtifact:
    return InMemoryPostCoreArtifact(path, content, media_type, record_count)


class _CanonicalValue(RootModel[Any]):
    pass


def _canonical_value_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        return canonical_json_bytes(value)
    return canonical_json_bytes(_CanonicalValue(root=value))


def _jsonl(models: list[Any] | tuple[Any, ...]) -> bytes:
    return b"".join(_canonical_value_bytes(item) + b"\n" for item in models)


def _core_payload(core: PostCoreSemanticCore) -> dict[str, Any]:
    return {
        "expansion_id": core.expansion_id,
        "family_id": core.family_id,
        "difficulty": core.difficulty.value,
        "core_index": core.core_index,
        "domain": core.domain,
        "attribute": core.attribute,
        "family_axes": dict(core.family_axes),
        "metadata": dict(core.metadata),
        "profile": dict(core.profile),
        "stratification": dict(core.stratification),
    }


def _catalog_payload(config: PostCoreDataConfig) -> dict[str, Any]:
    return {
        "release_id": config.release_id,
        "surface_catalog_version": config.surface_catalog_version,
        "surfaces": [surface.model_dump(mode="json") for surface in config.surfaces],
        "domains": list(config.domains),
        "attributes": list(config.attributes),
        "families": [
            {
                "family_id": family_id,
                "domains": list(config.family_domain_matrix[family_id]),
                "semantic_core_count": config.family_core_counts[family_id],
            }
            for family_id in config.family_ids
        ],
    }


def _core_order(config: PostCoreDataConfig, split_by_core: dict[str, Split], cores: tuple[PostCoreSemanticCore, ...]):
    split_order = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
    family_order = {family: index for index, family in enumerate(config.family_ids)}
    return tuple(sorted(cores, key=lambda core: (split_order[split_by_core[core.expansion_id]], family_order[core.family_id], core.core_index, core.expansion_id)))


def _task_order(config: PostCoreDataConfig, tasks: list[MemUpdateTaskV3]) -> tuple[MemUpdateTaskV3, ...]:
    split_order = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
    family_order = {family: index for index, family in enumerate(config.family_ids)}
    return tuple(sorted(tasks, key=lambda task: (split_order[task.metadata.split], family_order[task.task_family], task.metadata.split_key.semantic_core_id, int(task.metadata.extra["surface_variant"]))))


def _validate_task(task: MemUpdateTaskV3) -> None:
    if not isinstance(task, MemUpdateTaskV3):
        raise TypeError("post-core task is not MemUpdateTaskV3")
    replay = replay_task_v3(task)
    if replay.issues:
        raise ValueError(f"task {task.task_id} failed v3 replay: {replay.issues[0].code}")
    for query, evidence in zip(task.queries, task.gold_evidence, strict=True):
        evaluation = evaluate_evidence_v3(evidence, replay, evidence.stale_alternative, query, task.events)
        if evaluation.issues:
            raise ValueError(f"task {task.task_id} failed evidence evaluation: {evaluation.issues[0].code}")


def _manifest(config: PostCoreDataConfig, tasks: tuple[MemUpdateTaskV3, ...], *, config_ref: ArtifactRef, catalog_ref: ArtifactRef, split_ref: ArtifactRef, task_ref: ArtifactRef, code_revision: str) -> TaskManifestV3:
    split_counts = Counter(task.metadata.split.value for task in tasks)
    family_difficulty = Counter(f"{task.task_family}|{task.difficulty.value}" for task in tasks)
    core_split = {}
    for task in tasks:
        core_split.setdefault(task.metadata.split_key.semantic_core_id, task.metadata.split.value)
    core_counts = Counter(core_split.values())
    domain_counts = Counter(str(task.metadata.extra["domain"]) for task in tasks)
    attribute_counts = Counter(str(task.metadata.extra["attribute"]) for task in tasks)
    surface_counts = Counter(str(task.metadata.extra["surface_key"]) for task in tasks)
    task_hashes = {task.task_id: sha256_model(task) for task in tasks}
    task_metadata = {
        task.task_id: {
            "family": task.task_family,
            "domain": task.metadata.extra["domain"],
            "attribute": task.metadata.extra["attribute"],
            "difficulty": task.difficulty.value,
            "split": task.metadata.split.value,
            "surface": task.metadata.extra["surface_key"],
        }
        for task in tasks
    }
    return TaskManifestV3(
        data_release_id=config.release_id,
        split_policy_version=_SPLIT_POLICY_VERSION,
        compiler_versions={_GENERATOR_NAME: config.compiler_version},
        source_manifest_paths_and_hashes=(catalog_ref,),
        generation_configs_and_hashes=(config_ref,),
        split_counts={name: split_counts[name] for name in ("train", "dev", "test")},
        family_difficulty_counts=dict(sorted(family_difficulty.items())),
        semantic_core_counts={name: core_counts[name] for name in ("train", "dev", "test")},
        task_file_paths_and_hashes=(task_ref,),
        leakage_check_summary={
            "config_sha256": config_ref.sha256,
            "catalog_sha256": catalog_ref.sha256,
            "split_balance_sha256": split_ref.sha256,
            "split_policy_version": _SPLIT_POLICY_VERSION,
            "code_revision": code_revision,
            "code_revision_sha256": hashlib.sha256(code_revision.encode("utf-8")).hexdigest(),
            "compiler_version": config.compiler_version,
            "generator_name": _GENERATOR_NAME,
            "surface_variants_per_core": len(config.surfaces),
            "cross_split_overlap_count": 0,
            "domain_counts": dict(sorted(domain_counts.items())),
            "attribute_counts": dict(sorted(attribute_counts.items())),
            "surface_counts": dict(sorted(surface_counts.items())),
            "task_metadata": task_metadata,
            "task_record_hashes_bound": True,
        },
        human_audit_artifacts=(),
        created_at="deterministic-post-core-main-track",
        code_revision=code_revision,
        task_record_hashes=task_hashes,
    )


def _validation_report(config: PostCoreDataConfig, cores: tuple[PostCoreSemanticCore, ...], tasks: tuple[MemUpdateTaskV3, ...], split_result, *, code_revision: str, catalog_hash: str) -> dict[str, Any]:
    split_core_counts = {split.value: 0 for split in (Split.TRAIN, Split.DEV, Split.TEST)}
    core_to_split = {assignment.expansion_id: assignment.split.value for assignment in split_result.assignments}
    for split in core_to_split.values():
        split_core_counts[split] += 1
    split_task_counts = dict(Counter(task.metadata.split.value for task in tasks))
    for split in ("train", "dev", "test"):
        split_task_counts.setdefault(split, 0)

    group_fields = (
        "semantic_core_id",
        "trajectory_id",
        "source_group_id",
        "version_group_id",
        "paraphrase_group_id",
        "source_document_id",
        "source_id",
    )
    groups_by_split: dict[str, dict[str, set[str]]] = {
        field: {"train": set(), "dev": set(), "test": set()}
        for field in group_fields
    }
    for task in tasks:
        split = task.metadata.split.value
        key = task.metadata.split_key
        values = {
            "semantic_core_id": key.semantic_core_id,
            "trajectory_id": key.trajectory_id,
            "source_group_id": key.source_group_id,
            "version_group_id": key.version_group_id,
            "paraphrase_group_id": key.paraphrase_group_id,
            "source_document_id": key.source_document_id,
            "source_id": task.source.source_id,
        }
        for field_name, value in values.items():
            if value is not None:
                groups_by_split[field_name][split].add(value)
    overlaps = {
        field: sorted(
            set.intersection(*(set(values) for values in by_split.values()))
        )
        for field, by_split in groups_by_split.items()
    }
    overlap_count = sum(len(values) for values in overlaps.values())
    semantic_hashes = {}
    surface_issues = []
    by_core: dict[str, list[MemUpdateTaskV3]] = {}
    for task in tasks:
        by_core.setdefault(task.metadata.split_key.semantic_core_id, []).append(task)
    for core_id, variants in by_core.items():
        hashes = {semantic_task_hash_v3(task) for task in variants}
        if len(hashes) != 1 or len(variants) != len(config.surfaces):
            surface_issues.append(core_id)
        semantic_hashes[core_id] = next(iter(hashes))
    family_counts = dict(Counter(core.family_id for core in cores))
    expected_split_cores = {"train": 630, "dev": 90, "test": 180}
    expected_split_tasks = {"train": 2520, "dev": 360, "test": 720}
    expected_family_counts = {family_id: 300 for family_id in config.family_ids}
    valid = (
        len(cores) == 900
        and len(tasks) == 3600
        and split_core_counts == expected_split_cores
        and split_task_counts == expected_split_tasks
        and family_counts == expected_family_counts
        and overlap_count == 0
        and not surface_issues
    )
    return {
        "valid": valid,
        "review_status": "NOT_STARTED",
        "semantic_core_count": len(cores),
        "task_count": len(tasks),
        "split_core_counts": split_core_counts,
        "split_task_counts": split_task_counts,
        "family_core_counts": family_counts,
        "surface_variants_per_core": len(config.surfaces),
        "cross_split_overlap_count": overlap_count,
        "cross_split_overlaps": overlaps,
        "surface_equivalence_issue_core_ids": sorted(surface_issues),
        "semantic_hashes": semantic_hashes,
        "checks": ["memupdate_task_v3", "v3_replay", "normative_evidence_evaluation", "canonical_bytes", "group_first_splits", "cross_split_overlap", "four_surface_semantic_equivalence", "frozen_root_exclusion"],
        "code_revision": code_revision,
        "catalog_sha256": catalog_hash,
    }


def build_post_core_artifact_bundle(config: PostCoreDataConfig, *, code_revision: str) -> PostCoreArtifactBundle:
    if not isinstance(config, PostCoreDataConfig):
        raise TypeError("config must be a PostCoreDataConfig")
    if type(code_revision) is not str or not code_revision.strip():
        raise ValueError("code_revision must be a nonblank string")
    cores = tuple(generate_post_core_cores(config))
    split_result = assign_post_core_splits(config, cores)
    split_by_core = dict(split_result.split_by_expansion_id)
    ordered_cores = _core_order(config, split_by_core, cores)
    tasks = _task_order(config, [task for core in ordered_cores for task in render_post_core_tasks_v3(core, config=config, split=split_by_core[core.expansion_id], code_revision=code_revision)])
    for task in tasks:
        _validate_task(task)
    config_artifact = _artifact("generation_config.json", canonical_json_bytes(config), _JSON, 1)
    catalog_artifact = _artifact(
        "catalog_manifest.json",
        _canonical_value_bytes(_catalog_payload(config)),
        _JSON,
        1,
    )
    cores_artifact = _artifact("semantic_cores.jsonl", _jsonl([_core_payload(core) for core in cores]), _JSONL, len(cores))
    tasks_artifact = _artifact("tasks.jsonl", _jsonl(tasks), _JSONL, len(tasks))
    split_artifact = _artifact("split_balance.json", canonical_json_bytes(split_result.split_balance), _JSON, 1)
    manifest = _manifest(
        config,
        tasks,
        config_ref=config_artifact.ref,
        catalog_ref=catalog_artifact.ref,
        split_ref=split_artifact.ref,
        task_ref=tasks_artifact.ref,
        code_revision=code_revision,
    )
    manifest_artifact = _artifact("task_manifest.json", canonical_json_bytes(manifest), _JSON, 1)
    report_payload = _validation_report(config, cores, tasks, split_result, code_revision=code_revision, catalog_hash=catalog_artifact.ref.sha256)
    report_artifact = _artifact("validation_report.json", json.dumps(report_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), _JSON, 1)
    release_payload = {
        "release_id": config.release_id,
        "review_status": "NOT_STARTED",
        "artifact_order": list(POST_CORE_ARTIFACT_NAMES),
        "artifacts": [{"path": artifact.path, "sha256": artifact.ref.sha256, "media_type": artifact.media_type, "record_count": artifact.record_count} for artifact in (config_artifact, catalog_artifact, cores_artifact, tasks_artifact, split_artifact, manifest_artifact, report_artifact)],
        "release_index_self_hash": None,
        "counts": {"semantic_cores": len(cores), "tasks": len(tasks), "split_cores": report_payload["split_core_counts"], "split_tasks": report_payload["split_task_counts"]},
    }
    release_artifact = _artifact("release_index.json", json.dumps(release_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), _JSON, 1)
    artifacts = (config_artifact, catalog_artifact, cores_artifact, tasks_artifact, split_artifact, manifest_artifact, report_artifact, release_artifact)
    if tuple(item.path for item in artifacts) != POST_CORE_ARTIFACT_NAMES:
        raise AssertionError("post-core artifact order is not canonical")
    return PostCoreArtifactBundle(config, cores, tasks, manifest, report_payload, artifacts)


def _assert_not_frozen_output(output_dir: Path) -> None:
    requested = Path(os.path.abspath(output_dir))
    resolved = requested.resolve(strict=False)
    frozen = tuple((_PROJECT_ROOT / "data" / "vnext" / name).resolve() for name in _FROZEN_ROOT_NAMES)
    if any(resolved == root or root in resolved.parents for root in frozen):
        raise ValueError("main-track output must not overlap frozen Core or Pilot roots")


def _validate_exact_tree(root: Path) -> None:
    if not root.is_dir():
        raise ValueError("main-track artifact root must be a directory")
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != set(POST_CORE_ARTIFACT_NAMES) or len(entries) != len(POST_CORE_ARTIFACT_NAMES):
        raise ValueError("main-track artifact root must contain exactly eight artifacts")
    for entry in entries:
        metadata = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_nlink", 1) != 1:
            raise ValueError("main-track artifacts must be single-link regular files")


def validate_post_core_artifact_tree(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    _assert_not_frozen_output(root)
    _validate_exact_tree(root)
    config_bytes = (root / "generation_config.json").read_bytes()
    config = PostCoreDataConfig.model_validate_json(config_bytes)
    if canonical_json_bytes(config) != config_bytes:
        raise ValueError("generation_config.json is not canonical")
    catalog_bytes = (root / "catalog_manifest.json").read_bytes()
    try:
        catalog = json.loads(catalog_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("catalog_manifest.json is invalid JSON") from exc
    if json.dumps(catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") != catalog_bytes:
        raise ValueError("catalog_manifest.json is not canonical")
    expected_catalog = _catalog_payload(config)
    if catalog != expected_catalog:
        raise ValueError("catalog manifest disagrees with generation config")
    cores_raw = (root / "semantic_cores.jsonl").read_bytes()
    if b"\r" in cores_raw or not cores_raw.endswith(b"\n"):
        raise ValueError("semantic core JSONL framing is not canonical")
    core_rows = [json.loads(line) for line in cores_raw.splitlines()]
    expected_cores = [_core_payload(core) for core in generate_post_core_cores(config)]
    if core_rows != expected_cores:
        raise ValueError("semantic_cores.jsonl does not match canonical generation")
    tasks_raw = (root / "tasks.jsonl").read_bytes()
    if b"\r" in tasks_raw or not tasks_raw.endswith(b"\n"):
        raise ValueError("tasks.jsonl framing is not canonical")
    tasks = tuple(read_models(root / "tasks.jsonl", MemUpdateTaskV3, id_field="task_id"))
    if _jsonl(tasks) != tasks_raw:
        raise ValueError("tasks.jsonl is not canonical")
    split_raw = (root / "split_balance.json").read_bytes()
    split_payload = json.loads(split_raw)
    if _canonical_value_bytes(split_payload) != split_raw:
        raise ValueError("split_balance.json is not canonical")
    split_balance = PostCoreSplitBalanceReport.model_validate(split_payload)
    generated_cores = tuple(generate_post_core_cores(config))
    generated_split = assign_post_core_splits(config, generated_cores)
    if split_balance != generated_split.split_balance:
        raise ValueError("split_balance.json does not match canonical group-first allocation")
    manifest_bytes = (root / "task_manifest.json").read_bytes()
    manifest = TaskManifestV3.model_validate_json(manifest_bytes)
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("task_manifest.json is not canonical")
    refs = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in POST_CORE_ARTIFACT_NAMES[:-1]
    }
    summary = manifest.leakage_check_summary
    if (
        manifest.data_release_id != config.release_id
        or manifest.generation_configs_and_hashes[0].sha256 != refs["generation_config.json"]
        or manifest.source_manifest_paths_and_hashes[0].sha256 != refs["catalog_manifest.json"]
        or manifest.task_file_paths_and_hashes[0].sha256 != refs["tasks.jsonl"]
        or summary["split_balance_sha256"] != refs["split_balance.json"]
        or summary["config_sha256"] != refs["generation_config.json"]
        or summary["catalog_sha256"] != refs["catalog_manifest.json"]
        or summary["code_revision"] != manifest.code_revision
        or summary["code_revision_sha256"] != hashlib.sha256(manifest.code_revision.encode("utf-8")).hexdigest()
        or summary["compiler_version"] != config.compiler_version
    ):
        raise ValueError("task manifest artifact or provenance hash binding is invalid")
    expected_manifest_split_counts = dict(Counter(task.metadata.split.value for task in tasks))
    core_split_by_id: dict[str, str] = {}
    for task in tasks:
        core_id = task.metadata.split_key.semantic_core_id
        observed_split = task.metadata.split.value
        if core_id in core_split_by_id and core_split_by_id[core_id] != observed_split:
            raise ValueError("task variants disagree on semantic-core split")
        core_split_by_id[core_id] = observed_split
    expected_manifest_core_split_counts = dict(Counter(core_split_by_id.values()))
    expected_manifest_family_difficulty = dict(
        Counter(f"{task.task_family}|{task.difficulty.value}" for task in tasks)
    )
    if (
        dict(manifest.split_counts) != expected_manifest_split_counts
        or dict(manifest.semantic_core_counts) != expected_manifest_core_split_counts
        or dict(manifest.family_difficulty_counts) != expected_manifest_family_difficulty
    ):
        raise ValueError("task manifest split counts do not bind tasks.jsonl")
    expected_task_hashes = {task.task_id: sha256_model(task) for task in tasks}
    if dict(manifest.task_record_hashes) != expected_task_hashes:
        raise ValueError("task manifest task_record_hashes do not bind tasks.jsonl")
    expected_task_metadata = {
        task.task_id: {
            "family": task.task_family,
            "domain": task.metadata.extra["domain"],
            "attribute": task.metadata.extra["attribute"],
            "difficulty": task.difficulty.value,
            "split": task.metadata.split.value,
            "surface": task.metadata.extra["surface_key"],
        }
        for task in tasks
    }
    if dict(summary.get("task_metadata", {})) != expected_task_metadata:
        raise ValueError("task manifest per-task metadata does not bind tasks.jsonl")
    if (
        dict(summary.get("domain_counts", {}))
        != dict(Counter(item["domain"] for item in expected_task_metadata.values()))
        or dict(summary.get("attribute_counts", {}))
        != dict(Counter(item["attribute"] for item in expected_task_metadata.values()))
        or dict(summary.get("surface_counts", {}))
        != dict(Counter(item["surface"] for item in expected_task_metadata.values()))
    ):
        raise ValueError("task manifest axis counts do not bind tasks.jsonl")
    report_raw = (root / "validation_report.json").read_bytes()
    report = json.loads(report_raw)
    if _canonical_value_bytes(report) != report_raw:
        raise ValueError("validation_report.json is not canonical")
    release_raw = (root / "release_index.json").read_bytes()
    release = json.loads(release_raw)
    if _canonical_value_bytes(release) != release_raw:
        raise ValueError("release_index.json is not canonical")
    if (
        release.get("release_id") != config.release_id
        or release.get("review_status") != "NOT_STARTED"
        or release.get("artifact_order") != list(POST_CORE_ARTIFACT_NAMES)
        or release.get("release_index_self_hash") is not None
    ):
        raise ValueError("release index artifact order or self-hash policy is invalid")
    release_rows = release.get("artifacts")
    if not isinstance(release_rows, list) or len(release_rows) != len(POST_CORE_ARTIFACT_NAMES) - 1:
        raise ValueError("release index must bind exactly the seven non-self artifacts")
    release_refs = {row.get("path"): row.get("sha256") for row in release_rows}
    if set(release_refs) != set(POST_CORE_ARTIFACT_NAMES[:-1]) or any(release_refs.get(name) != refs[name] for name in POST_CORE_ARTIFACT_NAMES[:-1]):
        raise ValueError("release index hash binding is invalid")
    for task in tasks:
        _validate_task(task)
    expected = _validation_report(config, generated_cores, tasks, generated_split, code_revision=manifest.code_revision, catalog_hash=refs["catalog_manifest.json"])
    if report != expected or report.get("valid") is not True or report.get("review_status") != "NOT_STARTED":
        raise ValueError("validation report disagrees with canonical artifacts")
    return report


def _stage_exact_bytes(staged: Path, expected: bytes) -> None:
    try:
        actual = staged.read_bytes()
    except OSError as exc:
        raise ValueError("staged artifact could not be read") from exc
    if actual != expected:
        raise ValueError("staged artifact bytes changed")


def publish_post_core_artifact_bundle(
    bundle: PostCoreArtifactBundle,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> PublishedPostCoreArtifacts:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    if overwrite:
        raise ValueError("main-track publication is always no-clobber")
    if type(bundle) is not PostCoreArtifactBundle:
        raise TypeError("bundle must be a PostCoreArtifactBundle")
    output_dir = Path(output_dir)
    _assert_not_frozen_output(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("main-track output directory must be empty for no-clobber publication")
    destinations = tuple(output_dir / artifact.path for artifact in bundle.artifacts)
    payloads = {destination: artifact.content for destination, artifact in zip(destinations, bundle.artifacts, strict=True)}
    validators = {
        destination: (lambda staged, expected=artifact.content: _stage_exact_bytes(staged, expected))
        for destination, artifact in zip(destinations, bundle.artifacts, strict=True)
    }
    publish_files_atomically(payloads, overwrite=False, validators=validators)
    validate_post_core_artifact_tree(output_dir)
    return PublishedPostCoreArtifacts(output_dir, destinations, tuple(artifact.ref for artifact in bundle.artifacts))


# Explicit aliases used by command-line callers and downstream qualification code.
build_main_track_artifacts = build_post_core_artifact_bundle
build_main_track_artifact_bundle = build_post_core_artifact_bundle
validate_main_track_artifacts = validate_post_core_artifact_tree
validate_main_track_artifact_tree = validate_post_core_artifact_tree
publish_main_track_artifacts = publish_post_core_artifact_bundle
publish_main_track_artifact_bundle = publish_post_core_artifact_bundle


__all__ = [
    "POST_CORE_ARTIFACT_NAMES",
    "InMemoryPostCoreArtifact",
    "PostCoreArtifactBundle",
    "PublishedPostCoreArtifacts",
    "build_main_track_artifacts",
    "build_main_track_artifact_bundle",
    "build_post_core_artifact_bundle",
    "publish_main_track_artifacts",
    "publish_main_track_artifact_bundle",
    "publish_post_core_artifact_bundle",
    "validate_main_track_artifacts",
    "validate_main_track_artifact_tree",
    "validate_post_core_artifact_tree",
]
