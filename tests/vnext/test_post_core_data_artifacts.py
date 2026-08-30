from __future__ import annotations

import json
from pathlib import Path

import pytest

from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.post_core_artifacts import (
    POST_CORE_ARTIFACT_NAMES,
    build_post_core_artifact_bundle,
    publish_post_core_artifact_bundle,
    validate_post_core_artifact_tree,
)
from mub.vnext.generation.post_core_config import load_post_core_data_config
from mub.vnext.io import semantic_task_hash_v3


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "post_core_data.yaml"
CODE_REVISION = "post-core-artifacts-test"


@pytest.fixture(scope="module")
def config():
    return load_post_core_data_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def bundle(config):
    return build_post_core_artifact_bundle(config, code_revision=CODE_REVISION)


def test_main_track_artifact_counts_and_exact_order(bundle) -> None:
    assert tuple(artifact.path for artifact in bundle.artifacts) == POST_CORE_ARTIFACT_NAMES
    assert bundle.semantic_core_count == 900
    assert bundle.task_count == 3600
    assert bundle.split_core_counts == {"train": 630, "dev": 90, "test": 180}
    assert bundle.split_task_counts == {"train": 2520, "dev": 360, "test": 720}
    assert bundle.validation_report["valid"] is True
    assert bundle.validation_report["review_status"] == "NOT_STARTED"


def test_task_manifest_binds_all_generation_inputs_and_task_hashes(bundle) -> None:
    manifest = bundle.task_manifest.model_dump(mode="json")
    assert manifest["data_release_id"] == "main_track_v1"
    summary = manifest["leakage_check_summary"]
    assert summary["config_sha256"] == bundle.artifact("generation_config.json").ref.sha256
    assert summary["catalog_sha256"] == bundle.artifact("catalog_manifest.json").ref.sha256
    assert summary["code_revision"] == CODE_REVISION
    assert summary["code_revision_sha256"]
    assert summary["compiler_version"] == "2.0.0"
    assert summary["split_policy_version"] == "vnext-post-core-data-splits-v1"
    assert len(manifest["task_record_hashes"]) == 3600
    assert len(manifest["leakage_check_summary"]["task_metadata"]) == 3600
    assert set(manifest["leakage_check_summary"]["surface_counts"]) == {
        "en-US/explicit_canonical",
        "en-US/concise_natural",
        "es-ES/concise_natural",
        "ja-JP/concise_natural",
    }


def test_rebuild_is_byte_for_byte_deterministic(config, bundle) -> None:
    rebuilt = build_post_core_artifact_bundle(config, code_revision=CODE_REVISION)
    assert tuple(item.content for item in rebuilt.artifacts) == tuple(
        item.content for item in bundle.artifacts
    )


def test_validator_rejects_tampered_artifact(tmp_path: Path, bundle) -> None:
    output = tmp_path / "main_track_v1"
    publish_post_core_artifact_bundle(bundle, output)
    tasks_path = output / "tasks.jsonl"
    tasks_path.write_bytes(tasks_path.read_bytes().replace(b"ADD", b"BAD", 1))
    with pytest.raises(ValueError, match="canonical|hash|manifest|task"):
        validate_post_core_artifact_tree(output)


def test_publisher_rejects_frozen_core_or_pilot_roots(bundle) -> None:
    with pytest.raises(ValueError, match="frozen|immutable|Core|Pilot"):
        publish_post_core_artifact_bundle(bundle, ROOT / "data" / "vnext" / "core")


def test_four_surfaces_share_semantics_but_not_surface_text(bundle) -> None:
    rows = [
        json.loads(line)
        for line in bundle.artifact("tasks.jsonl").content.splitlines()
    ]
    by_core: dict[str, list] = {}
    for row in rows:
        by_core.setdefault(row["metadata"]["split_key"]["semantic_core_id"], []).append(row)
    assert len(by_core) == 900
    for tasks in by_core.values():
        assert len(tasks) == 4
        assert len({task["metadata"]["extra"]["surface_key"] for task in tasks}) == 4
        assert len({task["source"]["raw_hash"] for task in tasks}) == 4
        assert len({semantic_task_hash_v3(MemUpdateTaskV3.model_validate(task)) for task in tasks}) == 1


def test_all_semantic_cores_have_distinct_semantic_payloads(bundle) -> None:
    rows = [json.loads(line) for line in bundle.artifact("tasks.jsonl").content.splitlines()]
    by_core = {}
    for row in rows:
        by_core.setdefault(row["metadata"]["split_key"]["semantic_core_id"], []).append(row)
    hashes = {
        next(iter({semantic_task_hash_v3(MemUpdateTaskV3.model_validate(task)) for task in tasks}))
        for tasks in by_core.values()
    }
    assert len(by_core) == 900
    assert len(hashes) == 900


def test_all_rendered_cores_have_unique_semantic_hashes(bundle) -> None:
    by_core: dict[str, set[str]] = {}
    for task in bundle.tasks:
        by_core.setdefault(task.metadata.split_key.semantic_core_id, set()).add(
            semantic_task_hash_v3(task)
        )

    assert len(by_core) == 900
    assert all(len(hashes) == 1 for hashes in by_core.values())
    assert len({next(iter(hashes)) for hashes in by_core.values()}) == 900
