from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from mub.vnext.audit.core import (
    CORE_AUDIT_FAMILIES,
    CORE_AUDIT_SELECTOR_CONFIG,
    CORE_AUDIT_SURFACES,
    CoreAuditSelectionPackage,
    core_audit_selection_hash,
    select_core_audit_sample,
    selector_config_hash,
)
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import sha256_model
from mub.vnext.contracts import Difficulty


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bounded_core_release():
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(
        config, cores_per_family=40, code_revision=revision
    )
    bundle = build_core_artifact_bundle(snapshot, config)
    return snapshot, bundle.task_manifest


def test_core_selection_has_exact_quota_coverage_and_unique_cores(
    bounded_core_release,
) -> None:
    snapshot, manifest = bounded_core_release

    package = select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=sha256_model(manifest),
    )

    assert package.release_ready is False
    assert package.selection_policy_version == "core-audit-v3"
    assert package.source_task_manifest_hash == sha256_model(manifest)
    assert package.selector_config == CORE_AUDIT_SELECTOR_CONFIG
    assert package.selector_config_hash == selector_config_hash(package.selector_config)
    assert package.selection_hash == core_audit_selection_hash(package)
    assert len(package.selections) == 224
    assert len({item.task_id for item in package.selections}) == 224
    assert len({item.audit_id for item in package.selections}) == 224
    assert len({item.semantic_core_id for item in package.selections}) == 224

    by_family = defaultdict(list)
    for item in package.selections:
        by_family[item.family].append(item)
    assert set(by_family) == set(CORE_AUDIT_FAMILIES)
    for family, selections in by_family.items():
        assert len(selections) == 32
        assert Counter(item.surface_id for item in selections) == {
            surface: 8 for surface in CORE_AUDIT_SURFACES
        }
        assert Counter(item.split.value for item in selections) == {
            "train": 22,
            "dev": 3,
            "test": 7,
        }
        report = next(report for report in package.family_reports if report.family == family)
        assert report.uncovered_required_conditions == ()
        covered = {
            condition for item in selections for condition in item.covered_conditions
        }
        assert set(report.required_conditions) <= covered
        assert report.selected_task_ids == tuple(sorted(item.task_id for item in selections))


def test_core_selection_is_order_independent_and_manifest_bound(
    bounded_core_release,
) -> None:
    snapshot, manifest = bounded_core_release
    manifest_hash = sha256_model(manifest)

    first = select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=manifest_hash,
    )
    second = select_core_audit_sample(
        tuple(reversed(snapshot.tasks)),
        manifest,
        source_task_manifest_hash=manifest_hash,
    )

    assert first == second
    with pytest.raises(ValueError, match="source task manifest hash"):
        select_core_audit_sample(
            snapshot.tasks,
            manifest,
            source_task_manifest_hash="0" * 64,
        )


def test_core_selection_rejects_task_hash_corruption_and_duplicate_task(
    bounded_core_release,
) -> None:
    snapshot, manifest = bounded_core_release
    manifest_hash = sha256_model(manifest)
    victim = snapshot.tasks[0]
    corrupted_difficulty = (
        Difficulty.HARD if victim.difficulty is not Difficulty.HARD else Difficulty.EASY
    )
    corrupted = victim.model_copy(update={"difficulty": corrupted_difficulty})

    with pytest.raises(ValueError, match="task record hash"):
        select_core_audit_sample(
            (corrupted, *snapshot.tasks[1:]),
            manifest,
            source_task_manifest_hash=manifest_hash,
        )
    with pytest.raises(ValueError, match="unique"):
        select_core_audit_sample(
            (*snapshot.tasks, snapshot.tasks[0]),
            manifest,
            source_task_manifest_hash=manifest_hash,
        )


def test_core_selection_package_fails_closed_on_hash_or_schema_tampering(
    bounded_core_release,
) -> None:
    snapshot, manifest = bounded_core_release
    package = select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=sha256_model(manifest),
    )
    payload = package.model_dump(mode="python", exclude_computed_fields=True)

    with pytest.raises(ValidationError):
        CoreAuditSelectionPackage.model_validate(
            {**payload, "selector_config_hash": "f" * 64}
        )
    with pytest.raises(ValidationError):
        CoreAuditSelectionPackage.model_validate(
            {**payload, "selection_hash": "f" * 64}
        )
    with pytest.raises(ValidationError):
        CoreAuditSelectionPackage.model_validate(
            {**payload, "schema_version": "memupdatebench.task.v2"}
        )
