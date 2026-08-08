from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import os
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
import mub.vnext.audit.core as core_audit
import mub.vnext.audit.core_candidate as core_candidate
import mub.vnext.audit.core_stage as core_stage
from mub.vnext.audit.core_candidate import (
    CoreCandidateValidationReceipt,
    build_core_candidate_validation_receipt,
    core_candidate_receipt_hash,
    core_candidate_root_digest,
    verify_core_candidate_validation_receipt,
)
from mub.vnext.audit.core_stage import (
    core_audit_review_task_ids,
    gate_core_audit_files,
    load_core_audit_selection_package,
)
from mub.vnext.audit.core_review import (
    core_audit_decision_templates,
    validate_core_audit_review_context,
)
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.contracts import Difficulty, Split, TaskFamily
from mub.vnext.contracts.common import ArtifactRef


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
    for item in package.selections:
        assert len(item.surface_variants) == 4
        assert {variant.surface_id for variant in item.surface_variants} == set(
            CORE_AUDIT_SURFACES
        )
        selected_variant = next(
            variant for variant in item.surface_variants if variant.surface_id == item.surface_id
        )
        assert (selected_variant.task_id, selected_variant.task_hash) == (
            item.task_id,
            item.task_hash,
        )
    review_task_ids = core_audit_review_task_ids(package)
    assert len(review_task_ids) == 896
    assert len(set(review_task_ids)) == 896
    assert set(review_task_ids) == {
        variant.task_id
        for item in package.selections
        for variant in item.surface_variants
    }

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


def test_core_selection_bounds_generator_consumption_before_materializing_release(
    bounded_core_release,
) -> None:
    snapshot, manifest = bounded_core_release
    consumed = 0

    def oversized():
        nonlocal consumed
        for _ in range(core_audit._MAX_TASKS + 10):
            consumed += 1
            yield snapshot.tasks[0]

    with pytest.raises(ValueError, match="12,000-task"):
        select_core_audit_sample(
            oversized(),
            manifest,
            source_task_manifest_hash=sha256_model(manifest),
        )
    assert consumed == core_audit._MAX_TASKS + 1


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


def _jsonl(models) -> bytes:
    return b"".join(canonical_json_bytes(model) + b"\n" for model in models)


def _dummy_artifact_refs(manifest_hash: str, tasks_hash: str):
    hashes = {
        path: hashlib.sha256(path.encode("ascii")).hexdigest()
        for path in core_candidate._PATHS
    }
    hashes["task_manifest.json"] = manifest_hash
    hashes["tasks.jsonl"] = tasks_hash
    return tuple(
        ArtifactRef(
            path=path,
            sha256=hashes[path],
            media_type=(
                "application/x-ndjson" if path.endswith(".jsonl") else "application/json"
            ),
        )
        for path in core_candidate._PATHS
    )


def test_candidate_receipt_cannot_authenticate_without_current_root_validation(
    tmp_path,
) -> None:
    candidate = tmp_path / "old-candidate"
    candidate.mkdir()
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    artifacts = _dummy_artifact_refs("a" * 64, "b" * 64)
    payload = {
        "receipt_version": "core-candidate-validation-v1",
        "candidate_dir": str(candidate.resolve()),
        "expected_full": True,
        "source_task_manifest_hash": "a" * 64,
        "tasks_artifact_hash": "b" * 64,
        "task_count": 12_000,
        "code_revision": revision,
        "trusted_audit_tooling_revision": revision,
        "candidate_artifacts": artifacts,
        "candidate_root_digest": core_candidate_root_digest(artifacts),
    }
    receipt = CoreCandidateValidationReceipt(
        **payload,
        receipt_hash=core_candidate_receipt_hash(payload),
    )

    assert verify_core_candidate_validation_receipt(
        receipt, trusted_candidate_root=Path(receipt.candidate_dir)
    ) is False


def _write_bounded_candidate(root, snapshot):
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    bundle = build_core_artifact_bundle(snapshot, config)
    root.mkdir()
    for artifact in bundle.artifacts:
        (root / artifact.path).write_bytes(artifact.content)


@pytest.mark.parametrize(
    ("artifact_name", "operation"),
    (
        ("generation_config.json", "mutate"),
        ("core-hard-v1.json", "delete"),
        ("unexpected.json", "extra"),
    ),
)
def test_cached_receipt_rechecks_all_required_candidate_artifacts(
    bounded_core_release,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    operation: str,
) -> None:
    snapshot, _ = bounded_core_release
    candidate = tmp_path / f"candidate-{artifact_name}"
    _write_bounded_candidate(candidate, snapshot)

    class ValidReport:
        valid = True

    monkeypatch.setattr(
        core_candidate, "validate_core_release", lambda *args, **kwargs: ValidReport()
    )
    monkeypatch.setattr(
        core_candidate,
        "_assert_tracked_core_sources_clean",
        lambda: None,
        raising=False,
    )
    *_, receipt = core_stage._load_candidate(candidate, expected_full=False)
    target = candidate / artifact_name
    if operation == "delete":
        target.unlink()
    elif operation == "extra":
        target.write_bytes(b"{}")
    else:
        target.write_bytes(target.read_bytes() + b" ")

    assert verify_core_candidate_validation_receipt(
        receipt, trusted_candidate_root=Path(receipt.candidate_dir)
    ) is False


def test_candidate_boundary_rejects_hardlinked_artifact_before_read(
    bounded_core_release,
    tmp_path,
) -> None:
    snapshot, _ = bounded_core_release
    candidate = tmp_path / "candidate-hardlink"
    _write_bounded_candidate(candidate, snapshot)
    alias = tmp_path / "tasks-hardlink.jsonl"
    try:
        os.link(candidate / "tasks.jsonl", alias)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="single-link regular files"):
        core_candidate._guarded_candidate_tree_snapshot(candidate)


def test_cached_receipt_rechecks_tracked_core_cleanliness(
    bounded_core_release,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _ = bounded_core_release
    candidate = tmp_path / "candidate-dirty-tooling"
    _write_bounded_candidate(candidate, snapshot)

    class ValidReport:
        valid = True

    dirty = False

    def cleanliness_check():
        if dirty:
            raise ValueError("tracked Core source is dirty")

    monkeypatch.setattr(
        core_candidate, "validate_core_release", lambda *args, **kwargs: ValidReport()
    )
    monkeypatch.setattr(
        core_candidate,
        "_assert_tracked_core_sources_clean",
        cleanliness_check,
        raising=False,
    )
    *_, receipt = core_stage._load_candidate(candidate, expected_full=False)
    dirty = True

    assert verify_core_candidate_validation_receipt(
        receipt, trusted_candidate_root=Path(receipt.candidate_dir)
    ) is False


def _unregistered_receipt(candidate, snapshot, manifest):
    tree = core_candidate._guarded_candidate_tree_snapshot(candidate)
    return build_core_candidate_validation_receipt(
        candidate_dir=candidate,
        manifest_bytes=tree.content("task_manifest.json"),
        tasks_bytes=tree.content("tasks.jsonl"),
        manifest=manifest,
        expected_full=False,
        candidate_snapshot=tree,
    )


def test_receipt_build_does_not_bypass_authoritative_validation(
    bounded_core_release,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, manifest = bounded_core_release
    candidate = tmp_path / "candidate-registration-bypass"
    _write_bounded_candidate(candidate, snapshot)
    monkeypatch.setattr(
        core_candidate, "_assert_tracked_core_sources_clean", lambda: None
    )
    receipt = _unregistered_receipt(candidate, snapshot, manifest)
    core_candidate._VERIFIED_RECEIPT_SNAPSHOTS.pop(receipt.receipt_hash, None)
    calls = 0

    class ValidReport:
        valid = True

    def count_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return ValidReport()

    monkeypatch.setattr(core_candidate, "validate_core_release", count_validation)

    assert verify_core_candidate_validation_receipt(
        receipt, trusted_candidate_root=Path(receipt.candidate_dir)
    ) is True
    assert calls == 1


def test_receipt_rejects_trusted_head_change_during_validation(
    bounded_core_release,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, manifest = bounded_core_release
    candidate = tmp_path / "candidate-head-race"
    _write_bounded_candidate(candidate, snapshot)
    state = {"revision": manifest.code_revision}
    monkeypatch.setattr(
        core_candidate, "_assert_tracked_core_sources_clean", lambda: None
    )
    monkeypatch.setattr(
        core_candidate, "_trusted_code_revision", lambda: state["revision"]
    )
    receipt = _unregistered_receipt(candidate, snapshot, manifest)
    core_candidate._VERIFIED_RECEIPT_SNAPSHOTS.pop(receipt.receipt_hash, None)

    class ValidReport:
        valid = True

    def change_head(*args, **kwargs):
        state["revision"] = "f" * 40
        return ValidReport()

    monkeypatch.setattr(core_candidate, "validate_core_release", change_head)

    assert verify_core_candidate_validation_receipt(
        receipt, trusted_candidate_root=Path(receipt.candidate_dir)
    ) is False


def test_receipt_verification_rejects_swap_after_authoritative_validation(
    bounded_core_release,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, manifest = bounded_core_release
    candidate = tmp_path / "candidate-swap"
    _write_bounded_candidate(candidate, snapshot)
    current_manifest_bytes = (candidate / "task_manifest.json").read_bytes()
    current_tasks_bytes = (candidate / "tasks.jsonl").read_bytes()
    replacement_manifest = manifest.model_copy(
        update={"data_release_id": "replacement-after-validation"}
    )
    replacement_manifest_bytes = canonical_json_bytes(replacement_manifest)
    replacement_tasks_bytes = _jsonl(tuple(reversed(snapshot.tasks)))
    (candidate / "task_manifest.json").write_bytes(replacement_manifest_bytes)
    (candidate / "tasks.jsonl").write_bytes(replacement_tasks_bytes)
    monkeypatch.setattr(
        core_candidate, "_assert_tracked_core_sources_clean", lambda: None
    )
    receipt = build_core_candidate_validation_receipt(
        candidate_dir=candidate,
        manifest_bytes=replacement_manifest_bytes,
        tasks_bytes=replacement_tasks_bytes,
        manifest=replacement_manifest,
        expected_full=False,
    )
    assert receipt.code_revision == receipt.trusted_audit_tooling_revision
    (candidate / "task_manifest.json").write_bytes(current_manifest_bytes)
    (candidate / "tasks.jsonl").write_bytes(current_tasks_bytes)
    class ValidReport:
        valid = True

    def validate_then_swap(*args, **kwargs):
        (candidate / "task_manifest.json").write_bytes(
            replacement_manifest_bytes
        )
        (candidate / "tasks.jsonl").write_bytes(replacement_tasks_bytes)
        return ValidReport()

    monkeypatch.setattr(
        core_candidate, "validate_core_release", validate_then_swap
    )

    assert verify_core_candidate_validation_receipt(
        receipt, trusted_candidate_root=Path(receipt.candidate_dir)
    ) is False


def test_selection_and_gate_loaders_require_canonical_authenticated_bytes(
    bounded_core_release,
    tmp_path,
) -> None:
    snapshot, manifest = bounded_core_release
    package = select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=sha256_model(manifest),
    )
    selection_path = tmp_path / "audit_selection.json"
    selection_path.write_bytes(canonical_json_bytes(package))
    assert load_core_audit_selection_package(selection_path) == package
    canonical = selection_path.read_bytes()
    selection_path.write_bytes(b" " + canonical)
    with pytest.raises(ValueError, match="canonical"):
        load_core_audit_selection_package(selection_path)
    selection_path.write_bytes(
        canonical.replace(
            b'"schema_version":',
            b'"schema_version":"memupdatebench.core.audit.v3","schema_version":',
            1,
        )
    )
    with pytest.raises(ValueError, match="canonical"):
        load_core_audit_selection_package(selection_path)


def test_candidate_loader_detects_manifest_or_task_change_during_validation(
    bounded_core_release,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, manifest = bounded_core_release
    candidate = tmp_path / "candidate"
    _write_bounded_candidate(candidate, snapshot)
    tasks_path = candidate / "tasks.jsonl"

    class Report:
        valid = True

    def mutate_after_initial_read(*args, **kwargs):
        tasks_path.write_bytes(tasks_path.read_bytes() + b"\n")
        return Report()

    monkeypatch.setattr(
        core_candidate, "validate_core_release", mutate_after_initial_read
    )
    monkeypatch.setattr(
        core_candidate, "_assert_tracked_core_sources_clean", lambda: None
    )
    with pytest.raises(ValueError, match="tree or trusted revision changed"):
        core_stage._load_candidate(candidate, expected_full=False)


def test_gate_authenticates_manifest_selected_rows_and_four_surface_context(
    bounded_core_release,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, manifest = bounded_core_release
    package = select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=sha256_model(manifest),
    )
    monkeypatch.setattr(
        core_stage,
        "_load_candidate",
        lambda *args, **kwargs: (
            snapshot.tasks,
            manifest,
            sha256_model(manifest),
            None,
        ),
    )
    by_id = {task.task_id: task for task in snapshot.tasks}
    selected = tuple(by_id[item.task_id] for item in package.selections)
    surfaces = tuple(by_id[task_id] for task_id in core_audit_review_task_ids(package))
    assert len(validate_core_audit_review_context(package, selected, surfaces)) == 896
    paths = {
        "selection": tmp_path / "selection.json",
        "manifest": tmp_path / "task_manifest.json",
        "selected": tmp_path / "selected.jsonl",
        "surfaces": tmp_path / "surfaces.jsonl",
        "decisions": tmp_path / "decisions.jsonl",
    }
    paths["selection"].write_bytes(canonical_json_bytes(package))
    paths["manifest"].write_bytes(canonical_json_bytes(manifest))
    paths["selected"].write_bytes(_jsonl(selected))
    paths["surfaces"].write_bytes(_jsonl(surfaces))
    paths["decisions"].write_bytes(_jsonl(core_audit_decision_templates(package)))
    trusted_candidate = tmp_path / "trusted-candidate"
    _write_bounded_candidate(trusted_candidate, snapshot)

    report = gate_core_audit_files(
        selection_package_path=paths["selection"],
        candidate_dir=tmp_path / "trusted-candidate",
        selected_tasks_path=paths["selected"],
        surface_context_path=paths["surfaces"],
        decisions_path=paths["decisions"],
        adjudications_path=None,
        output_dir=tmp_path / "gate-output",
    )
    assert report.release_ready is False
    assert len(report.surface_context_evidence) == 896

    mutated = selected[0].model_copy(
        update={
            "difficulty": (
                Difficulty.HARD
                if selected[0].difficulty is not Difficulty.HARD
                else Difficulty.EASY
            )
        }
    )
    paths["selected"].write_bytes(_jsonl((mutated, *selected[1:])))
    with pytest.raises(ValueError, match="authenticated|hash|differ"):
        gate_core_audit_files(
            selection_package_path=paths["selection"],
            candidate_dir=tmp_path / "trusted-candidate",
            selected_tasks_path=paths["selected"],
            surface_context_path=paths["surfaces"],
            decisions_path=paths["decisions"],
            adjudications_path=None,
            output_dir=tmp_path / "mutated-output",
        )
    paths["selected"].write_bytes(_jsonl(selected))
    paths["surfaces"].write_bytes(_jsonl((mutated, *surfaces[1:])))
    with pytest.raises(ValueError, match="authenticated|hash|context"):
        gate_core_audit_files(
            selection_package_path=paths["selection"],
            candidate_dir=tmp_path / "trusted-candidate",
            selected_tasks_path=paths["selected"],
            surface_context_path=paths["surfaces"],
            decisions_path=paths["decisions"],
            adjudications_path=None,
            output_dir=tmp_path / "mutated-context-output",
        )
    paths["surfaces"].write_bytes(_jsonl(surfaces))
    monkeypatch.setattr(
        core_candidate, "_assert_tracked_core_sources_clean", lambda: None
    )
    tree = core_candidate._guarded_candidate_tree_snapshot(trusted_candidate)
    validation_receipt = build_core_candidate_validation_receipt(
        candidate_dir=trusted_candidate,
        manifest_bytes=tree.content("task_manifest.json"),
        tasks_bytes=tree.content("tasks.jsonl"),
        manifest=manifest,
        expected_full=False,
        candidate_snapshot=tree,
    )
    core_candidate._register_validated_core_candidate_receipt(
        validation_receipt, validated_snapshot=tree
    )
    monkeypatch.setattr(
        core_stage,
        "_load_candidate",
        lambda *args, **kwargs: (
            snapshot.tasks,
            manifest,
            sha256_model(manifest),
            validation_receipt,
        ),
    )

    def inject_source_change(*args, pre_publish, **kwargs):
        ancillary = trusted_candidate / "generation_config.json"
        ancillary.write_bytes(ancillary.read_bytes() + b" ")
        pre_publish()

    monkeypatch.setattr(
        core_stage, "publish_files_atomically", inject_source_change
    )
    with pytest.raises(ValueError, match="validated candidate root changed"):
        gate_core_audit_files(
            selection_package_path=paths["selection"],
            candidate_dir=trusted_candidate,
            selected_tasks_path=paths["selected"],
            surface_context_path=paths["surfaces"],
            decisions_path=paths["decisions"],
            adjudications_path=None,
            output_dir=tmp_path / "source-race-output",
        )


@pytest.mark.parametrize("field", ("family", "difficulty", "split", "conditions"))
def test_review_context_recomputes_selection_projection_fields(
    bounded_core_release,
    field: str,
) -> None:
    snapshot, manifest = bounded_core_release
    package = select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=sha256_model(manifest),
    )
    victim = package.selections[0]
    updates = {
        "family": TaskFamily.INTERLEAVED_MULTI_SLOT,
        "difficulty": (
            Difficulty.HARD
            if victim.difficulty is not Difficulty.HARD
            else Difficulty.EASY
        ),
        "split": Split.DEV if victim.split is not Split.DEV else Split.TEST,
        "conditions": tuple(
            token for token in victim.covered_conditions if not token.startswith("difficulty=")
        ),
    }
    row_payload = victim.model_dump(mode="python")
    row_payload["surface_variants"] = victim.surface_variants
    if field == "conditions":
        row_payload["covered_conditions"] = updates[field]
    else:
        row_payload[field] = updates[field]
    corrupted_row = type(victim).model_construct(**row_payload)
    package_payload = object.__getattribute__(package, "__dict__").copy()
    package_payload["selections"] = (corrupted_row, *package.selections[1:])
    corrupted = type(package).model_construct(**package_payload)
    by_id = {task.task_id: task for task in snapshot.tasks}
    selected = tuple(by_id[item.task_id] for item in package.selections)
    surfaces = tuple(by_id[task_id] for task_id in core_audit_review_task_ids(package))

    with pytest.raises(ValueError, match="selection fields"):
        validate_core_audit_review_context(corrupted, selected, surfaces)


def test_exact_quota_cover_recovers_when_greedy_consumes_scarce_split() -> None:
    def candidate(core_id, split, conditions):
        return core_audit._CoreCandidate(
            core_id=core_id,
            family=TaskFamily.REPEATED_SAME_SLOT,
            difficulty=Difficulty.EASY,
            split=split,
            conditions=tuple(conditions),
            tasks_by_surface={},
        )

    candidates = (
        candidate("train-greedy", Split.TRAIN, ("a", "b")),
        candidate("train-scarce", Split.TRAIN, ("c",)),
        candidate("dev-a", Split.DEV, ("a",)),
        candidate("dev-b", Split.DEV, ("b",)),
    )
    cover = core_audit._exact_quota_cover(
        candidates,
        {"a", "b", "c"},
        {Split.TRAIN: 1, Split.DEV: 2, Split.TEST: 0},
        3,
    )

    assert cover is not None
    assert {item.core_id for item in cover} == {
        "train-scarce",
        "dev-a",
        "dev-b",
    }
    filler = candidate("dev-filler", Split.DEV, ())
    labelled = core_audit._selection_reasons((*cover, filler), {"a", "b", "c"})
    assert [reason for _, reason in labelled[:-1]] == [
        "quota_set_cover",
        "quota_set_cover",
        "quota_set_cover",
    ]
    assert labelled[-1] == (filler, "quota_spread_fill")


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
