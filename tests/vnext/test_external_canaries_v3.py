from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import pytest

from mub.vnext.contracts.common import ArtifactRef


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "data" / "vnext" / "core" / "v3"


def _tree_snapshot(root: Path) -> dict[str, tuple[str, str | None]]:
    snapshot = {".": ("directory", None)}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", str(path.readlink()))
        elif path.is_file():
            snapshot[relative] = (
                "file",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        elif path.is_dir():
            snapshot[relative] = ("directory", None)
        else:
            snapshot[relative] = ("other", None)
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def immutable_release_snapshot():
    before = _tree_snapshot(RELEASE_ROOT)
    yield before
    assert _tree_snapshot(RELEASE_ROOT) == before


@pytest.fixture(scope="session")
def authenticated_release():
    from mub.vnext.external.canaries_v3 import authenticate_core_release

    return authenticate_core_release(RELEASE_ROOT)


@pytest.fixture(scope="session")
def canary_set(authenticated_release):
    from mub.vnext.external.canaries_v3 import _build_canary_set_authenticated

    return _build_canary_set_authenticated(authenticated_release)


def test_canary_policy_is_immutable_and_version_bound():
    import mub.vnext.external.canaries_v3 as canary_module
    from mub.vnext.external import (
        AuthenticatedCoreReleaseV1 as PublicAuthenticatedCoreReleaseV1,
    )

    assert PublicAuthenticatedCoreReleaseV1 is canary_module.AuthenticatedCoreReleaseV1
    assert isinstance(canary_module.FAMILY_LETTERS, MappingProxyType)
    assert isinstance(canary_module.CANARY_FAMILY_QUOTAS, MappingProxyType)
    assert len(canary_module.CANARY_SELECTION_POLICY_SHA256) == 64
    assert (
        canary_module.CANARY_SELECTION_POLICY_SHA256
        in canary_module.CANARY_SELECTION_VERSION
    )
    assert "validate_canary_manifest" not in canary_module.__all__


def test_canaries_are_dev_only_quota_bound_and_independent(canary_set):
    from mub.vnext.external.canaries_v3 import CANARY_FAMILY_QUOTAS, FAMILY_LETTERS

    assert FAMILY_LETTERS == {
        "repeated_same_slot_update": "A",
        "interleaved_multi_slot_update": "B",
        "entity_attribute_grounding": "C",
        "noop_write_discipline": "D",
        "deletion_forgetting": "E",
        "current_historical_query": "F",
        "long_horizon_memory_synthesis": "G",
    }
    assert CANARY_FAMILY_QUOTAS == {
        "A": 8,
        "B": 8,
        "C": 8,
        "D": 8,
        "E": 12,
        "F": 12,
        "G": 8,
    }
    assert len(canary_set.canaries) == 2
    assert {
        bundle.manifest.canary_id for bundle in canary_set.canaries
    } == {"canary_a", "canary_b"}

    field_values = {
        field: []
        for field in (
            "task_id",
            "semantic_core_id",
            "trajectory_id",
            "version_group_id",
            "source_group_id",
            "source_document_id",
            "paraphrase_group_id",
        )
    }
    for bundle in canary_set.canaries:
        assert len(bundle.records) == 64
        assert len(bundle.manifest.selected_tasks) == 64
        assert (
            Counter(
                item.family_letter
                for item in bundle.manifest.selected_tasks
            )
            == CANARY_FAMILY_QUOTAS
        )
        assert all(
            task.metadata.split.value == "dev" for task in bundle.tasks
        )
        assert (
            len(
                {
                    item.semantic_core_id
                    for item in bundle.manifest.selected_tasks
                }
            )
            == 64
        )
        for field in field_values:
            field_values[field].append(
                {
                    getattr(item, field)
                    for item in bundle.manifest.selected_tasks
                }
            )
    for field, values in field_values.items():
        assert values[0].isdisjoint(values[1]), field


def test_family_f_groups_are_deterministically_partitioned_before_hash_fill(
    authenticated_release,
    canary_set,
):
    from mub.vnext.external.canaries_v3 import _family_f_group_partition

    partition = _family_f_group_partition(authenticated_release)
    source_groups = {
        (
            task.metadata.split_key.trajectory_id,
            task.metadata.split_key.version_group_id,
        )
        for task in authenticated_release.dev_tasks
        if task.task_family == "current_historical_query"
    }
    assert set(partition) == {"canary_a", "canary_b"}
    assert partition["canary_a"].isdisjoint(partition["canary_b"])
    assert partition["canary_a"] | partition["canary_b"] == source_groups
    assert len(partition["canary_a"]) == len(partition["canary_b"]) == 3

    for bundle in canary_set.canaries:
        selected_groups = {
            (item.trajectory_id, item.version_group_id)
            for item in bundle.manifest.selected_tasks
            if item.family_letter == "F"
        }
        assert selected_groups <= partition[bundle.manifest.canary_id]


def test_canary_binds_authenticated_source_and_preserves_exact_lines(
    authenticated_release,
    canary_set,
):
    from mub.vnext.external.canaries_v3 import (
        _validate_canary_bundle_authenticated,
    )

    assert authenticated_release.release_manifest.release_stage == "task_release"
    assert authenticated_release.release_manifest.release_status == "FINAL_APPROVED"
    assert (
        authenticated_release.release_manifest_hash
        == authenticated_release.release_manifest.release_manifest_hash
    )
    for bundle in canary_set.canaries:
        _validate_canary_bundle_authenticated(bundle, authenticated_release)
        assert (
            bundle.manifest.source_release_manifest_hash
            == authenticated_release.release_manifest_hash
        )
        assert (
            bundle.manifest.source_task_manifest_ref.sha256
            == authenticated_release.task_manifest_ref.sha256
        )
        assert (
            bundle.manifest.source_tasks_ref.sha256
            == authenticated_release.tasks_ref.sha256
        )
        assert bundle.manifest.tasks_ref.sha256 == hashlib.sha256(
            b"".join(bundle.records)
        ).hexdigest()
        assert bundle.manifest.tasks_ref.record_count == 64
        selected = {
            item.task_id: item.task_record_hash
            for item in bundle.manifest.selected_tasks
        }
        for raw, task in zip(bundle.records, bundle.tasks, strict=True):
            assert raw == authenticated_release.raw_record_by_task_id[
                task.task_id
            ]
            assert selected[task.task_id] == (
                authenticated_release.task_manifest.task_record_hashes[
                    task.task_id
                ]
            )


def test_authenticated_release_mappings_are_read_only(authenticated_release):
    with pytest.raises(TypeError):
        authenticated_release.raw_record_by_task_id["forged"] = b"{}\n"
    with pytest.raises(TypeError):
        authenticated_release.task_by_id["forged"] = next(
            iter(authenticated_release.task_by_id.values())
        )


def test_authenticated_release_retains_only_dev_payloads(authenticated_release):
    assert len(authenticated_release.task_by_id) == 1200
    assert len(authenticated_release.raw_record_by_task_id) == 1200
    assert set(authenticated_release.task_by_id) == set(
        authenticated_release.raw_record_by_task_id
    )
    assert all(
        task.metadata.split.value == "dev"
        for task in authenticated_release.task_by_id.values()
    )


def test_build_reauthenticates_constructed_release_snapshots(
    authenticated_release,
    canary_set,
):
    from dataclasses import replace

    from mub.vnext.external.canaries_v3 import build_canary_set

    forged_records = dict(authenticated_release.raw_record_by_task_id)
    forged_records[canary_set.canaries[0].tasks[0].task_id] = b"{}\n"
    forged_release = replace(
        authenticated_release,
        raw_record_by_task_id=forged_records,
    )
    rebuilt = build_canary_set(forged_release)
    assert tuple(
        canary.records for canary in rebuilt.canaries
    ) == tuple(canary.records for canary in canary_set.canaries)


def test_validation_rejects_noncanonical_authenticated_selection(
    authenticated_release,
    canary_set,
):
    from mub.vnext.external.canaries_v3 import (
        CanarySetBundleV1,
        CanarySetManifestV1,
        _make_bundle,
        validate_canary_set,
    )
    from mub.vnext.io import canonical_json_bytes

    left_tasks = list(canary_set.canaries[0].tasks)
    right_tasks = list(canary_set.canaries[1].tasks)
    left_index = next(
        index
        for index, task in enumerate(left_tasks)
        if task.task_family == "repeated_same_slot_update"
    )
    right_index = next(
        index
        for index, task in enumerate(right_tasks)
        if task.task_family == "repeated_same_slot_update"
    )
    left_tasks[left_index], right_tasks[right_index] = (
        right_tasks[right_index],
        left_tasks[left_index],
    )
    bundles = (
        _make_bundle(
            authenticated_release,
            "canary_a",
            tuple(sorted(left_tasks, key=lambda task: task.task_id)),
        ),
        _make_bundle(
            authenticated_release,
            "canary_b",
            tuple(sorted(right_tasks, key=lambda task: task.task_id)),
        ),
    )
    refs = tuple(
        ArtifactRef(
            path=f"{bundle.manifest.canary_id}/canary_manifest.json",
            sha256=hashlib.sha256(bundle.manifest_bytes).hexdigest(),
            media_type="application/json",
            record_count=1,
        )
        for bundle in bundles
    )
    set_manifest = CanarySetManifestV1(
        selection_version=canary_set.set_manifest.selection_version,
        source_release_manifest_hash=(
            canary_set.set_manifest.source_release_manifest_hash
        ),
        canary_manifest_refs=refs,
        canary_ids=canary_set.set_manifest.canary_ids,
        independence_fields=canary_set.set_manifest.independence_fields,
    )
    forged = CanarySetBundleV1(
        canaries=bundles,
        set_manifest=set_manifest,
        set_manifest_bytes=canonical_json_bytes(set_manifest),
    )
    with pytest.raises(ValueError, match="canonical canary selection"):
        validate_canary_set(forged, authenticated_release)


def test_release_authentication_rejects_extra_entries_and_hardlinks(tmp_path):
    from mub.vnext.external.canaries_v3 import authenticate_core_release

    copied_release = tmp_path / "release-with-extra-entry"
    shutil.copytree(RELEASE_ROOT, copied_release)

    extra_directory = copied_release / "unmanifested-empty-directory"
    extra_directory.mkdir()
    with pytest.raises(ValueError, match="release tree"):
        authenticate_core_release(copied_release)
    extra_directory.rmdir()

    extra_file = copied_release / "unmanifested.txt"
    extra_file.write_text("not part of the approved release", encoding="utf-8")
    with pytest.raises(ValueError, match="release tree"):
        authenticate_core_release(copied_release)
    extra_file.unlink()

    artifact = copied_release / "candidate" / "generation_config.json"
    external_alias_source = tmp_path / "external-generation-config.json"
    shutil.copy2(artifact, external_alias_source)
    artifact.unlink()
    try:
        os.link(external_alias_source, artifact)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    with pytest.raises(ValueError, match="single-link"):
        authenticate_core_release(copied_release)


def test_coverage_first_selection_has_literal_required_dev_coverage(canary_set):
    for bundle in canary_set.canaries:
        operations = {
            action.operation.value
            for task in bundle.tasks
            for action in task.actions
        }
        assert {"ADD", "UPDATE", "NOOP"} <= operations

        family_e_scopes = {
            action.scope.value
            for task in bundle.tasks
            if task.task_family == "deletion_forgetting"
            for action in task.actions
            if action.operation.value == "DELETE" and action.scope is not None
        }
        assert {"object", "attribute", "entity", "namespace", "ttl"} <= (
            family_e_scopes
        )

        family_f_selectors = {
            query.selector.kind
            for task in bundle.tasks
            if task.task_family == "current_historical_query"
            for query in task.queries
        }
        assert "current" in family_f_selectors
        assert any(selector != "current" for selector in family_f_selectors)

        assert any(
            len(query.target_object_keys) > 1
            for task in bundle.tasks
            for query in task.queries
        )
        family_g_queries = [
            query
            for task in bundle.tasks
            if task.task_family == "long_horizon_memory_synthesis"
            for query in task.queries
        ]
        assert family_g_queries
        assert any(query.synthesis is not None for query in family_g_queries)


def test_rebuilds_byte_identically_and_publishes_atomically(
    tmp_path,
    monkeypatch,
    authenticated_release,
    canary_set,
):
    import mub.vnext.external.canaries_v3 as canary_module

    rebuilt = canary_module._build_canary_set_authenticated(
        authenticated_release
    )
    for first, second in zip(canary_set.canaries, rebuilt.canaries):
        assert first.records == second.records
        assert first.manifest_bytes == second.manifest_bytes

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    canary_module.publish_canary_set(
        canary_set,
        first_root,
        release=authenticated_release,
    )
    monkeypatch.setattr(
        canary_module,
        "_IMMUTABLE_CORE_ROOT",
        tmp_path / "checkout-without-core-data",
    )
    canary_module.publish_canary_set(
        rebuilt,
        second_root,
        release=authenticated_release,
    )
    first_artifacts = {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file()
    }
    second_artifacts = {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file()
    }
    assert set(first_artifacts) == {
        "canary_a/tasks.jsonl",
        "canary_a/canary_manifest.json",
        "canary_b/tasks.jsonl",
        "canary_b/canary_manifest.json",
        "canary_set_manifest.json",
    }
    assert first_artifacts == second_artifacts
    for bundle in canary_set.canaries:
        directory = first_root / bundle.manifest.canary_id
        assert (directory / "tasks.jsonl").read_bytes() == b"".join(
            bundle.records
        )
        assert (directory / "canary_manifest.json").read_bytes() == (
            bundle.manifest_bytes
        )
    assert (first_root / "canary_set_manifest.json").read_bytes() == (
        canary_set.set_manifest_bytes
    )
    with pytest.raises(FileExistsError):
        canary_module.publish_canary_set(
            canary_set,
            first_root,
            release=authenticated_release,
        )


def test_atomic_install_never_replaces_existing_destination(tmp_path):
    from mub.vnext.external.canaries_v3 import _rename_no_replace

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    (destination / "destination.txt").write_text(
        "destination",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        _rename_no_replace(source, destination)
    assert (source / "source.txt").read_text(encoding="utf-8") == "source"
    assert (destination / "destination.txt").read_text(
        encoding="utf-8"
    ) == "destination"

    shutil.rmtree(destination)
    _rename_no_replace(source, destination)
    assert not source.exists()
    assert (destination / "source.txt").read_text(encoding="utf-8") == "source"


def test_publication_cleans_staging_after_write_failure(
    tmp_path,
    monkeypatch,
    authenticated_release,
    canary_set,
):
    import mub.vnext.external.canaries_v3 as canary_module

    original_write = canary_module._write_fsynced
    calls = 0

    def fail_second_write(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        original_write(path, content)

    monkeypatch.setattr(canary_module, "_write_fsynced", fail_second_write)
    output = tmp_path / "failed-publication"
    with pytest.raises(OSError, match="injected publication failure"):
        canary_module.publish_canary_set(
            canary_set,
            output,
            release=authenticated_release,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed-publication.staging-*"))


def test_tampering_and_public_trust_boundary_attacks_are_rejected(
    tmp_path,
    authenticated_release,
    canary_set,
):
    from mub.vnext.external.canaries_v3 import (
        CanaryBundleV1,
        CanaryManifestV1,
        _validate_canary_bundle_authenticated,
        _validate_canary_manifest_shape,
        authenticate_core_release,
    )
    from mub.vnext.io import canonical_json_bytes

    corrupted = tmp_path / "corrupt-release"
    corrupted.mkdir()
    payload = json.loads(
        (RELEASE_ROOT / "task_release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    payload["release_status"] = "NOT_APPROVED"
    (corrupted / "task_release_manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        authenticate_core_release(corrupted)

    manifest = canary_set.canaries[0].manifest
    stored_manifest = {
        field_name: manifest.__dict__[field_name]
        for field_name in CanaryManifestV1.model_fields
    }
    forged_hash = CanaryManifestV1.model_construct(**{
        **stored_manifest,
        "tasks_ref": ArtifactRef(
            path="tasks.jsonl", sha256="0" * 64,
            media_type="application/x-ndjson", record_count=64,
        ),
    })
    with pytest.raises(ValueError):
        _validate_canary_bundle_authenticated(
            CanaryBundleV1(
                manifest=forged_hash,
                tasks=canary_set.canaries[0].tasks,
                records=canary_set.canaries[0].records,
                manifest_bytes=canonical_json_bytes(forged_hash),
            ),
            authenticated_release,
        )

    class ArtifactRefSubclass(ArtifactRef):
        pass

    forged_subclass = CanaryManifestV1.model_construct(**{
        **stored_manifest,
        "tasks_ref": ArtifactRefSubclass(
            path="tasks.jsonl", sha256=manifest.tasks_ref.sha256,
            media_type="application/x-ndjson", record_count=64,
        ),
    })
    with pytest.raises(ValueError):
        _validate_canary_manifest_shape(forged_subclass)


def test_output_safety_rejects_dangling_and_parent_reparse_points(tmp_path):
    import mub.vnext.external.canaries_v3 as canary_module

    target = tmp_path / "target"
    target.mkdir()
    parent_link = tmp_path / "parent-link"
    dangling_output = tmp_path / "dangling-output"
    try:
        parent_link.symlink_to(target, target_is_directory=True)
        dangling_output.symlink_to(
            tmp_path / "missing-target",
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(ValueError, match="reparse"):
        canary_module._ensure_output_root_is_safe(
            parent_link / "canaries",
            RELEASE_ROOT,
        )
    with pytest.raises(ValueError, match="reparse"):
        canary_module._ensure_output_root_is_safe(
            dangling_output,
            RELEASE_ROOT,
        )


def test_cli_refuses_core_destination_and_preserves_immutable_release(
    tmp_path,
    immutable_release_snapshot,
):
    output = tmp_path / "cli-金丝雀"
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "ascii:strict"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "vnext_prepare_external_canaries.py"),
        "--release-root", str(RELEASE_ROOT),
        "--output-root", str(output),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "canary_a" / "tasks.jsonl").is_file()
    unsafe = subprocess.run(
        command[:-1] + [str(RELEASE_ROOT / "forbidden-canaries")],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert unsafe.returncode != 0
    assert _tree_snapshot(RELEASE_ROOT) == immutable_release_snapshot
