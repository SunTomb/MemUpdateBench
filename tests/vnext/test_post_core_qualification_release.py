from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import shutil

import pytest

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.identity_v1 import IdentityEvidenceBundleV1
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilityAttemptPlanV1,
    CapabilityBudgetV1,
    CapabilityFixtureV1,
    CapabilitySmokePlanV1,
    DecisionScope,
    QualificationDecisionBundleV1,
    QualificationDecisionV1,
    QualificationStatus,
    QUALIFICATION_ARTIFACT_ORDER,
    QUALIFICATION_INDEX_PATH,
)
from mub.vnext.post_core.qualification_planning_v1 import (
    CapabilitySmokePlanConfigV1,
    build_capability_smoke_plan_v1,
)
from tests.vnext.qualification_fixtures import open_runtime_receipts, provider_attestations
from mub.vnext.post_core import qualification_release_v1
from mub.vnext.post_core.qualification_release_v1 import (
    BASE_COMMIT,
    QUALIFICATION_ARTIFACTS,
    CommittedQualificationReleaseError,
    QualificationReleaseError,
    build_qualification_release_v1,
    load_qualification_release_config_v1,
    publish_qualification_release_v1,
    verify_qualification_release_v1,
    _verify_qualification_artifact_bytes_v1,
)


def _source_inputs(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    source_ids = (
        "core_manifest", "handoff_source", "identity_evidence", "open_snapshot_audit_receipt",
        "open_snapshot_closure_receipt", "phase0_index", "qwen_load_receipt", "task14_index", "workflow_source",
    )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for index, source_id in enumerate(source_ids):
        path = tmp_path / f"{source_id}.blob"
        raw = (
            (Path(__file__).resolve().parents[2] / "configs" / "vnext" / "post_core" / "official_identity_evidence_v1.json").read_bytes()
            if source_id == "identity_evidence"
            else f"source-{index}".encode()
        )
        path.write_bytes(raw)
        paths[source_id] = path
        hashes[source_id] = hashlib.sha256(raw).hexdigest()
    payload = {
        "base_attempts_per_role": 8,
        "base_commit": BASE_COMMIT,
        "escalation_attempts_per_role": 8,
        "max_retries": 0,
        "publisher_network_allowed": False,
        "registry_keys": [
            "qwen35_9b_bf16", "meta_muse_glimmer_30b_int4", "meta_muse_glimmer_30b_bf16",
            "claude_sonnet_4_6", "claude_opus_4_8", "gemini_3_6_flash", "grok_4_5", "gpt_5_5",
        ],
        "release_id": "memupdatebench.post-core.qualification.v1",
        "required_source_sha256": hashes,
        "schema_version": "memupdatebench.post-core.qualification-config.v1",
        "scientific_execution_allowed": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_bytes(payload))
    return paths, config_path


def _smoke_budget() -> CapabilityBudgetV1:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_budget_v1

    return build_capability_budget_v1()


def _fixtures() -> tuple[CapabilityFixtureV1, ...]:
    from mub.vnext.post_core.qualification_planning_v1 import build_capability_fixtures_v1

    return build_capability_fixtures_v1()


def _identity_bundle() -> IdentityEvidenceBundleV1:
    path = Path(__file__).resolve().parents[2] / "configs" / "vnext" / "post_core" / "official_identity_evidence_v1.json"
    return IdentityEvidenceBundleV1.model_validate_json(path.read_bytes())


def _smoke_plan(release_id: str, keys: tuple[str, ...]) -> CapabilitySmokePlanV1:
    return build_capability_smoke_plan_v1(
        CapabilitySmokePlanConfigV1(release_id=release_id, registry_keys=keys, budget=_smoke_budget()),
        _fixtures(),
    )


def _decisions(release_id: str, keys: tuple[str, ...]) -> QualificationDecisionBundleV1:
    return QualificationDecisionBundleV1(
        release_id=release_id,
        decisions=tuple(
            QualificationDecisionV1(
                registry_key=key, scope=scope, status=QualificationStatus.BLOCKED,
                reasons=("not run",), evidence_binding_ids=("core_manifest",),
            )
            for key in keys for scope in DecisionScope
        ),
    )


def _inputs(tmp_path: Path) -> dict[str, object]:
    paths, config_path = _source_inputs(tmp_path)
    config = load_qualification_release_config_v1(config_path)
    provider_path = tmp_path / "provider.jsonl"
    runtime_path = tmp_path / "runtime.jsonl"
    provider_path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in provider_attestations()))
    runtime_path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in open_runtime_receipts()))
    return {
        "config": config,
        "source_paths": paths,
        "provider_attestations_path": provider_path,
        "runtime_receipts_path": runtime_path,
        "capability_fixtures": _fixtures(),
        "capability_budget": _smoke_budget(),
        "smoke_plan": _smoke_plan(config.release_id, config.registry_keys),
        "identity_bundle": _identity_bundle(),
    }


def test_builder_emits_exact_deterministic_artifacts_and_zero_counters(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    left = build_qualification_release_v1(**inputs)
    right = build_qualification_release_v1(**inputs)
    assert tuple(left.artifact_bytes) == QUALIFICATION_ARTIFACTS
    assert tuple(left.artifact_bytes) == (*QUALIFICATION_ARTIFACT_ORDER, QUALIFICATION_INDEX_PATH)
    assert dict(left.artifact_bytes) == dict(right.artifact_bytes)
    assert left.index_sha256 == right.index_sha256
    assert (left.provider_calls, left.model_loads, left.network_calls, left.credential_reads, left.benchmark_generations) == (0, 0, 0, 0, 0)
    _verify_qualification_artifact_bytes_v1(left)
    assert "stale_copied" not in left.artifact_bytes["qualification_validation_receipt.json"].decode()


def test_release_requires_source_backed_provider_and_runtime_evidence(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    assert build_qualification_release_v1(
        **{**inputs, "provider_attestations": provider_attestations(), "runtime_receipts": open_runtime_receipts()}
    )
    typed_only = dict(inputs)
    typed_only.pop("provider_attestations_path")
    typed_only.pop("runtime_receipts_path")
    typed_only["provider_attestations"] = provider_attestations()
    typed_only["runtime_receipts"] = open_runtime_receipts()
    with pytest.raises(ValueError, match="provider attestations path"):
        build_qualification_release_v1(**typed_only)


def test_release_binds_dynamic_evidence_hashes_and_rejects_forged_identity(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    publication = build_qualification_release_v1(**inputs)
    source_bundle = qualification_release_v1.SourceBindingBundleV1.model_validate_json(
        publication.artifact_bytes["source_bindings.json"]
    )
    bindings = {item.source_id: item for item in source_bundle.sources}
    assert bindings["provider_attestations"].sha256 == hashlib.sha256(
        inputs["provider_attestations_path"].read_bytes()
    ).hexdigest()
    assert bindings["runtime_receipts"].sha256 == hashlib.sha256(
        inputs["runtime_receipts_path"].read_bytes()
    ).hexdigest()
    assert {"qualification_config", "capability_fixtures", "capability_parser_contract", "qualification_planner"} <= set(bindings)
    manifest = qualification_release_v1.QualificationReleaseManifestV1.model_validate_json(
        publication.artifact_bytes["qualification_release_manifest.json"]
    )
    assert dict(manifest.source_hashes) == {source_id: binding.sha256 for source_id, binding in bindings.items()}

    bundle = _identity_bundle()
    forged_record = type(bundle.records[-1]).model_construct(**{
        **bundle.records[-1].model_dump(mode="python"), "official_model_id": "gpt-5.5",
    })
    forged = IdentityEvidenceBundleV1.model_construct(**{
        **bundle.model_dump(mode="python"), "records": (*bundle.records[:-1], forged_record),
    })
    with pytest.raises(ValueError, match="identity bundle"):
        build_qualification_release_v1(**{**inputs, "identity_bundle": forged})


def test_release_rejects_dangling_decision_evidence_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    original = qualification_release_v1.derive_qualification_decisions_v1

    def forged(*args: object):
        decisions = original(*args)
        return (decisions[0].model_copy(update={"evidence_binding_ids": ("missing_binding",)}), *decisions[1:])

    monkeypatch.setattr(qualification_release_v1, "derive_qualification_decisions_v1", forged)
    with pytest.raises(ValueError, match="decision evidence binding"):
        build_qualification_release_v1(**inputs)


def test_builder_rejects_source_substitution_and_missing_source(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"different")
    paths = dict(inputs["source_paths"])
    paths["workflow_source"] = replacement
    with pytest.raises(ValueError, match="workflow_source"):
        build_qualification_release_v1(**{**inputs, "source_paths": paths})
    paths.pop("workflow_source")
    with pytest.raises(ValueError, match="exact nine"):
        build_qualification_release_v1(**{**inputs, "source_paths": paths})


def test_builder_rejects_secret_like_and_metric_fields_before_exposing_bytes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="decision"):
        build_qualification_release_v1(
            **inputs,
            decision_bundle=_decisions(inputs["config"].release_id, inputs["config"].registry_keys),
        )


def test_builder_uses_source_identity_without_compatibility_input(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    without_compatibility_input = {key: value for key, value in inputs.items() if key != "identity_bundle"}
    publication = build_qualification_release_v1(**without_compatibility_input)
    assert publication.release_id == inputs["config"].release_id


def test_builder_rejects_file_backed_config_without_its_snapshot(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="source snapshot"):
        build_qualification_release_v1(**{**inputs, "config": replace(inputs["config"], source_snapshot=None)})


def test_builder_revalidates_source_after_artifact_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    source = inputs["source_paths"]["workflow_source"]
    original = qualification_release_v1._reject_metrics
    def mutate_once(value: object) -> None:
        source.write_bytes(b"changed")
        original(value)
    monkeypatch.setattr(qualification_release_v1, "_reject_metrics", mutate_once)
    with pytest.raises(ValueError, match="workflow_source"):
        build_qualification_release_v1(**inputs)


def test_publish_reopens_exact_artifacts_and_refuses_clobber(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "published"
    published = publish_qualification_release_v1(output, **inputs)
    reopened = verify_qualification_release_v1(output, **inputs)
    assert published.output_root == output.resolve()
    assert dict(reopened.artifact_bytes) == dict(published.artifact_bytes)
    with pytest.raises(FileExistsError):
        publish_qualification_release_v1(output, **inputs)


def test_publish_rejects_output_overlap_and_source_mutation(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    source = inputs["source_paths"]["workflow_source"]
    with pytest.raises(ValueError, match="overlaps"):
        publish_qualification_release_v1(source, **inputs)
    output = tmp_path / "mutated"
    def mutate_source() -> None:
        source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="workflow_source"):
        publish_qualification_release_v1(output, before_commit=mutate_source, **inputs)
    assert not output.exists()


def test_publish_rejects_file_backed_config_mutation_before_commit(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config_path = inputs["config"].config_path
    assert config_path is not None
    def mutate_config() -> None:
        config_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="qualification config"):
        publish_qualification_release_v1(tmp_path / "mutated-config", before_commit=mutate_config, **inputs)


def test_publish_rejects_provider_receipt_mutation_before_commit(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    provider_path = inputs["provider_attestations_path"]
    assert isinstance(provider_path, Path)
    def mutate_provider() -> None:
        provider_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provider attestations"):
        publish_qualification_release_v1(tmp_path / "mutated-provider", before_commit=mutate_provider, **inputs)


def test_publish_preserves_tampered_stage_for_quarantine(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "tampered"
    seen: list[Path] = []
    def mutate_stage() -> None:
        stage = next(tmp_path.glob(".mub-post-core-qualification-stage-*"))
        seen.append(stage)
        (stage / "source_bindings.json").write_bytes(b"tampered")
    with pytest.raises(QualificationReleaseError, match="staging"):
        publish_qualification_release_v1(output, before_commit=mutate_stage, **inputs)
    assert seen and seen[0].exists()


def test_publish_rejects_extra_stage_member_before_commit(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    seen: list[Path] = []
    def add_member() -> None:
        stage = next(tmp_path.glob(".mub-post-core-qualification-stage-*"))
        seen.append(stage)
        (stage / "unexpected").write_bytes(b"extra")
    with pytest.raises(QualificationReleaseError, match="staging"):
        publish_qualification_release_v1(tmp_path / "extra-member", before_commit=add_member, **inputs)
    assert seen and seen[0].exists()


def test_verify_preserves_committed_root_when_artifact_is_tampered(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "post-commit"
    publish_qualification_release_v1(output, **inputs)
    (output / "source_bindings.json").write_bytes(b"tampered")
    with pytest.raises(CommittedQualificationReleaseError):
        verify_qualification_release_v1(output, **inputs)
    assert output.exists()


def test_collective_capture_rejects_early_artifact_mutation_during_later_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == "nt":
        pytest.skip("Windows descriptor sharing prevents the concurrent replacement probe")
    inputs = _inputs(tmp_path)
    output = tmp_path / "collective-mutation"
    publication = publish_qualification_release_v1(output, **inputs)
    original_read = qualification_release_v1.os.read
    calls = 0
    def mutate_during_later_read(descriptor: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 3:
            (output / "qualification_release_manifest.json").write_bytes(b"tampered")
        return original_read(descriptor, size)
    monkeypatch.setattr(qualification_release_v1.os, "read", mutate_during_later_read)
    with pytest.raises(QualificationReleaseError):
        qualification_release_v1._capture_all_artifacts(output, publication.artifact_bytes)


def test_publish_revalidates_sources_after_final_collective_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "final-source-order"
    source = inputs["source_paths"]["workflow_source"]
    original = qualification_release_v1._read_published_root
    calls = 0
    def mutate_on_second_check(root: Path, expected: object):
        nonlocal calls
        calls += 1
        result = original(root, expected)
        if calls == 2:
            source.write_bytes(b"changed")
        return result
    monkeypatch.setattr(qualification_release_v1, "_read_published_root", mutate_on_second_check)
    with pytest.raises(CommittedQualificationReleaseError):
        publish_qualification_release_v1(output, **inputs)
    assert output.exists()


def test_publish_resolves_provider_jsonl_once_before_final_revalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    provider_path = inputs["provider_attestations_path"]
    assert isinstance(provider_path, Path)
    original = qualification_release_v1._load_provider_rows
    calls = 0
    def count_provider(path: Path):
        nonlocal calls
        calls += 1
        return original(path)
    monkeypatch.setattr(qualification_release_v1, "_load_provider_rows", count_provider)
    publish_qualification_release_v1(tmp_path / "resolve-once", **inputs)
    assert calls == 1


def test_build_and_verify_resolve_provider_jsonl_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    provider_path = inputs["provider_attestations_path"]
    assert isinstance(provider_path, Path)
    original = qualification_release_v1._load_provider_rows
    calls = 0
    def count_provider(path: Path):
        nonlocal calls
        calls += 1
        return original(path)
    monkeypatch.setattr(qualification_release_v1, "_load_provider_rows", count_provider)
    built = build_qualification_release_v1(**inputs)
    assert calls == 1
    monkeypatch.setattr(qualification_release_v1, "_load_provider_rows", original)
    output = tmp_path / "verify-resolve-once"
    publish_qualification_release_v1(output, **inputs)
    calls = 0
    monkeypatch.setattr(qualification_release_v1, "_load_provider_rows", count_provider)
    verify_qualification_release_v1(output, **inputs)
    assert calls == 1
    assert built.artifact_bytes["provider_capability_attestations.jsonl"] == provider_path.read_bytes()


def test_public_apis_reject_unknown_input_keys(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(TypeError, match="smoke_plna"):
        build_qualification_release_v1(**inputs, smoke_plna=None)
    with pytest.raises(TypeError, match="decison_bundle"):
        publish_qualification_release_v1(tmp_path / "unknown-publish", **inputs, decison_bundle=None)
    output = tmp_path / "unknown-verify"
    publish_qualification_release_v1(output, **inputs)
    with pytest.raises(TypeError, match="validation_reciept"):
        verify_qualification_release_v1(output, **inputs, validation_reciept=None)


def test_resolver_rejects_conflicting_input_forms(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    replacement = tmp_path / "different-workflow"
    replacement.write_bytes(b"different")
    with pytest.raises(ValueError, match="conflicting source path"):
        build_qualification_release_v1(**inputs, workflow_source_path=replacement)
    with pytest.raises(ValueError, match="provider attestations"):
        build_qualification_release_v1(
            **inputs,
            provider_attestations=(provider_attestations()[0],),
        )
    with pytest.raises(ValueError, match="runtime receipts"):
        build_qualification_release_v1(
            **inputs,
            runtime_receipts=(open_runtime_receipts()[0],),
        )


def test_source_hardlink_is_rejected_when_host_supports_it(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    source = inputs["source_paths"]["workflow_source"]
    alias = tmp_path / "workflow-hardlink"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="single-link"):
        build_qualification_release_v1(**inputs)


def test_output_symlink_is_rejected_when_host_supports_it(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output-link"
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="unsafe"):
        publish_qualification_release_v1(output, **inputs)


def test_missing_no_replace_primitive_raises_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.delattr(qualification_release_v1.ctypes, "windll", raising=False)
    with pytest.raises(qualification_release_v1.NoReplacePrimitiveUnavailableError):
        publish_qualification_release_v1(tmp_path / "no-primitive", **inputs)


def test_fsync_directory_does_not_swallow_open_eio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == "nt":
        pytest.skip("Windows directory fsync is intentionally unavailable")
    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise OSError(5, "EIO")
    monkeypatch.setattr(qualification_release_v1.os, "open", fail_open)
    with pytest.raises(OSError):
        qualification_release_v1._fsync_directory(tmp_path)


def test_post_rename_fsync_failure_preserves_committed_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "fsync-committed"
    original = qualification_release_v1._fsync_directory
    calls = 0
    def fail_final(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(5, "EIO")
        original(path)
    monkeypatch.setattr(qualification_release_v1, "_fsync_directory", fail_final)
    with pytest.raises(CommittedQualificationReleaseError):
        publish_qualification_release_v1(output, **inputs)
    assert output.exists()


def test_write_failure_after_first_member_cleans_verified_partial_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    original_open = Path.open
    calls = 0
    def fail_second_stage_open(path: Path, *args: object, **kwargs: object):
        nonlocal calls
        if path.parent.name.startswith(".mub-post-core-qualification-stage-"):
            calls += 1
            if calls == 2:
                raise OSError(5, "EIO")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_second_stage_open)
    with pytest.raises(OSError):
        publish_qualification_release_v1(tmp_path / "write-failure", **inputs)
    assert not tuple(tmp_path.glob(".mub-post-core-qualification-stage-*"))


def test_tampered_partial_stage_is_preserved_for_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    original_open = Path.open
    calls = 0
    seen: list[Path] = []
    def tamper_then_fail(path: Path, *args: object, **kwargs: object):
        nonlocal calls
        if path.parent.name.startswith(".mub-post-core-qualification-stage-"):
            calls += 1
            if calls == 2:
                seen.append(path.parent)
                (path.parent / "unexpected").write_bytes(b"foreign")
                raise OSError(5, "EIO")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tamper_then_fail)
    with pytest.raises(OSError):
        publish_qualification_release_v1(tmp_path / "write-tampered", **inputs)
    assert seen and seen[0].exists()


def test_intact_complete_stage_is_cleaned_after_precommit_failure(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    def fail_precommit() -> None:
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError, match="stop"):
        publish_qualification_release_v1(tmp_path / "complete-clean", before_commit=fail_precommit, **inputs)
    assert not tuple(tmp_path.glob(".mub-post-core-qualification-stage-*"))


def test_substituted_complete_stage_is_preserved_after_precommit_failure(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    replacement: list[Path] = []
    def substitute_stage() -> None:
        stage = next(tmp_path.glob(".mub-post-core-qualification-stage-*"))
        parked = tmp_path / "parked-stage"
        stage.rename(parked)
        stage.mkdir()
        for member in parked.iterdir():
            shutil.copy2(member, stage / member.name)
        replacement.append(stage)
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError, match="stop"):
        publish_qualification_release_v1(tmp_path / "complete-substitution", before_commit=substitute_stage, **inputs)
    assert replacement and replacement[0].exists()
