from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.preparation.task12 import admit_task12_dry_run
from mub.vnext.runtime.task12_execution_v3 import (
    Task12RuntimeCodeBindingV1,
    load_task12_control_json_v3,
)
from mub.vnext.runtime.task12_matrix_v3 import (
    _execute_task12_matrix_bundles_for_test_v3,
    Task12MatrixBundleManifestV1,
    Task12MatrixRunSummaryV1,
)
from tests.vnext.task12_fixtures import (
    build_task12_inputs,
    build_task12_manifest,
)
from tests.vnext.test_core_task12_preparation import _authorize_fixture_release


def test_task13_authenticated_input_contracts_are_frozen_dataclasses():
    from dataclasses import fields, is_dataclass

    from mub.vnext.statistics.input_v3 import (
        Task13AuthenticatedMatrixV1,
        Task13AuthenticatedObservationV1,
        Task13AuthenticatedRunV1,
        Task13IntegrityAuditV1,
        load_task13_authenticated_matrix_v1,
    )

    assert callable(load_task13_authenticated_matrix_v1)
    for contract in (
        Task13AuthenticatedObservationV1,
        Task13AuthenticatedRunV1,
        Task13AuthenticatedMatrixV1,
    ):
        assert is_dataclass(contract)
        assert contract.__dataclass_params__.frozen
        assert fields(contract)
    assert Task13IntegrityAuditV1.model_config["frozen"] is True


def test_task13_loader_has_one_unambiguous_control_path_signature():
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    assert set(inspect.signature(load_task13_authenticated_matrix_v1).parameters) == {
        "preparation_manifest_path",
        "plan_path",
        "core_root",
        "evidence_root",
        "matrix_root",
        "matrix_manifest_path",
        "matrix_summary_path",
        "integrity_audit_path",
        "repository_root",
        "expected_preparation_manifest_sha256",
        "expected_plan_sha256",
        "expected_matrix_manifest_sha256",
        "expected_matrix_summary_sha256",
        "expected_integrity_audit_sha256",
    }


def test_task13_uses_shared_task12_control_loader_boundary(tmp_path):
    path = tmp_path / "control.json"
    model = Task12RuntimeCodeBindingV1(
        code_revision="a" * 40,
        code_tree_sha256="b" * 64,
    )
    path.write_bytes(canonical_json_bytes(model) + b"\n")

    assert load_task12_control_json_v3(
        path,
        Task12RuntimeCodeBindingV1,
        allow_trailing_lf=True,
    ) == model
    with pytest.raises(ValueError, match="noncanonical artifact"):
        load_task12_control_json_v3(path, Task12RuntimeCodeBindingV1)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Task12RuntimeCodeBindingV1(
    code_revision="8" * 40,
    code_tree_sha256="9" * 64,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _approved_core_source() -> Path | None:
    candidates = (
        ROOT / "data" / "vnext" / "core" / "v3",
        ROOT.parent / "vnext-phase0" / "data" / "vnext" / "core" / "v3",
    )
    return next(
        (candidate for candidate in candidates if (candidate / "task_release_manifest.json").is_file()),
        None,
    )


def _build_task12_inputs_without_regeneration(root: Path) -> dict[str, object]:
    """Use the approved full Core release while retaining the existing fixture setup."""
    source_root = _approved_core_source()
    if source_root is None:
        from tests.vnext.task12_fixtures import build_task12_inputs

        return build_task12_inputs(root)

    import tests.vnext.task12_fixtures as fixture_module
    from mub.vnext.generation.artifacts import InMemoryPilotArtifact
    from mub.vnext.generation.core_hard_suite import CoreHardSuiteManifest
    from mub.vnext.contracts.v3.task import MemUpdateTaskV3

    release = json.loads((source_root / "task_release_manifest.json").read_bytes())
    candidate_root = source_root / "candidate"
    artifact_refs = {
        str(ref["path"]): str(ref["sha256"])
        for ref in release["artifact_refs"]
        if str(ref["path"]).startswith("candidate/")
    }
    artifact_specs = {
        "tasks.jsonl": ("application/x-ndjson", 12000),
        "semantic_cores.jsonl": ("application/x-ndjson", 3000),
        "generation_config.json": ("application/json", 1),
        "split_balance.json": ("application/json", 1),
        "task_manifest.json": ("application/json", 1),
        "core-hard-v1.json": ("application/json", 1),
        "validation_report.json": ("application/json", 1),
    }
    artifacts = []
    for name, (media_type, record_count) in artifact_specs.items():
        path = candidate_root / name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact_refs[f"candidate/{name}"]
        artifacts.append(InMemoryPilotArtifact(name, raw, media_type, record_count))

    hard_suite = CoreHardSuiteManifest.model_validate_json(
        (candidate_root / "core-hard-v1.json").read_bytes()
    )
    hard_ids = set(hard_suite.task_ids)
    selected_tasks = []
    with (candidate_root / "tasks.jsonl").open("rb") as handle:
        for line in handle:
            payload = json.loads(line)
            if (
                payload["task_id"] in hard_ids
                or payload["metadata"]["split"] == "test"
            ):
                selected_tasks.append(MemUpdateTaskV3.model_validate(payload))
    bundle = SimpleNamespace(
        artifacts=tuple(artifacts),
        hard_suite=hard_suite,
        snapshot=SimpleNamespace(tasks=tuple(selected_tasks)),
    )
    original_compile = fixture_module.compile_core_snapshot
    original_bundle = fixture_module.build_core_artifact_bundle
    fixture_module.compile_core_snapshot = lambda *args, **kwargs: object()
    fixture_module.build_core_artifact_bundle = lambda *args, **kwargs: bundle
    try:
        inputs = fixture_module.build_task12_inputs(root)
    finally:
        fixture_module.compile_core_snapshot = original_compile
        fixture_module.build_core_artifact_bundle = original_bundle

    import shutil

    for relative in ("audit", "schemas"):
        shutil.copytree(source_root / relative, inputs["core_root"] / relative)
    release_path = inputs["core_root"] / "task_release_manifest.json"
    shutil.copy2(source_root / "task_release_manifest.json", release_path)
    inputs["release_ref"] = {
        "path": "task_release_manifest.json",
        "sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
        "media_type": "application/json",
        "record_count": 1,
    }
    inputs["release_manifest_hash"] = release["release_manifest_hash"]
    inputs["approved_release_root_digest"] = release["release_root_digest"]
    return inputs


@pytest.fixture(scope="module")
def authenticated_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("task13-input")
    inputs = _build_task12_inputs_without_regeneration(root)
    manifest = build_task12_manifest(inputs)
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    _authorize_fixture_release(patch, inputs, manifest)
    if "approved_release_root_digest" in inputs:
        import mub.vnext.preparation.task12 as task12

        patch.setattr(
            task12,
            "_APPROVED_CORE_RELEASE_ROOT_DIGEST",
            inputs["approved_release_root_digest"],
        )
    try:
        preparation_root = root / "preparation"
        preparation_root.mkdir()
        preparation_manifest_path = _write(
            preparation_root / "task12_preparation_manifest.json",
            canonical_json_bytes(manifest),
        )
        plan = admit_task12_dry_run(
            manifest=manifest,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            output_dir=preparation_root,
        )
        plan_path = _write(
            preparation_root / "dry_run_plan.json",
            canonical_json_bytes(plan) + b"\n",
        )
        from mub.vnext.adapters.core_v3 import RawAppendAdapterV3
        from tests.vnext.test_core_task12_matrix_bundle import _FakePromptedAnswerModel
        from mub.vnext.runtime.task12_matrix_v3 import build_task12_matrix_bundles_v3

        matrix = build_task12_matrix_bundles_v3(
            manifest=manifest,
            plan=plan,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME,
            output_root=root / "matrix",
        )
        models = {
            "answer_model_a": _FakePromptedAnswerModel("city-80", "answer_model_a"),
            "answer_model_b": _FakePromptedAnswerModel("city-80", "answer_model_b"),
        }
        result = _execute_task12_matrix_bundles_for_test_v3(
            manifest=manifest,
            plan=plan,
            matrix_bundle_manifest=matrix.manifest,
            matrix_root=matrix.matrix_root,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME,
            adapter_factory=lambda task: RawAppendAdapterV3(
                task,
                retrieval_policy="normal_topk",
            ),
            prompted_answer_models=models,
        )
        matrix_manifest_path = matrix.matrix_manifest_path
        summary_path = result.summary_path
        audit = {
            "status": "verified",
            "runtime_code_binding": RUNTIME.model_dump(mode="json"),
            "matrix_bundle_manifest_sha256": _hash(matrix_manifest_path),
            "matrix_summary_sha256": _hash(summary_path),
            "counts": {
                "run_count": 18,
                "total_task_rows": 1440,
                "total_score_rows": 1440,
                "failed": 0,
                "partial": 0,
                "semantic_multiset_mismatches": 0,
            },
        }
        audit_path = _write(
            root / "logs" / "matrix_integrity_audit.json",
            json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        return {
            "root": root,
            "inputs": inputs,
            "manifest": manifest,
            "plan": plan,
            "preparation_manifest_path": preparation_manifest_path,
            "plan_path": plan_path,
            "matrix": matrix,
            "matrix_manifest_path": matrix_manifest_path,
            "summary_path": summary_path,
            "audit_path": audit_path,
        }
    finally:
        patch.undo()


def _load_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "preparation_manifest_path": fixture["preparation_manifest_path"],
        "plan_path": fixture["plan_path"],
        "core_root": fixture["inputs"]["core_root"],
        "evidence_root": fixture["inputs"]["evidence_root"],
        "matrix_root": fixture["matrix"].matrix_root,
        "matrix_manifest_path": fixture["matrix_manifest_path"],
        "matrix_summary_path": fixture["summary_path"],
        "integrity_audit_path": fixture["audit_path"],
        "repository_root": ROOT,
        "expected_preparation_manifest_sha256": _hash(fixture["preparation_manifest_path"]),
        "expected_plan_sha256": _hash(fixture["plan_path"]),
        "expected_matrix_manifest_sha256": _hash(fixture["matrix_manifest_path"]),
        "expected_matrix_summary_sha256": _hash(fixture["summary_path"]),
        "expected_integrity_audit_sha256": _hash(fixture["audit_path"]),
    }


def test_task13_loader_returns_18_exact_runs_and_shared_20x4_core_tasks(authenticated_fixture):
    from collections import Counter
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    loaded = load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))

    assert len(loaded.runs) == 18
    assert all(len(run.observations) == 80 for run in loaded.runs)
    assert len(loaded.canonical_core_ids) == 20
    assert loaded.canonical_core_ids == tuple(
        sorted(loaded.canonical_core_ids, key=lambda value: value.encode("utf-8"))
    )
    assert all(
        tuple(obs.semantic_core_id for obs in run.observations)
        == tuple(
            sorted(
                (obs.semantic_core_id for obs in run.observations),
                key=lambda value: value.encode("utf-8"),
            )
        )
        for run in loaded.runs
    )
    assert all(
        Counter(obs.semantic_core_id for obs in run.observations)
        == Counter({core_id: 4 for core_id in loaded.canonical_core_ids})
        for run in loaded.runs
    )
    assert len({
        tuple(obs.task.task_id for obs in run.observations)
        for run in loaded.runs
    }) == 1
    assert len({
        tuple((obs.task.task_id, obs.semantic_core_id) for obs in run.observations)
        for run in loaded.runs
    }) == 1
    assert all(obs.source.score_artifact_sha256 for run in loaded.runs for obs in run.observations)
    from mub.vnext.statistics.input_v3 import Task13IntegrityAuditV1

    assert isinstance(loaded.integrity_audit, Task13IntegrityAuditV1)
    with pytest.raises((TypeError, ValueError)):
        loaded.integrity_audit.counts.failed = 1


def _load_with_tampered_authenticated_run(
    authenticated_fixture,
    monkeypatch,
    transform,
):
    import mub.vnext.statistics.input_v3 as input_v3

    original = input_v3._validate_task12_run_bundle_v3
    tampered = False

    def wrapped(*args, **kwargs):
        nonlocal tampered
        validated = original(*args, **kwargs)
        if tampered:
            return validated
        tampered = True
        forged_tasks = tuple(transform(validated.tasks))
        return replace(validated, tasks=forged_tasks)

    monkeypatch.setattr(input_v3, "_validate_task12_run_bundle_v3", wrapped)
    return input_v3.load_task13_authenticated_matrix_v1(
        **_load_kwargs(authenticated_fixture)
    )


def test_task13_loader_rejects_task_substitution_against_authenticated_scope(
    authenticated_fixture,
    monkeypatch,
):
    def substitute(tasks):
        forged = list(tasks)
        forged[0] = forged[0].model_copy(update={"task_id": "task-substitute"})
        return sorted(
            forged,
            key=lambda task: (
                task.metadata.split_key.semantic_core_id.encode("utf-8"),
                task.task_id.encode("utf-8"),
            ),
        )

    with pytest.raises(ValueError, match="task IDs or semantic-core assignments differ"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            substitute,
        )


def test_task13_loader_rejects_task_to_core_reassignment_against_authenticated_scope(
    authenticated_fixture,
    monkeypatch,
):
    def reassign_task_ids(tasks):
        forged = list(tasks)
        first_id = forged[0].task_id
        second_id = forged[4].task_id
        forged[0] = forged[0].model_copy(update={"task_id": second_id})
        forged[4] = forged[4].model_copy(update={"task_id": first_id})
        return sorted(
            forged,
            key=lambda task: (
                task.metadata.split_key.semantic_core_id.encode("utf-8"),
                task.task_id.encode("utf-8"),
            ),
        )

    with pytest.raises(ValueError, match="task IDs or semantic-core assignments differ"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            reassign_task_ids,
        )


def test_task13_loader_rejects_duplicate_task_ids_against_authenticated_scope(
    authenticated_fixture,
    monkeypatch,
):
    def duplicate_task_id(tasks):
        forged = list(tasks)
        forged[1] = forged[1].model_copy(update={"task_id": forged[0].task_id})
        return sorted(
            forged,
            key=lambda task: (
                task.metadata.split_key.semantic_core_id.encode("utf-8"),
                task.task_id.encode("utf-8"),
            ),
        )

    with pytest.raises(ValueError, match="unique task IDs"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            duplicate_task_id,
        )


def test_task13_loader_rejects_wrong_core_multiplicity(
    authenticated_fixture,
    monkeypatch,
):
    def change_one_task_core(tasks):
        forged = list(tasks)
        target_core = forged[4].metadata.split_key.semantic_core_id
        forged[0] = forged[0].model_copy(
            update={
                "metadata": forged[0].metadata.model_copy(
                    update={
                        "split_key": forged[0].metadata.split_key.model_copy(
                            update={"semantic_core_id": target_core}
                        )
                    }
                )
            }
        )
        return sorted(
            forged,
            key=lambda task: (
                task.metadata.split_key.semantic_core_id.encode("utf-8"),
                task.task_id.encode("utf-8"),
            ),
        )

    with pytest.raises(ValueError, match="exactly four task IDs"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            change_one_task_core,
        )



def test_task13_loader_delegates_controls_to_shared_task12_loader(
    authenticated_fixture,
    monkeypatch,
):
    import mub.vnext.statistics.input_v3 as input_v3

    original = input_v3.load_task12_control_json_v3
    calls = []

    def wrapped(path, model_type, *, allow_trailing_lf=False):
        calls.append((Path(path).name, allow_trailing_lf))
        return original(path, model_type, allow_trailing_lf=allow_trailing_lf)

    monkeypatch.setattr(input_v3, "load_task12_control_json_v3", wrapped)
    input_v3.load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    assert calls == [
        ("task12_preparation_manifest.json", False),
        ("dry_run_plan.json", True),
        ("matrix_bundle_manifest.json", False),
        ("matrix_run_summary.json", False),
    ]


def test_task13_loader_rejects_summary_hash_tampering(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    path = authenticated_fixture["summary_path"]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    try:
        with pytest.raises(ValueError, match="summary hash mismatch"):
            load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    finally:
        path.write_bytes(original)


def test_task13_loader_rejects_missing_score_row(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    ref = authenticated_fixture["matrix"].manifest.run_bundles[0]
    output = (
        authenticated_fixture["matrix"].matrix_root
        / ref.bundle_leaf
        / ref.output_leaf
        / "scores"
        / "scores.jsonl"
    )
    original = output.read_bytes()
    output.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
    try:
        with pytest.raises(ValueError, match="score"):
            load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    finally:
        output.write_bytes(original)


def test_task13_loader_rejects_reordered_matrix_manifest(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    manifest_path = authenticated_fixture["matrix_manifest_path"]
    summary_path = authenticated_fixture["summary_path"]
    audit_path = authenticated_fixture["audit_path"]
    original_manifest = manifest_path.read_bytes()
    original_summary = summary_path.read_bytes()
    original_audit = audit_path.read_bytes()
    model = Task12MatrixBundleManifestV1.model_validate_json(original_manifest)
    tampered = model.model_copy(update={"run_bundles": tuple(reversed(model.run_bundles))})
    tampered_manifest = canonical_json_bytes(tampered)
    manifest_path.write_bytes(tampered_manifest)
    summary = Task12MatrixRunSummaryV1.model_validate_json(original_summary)
    summary_path.write_bytes(
        canonical_json_bytes(
            summary.model_copy(
                update={
                    "matrix_bundle_manifest_sha256": _hash(manifest_path),
                }
            )
        )
    )
    audit = json.loads(original_audit)
    audit["matrix_bundle_manifest_sha256"] = _hash(manifest_path)
    audit["matrix_summary_sha256"] = _hash(summary_path)
    audit_path.write_bytes(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode())
    try:
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["expected_matrix_manifest_sha256"] = _hash(manifest_path)
        kwargs["expected_matrix_summary_sha256"] = _hash(summary_path)
        kwargs["expected_integrity_audit_sha256"] = _hash(audit_path)
        with pytest.raises(ValueError, match="matrix manifest order"):
            load_task13_authenticated_matrix_v1(**kwargs)
    finally:
        manifest_path.write_bytes(original_manifest)
        summary_path.write_bytes(original_summary)
        audit_path.write_bytes(original_audit)


def test_task13_loader_rejects_foreign_runtime_in_audit(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    path = authenticated_fixture["audit_path"]
    original = path.read_bytes()
    audit = json.loads(original)
    audit["runtime_code_binding"] = {
        "code_revision": "7" * 40,
        "code_tree_sha256": "6" * 64,
    }
    path.write_bytes(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode())
    try:
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["expected_integrity_audit_sha256"] = _hash(path)
        with pytest.raises(ValueError, match="runtime differs"):
            load_task13_authenticated_matrix_v1(**kwargs)
    finally:
        path.write_bytes(original)


def test_task13_loader_rejects_failed_audit_counts(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    path = authenticated_fixture["audit_path"]
    original = path.read_bytes()
    audit = json.loads(original)
    audit["counts"]["failed"] = 1
    path.write_bytes(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode())
    try:
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["expected_integrity_audit_sha256"] = _hash(path)
        with pytest.raises(ValueError, match="integrity audit is invalid"):
            load_task13_authenticated_matrix_v1(**kwargs)
    finally:
        path.write_bytes(original)


def test_task13_loader_rejects_symlink_and_hardlink_control_files(authenticated_fixture, tmp_path):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    source = authenticated_fixture["preparation_manifest_path"]
    symlink = tmp_path / "manifest-symlink.json"
    hardlink = tmp_path / "manifest-hardlink.json"
    try:
        symlink.symlink_to(source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    os.link(source, hardlink)
    for path in (symlink, hardlink):
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["preparation_manifest_path"] = path
        with pytest.raises(ValueError, match="(regular|reparse)"):
            load_task13_authenticated_matrix_v1(**kwargs)


def test_task13_current_repository_revision_need_not_match_task12_runtime(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    loaded = load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    assert loaded.runtime == RUNTIME
    assert loaded.runtime.code_revision != authenticated_fixture["plan"].code_revision
