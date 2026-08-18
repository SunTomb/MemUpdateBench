from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from mub.vnext.io import canonical_json_bytes
from mub.vnext.runtime.task12_execution_v3 import (
    Task12RuntimeCodeBindingV1,
    load_task12_control_json_v3,
)
from mub.vnext.runtime.task12_matrix_v3 import (
    Task12MatrixBundleManifestV1,
    Task12MatrixRunSummaryV1,
)
from tests.vnext.task13_input_fixtures import (
    build_compact_authenticated_task13_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Task12RuntimeCodeBindingV1(
    code_revision="8" * 40,
    code_tree_sha256="9" * 64,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


@pytest.fixture(scope="module")
def authenticated_fixture(tmp_path_factory):
    return build_compact_authenticated_task13_fixture(
        tmp_path_factory.mktemp("task13-input"),
        ROOT,
        RUNTIME,
    )


def _load_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    expected = fixture["expected_hashes"]
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
        "expected_preparation_manifest_sha256": expected["preparation_manifest"],
        "expected_plan_sha256": expected["plan"],
        "expected_matrix_manifest_sha256": expected["matrix_manifest"],
        "expected_matrix_summary_sha256": expected["matrix_summary"],
        "expected_integrity_audit_sha256": expected["integrity_audit"],
    }


def _ordered_pairs(records):
    return tuple(
        (
            record.cell_id if hasattr(record, "cell_id") else record.cell.cell_id,
            (
                record.answer_model_slot
                if hasattr(record, "answer_model_slot")
                else record.run_configuration.answer_model_slot
            ),
        )
        for record in records
    )


def test_task13_loader_returns_exact_ordered_18_runs_and_shared_20x4_core_tasks(
    authenticated_fixture,
):
    from collections import Counter

    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    loaded = load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    expected_pairs = _ordered_pairs(loaded.plan.admitted_answer_runs)

    assert len(loaded.runs) == 18
    assert _ordered_pairs(loaded.runs) == expected_pairs
    assert _ordered_pairs(loaded.matrix_manifest.run_bundles) == expected_pairs
    assert _ordered_pairs(loaded.summary.completed_runs) == expected_pairs
    assert all(len(run.observations) == 80 for run in loaded.runs)
    assert len(loaded.canonical_core_ids) == 20
    assert loaded.canonical_core_ids == tuple(
        sorted(loaded.canonical_core_ids, key=lambda value: value.encode("utf-8"))
    )
    expected_identity = tuple(
        (obs.semantic_core_id, obs.task.task_id)
        for obs in loaded.runs[0].observations
    )
    assert expected_identity == tuple(
        sorted(
            expected_identity,
            key=lambda pair: (pair[0].encode("utf-8"), pair[1].encode("utf-8")),
        )
    )
    for core_id in loaded.canonical_core_ids:
        task_ids = tuple(
            task_id for observed_core, task_id in expected_identity if observed_core == core_id
        )
        assert task_ids == tuple(sorted(task_ids, key=lambda value: value.encode("utf-8")))
        assert len(task_ids) == 4
    assert all(
        Counter(obs.semantic_core_id for obs in run.observations)
        == Counter({core_id: 4 for core_id in loaded.canonical_core_ids})
        for run in loaded.runs
    )
    assert {
        tuple((obs.task.task_id, obs.semantic_core_id) for obs in run.observations)
        for run in loaded.runs
    } == {
        tuple((task_id, core_id) for core_id, task_id in expected_identity)
    }
    assert all(
        obs.source.score_artifact_sha256
        for run in loaded.runs
        for obs in run.observations
    )


def _load_with_tampered_authenticated_run(
    authenticated_fixture,
    monkeypatch,
    transform,
    *,
    target_run_index: int,
):
    import mub.vnext.statistics.input_v3 as input_v3

    original = input_v3._validate_task12_run_bundle_v3
    call_index = -1

    def wrapped(*args, **kwargs):
        nonlocal call_index
        call_index += 1
        validated = original(*args, **kwargs)
        if call_index != target_run_index:
            return validated
        return replace(validated, tasks=tuple(transform(validated.tasks)))

    monkeypatch.setattr(input_v3, "_validate_task12_run_bundle_v3", wrapped)
    return input_v3.load_task13_authenticated_matrix_v1(
        **_load_kwargs(authenticated_fixture)
    )


def _canonical_tasks(tasks):
    return sorted(
        tasks,
        key=lambda task: (
            task.metadata.split_key.semantic_core_id.encode("utf-8"),
            task.task_id.encode("utf-8"),
        ),
    )


@pytest.mark.parametrize("target_run_index", [0, 17])
def test_task13_loader_rejects_task_substitution_in_first_or_last_run(
    authenticated_fixture,
    monkeypatch,
    target_run_index,
):
    def substitute(tasks):
        forged = list(tasks)
        forged[0] = forged[0].model_copy(update={"task_id": "task-substitute"})
        return _canonical_tasks(forged)

    with pytest.raises(ValueError, match="task IDs or semantic-core assignments differ"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            substitute,
            target_run_index=target_run_index,
        )


def test_task13_loader_rejects_task_to_core_reassignment(
    authenticated_fixture,
    monkeypatch,
):
    def reassign_task_ids(tasks):
        forged = _canonical_tasks(tasks)
        first_id = forged[0].task_id
        second_id = forged[4].task_id
        forged[0] = forged[0].model_copy(update={"task_id": second_id})
        forged[4] = forged[4].model_copy(update={"task_id": first_id})
        return _canonical_tasks(forged)

    with pytest.raises(ValueError, match="task IDs or semantic-core assignments differ"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            reassign_task_ids,
            target_run_index=17,
        )


def test_task13_loader_rejects_duplicate_task_ids(
    authenticated_fixture,
    monkeypatch,
):
    def duplicate_task_id(tasks):
        forged = _canonical_tasks(tasks)
        forged[1] = forged[1].model_copy(update={"task_id": forged[0].task_id})
        return _canonical_tasks(forged)

    with pytest.raises(ValueError, match="unique task IDs"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            duplicate_task_id,
            target_run_index=17,
        )


def test_task13_loader_rejects_wrong_core_multiplicity(
    authenticated_fixture,
    monkeypatch,
):
    def change_one_task_core(tasks):
        forged = _canonical_tasks(tasks)
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
        return _canonical_tasks(forged)

    with pytest.raises(ValueError, match="exactly four task IDs"):
        _load_with_tampered_authenticated_run(
            authenticated_fixture,
            monkeypatch,
            change_one_task_core,
            target_run_index=17,
        )


def test_task13_loader_parses_authenticated_bytes_with_shared_task12_parser(
    authenticated_fixture,
    monkeypatch,
):
    import mub.vnext.statistics.input_v3 as input_v3

    original = input_v3.parse_task12_control_json_bytes_v3
    calls = []

    def wrapped(raw, model_type, *, source, allow_trailing_lf=False):
        calls.append((Path(source).name, allow_trailing_lf, hashlib.sha256(raw).hexdigest()))
        return original(
            raw,
            model_type,
            source=source,
            allow_trailing_lf=allow_trailing_lf,
        )

    monkeypatch.setattr(input_v3, "parse_task12_control_json_bytes_v3", wrapped)
    input_v3.load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    expected = authenticated_fixture["expected_hashes"]
    assert calls == [
        ("task12_preparation_manifest.json", False, expected["preparation_manifest"]),
        ("dry_run_plan.json", True, expected["plan"]),
        ("matrix_bundle_manifest.json", False, expected["matrix_manifest"]),
        ("matrix_run_summary.json", False, expected["matrix_summary"]),
    ]


def test_task13_loader_does_not_reread_hashed_controls(
    authenticated_fixture,
    monkeypatch,
):
    import mub.vnext.statistics.input_v3 as input_v3

    path = authenticated_fixture["preparation_manifest_path"]
    original_raw = path.read_bytes()
    original_read = input_v3._read_expected
    swapped = False

    def read_then_replace(selected, expected_sha256, *, label):
        nonlocal swapped
        raw = original_read(selected, expected_sha256, label=label)
        if Path(selected) == path and not swapped:
            path.write_bytes(raw + b" ")
            swapped = True
        return raw

    monkeypatch.setattr(input_v3, "_read_expected", read_then_replace)
    try:
        loaded = input_v3.load_task13_authenticated_matrix_v1(
            **_load_kwargs(authenticated_fixture)
        )
        with pytest.raises(ValueError, match="noncanonical artifact"):
            load_task12_control_json_v3(path, type(loaded.manifest))
    finally:
        path.write_bytes(original_raw)
    assert swapped
    assert len(loaded.runs) == 18


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


def _first_output_file(authenticated_fixture, relative: str, *, run_index: int = 0):
    ref = authenticated_fixture["matrix"].manifest.run_bundles[run_index]
    return (
        authenticated_fixture["matrix"].matrix_root
        / ref.bundle_leaf
        / ref.output_leaf
        / relative
    )


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("scores/scores.jsonl", "score"),
        ("task_runs.jsonl", "complete|expected|row"),
    ],
)
def test_task13_loader_rejects_truncated_run_or_score_rows(
    authenticated_fixture,
    relative,
    message,
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    path = _first_output_file(authenticated_fixture, relative, run_index=17)
    original = path.read_bytes()
    path.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
    try:
        with pytest.raises(ValueError, match=message):
            load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    finally:
        path.write_bytes(original)


def _rewrite_audit_bindings(fixture, *, manifest_hash=None, summary_hash=None):
    path = fixture["audit_path"]
    audit = json.loads(path.read_bytes())
    if manifest_hash is not None:
        audit["matrix_bundle_manifest_sha256"] = manifest_hash
    if summary_hash is not None:
        audit["matrix_summary_sha256"] = summary_hash
    path.write_bytes(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


@pytest.mark.parametrize("mode", ["duplicate", "missing"])
def test_task13_loader_rejects_duplicate_or_missing_matrix_pair(
    authenticated_fixture,
    mode,
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    manifest_path = authenticated_fixture["matrix_manifest_path"]
    original = manifest_path.read_bytes()
    payload = json.loads(original)
    if mode == "duplicate":
        payload["run_bundles"][-1] = payload["run_bundles"][0]
    else:
        payload["run_bundles"].pop()
    manifest_path.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    try:
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["expected_matrix_manifest_sha256"] = _hash(manifest_path)
        with pytest.raises(ValueError, match="18 unique runs|18 ordered answer runs"):
            load_task13_authenticated_matrix_v1(**kwargs)
    finally:
        manifest_path.write_bytes(original)


def test_task13_loader_rejects_reordered_matrix_manifest(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    manifest_path = authenticated_fixture["matrix_manifest_path"]
    summary_path = authenticated_fixture["summary_path"]
    audit_path = authenticated_fixture["audit_path"]
    original_manifest = manifest_path.read_bytes()
    original_summary = summary_path.read_bytes()
    original_audit = audit_path.read_bytes()
    model = Task12MatrixBundleManifestV1.model_validate_json(original_manifest)
    manifest_path.write_bytes(
        canonical_json_bytes(
            model.model_copy(update={"run_bundles": tuple(reversed(model.run_bundles))})
        )
    )
    summary = Task12MatrixRunSummaryV1.model_validate_json(original_summary)
    summary_path.write_bytes(
        canonical_json_bytes(
            summary.model_copy(
                update={"matrix_bundle_manifest_sha256": _hash(manifest_path)}
            )
        )
    )
    _rewrite_audit_bindings(
        authenticated_fixture,
        manifest_hash=_hash(manifest_path),
        summary_hash=_hash(summary_path),
    )
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


@pytest.mark.parametrize("digest_field", ["run_manifest_sha256", "score_artifact_sha256"])
def test_task13_loader_rejects_per_run_summary_digest_tampering(
    authenticated_fixture,
    digest_field,
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    summary_path = authenticated_fixture["summary_path"]
    audit_path = authenticated_fixture["audit_path"]
    original_summary = summary_path.read_bytes()
    original_audit = audit_path.read_bytes()
    summary = Task12MatrixRunSummaryV1.model_validate_json(original_summary)
    records = list(summary.completed_runs)
    records[-1] = records[-1].model_copy(update={digest_field: "f" * 64})
    summary_path.write_bytes(
        canonical_json_bytes(summary.model_copy(update={"completed_runs": tuple(records)}))
    )
    _rewrite_audit_bindings(authenticated_fixture, summary_hash=_hash(summary_path))
    try:
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["expected_matrix_summary_sha256"] = _hash(summary_path)
        kwargs["expected_integrity_audit_sha256"] = _hash(audit_path)
        with pytest.raises(ValueError, match="run or score hash differs"):
            load_task13_authenticated_matrix_v1(**kwargs)
    finally:
        summary_path.write_bytes(original_summary)
        audit_path.write_bytes(original_audit)


def test_task13_loader_rejects_same_id_core_task_content_tampering(
    authenticated_fixture,
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    path = authenticated_fixture["inputs"]["core_root"] / "candidate" / "tasks.jsonl"
    original = path.read_bytes()
    rows = original.splitlines()
    payload = json.loads(rows[0])
    original_task_id = payload["task_id"]
    payload["events"][0]["raw_text"] += " tampered"
    assert payload["task_id"] == original_task_id
    rows[0] = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(b"\n".join(rows) + b"\n")
    try:
        with pytest.raises(ValueError, match="Core tasks differ|artifact digest mismatch"):
            load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    finally:
        path.write_bytes(original)


def test_task13_loader_rejects_foreign_runtime_in_audit(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    path = authenticated_fixture["audit_path"]
    original = path.read_bytes()
    audit = json.loads(original)
    audit["runtime_code_binding"] = {
        "code_revision": "7" * 40,
        "code_tree_sha256": "6" * 64,
    }
    path.write_bytes(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
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
    path.write_bytes(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    try:
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["expected_integrity_audit_sha256"] = _hash(path)
        with pytest.raises(ValueError, match="integrity audit is invalid"):
            load_task13_authenticated_matrix_v1(**kwargs)
    finally:
        path.write_bytes(original)


def test_task13_loader_rejects_symlink_control_file(
    authenticated_fixture,
    tmp_path,
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    source = authenticated_fixture["preparation_manifest_path"]
    symlink = tmp_path / "manifest-symlink.json"
    try:
        symlink.symlink_to(source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    kwargs = _load_kwargs(authenticated_fixture)
    kwargs["preparation_manifest_path"] = symlink
    with pytest.raises(ValueError, match="regular|reparse"):
        load_task13_authenticated_matrix_v1(**kwargs)


def test_task13_loader_rejects_hardlink_control_file(
    authenticated_fixture,
    tmp_path,
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    source = authenticated_fixture["preparation_manifest_path"]
    hardlink = tmp_path / "manifest-hardlink.json"
    os.link(source, hardlink)
    try:
        kwargs = _load_kwargs(authenticated_fixture)
        kwargs["preparation_manifest_path"] = hardlink
        with pytest.raises(ValueError, match="regular|hardlink"):
            load_task13_authenticated_matrix_v1(**kwargs)
    finally:
        hardlink.unlink()


def test_task13_current_repository_revision_need_not_match_task12_runtime(
    authenticated_fixture,
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    loaded = load_task13_authenticated_matrix_v1(**_load_kwargs(authenticated_fixture))
    assert loaded.runtime == RUNTIME
    assert loaded.runtime.code_revision != authenticated_fixture["plan"].code_revision
