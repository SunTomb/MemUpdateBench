from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from mub.vnext.io import sha256_model

from scripts import vnext_plan_main_track_factorial as planner
from scripts import vnext_run_main_track_factorial_manager as runner


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "vnext" / "main_track_v1_factorial_plan_v1" / "factorial_manifest.json"
CANDIDATE = ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1"
ATTESTATION = ROOT / "results" / "vnext" / "main_track_v1_audit_completion_attestation_v1" / "review_attestation.json"


def _args(
    output_root: Path,
    *,
    manager_kind: str = "letta",
    cell_id: str | None = None,
    candidate_root: Path = CANDIDATE,
    audit_attestation: Path = ATTESTATION,
    execution_mode: str = "injected_test_only",
):
    return runner.RunnerArgs(
        manifest=MANIFEST,
        candidate_root=candidate_root,
        audit_attestation=audit_attestation,
        manager_kind=manager_kind,
        cell_id=cell_id or f"{manager_kind}_profile__qwen35_answer",
        output_root=output_root,
        execution_mode=execution_mode,
    )


def _value_from_event(event) -> object | None:
    if event.metadata.get("semantic_effect") == "noop":
        return None
    match = re.search(r"=\s*(.+?)(?:\.|$)", event.normalized_text)
    assert match
    return json.loads(match.group(1))


class FakeExtractor:
    identity = {"extractor_id": "fake-visible-event", "extractor_version": "test-v1"}

    def __init__(self):
        self.calls = []

    def extract(self, event, object_key):
        self.calls.append(event.event_id)
        operation = event.metadata.get("semantic_effect", "noop")
        raw = runner.canonical_json_bytes({"operation": operation, "value": _value_from_event(event)})
        return {
            "operation": operation,
            "value": _value_from_event(event),
            "output_sha256": hashlib.sha256(raw).hexdigest(),
            "generated_tokens": 3,
            "latency_ms": 0.5,
        }


class FakeManager:
    identity = {"manager_id": "fake-manager", "manager_version": "test-v1"}

    def __init__(self):
        self.calls = []
        self.entries = {}
        self.closed = False

    def reset(self, task):
        self.entries = {}

    def ingest(self, event, *, operation, value, object_key):
        self.calls.append((event.event_id, operation))
        key = object_key.canonical_id
        if operation in {"add", "update"}:
            self.entries[key] = {
                "entry_id": f"entry:{key}",
                "object_key": object_key.model_dump(mode="json"),
                "value": value,
                "content": f"{key} = {value}",
                "source_event_ids": [event.event_id],
                "score": 1.0,
                "rank": 1,
                "version_metadata": {"version_index": 0},
            }
        elif operation == "delete":
            self.entries.pop(key, None)
        return {"effective_operation": operation, "affected_entry_ids": [f"entry:{key}"]}

    def export_entries(self):
        return list(self.entries.values())

    def retrieve(self, query):
        entries = list(self.entries.values())
        return {
            "entries": entries,
            "context_order": "fake_insertion_order",
            "version_metadata": {"source": "fake-manager", "version": 0},
        }

    def close(self):
        self.closed = True


def _run(tmp_path: Path, *, extractor: FakeExtractor | None = None, managers=None, manager_kind="letta"):
    extractor = extractor or FakeExtractor()
    managers = managers if managers is not None else []

    def manager_factory(task, cell):
        manager = FakeManager()
        managers.append(manager)
        return manager

    summary = runner.run(
        _args(tmp_path / "out", manager_kind=manager_kind),
        extractor_factory=lambda: extractor,
        manager_factory=manager_factory,
    )
    return summary, extractor, managers


def test_manifest_binding_tamper_is_rejected_before_fake_calls(tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_bytes())
    manifest["candidate_artifact_hashes"]["tasks.jsonl"] = "0" * 64
    manifest["payload_sha256"] = hashlib.sha256(
        runner.canonical_json_bytes({key: value for key, value in manifest.items() if key != "payload_sha256"})
    ).hexdigest()
    tampered = tmp_path / "factorial_manifest.json"
    tampered.write_bytes(runner.canonical_json_bytes(manifest))
    extractor = FakeExtractor()
    with pytest.raises(ValueError, match="candidate hashes"):
        runner.run(
            _args(tmp_path / "out"),
            extractor_factory=lambda: extractor,
            manager_factory=lambda task, cell: FakeManager(),
            manifest=tampered,
        )
    assert extractor.calls == []


def test_fake_adapter_runs_once_per_supported_task_and_emits_all_terminals(tmp_path):
    summary, extractor, managers = _run(tmp_path)
    assert summary["requested"] == 720
    assert summary["terminal_rows"] == 720
    assert summary["supported"] == 240
    assert summary["unsupported"] == 480
    assert len(managers) == 240
    assert sum(len(manager.calls) for manager in managers) == len(extractor.calls)
    rows = [json.loads(line) for line in (tmp_path / "out" / "manager_rows.jsonl").read_bytes().splitlines()]
    assert [row["task_id"] for row in rows] == [task.task_id for task in planner.select_test_tasks(CANDIDATE)]
    assert all(row["status"] in {"SUPPORTED", "UNSUPPORTED"} for row in rows)


def test_unsupported_rows_have_typed_reason_and_null_policy(tmp_path):
    _run(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "out" / "manager_rows.jsonl").read_bytes().splitlines()]
    unsupported = [row for row in rows if row["status"] == "UNSUPPORTED"]
    assert len(unsupported) == 480
    assert all(row["reason_code"] and row["reason_kind"] for row in unsupported)
    for row in unsupported:
        assert row["event_records"] is None
        assert row["reconciliation_count"] is None
        assert row["state"] is None
        assert row["retrieval"] is None


def test_retrieval_trace_hash_binds_actual_ordered_entries(tmp_path):
    _run(tmp_path)
    row = next(
        json.loads(line)
        for line in (tmp_path / "out" / "manager_rows.jsonl").read_bytes().splitlines()
        if json.loads(line)["status"] == "SUPPORTED"
    )
    trace = row["retrieval"]["trace"]
    assert row["retrieval"]["trace_sha256"] == hashlib.sha256(runner.canonical_json_bytes(trace)).hexdigest()
    assert [entry["rank"] for entry in trace["entries"]] == [1]
    assert trace["entries"][0]["source_event_ids"]
    assert trace["context_order"] == "fake_insertion_order"


def test_output_is_no_replace_and_public_payload_rejects_raw_fields(tmp_path):
    _run(tmp_path)
    with pytest.raises(FileExistsError):
        _run(tmp_path)
    with pytest.raises(ValueError, match="raw prompt/output/reasoning"):
        runner.validate_public_payload({"raw_output": "must not escape"})


def test_production_adapter_absence_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="production adapters"):
        runner.run(_args(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_candidate_task_hashes_are_bound_in_supported_rows(tmp_path):
    _run(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "out" / "manager_rows.jsonl").read_bytes().splitlines()]
    selected = {task.task_id: task for task in planner.select_test_tasks(CANDIDATE)}
    for row in rows:
        if row["status"] == "SUPPORTED":
            assert row["task_sha256"] == sha256_model(selected[row["task_id"]])


def _write_manifest_copy(tmp_path: Path, mutate):
    manifest = json.loads(MANIFEST.read_bytes())
    mutate(manifest)
    payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    manifest["payload_sha256"] = hashlib.sha256(runner.canonical_json_bytes(payload)).hexdigest()
    path = tmp_path / "factorial_manifest.json"
    path.write_bytes(runner.canonical_json_bytes(manifest))
    return path


def test_manifest_binds_candidate_root_and_audit_path_to_supplied_paths(tmp_path):
    wrong_root = _write_manifest_copy(tmp_path, lambda manifest: manifest["candidate"].update({"root": str(tmp_path)}))
    with pytest.raises(ValueError, match="candidate root binding"):
        runner.run(
            _args(tmp_path / "out"),
            extractor_factory=lambda: FakeExtractor(),
            manager_factory=lambda task, cell: FakeManager(),
            manifest=wrong_root,
        )
    wrong_audit = _write_manifest_copy(tmp_path, lambda manifest: manifest["audit_attestation"].update({"path": str(tmp_path / "wrong-audit.json")}))
    with pytest.raises(ValueError, match="audit attestation path binding"):
        runner.run(
            _args(tmp_path / "out2"),
            extractor_factory=lambda: FakeExtractor(),
            manager_factory=lambda task, cell: FakeManager(),
            manifest=wrong_audit,
        )


def test_manifest_binds_exact_manager_and_extractor_specs(tmp_path):
    tampered = _write_manifest_copy(
        tmp_path,
        lambda manifest: manifest["cells"][2]["manager"].update({"system_version": "tampered"}),
    )
    with pytest.raises(ValueError, match="manager specification"):
        runner.run(
            _args(tmp_path / "out"),
            extractor_factory=lambda: FakeExtractor(),
            manager_factory=lambda task, cell: FakeManager(),
            manifest=tampered,
        )


def test_manifest_support_reasons_and_order_are_recomputed(tmp_path):
    def mutate(manifest):
        manifest["cells"][2]["unsupported_tasks"][0]["reason_code"] = "forged_reason"

    tampered = _write_manifest_copy(tmp_path, mutate)
    with pytest.raises(ValueError, match="support reasons"):
        runner.run(
            _args(tmp_path / "out"),
            extractor_factory=lambda: FakeExtractor(),
            manager_factory=lambda task, cell: FakeManager(),
            manifest=tampered,
        )


def test_candidate_hashes_are_revalidated_after_task_parse_before_calls(tmp_path, monkeypatch):
    import shutil

    candidate = tmp_path / "candidate"
    shutil.copytree(CANDIDATE, candidate)
    manifest = json.loads(MANIFEST.read_bytes())
    manifest["candidate"]["root"] = str(candidate.resolve())
    manifest["audit_attestation"]["path"] = str(ATTESTATION.resolve())
    manifest["payload_sha256"] = hashlib.sha256(
        runner.canonical_json_bytes({key: value for key, value in manifest.items() if key != "payload_sha256"})
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(runner.canonical_json_bytes(manifest))
    original = planner.select_test_tasks
    mutated = False

    def select_then_mutate(root):
        nonlocal mutated
        tasks = original(root)
        if not mutated:
            mutated = True
            task_file = Path(root) / "tasks.jsonl"
            task_file.write_bytes(task_file.read_bytes() + b" ")
        return tasks

    monkeypatch.setattr(runner.planner, "select_test_tasks", select_then_mutate)
    extractor = FakeExtractor()
    with pytest.raises(ValueError, match="changed after parsing"):
        runner.run(
            _args(tmp_path / "out", candidate_root=candidate),
            extractor_factory=lambda: extractor,
            manager_factory=lambda task, cell: FakeManager(),
            manifest=manifest_path,
        )
    assert extractor.calls == []


def test_manager_result_preserves_affected_entry_ids_and_requires_field():
    assert runner._manager_result({"effective_operation": "add", "affected_entry_ids": ["entry-1"]}) == ("add", ["entry-1"])
    with pytest.raises(ValueError, match="affected_entry_ids"):
        runner._manager_result({"effective_operation": "add"})


def test_incomplete_arbitrary_entry_object_key_is_rejected():
    with pytest.raises(ValueError, match="object key"):
        runner._entry({
            "entry_id": "entry-1",
            "object_key": {"namespace": "post_core", "entity": "e", "attribute": "a"},
            "value": "v",
            "content": "e.a = v",
            "source_event_ids": ["event-1"],
            "score": 1.0,
            "rank": 1,
            "version_metadata": {},
        })


def test_publication_failure_cleans_staged_output_root(tmp_path, monkeypatch):
    output = tmp_path / "output"

    def fail_publish(*args, **kwargs):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(runner, "publish_files_atomically", fail_publish)
    with pytest.raises(OSError, match="synthetic publication failure"):
        runner._publish_output(output, {"manager_summary.json": b"{}"}, ())
    assert not output.exists()


def test_summary_and_index_bind_numeric_zero_execution_boundary(tmp_path):
    summary, _, _ = _run(tmp_path)
    assert summary["execution_boundary"] == {
        "provider_calls": 0,
        "model_loads": 0,
        "database_accesses": 0,
        "network_calls": 0,
        "gpu_calls": 0,
        "executable_calls": 0,
        "remote_operations": 0,
    }
    assert summary["status"] == "PASS"
    assert summary["scientific_evidence"] is False
    assert summary["execution_boundary_observed"] is False
    assert summary["evidence_class"] == "manager_state_retrieval_fixture_test_only"
    index = json.loads((tmp_path / "out" / "artifact_index.json").read_bytes())
    assert index["execution_boundary"] == summary["execution_boundary"]
    assert index["status"] == "PASS"
    assert index["scientific_evidence"] is False
    assert index["execution_boundary_observed"] is False
    assert index["evidence_class"] == "manager_state_retrieval_fixture_test_only"


class RaisingManager(FakeManager):
    def ingest(self, event, *, operation, value, object_key):
        raise RuntimeError("synthetic ingest failure")


class DuplicateTargetManager(FakeManager):
    def export_entries(self):
        entries = super().export_entries()
        if not entries:
            return entries
        duplicate = dict(entries[0])
        duplicate["entry_id"] = f"{entries[0]['entry_id']}:duplicate"
        return entries + [duplicate]


class DistractorSameValueManager(FakeManager):
    def retrieve(self, query):
        result = super().retrieve(query)
        if result["entries"]:
            distractor = dict(result["entries"][0])
            distractor["entry_id"] = "entry:distractor"
            distractor_key = dict(distractor["object_key"])
            distractor_key["entity"] = "different-entity"
            distractor["object_key"] = distractor_key
            result["entries"] = [distractor]
        return result


class CloseFailManager(FakeManager):
    def close(self):
        self.closed = True
        raise RuntimeError("synthetic manager close failure")


class CloseFailExtractor(FakeExtractor):
    def close(self):
        raise RuntimeError("synthetic extractor close failure")


class ProductionBoundFactory:
    production_bound = True

    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        return self.value


def test_failed_supported_rows_make_summary_and_cli_status_fail(tmp_path):
    def manager_factory(task, cell):
        return RaisingManager()

    summary = runner.run(
        _args(tmp_path / "out"),
        extractor_factory=ProductionBoundFactory(FakeExtractor()),
        manager_factory=manager_factory,
    )
    assert summary["status"] == "FAIL"
    assert summary["failed"] == 240
    index = json.loads((tmp_path / "out" / "artifact_index.json").read_bytes())
    assert index["status"] == "FAIL"


def test_duplicate_target_entries_are_ambiguous_not_last_wins(tmp_path):
    def manager_factory(task, cell):
        return DuplicateTargetManager()

    runner.run(
        _args(tmp_path / "duplicate-out"),
        extractor_factory=lambda: FakeExtractor(),
        manager_factory=manager_factory,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "duplicate-out" / "manager_rows.jsonl").read_bytes().splitlines()
    ]
    row = next(row for row in rows if row["status"] == "SUPPORTED")
    assert row["state"]["final_value"] is None
    assert row["state"]["state_accuracy"] is False
    assert row["state"]["stable_entry_id"] is False
    assert row["parsed_final_value"] is None
    assert row["stable_entry_id"] is False


def test_same_value_distractor_is_not_gold_retrieval(tmp_path):
    def manager_factory(task, cell):
        return DistractorSameValueManager()

    runner.run(
        _args(tmp_path / "distractor-out"),
        extractor_factory=lambda: FakeExtractor(),
        manager_factory=manager_factory,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "distractor-out" / "manager_rows.jsonl").read_bytes().splitlines()
    ]
    row = next(row for row in rows if row["status"] == "SUPPORTED")
    assert row["retrieval"]["gold_retrieved"] is False
    assert row["gold_retrieved"] is False


def test_manager_result_validates_requested_operation_and_mutation_ids():
    assert runner._manager_result(
        {"effective_operation": "update", "affected_entry_ids": ["entry-1"]},
        requested_operation="update",
    ) == ("update", ["entry-1"])
    assert runner._manager_result(
        {"effective_operation": "noop", "affected_entry_ids": []},
        requested_operation="noop",
    ) == ("noop", [])
    with pytest.raises(ValueError, match="requested operation"):
        runner._manager_result(
            {"effective_operation": "add", "affected_entry_ids": ["entry-1"]},
            requested_operation="update",
        )
    with pytest.raises(ValueError, match="affected_entry_ids"):
        runner._manager_result(
            {"effective_operation": "update", "affected_entry_ids": []},
            requested_operation="update",
        )


def test_manager_result_contract_binds_requested_action():
    task = planner.select_test_tasks(CANDIDATE)[0]
    key = task.target_objects[0]
    requested = {
        "operation": "ADD",
        "scope": "object",
        "target_object_keys": [key],
        "value": "value",
    }
    effective = {
        "operation": "UPDATE",
        "scope": "object",
        "target_object_keys": [key],
        "value": "value",
    }
    result = runner.AdapterActionResultV3(
        event_id="event-contract",
        requested_action=requested,
        effective_action=effective,
        execution_status="executed",
        affected_entry_ids=("entry-contract",),
    )
    with pytest.raises(ValueError, match="requested operation"):
        runner._manager_result(result, requested_operation="update")


def test_row_consistency_rejects_drift_between_event_aliases():
    row = {
        "state": None,
        "state_accuracy": None,
        "parsed_final_value": None,
        "final_memory_size": None,
        "stable_entry_id": None,
        "retrieval": None,
        "retrieval_trace": None,
        "retrieval_trace_sha256": None,
        "gold_retrieved": None,
        "event_records": [{"event_id": "e1"}],
        "extractions": [{"event_id": "e2"}],
    }
    with pytest.raises(ValueError, match="event records"):
        runner._validate_row_consistency(row)


@pytest.mark.parametrize(
    [
        (CloseFailManager, FakeExtractor),
        (FakeManager, CloseFailExtractor),
    ],
)
def test_resource_cleanup_failures_fail_run_and_block_publication(
    tmp_path, manager_type, extractor_type
):
    with pytest.raises(RuntimeError, match="cleanup"):
        runner.run(
            _args(tmp_path / "out"),
            extractor_factory=ProductionBoundFactory(extractor_type()),
            manager_factory=lambda task, cell: manager_type(),
        )
    assert not (tmp_path / "out").exists()


def test_production_requires_bound_factories_and_exact_runtime_identities(tmp_path):
    args = _args(tmp_path / "production-out", execution_mode="production")
    with pytest.raises(RuntimeError, match="production-bound"):
        runner.run(
            args,
            extractor_factory=lambda: FakeExtractor(),
            manager_factory=lambda task, cell: FakeManager(),
        )

    cell = next(item for item in json.loads(MANIFEST.read_bytes())["cells"] if item["cell_id"] == args.cell_id)
    extractor = FakeExtractor()
    extractor.identity = dict(cell["extractor"]["identity"])
    manager = FakeManager()
    manager.identity = dict(cell["manager"])
    summary = runner.run(
        args,
        extractor_factory=ProductionBoundFactory(extractor),
        manager_factory=ProductionBoundFactory(manager),
    )
    assert summary["status"] == "PASS"
    assert summary["scientific_evidence"] is True
    assert summary["execution_boundary_observed"] is True
