from pathlib import Path
import pytest
from scripts.vnext_run_letta_qwen_extraction_canary import (
    canonical_json_bytes,
    validate_extraction,
    validate_output_root,
    validate_qualification_artifacts,
    validate_worker_runtime_binding,
)

def test_missing_qualification_validator_is_callable(tmp_path):
    with pytest.raises(ValueError):
        validate_qualification_artifacts(tmp_path)



def test_qualification_requires_schema_and_affirmative_gates(tmp_path):
    for name in ('letta_runtime_qualification.json','letta_runtime_preflight.json','letta_runtime_admission.json'):
        (tmp_path/name).write_text('{}')
    with pytest.raises(ValueError):
        validate_qualification_artifacts(tmp_path)


    import hashlib
    import subprocess
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    project = tmp_path / "letta-project"
    project.mkdir()
    runner = project / "mub" / "vnext" / "external" / "workers"
    runner.mkdir(parents=True)
    runner_file = runner / "letta_worker.py"
    runner_file.write_text("print('worker')\n")
    subprocess.run(("git", "-C", str(project), "init"), check=True, capture_output=True)
    subprocess.run(("git", "-C", str(project), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(project), "config", "user.name", "Test"), check=True)
    subprocess.run(("git", "-C", str(project), "add", "."), check=True)
    subprocess.run(("git", "-C", str(project), "commit", "-m", "fixture"), check=True, capture_output=True)
    revision = subprocess.check_output(("git", "-C", str(project), "rev-parse", "HEAD"), text=True).strip()
    tree_hash, file_count = module._tracked_tree_identity(project)
    closure = {"project_source": {"commit": revision, "tree_sha256": tree_hash, "file_count": file_count}, "runner_source_sha256": "a" * 64, "worker_source_sha256": hashlib.sha256(runner_file.read_bytes()).hexdigest()}
    executable = tmp_path / "python"
    executable.write_bytes(b"python")

    binding = module.validate_worker_runtime_binding(executable, project, closure)
    assert binding["project_revision"] == revision
    assert binding["worker_source_sha256"] == closure["worker_source_sha256"]
    assert binding["qualification_runner_source_sha256"] == closure["runner_source_sha256"]

    with pytest.raises(ValueError, match="absolute"):
        module.validate_worker_runtime_binding(Path("relative-python"), project, closure)


def test_worker_command_uses_explicit_runtime_and_project_paths(tmp_path):
    import scripts.vnext_run_letta_qwen_extraction_canary as module

    executable = tmp_path / "python"
    executable.write_bytes(b"python")
    project = tmp_path / "project"
    worker = project / "mub" / "vnext" / "external" / "workers" / "letta_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("worker\n")
    command = module.build_worker_command(executable, project, "{}")
    assert command[:2] == (str(executable), str(worker))
    assert "--configuration-json" in command

def test_output_root_rejects_frozen_overlap(tmp_path):
    frozen=tmp_path/'frozen'; frozen.mkdir()
    with pytest.raises(ValueError): validate_output_root(frozen/'out', frozen_roots=(frozen,))

def test_qualification_hashes_are_mandatory(tmp_path):
    closure = {"schema_version":"memupdatebench.external.letta.runtime_qualification.v1","candidate_id":"letta_0_16_8_song1_local_linux","outcome":"PASS","identity":{"package_name":"letta","package_version":"0.16.8","source_commit":"1131535716e8a31c9a437f8695e25ac98f203a24"},"source":{},"project_source":{},"runner_source_sha256":"a"*64,"worker_source_sha256":"b"*64,"runtime":{"loopback_only":True},"boundary":{"llm_used":False,"api_used":False,"gpu_used":False},"cleanup":{"status":"PASS"},"preflight":{},"admission":{}}
    preflight = {"schema_version":"memupdatebench.external.letta.preflight.v2","candidate_id":"letta_0_16_8_profile","mode":"profile_single_record_runtime","outcome":"pass","passed":True,"identity":{},"official_health":{},"runtime":{},"namespace_reset_probe":{},"lifecycle":{},"clean_close":{},"security":{},"boundary":{},"unsupported":{}}
    admission = {"schema_version":"memupdatebench.external.letta.admission.v2","candidate_id":"letta_0_16_8_profile","admission_scope":"profile_single_record_runtime","outcome":"pass","admitted":True,"gates":{},"reasons":[]}
    for name, value in (("letta_runtime_qualification.json",closure),("letta_runtime_preflight.json",preflight),("letta_runtime_admission.json",admission)):
        (tmp_path/name).write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="hash"):
        validate_qualification_artifacts(tmp_path)


def test_scope_selection_preserves_canary_and_selects_exact_full_family_a(monkeypatch):
    import scripts.vnext_run_letta_qwen_extraction_canary as module
    from types import SimpleNamespace

    class JsonTask:
        def __init__(self, core_id, task_id):
            self.task_id = task_id
            self.metadata = SimpleNamespace(extra={"semantic_core_id": core_id})

    raw = b"authenticated"
    monkeypatch.setattr(module, "TASK_SHA256", module.sha256_bytes(raw))
    tasks = [JsonTask(f"core-{core}", f"task-{core}-{variant}") for core in range(20) for variant in range(4)]
    monkeypatch.setattr(module, "_parse_authenticated_tasks", lambda _: tasks)

    assert len(module.select_tasks(raw)) == 32
    full = module.select_tasks(raw, scope="full-family-a80")
    assert len(full) == 80
    assert [task.task_id for task in full] == sorted((task.task_id for task in tasks), key=lambda x: x.encode())


def test_scope_rejects_wrong_authenticated_cardinality(monkeypatch):
    import scripts.vnext_run_letta_qwen_extraction_canary as module
    raw = b"authenticated"
    monkeypatch.setattr(module, "TASK_SHA256", module.sha256_bytes(raw))
    monkeypatch.setattr(module, "_parse_authenticated_tasks", lambda _: [])
    with pytest.raises(ValueError, match="80"):
        module.select_tasks(raw, scope="full-family-a80")


def test_full_scope_summary_has_null_prompted_metrics_and_operation_counts():
    import scripts.vnext_run_letta_qwen_extraction_canary as module
    rows = [
        {"status": "PASS", "state_accuracy": True, "gold_retrieved_k16": True, "final_memory_size": 1,
         "reconciliation_count": 1, "extractions": [{"effective_operation": "add", "operation": "update"}]},
        {"status": "NOT_SUPPORTED", "state_accuracy": None, "gold_retrieved_k16": None, "final_memory_size": None,
         "reconciliation_count": 0, "extractions": []},
        {"status": "FAIL", "state_accuracy": None, "gold_retrieved_k16": None, "final_memory_size": None,
         "reconciliation_count": 0, "extractions": [{"effective_operation": "noop", "operation": "noop"}]},
    ]
    summary = module.build_summary(rows, scope="full-family-a80", requested=3, rows_sha256="x" * 64,
                                   qualification_hashes={}, qualification_identity={}, letta_binding={},
                                   endpoint="http://127.0.0.1:8000", model_provenance={})
    assert summary["scope"] == "full-family-a80"
    assert summary["prompted_exact_match"] is None
    assert summary["prompted_metrics_denominator"] == 0
    assert summary["operation_counts"]["requested"]["update"] == 1
    assert summary["operation_counts"]["requested"]["noop"] == 1
    assert summary["operation_counts"]["effective"]["add"] == 1
    assert summary["operation_counts"]["effective"]["noop"] == 1
    assert summary["total_reconciliation_count"] == 1
