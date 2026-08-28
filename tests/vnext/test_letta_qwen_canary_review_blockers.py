from pathlib import Path
import pytest
from scripts.vnext_run_letta_qwen_extraction_canary import validate_loopback_binding

def test_loopback_endpoint_must_match_qualification_server_port():
    closure={"runtime":{"loopback_only":True,"measured":{"server_port":"8123"}}}
    with pytest.raises(ValueError): validate_loopback_binding("http://127.0.0.1:8124", closure)

def test_qualification_requires_exact_v2_schema():
    from scripts.vnext_run_letta_qwen_extraction_canary import validate_qualification_artifacts
    assert callable(validate_qualification_artifacts)

def test_run_assigns_task_path_before_selection(tmp_path, monkeypatch):
    import types
    import scripts.vnext_run_letta_qwen_extraction_canary as module
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_bytes(b"{}")
    (tmp_path / "python").write_bytes(b"python")
    monkeypatch.setattr(module, "validate_output_root", lambda path, frozen_roots=(): path)
    monkeypatch.setattr(module, "verify_model_provenance", lambda *args: {})
    monkeypatch.setattr(module, "validate_qualification_artifacts", lambda root: {"closure":{"runtime":{"measured":{"server_port":8000}}},"hashes":{}})
    monkeypatch.setattr(module, "validate_loopback_binding", lambda url, closure: url)
    monkeypatch.setattr(module, "validate_worker_runtime_binding", lambda *args, **kwargs: {"project_root": str(tmp_path), "python_executable": str(tmp_path / "python"), "runner_source_sha256": "a" * 64})
    monkeypatch.setattr(module, "select_tasks", lambda raw: [])
    monkeypatch.setattr(module, "QwenExtractor", lambda snapshot: types.SimpleNamespace(load=lambda: None, close=lambda: None))
    args = types.SimpleNamespace(tasks=str(tasks), output_root=str(tmp_path / "out"), model_snapshot=str(tmp_path), model_runtime_receipt=str(tasks), model_snapshot_binding=str(tasks), qualification_root=str(tmp_path), letta_base_url="http://127.0.0.1:8000", letta_python_executable=str(tmp_path / "python"), letta_project_root=str(tmp_path), expected_letta_project_revision=None)
    with pytest.raises(RuntimeError, match="terminal row count"):
        module.run(args)
