from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from scripts.vnext_run_letta_runtime_qualification import (
    PACKAGE_IDENTITY,
    QualificationConfig,
    SubprocessRunner,
    build_initdb_command,
    build_launcher_source,
    find_free_loopback_port,
    make_password_file,
    run_qualification,
)


def _config(tmp_path: Path) -> QualificationConfig:
    root = tmp_path / "project"
    root.mkdir()
    (root / "scripts").mkdir()
    (tmp_path / "python").touch()
    (tmp_path / "alembic").touch()
    (tmp_path / "pgbin").mkdir()
    source = tmp_path / "letta-source"
    source.mkdir()
    nltk = tmp_path / "nltk"
    (nltk / "tokenizers" / "punkt_tab").mkdir(parents=True)
    output = tmp_path / "evidence"
    output.mkdir()
    return QualificationConfig(
        python_executable=tmp_path / "python",
        postgres_bin=tmp_path / "pgbin",
        alembic_executable=tmp_path / "alembic",
        letta_source=source,
        project_root=root,
        nltk_cache=nltk,
        output_root=output,
        project_revision="a" * 40,
    )


def test_direct_script_invocation_help_works_without_project_cwd_or_pythonpath(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "vnext_run_letta_runtime_qualification.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_subprocess_runner_forwards_caller_check_once(monkeypatch) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("scripts.vnext_run_letta_runtime_qualification.subprocess.run", fake_run)
    assert SubprocessRunner().run(("tool",), check=False, capture_output=True) == "ok"
    assert observed["kwargs"]["check"] is False


def test_config_requires_explicit_alembic_executable_and_source_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.alembic_executable.is_absolute()
    assert PACKAGE_IDENTITY["source_commit"] == "1131535716e8a31c9a437f8695e25ac98f203a24"


def test_sql_identifiers_reject_injection() -> None:
    from scripts.vnext_run_letta_runtime_qualification import validate_postgres_identifier

    assert validate_postgres_identifier("mub_letta_v1") == "mub_letta_v1"
    with pytest.raises(ValueError):
        validate_postgres_identifier("ok; DROP DATABASE other;--")


def test_main_does_not_publish_when_output_root_is_unsafe(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_runtime_qualification import main

    config = _config(tmp_path)
    frozen = config.project_root / "data" / "vnext" / "core" / "v3"
    frozen.mkdir(parents=True)
    assert main([
        "--python-executable", str(config.python_executable),
        "--postgres-bin", str(config.postgres_bin),
        "--alembic-executable", str(config.alembic_executable),
        "--letta-source", str(config.letta_source),
        "--project-root", str(config.project_root),
        "--nltk-cache", str(config.nltk_cache),
        "--output-root", str(frozen),
        "--project-revision", "a" * 40,
    ]) == 1
    assert not (frozen / "letta_runtime_qualification.json").exists()


def test_build_postgres_environment_excludes_inherited_pg_variables(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_runtime_qualification import build_postgres_environment

    env = build_postgres_environment({"PATH": "x", "PGSERVICE": "evil", "PGPASSFILE": "secret"}, password="pw")
    assert "PGSERVICE" not in env and "PGPASSFILE" not in env
    assert env["PGPASSWORD"] == "pw"


def test_readiness_requires_json_list_and_live_process() -> None:
    from scripts.vnext_run_letta_runtime_qualification import SubprocessRunner

    class Response:
        status = 200

        def read(self):
            return b'{}'

    class Process:
        def poll(self):
            return None

    runner = SubprocessRunner()
    assert runner._valid_ready_response(Response(), Process()) is False
    command = build_initdb_command(Path("/pg/initdb"), Path("/tmp/cluster"), "mub_letta", Path("/tmp/pw"))
    normalized = tuple(part.replace("\\", "/") for part in command)
    assert normalized == ("/pg/initdb", "-D", "/tmp/cluster", "-U", "mub_letta", "--pwfile=/tmp/pw", "--auth-local=md5", "--auth-host=md5")
    assert all("secret" not in part and "password" not in part for part in command)


def test_free_port_is_loopback_and_released() -> None:
    port = find_free_loopback_port()
    assert isinstance(port, int) and 1024 < port < 65536


def test_password_file_is_mode_0600_and_deleted(tmp_path: Path) -> None:
    path, password = make_password_file(tmp_path, token_factory=lambda: "random-secret")
    assert path.read_text() == "random-secret\n"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert password == "random-secret"
    path.unlink()
    assert not path.exists()


def test_launcher_requires_local_punkt_tab_cache_and_disables_download(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="punkt_tab"):
        build_launcher_source(cache, 18285)
    (cache / "tokenizers" / "punkt_tab").mkdir(parents=True)
    source = build_launcher_source(cache, 18285)
    assert "nltk.download = lambda *args, **kwargs: True" in source
    assert "start_server(port=18285, host=\"127.0.0.1\"" in source


def test_source_identity_rejects_commit_mismatch(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_runtime_qualification import _source_identity

    class MismatchRunner:
        def run(self, command, **kwargs):
            return type("Result", (), {"stdout": "b" * 40 + "\n", "returncode": 0})()

    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="commit mismatch"):
        _source_identity(source, MismatchRunner(), PACKAGE_IDENTITY["source_commit"])


def test_formal_receipt_rejects_secret_payload(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_runtime_qualification import _formal_json

    output = tmp_path / "preflight.json"
    output.write_bytes(b'{"error":"password=hunter2"}')
    class Runner:
        def run(self, command, **kwargs):
            return type("Result", (), {"returncode": 1})()

    with pytest.raises(ValueError, match="secret"):
        _formal_json(Runner(), ("tool",), output, cwd=tmp_path, env={})


def test_formal_receipt_rejects_noncanonical_json(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_runtime_qualification import _formal_json

    output = tmp_path / "preflight.json"
    output.write_bytes(b'{"passed": true}')
    class Runner:
        def run(self, command, **kwargs):
            return type("Result", (), {"returncode": 1})()

    with pytest.raises(ValueError, match="canonical"):
        _formal_json(Runner(), ("tool",), output, cwd=tmp_path, env={})


def test_run_rejects_frozen_core_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = config.__class__(**{**config.__dict__, "output_root": config.project_root / "data" / "vnext" / "core" / "v3"})
    config.output_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="frozen"):
        run_qualification(config, runner=object())


class FakeRunner:
    def __init__(self, *, fail_at: str | None = None, stop_returncode: int = 0, blocked_formal: bool = False, post_lifecycle_marker_count: str = "0"):
        self.calls: list[tuple[str, ...]] = []
        self.fail_at = fail_at
        self.stop_returncode = stop_returncode
        self.blocked_formal = blocked_formal
        self.post_lifecycle_marker_count = post_lifecycle_marker_count
        self.processes = []

    def run(self, command, *, check=True, **kwargs):
        self.calls.append(tuple(command))
        if self.fail_at and any(self.fail_at in part for part in command):
            raise RuntimeError("simulated failure")
        joined = " ".join(command)
        if command and command[0] == "git":
            output = "1131535716e8a31c9a437f8695e25ac98f203a24" if "letta-source" in joined else "a" * 40
            return type("Result", (), {"stdout": output + "\n", "stderr": "", "returncode": 0})()
        if "-c" in command:
            sql = command[command.index("-c") + 1]
            output = {
                "SHOW server_version;": "18.6",
                "SELECT extversion FROM pg_extension WHERE extname='vector';": "0.8.6",
                "SELECT pg_backend_pid() || '|' || coalesce(inet_server_addr()::text,'') || '|' || coalesce(inet_server_port()::text,'') || '|' || current_database() || '|' || current_user || '|' || (SELECT rolsuper::text FROM pg_roles WHERE rolname=current_user);": "123|127.0.0.1|5432|mub_letta_v1|mub_letta_v1|true",
                "SELECT current_database();": "mub_letta_v1",
                "SELECT count(*) FROM block WHERE description LIKE 'MemUpdateBench namespace marker v1:%';": self.post_lifecycle_marker_count,
            }.get(sql, "")
            return type("Result", (), {"stdout": output + "\n", "stderr": "", "returncode": 0})()
        if "vnext_admit_letta_runtime.py" in joined:
            payload = {"admitted": not self.blocked_formal, "outcome": "pass" if not self.blocked_formal else "blocked"}
            Path(command[command.index("--output") + 1]).write_bytes(
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            )
            return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0 if not self.blocked_formal else 1})()
        if "preflight" in joined:
            payload = {"passed": not self.blocked_formal, "outcome": "pass" if not self.blocked_formal else "blocked"}
            Path(command[command.index("--output") + 1]).write_bytes(
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            )
            return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0 if not self.blocked_formal else 1})()
        return type("Result", (), {"stdout": "", "stderr": "", "returncode": self.stop_returncode if "pg_ctl" in joined and "stop" in joined else 0})()

    def popen(self, command, **kwargs):
        self.calls.append(tuple(command))
        class Process:
            def __init__(self):
                self.returncode = None
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = 0
            def wait(self, timeout=None):
                self.returncode = 0
                return 0
            def kill(self):
                self.returncode = -9
        process = Process()
        self.processes.append(process)
        return process

    def ready(self, url, **kwargs):
        return True


def test_cleanup_failure_downgrades_pass_to_blocked(tmp_path: Path) -> None:
    result = run_qualification(_config(tmp_path), runner=FakeRunner(stop_returncode=1))
    assert result["outcome"] == "BLOCKED"
    assert result["cleanup"]["status"] == "BLOCKED"
    assert "postgres_stop_failed" in result["cleanup"]["errors"]


def test_run_invokes_vector_before_alembic_and_formal_scripts_and_publishes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = FakeRunner()
    result = run_qualification(config, runner=runner)
    assert result["outcome"] == "PASS"
    assert result["boundary"] == {"llm_used": False, "api_used": False, "gpu_used": False}
    calls = [" ".join(call) for call in runner.calls]
    assert any("CREATE EXTENSION vector" in call for call in calls)
    assert calls.index(next(c for c in calls if "CREATE EXTENSION vector" in c)) < calls.index(next(c for c in calls if "alembic" in c))
    assert any("vnext_preflight_letta_runtime.py" in call for call in calls)
    assert any("vnext_admit_letta_runtime.py" in call for call in calls)
    assert any(call[0] == str(config.alembic_executable) and call[1] == "-c" and call[-2:] == ("upgrade", "head") for call in runner.calls)
    assert (config.output_root / "letta_runtime_qualification.json").exists()
    assert all("random-secret" not in part for call in runner.calls for part in call)


def test_post_lifecycle_marker_probe_uses_singular_block_table_and_zero_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = FakeRunner(post_lifecycle_marker_count="0")

    result = run_qualification(config, runner=runner)

    marker_queries = [
        call[call.index("-c") + 1]
        for call in runner.calls
        if "-c" in call and "MemUpdateBench namespace marker v1:" in call[call.index("-c") + 1]
    ]
    assert marker_queries == ["SELECT count(*) FROM block WHERE description LIKE 'MemUpdateBench namespace marker v1:%';"]
    assert result["outcome"] == "PASS"
    assert result["runtime"]["measured"]["post_lifecycle_marker_count"] == "0"


def test_nonzero_post_lifecycle_marker_count_blocks_cleanup_leak(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = FakeRunner(post_lifecycle_marker_count="2")

    result = run_qualification(config, runner=runner)

    assert result["outcome"] == "BLOCKED"
    assert result["runtime"]["measured"]["post_lifecycle_marker_count"] == "2"
    assert result["diagnostic"]["error_type"] == "ValueError"
    assert "post-lifecycle database isolation/cleanup probe failed" in result["diagnostic"]["message"]


def test_blocked_formal_receipts_are_preserved_with_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_qualification(config, runner=FakeRunner(blocked_formal=True))
    assert result["outcome"] == "BLOCKED"
    preflight = config.output_root / "letta_runtime_preflight.json"
    admission = config.output_root / "letta_runtime_admission.json"
    assert preflight.read_bytes() == b'{"outcome":"blocked","passed":false}'
    assert admission.read_bytes() == b'{"admitted":false,"outcome":"blocked"}'
    assert result["preflight"]["sha256"] == hashlib.sha256(preflight.read_bytes()).hexdigest()
    assert result["admission"]["sha256"] == hashlib.sha256(admission.read_bytes()).hexdigest()


def test_run_cleans_up_on_failure_and_publishes_redacted_diagnostic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = FakeRunner(fail_at="alembic")
    result = run_qualification(config, runner=runner)
    assert result["outcome"] == "BLOCKED"
    assert result["diagnostic"]["status"] == "NOT_RUN"
    assert result["diagnostic"]["error_type"] == "RuntimeError"
    assert "simulated failure" in result["diagnostic"]["message"]
    assert result["cleanup"]["status"] == "PASS"
    assert (config.output_root / "letta_runtime_qualification.json").exists()


def test_run_never_replaces_existing_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = config.output_root / "letta_runtime_qualification.json"
    target.write_text("existing")
    with pytest.raises(FileExistsError):
        run_qualification(config, runner=FakeRunner())
