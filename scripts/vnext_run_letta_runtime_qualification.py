from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.security import redact_sensitive_text, scan_for_secrets


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


SCHEMA_VERSION = "memupdatebench.external.letta.runtime_qualification.v1"
PACKAGE_IDENTITY = {
    "package_name": "letta",
    "package_version": "0.16.8",
    "source_repository": "letta-ai/letta",
    "source_commit": "1131535716e8a31c9a437f8695e25ac98f203a24",
}
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class QualificationConfig:
    python_executable: Path
    postgres_bin: Path
    alembic_executable: Path
    letta_source: Path
    project_root: Path
    nltk_cache: Path
    output_root: Path
    project_revision: str
    role: str = "mub_letta_v1"
    database: str = "mub_letta_v1"
    timeout_seconds: float = 60.0
    expected_postgres_version: str | None = None
    expected_pgvector_version: str | None = None
    letta_source_commit: str = PACKAGE_IDENTITY["source_commit"]


class Runner(Protocol):
    def run(self, command: tuple[str, ...], **kwargs): ...
    def popen(self, command: tuple[str, ...], **kwargs): ...
    def ready(self, url: str, **kwargs) -> bool: ...


class SubprocessRunner:
    def run(self, command: tuple[str, ...], **kwargs):
        return subprocess.run(command, **kwargs)

    def popen(self, command: tuple[str, ...], **kwargs):
        return subprocess.Popen(command, **kwargs)

    @staticmethod
    def _valid_ready_response(response, process=None) -> bool:
        if process is not None and process.poll() is not None:
            return False
        if getattr(response, "status", None) != 200:
            return False
        try:
            payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return False
        if isinstance(payload, list):
            return True
        return isinstance(payload, dict) and isinstance(payload.get("blocks", payload.get("data")), list)

    def ready(self, url: str, *, timeout_seconds: float, process=None) -> bool:
        import urllib.request
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if self._valid_ready_response(response, process):
                        return True
            except Exception:
                pass
            time.sleep(0.25)
        return False


def find_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distinct_ports() -> tuple[int, int]:
    for _ in range(8):
        db_port, server_port = find_free_loopback_port(), find_free_loopback_port()
        if db_port != server_port:
            return db_port, server_port
    raise RuntimeError("could not allocate distinct loopback ports")


def make_password_file(directory: Path, *, token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32)) -> tuple[Path, str]:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    password = token_factory()
    if type(password) is not str or not password or any(ch in password for ch in "\r\n"):
        raise ValueError("password generator returned an invalid value")
    path = directory / f"postgres-password-{secrets.token_hex(8)}"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, (password + "\n").encode())
    finally:
        os.close(fd)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path, password


def validate_postgres_identifier(value: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("invalid PostgreSQL identifier")
    return value


def build_initdb_command(initdb: Path, cluster: Path, role: str, pwfile: Path) -> tuple[str, ...]:
    validate_postgres_identifier(role)
    command = (str(initdb), "-D", str(cluster), "-U", role, f"--pwfile={pwfile}", "--auth-local=md5", "--auth-host=md5")
    if scan_for_secrets(command):
        raise ValueError("initdb command failed secret scan")
    return command


def build_launcher_source(nltk_cache: Path, port: int, letta_dir: Path | None = None) -> str:
    cache = nltk_cache.resolve()
    if not cache.is_dir() or not (cache / "tokenizers" / "punkt_tab").is_dir():
        raise ValueError("local punkt_tab NLTK cache is required")
    directory = (letta_dir or cache.parent / "letta").resolve()
    return f'''import os\nimport nltk\nos.environ["LETTA_DIR"] = {str(directory)!r}\nnltk.data.path.insert(0, {str(cache)!r})\nnltk.download = lambda *args, **kwargs: True\nfrom letta.server.rest_api.app import start_server\nstart_server(port={int(port)}, host="127.0.0.1", debug=False, reload=False)\n'''


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ensure_safe_output_root(config: QualificationConfig) -> Path:
    root = config.output_root
    if not root.is_absolute():
        raise ValueError("output root must be absolute")
    assert_no_reparse_components(root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("output root must be a real directory")
    frozen = (
        config.project_root / "data" / "vnext" / "core",
        config.project_root / "data" / "vnext" / "phase0",
        config.project_root / "configs" / "vnext" / "post_core",
    )
    if any(_contains(candidate, root) or _contains(root, candidate) for candidate in frozen):
        raise ValueError("output root is inside or contains a frozen root")
    protected = (*frozen, config.project_root, config.letta_source)
    if any(_contains(candidate, root) or _contains(root, candidate) for candidate in protected if candidate not in frozen):
        raise ValueError("output root overlaps protected source")
    root.mkdir(parents=True, exist_ok=True)
    assert_no_reparse_components(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("output root must be a real directory")
    return root.resolve()


def _git_head(path: Path, runner: Runner) -> str:
    result = runner.run(("git", "-C", str(path), "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    if getattr(result, "returncode", 1) != 0:
        raise ValueError("source git identity unavailable")
    value = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("source git identity unavailable")
    return value


def _verify_real_tree(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("source path must be absolute")
    assert_no_reparse_components(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("source path must be a real directory")
    resolved = path.resolve(strict=True)
    for item in resolved.rglob("*"):
        assert_no_reparse_components(item)
        if item.is_symlink():
            raise ValueError("source tree must not contain symlinks")
    return resolved


def _framed_tree_hash(source: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in source.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(source).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        digest.update(len(content).to_bytes(8, "big")); digest.update(content)
    return digest.hexdigest(), len(files)


def _stable_tree_identity(source: Path) -> tuple[str, int]:
    first = _framed_tree_hash(source)
    second = _framed_tree_hash(source)
    if first != second:
        raise ValueError("source tree changed during hashing")
    return first


def _source_identity(source: Path, runner: Runner, expected_commit: str) -> dict:
    resolved = _verify_real_tree(source)
    commit = _git_head(resolved, runner)
    if commit != expected_commit:
        raise ValueError("Letta source commit mismatch")
    tree_sha256, file_count = _stable_tree_identity(resolved)
    return {"commit": commit, "tree_sha256": tree_sha256, "file_count": file_count, "commit_verified": True}


def _project_identity(project: Path, runner: Runner, expected_revision: str) -> dict:
    resolved = _verify_real_tree(project)
    commit = _git_head(resolved, runner)
    if commit != expected_revision:
        raise ValueError("project source revision mismatch")
    tree_sha256, file_count = _stable_tree_identity(resolved)
    return {"commit": commit, "tree_sha256": tree_sha256, "file_count": file_count, "commit_verified": True}


def _publish_bytes_no_replace(path: Path, raw: bytes) -> str:
    if not path.is_absolute():
        raise ValueError("qualification output must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("qualification output parent must be a real directory")
    identity = (parent.stat().st_dev, parent.stat().st_ino)
    with path.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    if (parent.stat().st_dev, parent.stat().st_ino) != identity:
        raise ValueError("qualification output parent changed")
    if path.read_bytes() != raw:
        raise ValueError("qualification output reread mismatch")
    if os.name != "nt":
        fd = os.open(parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def _publish_no_replace(path: Path, payload: dict) -> None:
    if scan_for_secrets(payload):
        raise ValueError("qualification evidence failed secret scan")
    _publish_bytes_no_replace(path, _canonical_json_bytes(payload))


def build_postgres_environment(source_environment: Mapping[str, str], *, password: str) -> dict[str, str]:
    selected = {name: value for name, value in source_environment.items() if name in {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH"} and type(value) is str}
    selected["PGPASSWORD"] = password
    return selected


def _runtime_environment(config: QualificationConfig, *, port: int, db_uri: str, private: Path) -> dict[str, str]:
    home = private / "home"
    letta_dir = private / "letta"
    home.mkdir(mode=0o700); letta_dir.mkdir(mode=0o700)
    env = {
        "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(config.project_root), "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1", "NLTK_DATA": str(config.nltk_cache), "HOME": str(home), "LETTA_DIR": str(letta_dir),
        "LETTA_PG_URI": db_uri, "LETTA_NATIVE_API_BASE_URL": f"http://127.0.0.1:{port}",
        "LETTA_DISABLE_TRACING": "true", "LETTA_LLM_API_LOGGING": "false", "LETTA_TRACK_PROVIDER_TRACE": "false",
    }
    return env


def _port_closed(port: int | None) -> bool:
    if port is None: return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _psql(runner: Runner, psql: Path, socket_dir: Path, port: int, database: str, role: str, password: str, sql: str, env: Mapping[str, str], *, tcp: bool = False) -> str:
    host = "127.0.0.1" if tcp else str(socket_dir)
    result = runner.run((str(psql), "-h", host, "-p", str(port), "-U", role, "-d", database, "-v", "ON_ERROR_STOP=1", "-X", "-At", "-c", sql), env=env, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _formal_json(runner: Runner, command: tuple[str, ...], output: Path, *, cwd: Path, env: Mapping[str, str]) -> tuple[dict, bytes, int]:
    result = runner.run(command, cwd=str(cwd), env=env, check=False, capture_output=True, text=True)
    if not output.is_file() or output.is_symlink():
        raise ValueError("formal qualification artifact missing")
    raw = output.read_bytes()
    try: value = json.loads(raw)
    except Exception: raise ValueError("formal qualification artifact is invalid JSON") from None
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ValueError("formal qualification artifact must be canonical JSON")
    if scan_for_secrets(value):
        raise ValueError("formal qualification artifact failed secret scan")
    return value, raw, int(getattr(result, "returncode", 1))


def _validate_absolute_executable(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    assert_no_reparse_components(path)
    return path.resolve(strict=True)


def run_qualification(config: QualificationConfig, *, runner: Runner | None = None) -> dict:
    runner = runner or SubprocessRunner()
    output_root = _ensure_safe_output_root(config)
    _validate_absolute_executable(config.alembic_executable, "alembic executable")
    if (output_root / "letta_runtime_qualification.json").exists():
        raise FileExistsError(output_root / "letta_runtime_qualification.json")
    source_identity = _source_identity(config.letta_source, runner, config.letta_source_commit)
    project_identity = _project_identity(config.project_root, runner, config.project_revision)
    runner_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cluster = Path(tempfile.mkdtemp(prefix="mub-letta-pg-"))
    private = Path(tempfile.mkdtemp(prefix="mub-letta-private-"))
    socket_dir = private / "socket"; socket_dir.mkdir(mode=0o700)
    os.chmod(socket_dir, 0o700)
    pg_ctl = config.postgres_bin / "pg_ctl"
    password_file = None; process = None; log_handles = []
    cleanup_errors: list[str] = []; password_deleted = False; process_stopped = True
    logs_deleted = False; preflight_value = admission_value = None; preflight_raw = admission_raw = None
    preflight_hash = admission_hash = None; measured: dict[str, object] = {}
    outcome = "BLOCKED"; diagnostic: dict[str, object] = {"status": "NOT_RUN"}
    db_port = server_port = None
    try:
        validate_postgres_identifier(config.role); validate_postgres_identifier(config.database)
        password_file, password = make_password_file(private)
        db_port, server_port = _distinct_ports()
        initdb = config.postgres_bin / "initdb"; psql = config.postgres_bin / "psql"
        initdb_cmd = build_initdb_command(initdb, cluster, config.role, password_file)
        pg_env = build_postgres_environment(os.environ, password=password)
        runner.run(initdb_cmd, env=build_postgres_environment(os.environ, password=password), check=True, capture_output=True, text=True)
        password_file.unlink(); password_file = None; password_deleted = True
        runner.run((str(pg_ctl), "-D", str(cluster), "-o", f"-h 127.0.0.1 -k {socket_dir} -p {db_port}", "-l", str(private / "postgres.log"), "-w", "start"), env=pg_env, check=True, capture_output=True, text=True)
        _psql(runner, psql, socket_dir, db_port, "postgres", config.role, password, f"CREATE DATABASE {config.database} OWNER {config.role};", pg_env)
        _psql(runner, psql, socket_dir, db_port, config.database, config.role, password, "CREATE EXTENSION vector;", pg_env)
        version = _psql(runner, psql, socket_dir, db_port, config.database, config.role, password, "SHOW server_version;", pg_env)
        vector_version = _psql(runner, psql, socket_dir, db_port, config.database, config.role, password, "SELECT extversion FROM pg_extension WHERE extname='vector';", pg_env)
        identity_raw = _psql(runner, psql, socket_dir, db_port, config.database, config.role, password, "SELECT pg_backend_pid() || '|' || coalesce(inet_server_addr()::text,'') || '|' || coalesce(inet_server_port()::text,'') || '|' || current_database() || '|' || current_user || '|' || (SELECT rolsuper::text FROM pg_roles WHERE rolname=current_user);", pg_env, tcp=True)
        fields = identity_raw.split("|")
        if len(fields) != 6 or fields[3] != config.database or fields[4] != config.role or fields[5].lower() != "true":
            raise ValueError("PostgreSQL identity probe failed")
        if config.expected_postgres_version is not None and version != config.expected_postgres_version:
            raise ValueError("PostgreSQL version mismatch")
        if config.expected_pgvector_version is not None and vector_version != config.expected_pgvector_version:
            raise ValueError("pgvector version mismatch")
        measured = {"postgres_version": version, "pgvector_version": vector_version, "server_pid": fields[0], "server_address": fields[1], "server_port": fields[2], "database": fields[3], "current_user": fields[4], "rolsuper": True}
        database_identity = f"postgresql-{version}-pgvector-{vector_version}-dedicated-loopback"
        db_uri = f"postgresql+asyncpg://{config.role}:{password}@127.0.0.1:{db_port}/{config.database}"
        env = _runtime_environment(config, port=server_port, db_uri=db_uri, private=private)
        runner.run((str(config.alembic_executable), "-c", str(config.letta_source / "alembic.ini"), "upgrade", "head"), cwd=str(config.letta_source), env=env, check=True, capture_output=True, text=True)
        launcher = private / "letta_server_launcher.py"; launcher.write_text(build_launcher_source(config.nltk_cache, server_port, private / "letta"), encoding="utf-8")
        stdout_handle = (private / "server.stdout.log").open("wb"); stderr_handle = (private / "server.stderr.log").open("wb"); log_handles = [stdout_handle, stderr_handle]
        process = runner.popen((str(config.python_executable), str(launcher)), cwd=str(config.letta_source), env=env, stdout=stdout_handle, stderr=stderr_handle)
        stdout_handle.close(); stderr_handle.close(); log_handles = []
        if not runner.ready(f"http://127.0.0.1:{server_port}/v1/blocks/", timeout_seconds=config.timeout_seconds, process=process):
            raise RuntimeError("Letta server readiness timeout")
        preflight = private / "preflight.json"; admission = private / "admission.json"
        preflight_value, preflight_raw, preflight_rc = _formal_json(runner, (str(config.python_executable), str(config.project_root / "scripts" / "vnext_preflight_letta_runtime.py"), "--python-executable", str(config.python_executable), "--worker-command", str(config.project_root / "mub" / "vnext" / "external" / "workers" / "letta_worker.py"), "--server-identity", f"http://127.0.0.1:{server_port}", "--database-identity", database_identity, "--output", str(preflight), "--run-prefix", "letta-runtime-song1", "--timeout-seconds", str(config.timeout_seconds), "--database-isolation-verified"), preflight, cwd=config.project_root, env=env)
        admission_value, admission_raw, admission_rc = _formal_json(runner, (str(config.python_executable), str(config.project_root / "scripts" / "vnext_admit_letta_runtime.py"), "--preflight", str(preflight), "--output", str(admission)), admission, cwd=config.project_root, env=env)
        post_lifecycle_marker_count = _psql(runner, psql, socket_dir, db_port, config.database, config.role, password, "SELECT count(*) FROM block WHERE description LIKE 'MemUpdateBench namespace marker v1:%';", pg_env)
        measured["post_lifecycle_marker_count"] = post_lifecycle_marker_count
        try:
            marker_count_is_zero = int(post_lifecycle_marker_count) == 0
        except (TypeError, ValueError):
            marker_count_is_zero = False
        if not marker_count_is_zero:
            raise ValueError("post-lifecycle database isolation/cleanup probe failed: marker count must be zero")
        measured["post_lifecycle_database"] = _psql(runner, psql, socket_dir, db_port, config.database, config.role, password, "SELECT current_database();", pg_env)
        if measured["post_lifecycle_database"] != config.database:
            raise ValueError("post-lifecycle database isolation probe failed")
        if _source_identity(config.letta_source, runner, config.letta_source_commit) != source_identity:
            raise ValueError("Letta source changed during qualification")
        if _project_identity(config.project_root, runner, config.project_revision) != project_identity:
            raise ValueError("project source changed during qualification")
        preflight_hash = hashlib.sha256(preflight_raw).hexdigest(); admission_hash = hashlib.sha256(admission_raw).hexdigest()
        passed = preflight_value.get("passed") is True and preflight_value.get("outcome") == "pass" and admission_value.get("admitted") is True and admission_value.get("outcome") == "pass" and preflight_rc in (0, 1) and admission_rc in (0, 1)
        outcome = "PASS" if passed else "BLOCKED"
        diagnostic = {"status": "PASS" if passed else "BLOCKED", "preflight_returncode": preflight_rc, "admission_returncode": admission_rc}
    except Exception as exc:
        outcome = "BLOCKED"; diagnostic = {"status": "NOT_RUN", "error_type": type(exc).__name__, "message": redact_sensitive_text(str(exc))}
    finally:
        for handle in log_handles:
            try: handle.close()
            except Exception: cleanup_errors.append("log_close_failed")
        if process is not None:
            try:
                process.terminate(); process.wait(timeout=10)
                process_stopped = process.poll() is not None
            except Exception:
                try: process.kill(); process.wait(timeout=10); process_stopped = process.poll() is not None
                except Exception: cleanup_errors.append("letta_stop_failed"); process_stopped = False
        if process is not None and not process_stopped: cleanup_errors.append("letta_process_running")
        try:
            stop_result = runner.run((str(pg_ctl), "-D", str(cluster), "-m", "fast", "-w", "stop"), env=build_postgres_environment(os.environ, password=password if 'password' in locals() else ""), check=False, capture_output=True, text=True)
            if getattr(stop_result, "returncode", 1) != 0:
                cleanup_errors.append("postgres_stop_failed")
        except Exception:
            cleanup_errors.append("postgres_stop_failed")
        if db_port is not None and not _port_closed(db_port): cleanup_errors.append("postgres_port_open")
        if server_port is not None and not _port_closed(server_port): cleanup_errors.append("letta_port_open")
        if password_file is not None:
            try: password_file.unlink(); password_deleted = True
            except FileNotFoundError: password_deleted = True
            except Exception: cleanup_errors.append("password_file_delete_failed")
        try: shutil.rmtree(cluster)
        except FileNotFoundError: pass
        except Exception: cleanup_errors.append("postgres_private_dir_delete_failed")
        try: shutil.rmtree(private)
        except FileNotFoundError: pass
        except Exception: cleanup_errors.append("private_dir_delete_failed")
        logs_deleted = not private.exists()
        if not logs_deleted: cleanup_errors.append("raw_logs_remain")
    cleanup_passed = not cleanup_errors and process_stopped and password_deleted and logs_deleted
    if not cleanup_passed:
        outcome = "BLOCKED"; diagnostic = {**diagnostic, "cleanup_blocked": True}
    payload = {"schema_version": SCHEMA_VERSION, "candidate_id": "letta_0_16_8_song1_local_linux", "outcome": outcome, "project_revision": config.project_revision, "identity": PACKAGE_IDENTITY, "source": source_identity, "project_source": project_identity, "runner_source_sha256": runner_source_sha256, "preflight": {"artifact": "letta_runtime_preflight.json", "sha256": preflight_hash, "bytes": len(preflight_raw) if preflight_raw else None}, "admission": {"artifact": "letta_runtime_admission.json", "sha256": admission_hash, "bytes": len(admission_raw) if admission_raw else None}, "runtime": {"python": str(config.python_executable), "postgres_bin": str(config.postgres_bin), "measured": measured, "loopback_only": True, "random_ports": True, "dedicated_role_superuser": measured.get("rolsuper") is True, "vector_before_alembic": True, "nltk_cache_verified": True}, "boundary": {"llm_used": False, "api_used": False, "gpu_used": False}, "cleanup": {"status": "PASS" if cleanup_passed else "BLOCKED", "errors": cleanup_errors, "password_file_deleted": password_deleted, "process_stopped": process_stopped, "logs_deleted": logs_deleted}, "diagnostic": diagnostic, "raw_logs_recorded": False}
    if scan_for_secrets(payload): raise ValueError("qualification closure failed secret scan")
    if preflight_raw is not None: _publish_bytes_no_replace(output_root / "letta_runtime_preflight.json", preflight_raw)
    if admission_raw is not None: _publish_bytes_no_replace(output_root / "letta_runtime_admission.json", admission_raw)
    _publish_no_replace(output_root / "letta_runtime_qualification.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local Linux Letta 0.16.8 runtime qualification.")
    for name in ("python-executable", "postgres-bin", "alembic-executable", "letta-source", "project-root", "nltk-cache", "output-root", "project-revision"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-postgres-version")
    parser.add_argument("--expected-pgvector-version")
    args = parser.parse_args(argv)
    config = QualificationConfig(Path(args.python_executable), Path(args.postgres_bin), Path(args.alembic_executable), Path(args.letta_source), Path(args.project_root), Path(args.nltk_cache), Path(args.output_root), args.project_revision, expected_postgres_version=args.expected_postgres_version, expected_pgvector_version=args.expected_pgvector_version)
    try:
        payload = run_qualification(config)
    except Exception:
        return 1
    return 0 if payload["outcome"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
