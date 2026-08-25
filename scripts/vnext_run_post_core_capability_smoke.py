from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilityAttemptReceiptV1,
)
from mub.vnext.post_core.qualification_validation_v1 import (
    load_capability_smoke_plan_v1,
    load_execution_authorization_v1,
    validate_capability_attempt_receipts_v1,
    validate_qualification_secret_free,
)


EXIT_SUCCESS = 0
EXIT_BLOCKED = 10
EXIT_USAGE = 11
EXIT_STALE_SOURCE = 12
EXIT_OUTPUT = 13
EXIT_ADAPTER = 14
_MAX_CAPTURE_BYTES = 4 * 1024 * 1024
_OS_ENVIRONMENT_ALLOWLIST = (
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
)


class _ArgumentUsageError(Exception):
    pass


class _AdapterProtocolError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentUsageError


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run an explicitly authorized post-Core capability smoke adapter.",
        allow_abbrev=False,
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--authorization-receipt")
    parser.add_argument("--adapter-executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _absolute_path(path: Path) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return Path(os.path.abspath(os.path.normpath(str(selected))))


def _reject_reparse_components(path: Path) -> None:
    selected = _absolute_path(path)
    current = Path(selected.anchor)
    for part in selected.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        if _is_reparse(current):
            raise ValueError("unsafe link or reparse path")


def _regular_single_link(path: Path, label: str) -> Path:
    selected = _absolute_path(path)
    _reject_reparse_components(selected)
    try:
        metadata = selected.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if (
        _is_reparse(selected)
        or not stat.S_ISREG(metadata.st_mode)
        or getattr(metadata, "st_nlink", 1) != 1
    ):
        raise ValueError(f"{label} must be a regular single-link file")
    return selected


def _safe_absent_output(path: Path) -> Path:
    selected = _absolute_path(path)
    _reject_reparse_components(selected.parent)
    if not selected.parent.is_dir() or _is_reparse(selected.parent):
        raise ValueError("unsafe output parent")
    if selected.exists() or selected.is_symlink() or _is_reparse(selected):
        raise FileExistsError("output path is unavailable")
    return selected


def _adapter_environment() -> dict[str, str]:
    return {
        key: value
        for key in _OS_ENVIRONMENT_ALLOWLIST
        if (value := os.environ.get(key)) is not None
    }


def _selected_attempts(plan: Any, authorization: Any) -> tuple[Any, ...]:
    allowed = set(authorization.authorized_call_ids)
    return tuple(attempt for attempt in plan.attempts if attempt.call_id in allowed)


def _parse_adapter_stdout(raw: bytes) -> tuple[CapabilityAttemptReceiptV1, ...]:
    if not raw or len(raw) > _MAX_CAPTURE_BYTES or not raw.endswith(b"\n"):
        raise _AdapterProtocolError
    rows: list[CapabilityAttemptReceiptV1] = []
    for line in raw[:-1].split(b"\n"):
        if not line:
            raise _AdapterProtocolError
        try:
            payload = json.loads(line)
            validate_qualification_secret_free(payload)
            receipt = CapabilityAttemptReceiptV1.model_validate(payload)
        except Exception as exc:
            raise _AdapterProtocolError from exc
        if canonical_bytes(receipt) != line:
            raise _AdapterProtocolError
        rows.append(receipt)
    return tuple(rows)


def _write_output(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("output write failed")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _is_stale_source_rejection(exception: Exception) -> bool:
    message = str(exception).lower()
    return (
        "changed while being read" in message
        or "changed after validation" in message
        or "authorization plan hash mismatch" in message
    )


def _summary(output: Path, selected: tuple[Any, ...]) -> str:
    return json.dumps(
        {
            "status": "SUCCESS",
            "output": str(output),
            "call_count": len(selected),
            "base_count": sum(item.phase is AttemptPhase.BASE for item in selected),
            "escalation_count": sum(item.phase is AttemptPhase.ESCALATION for item in selected),
            "retries": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except _ArgumentUsageError:
        print("capability smoke contract/usage rejected: invalid arguments", file=sys.stderr)
        return EXIT_USAGE
    except SystemExit as exception:
        return EXIT_SUCCESS if exception.code == 0 else EXIT_USAGE
    if not args.authorization_receipt:
        print("capability smoke blocked: missing execution authorization", file=sys.stderr)
        return EXIT_BLOCKED
    if not args.execute:
        print("capability smoke requires explicit --execute", file=sys.stderr)
        return EXIT_USAGE
    try:
        plan = load_capability_smoke_plan_v1(Path(args.plan))
        authorization = load_execution_authorization_v1(Path(args.authorization_receipt), plan)
        selected = _selected_attempts(plan, authorization)
    except ValueError as exception:
        if _is_stale_source_rejection(exception):
            print("capability smoke stale source rejected", file=sys.stderr)
            return EXIT_STALE_SOURCE
        print("capability smoke contract/usage rejected", file=sys.stderr)
        return EXIT_USAGE
    except (TypeError, OSError):
        print("capability smoke contract/usage rejected", file=sys.stderr)
        return EXIT_USAGE
    try:
        adapter = _regular_single_link(Path(args.adapter_executable), "adapter executable")
    except (ValueError, OSError):
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    try:
        output = _safe_absent_output(Path(args.output))
    except FileExistsError:
        print("capability smoke rejected: output path is unavailable", file=sys.stderr)
        return EXIT_OUTPUT
    except (ValueError, OSError):
        print("capability smoke rejected: unsafe output path", file=sys.stderr)
        return EXIT_OUTPUT

    command = (
        [sys.executable, str(adapter), "--jsonl-protocol-v1"]
        if adapter.suffix.lower() == ".py"
        else [str(adapter), "--jsonl-protocol-v1"]
    )
    payload = b"".join(canonical_bytes(attempt) + b"\n" for attempt in selected)
    timeout_seconds = min(300, max(attempt.budget.timeout_seconds for attempt in selected))
    try:
        completed = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            timeout=timeout_seconds,
            shell=False,
            env=_adapter_environment(),
        )
        if (
            completed.returncode != 0
            or completed.stderr
            or len(completed.stdout) > _MAX_CAPTURE_BYTES
            or len(completed.stderr) > _MAX_CAPTURE_BYTES
        ):
            raise _AdapterProtocolError
        receipts = _parse_adapter_stdout(completed.stdout)
        validate_capability_attempt_receipts_v1(selected, receipts)
        _write_output(output, completed.stdout)
    except (subprocess.TimeoutExpired, OSError, ValueError, _AdapterProtocolError):
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    except Exception:
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    print(_summary(output, selected))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
