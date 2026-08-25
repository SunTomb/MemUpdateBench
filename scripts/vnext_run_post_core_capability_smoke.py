from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilityAdapterResultV1,
    CapabilityAttemptReceiptV1,
    GateStatus,
)
from mub.vnext.post_core.qualification_validation_v1 import (
    _read_regular_single_link,
    load_capability_anomaly_receipt_v1,
    load_capability_smoke_plan_v1,
    load_canonical_jsonl_v1,
    load_execution_authorization_v1,
    validate_capability_attempt_receipts_v1,
    validate_escalation_anomaly_evidence_v1,
    validate_escalation_anomaly_receipt_v1,
    validate_qualification_secret_free,
)


EXIT_SUCCESS = 0
EXIT_BLOCKED = 10
EXIT_USAGE = 11
EXIT_STALE_SOURCE = 12
EXIT_OUTPUT = 13
EXIT_ADAPTER = 14
_MAX_CAPTURE_BYTES = 4 * 1024 * 1024
_OS_ENVIRONMENT_ALLOWLIST = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP")
_EXPECTED_RESPONSE_MODELS = {
    "claude_sonnet_4_6": "claude-sonnet-4-6",
    "claude_opus_4_8": "claude-opus-4-8",
    "gemini_3_6_flash": "Gemini 3.6 Flash (Low)",
    "grok_4_5": "grok-4.5",
    "gpt_5_5": "gpt-5.5",
}


class _ArgumentUsageError(Exception):
    pass


class _AdapterProtocolError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentUsageError


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run an explicitly authorized post-Core capability smoke adapter.", allow_abbrev=False
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--authorization-receipt")
    parser.add_argument("--escalation-anomaly-receipt")
    parser.add_argument("--base-receipts")
    parser.add_argument("--adapter-executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return bool(stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


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


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _regular_single_link(path: Path, label: str) -> Path:
    selected = _absolute_path(path)
    _reject_reparse_components(selected)
    try:
        metadata = selected.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if _is_reparse(selected) or not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"{label} must be a regular single-link file")
    return selected


def _write_all(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("write failed")
        remaining = remaining[written:]


def _adapter_environment() -> dict[str, str]:
    return {key: value for key in _OS_ENVIRONMENT_ALLOWLIST if (value := os.environ.get(key)) is not None}


def _run_python_adapter_bounded(
    source_text: str, payload: bytes, timeout_seconds: int
) -> tuple[int, bytes, bytes]:
    if len(payload) > _MAX_CAPTURE_BYTES:
        raise _AdapterProtocolError
    process = subprocess.Popen(
        [sys.executable, "-c", source_text, "--jsonl-protocol-v1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=_adapter_environment(),
    )
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()

    def read_pipe(name: str, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                buffer = captured[name]
                remaining = _MAX_CAPTURE_BYTES + 1 - len(buffer)
                if remaining > 0:
                    buffer.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
                    try:
                        process.kill()
                    except OSError:
                        pass
                    return
        finally:
            stream.close()

    stdout_thread = threading.Thread(target=read_pipe, args=("stdout", process.stdout), daemon=True)
    stderr_thread = threading.Thread(target=read_pipe, args=("stderr", process.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        try:
            if process.stdin is None:
                raise _AdapterProtocolError
            _write_all(process.stdin.fileno(), payload)
        except (OSError, _AdapterProtocolError):
            pass
        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise _AdapterProtocolError from exc
    finally:
        if process.poll() is None:
            process.kill()
        stdout_thread.join()
        stderr_thread.join()
    if overflow.is_set():
        raise _AdapterProtocolError
    return returncode, bytes(captured["stdout"]), bytes(captured["stderr"])


def _selected_attempts(plan: Any, authorization: Any) -> tuple[Any, ...]:
    allowed = set(authorization.authorized_call_ids)
    return tuple(attempt for attempt in plan.attempts if attempt.call_id in allowed)


def _capture_python_adapter(path: Path) -> tuple[Path, bytes, str]:
    """Capture the only v1 adapter form: bounded UTF-8 Python source."""
    adapter = _regular_single_link(path, "adapter executable")
    if adapter.suffix.lower() != ".py":
        raise ValueError("adapter executable must be a Python source file in v1")
    raw = _read_regular_single_link(adapter, "adapter executable")
    if len(raw) > _MAX_CAPTURE_BYTES:
        raise ValueError("adapter source exceeds the v1 size limit")
    try:
        source_text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("adapter source must use UTF-8") from exc
    return adapter, raw, source_text


def _after_adapter_pinned(original: Path, captured: Path) -> None:
    return None


def _before_adapter_run(output: Path) -> None:
    return None


def _reserve_output(path: Path) -> tuple[Path, int, tuple[int, int], tuple[int, int]]:
    selected = _absolute_path(path)
    _reject_reparse_components(selected.parent)
    try:
        parent_before = selected.parent.stat()
    except OSError as exc:
        raise ValueError("unsafe output parent") from exc
    if not selected.parent.is_dir() or _is_reparse(selected.parent):
        raise ValueError("unsafe output parent")
    try:
        descriptor = os.open(selected, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError:
        raise
    try:
        output_identity = _identity(os.fstat(descriptor))
        return selected, descriptor, _identity(parent_before), output_identity
    except Exception:
        os.close(descriptor)
        try:
            selected.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _reserved_output_stable(
    path: Path,
    descriptor: int,
    parent_identity: tuple[int, int],
    output_identity: tuple[int, int],
    expected_size: int,
) -> bool:
    try:
        _reject_reparse_components(path.parent)
        path_metadata = path.lstat()
        descriptor_metadata = os.fstat(descriptor)
        return bool(
            _identity(path.parent.stat()) == parent_identity
            and _identity(path_metadata) == output_identity
            and _identity(descriptor_metadata) == output_identity
            and stat.S_ISREG(path_metadata.st_mode)
            and stat.S_ISREG(descriptor_metadata.st_mode)
            and not _is_reparse(path)
            and not (getattr(descriptor_metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            and getattr(path_metadata, "st_nlink", 1) == 1
            and getattr(descriptor_metadata, "st_nlink", 1) == 1
            and path_metadata.st_size == expected_size
            and descriptor_metadata.st_size == expected_size
        )
    except (OSError, ValueError):
        return False


def _discard_reserved_output(path: Path, descriptor: int, parent_identity: tuple[int, int], output_identity: tuple[int, int]) -> None:
    try:
        expected_size = os.fstat(descriptor).st_size
    except OSError:
        expected_size = -1
    _reserved_output_stable(path, descriptor, parent_identity, output_identity, expected_size)
    try:
        os.close(descriptor)
    finally:
        try:
            _reject_reparse_components(path.parent)
            metadata = path.lstat()
            if (
                _identity(path.parent.stat()) == parent_identity
                and _identity(metadata) == output_identity
                and stat.S_ISREG(metadata.st_mode)
                and not _is_reparse(path)
            ):
                path.unlink()
        except (OSError, ValueError):
            pass


def _finalize_reserved_output(path: Path, descriptor: int, parent_identity: tuple[int, int], output_identity: tuple[int, int], raw: bytes) -> None:
    if not _reserved_output_stable(path, descriptor, parent_identity, output_identity, 0):
        raise _AdapterProtocolError
    _write_all(descriptor, raw)
    os.fsync(descriptor)
    if not _reserved_output_stable(path, descriptor, parent_identity, output_identity, len(raw)):
        raise _AdapterProtocolError
    os.close(descriptor)


_SINGLE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _parse_adapter_stdout(raw: bytes) -> tuple[CapabilityAdapterResultV1, ...]:
    if not raw or len(raw) > _MAX_CAPTURE_BYTES or not raw.endswith(b"\n"):
        raise _AdapterProtocolError
    rows: list[CapabilityAdapterResultV1] = []
    for line in raw[:-1].split(b"\n"):
        if not line:
            raise _AdapterProtocolError
        try:
            payload = json.loads(line)
            validate_qualification_secret_free(payload)
            result = CapabilityAdapterResultV1.model_validate(payload)
        except Exception as exc:
            raise _AdapterProtocolError from exc
        if canonical_bytes(result) != line:
            raise _AdapterProtocolError
        rows.append(result)
    return tuple(rows)


def _adapter_results_to_receipts(
    selected: tuple[Any, ...], results: tuple[CapabilityAdapterResultV1, ...]
) -> tuple[CapabilityAttemptReceiptV1, ...]:
    if len(results) != len(selected):
        raise _AdapterProtocolError
    receipts: list[CapabilityAttemptReceiptV1] = []
    for attempt, result in zip(selected, results):
        if result.call_id != attempt.call_id or result.registry_key != attempt.registry_key:
            raise _AdapterProtocolError
        if result.error_class is not None:
            if not result.error_class.strip() or result.response_projection is not None:
                raise _AdapterProtocolError
            receipts.append(
                CapabilityAttemptReceiptV1(
                    call_id=attempt.call_id,
                    registry_key=attempt.registry_key,
                    status=GateStatus.FAIL,
                    error_class=result.error_class,
                )
            )
            continue
        projection = result.response_projection
        if projection is None or not _transport_result_is_complete(attempt, result):
            raise _AdapterProtocolError
        if _projection_matches_fixture(attempt, projection):
            receipts.append(
                CapabilityAttemptReceiptV1(
                    call_id=attempt.call_id,
                    registry_key=attempt.registry_key,
                    status=GateStatus.PASS,
                    response_model=result.response_model,
                    response_format=result.response_format,
                    stop_reason=result.stop_reason,
                    usage_present=result.usage_present,
                    latency_ms=result.latency_ms,
                    redacted_response_sha256=hashlib.sha256(
                        canonical_bytes({"fixture_id": attempt.fixture_id, "projection": projection})
                    ).hexdigest(),
                )
            )
        else:
            receipts.append(
                CapabilityAttemptReceiptV1(
                    call_id=attempt.call_id,
                    registry_key=attempt.registry_key,
                    status=GateStatus.FAIL,
                    error_class="PARSER_MISMATCH",
                )
            )
    return tuple(receipts)


def _transport_result_is_complete(attempt: Any, result: CapabilityAdapterResultV1) -> bool:
    if attempt.runtime_or_endpoint_class == "api_transfer_station":
        return bool(
            result.response_format in {"ANTHROPIC_MESSAGE_JSON", "SSE"}
            and result.response_model == _EXPECTED_RESPONSE_MODELS.get(attempt.registry_key)
            and result.stop_reason is not None
            and result.stop_reason.strip()
            and result.usage_present is not None
        )
    return (
        result.response_format == "LOCAL_TEXT"
        and result.response_model is None
        and result.latency_ms is not None
    )


def _projection_matches_fixture(attempt: Any, projection: str) -> bool:
    if attempt.fixture_id == "exact_ok_1":
        return projection == "READY"
    if attempt.fixture_id == "exact_ok_2":
        return projection == "ACK"
    if attempt.fixture_id in {"parser_city_1", "parser_city_2"}:
        return bool(
            projection == projection.strip()
            and projection
            and len(projection) <= attempt.budget.max_output_tokens
            and "\n" not in projection
            and "\r" not in projection
            and _SINGLE_IDENTIFIER.fullmatch(projection)
        )
    return False


def _is_stale_source_rejection(exception: Exception) -> bool:
    message = str(exception).lower()
    return (
        "changed while being read" in message
        or "changed after validation" in message
        or "plan hash mismatch" in message
        or "receipt hash mismatch" in message
    )


def _summary(output: Path, selected: tuple[Any, ...], receipts: tuple[CapabilityAttemptReceiptV1, ...]) -> str:
    pass_count = sum(receipt.status is GateStatus.PASS for receipt in receipts)
    fail_count = len(receipts) - pass_count
    return json.dumps({"status": "SUCCESS" if fail_count == 0 else "BLOCKED", "output": str(output), "call_count": len(selected), "base_count": sum(item.phase is AttemptPhase.BASE for item in selected), "escalation_count": sum(item.phase is AttemptPhase.ESCALATION for item in selected), "pass_count": pass_count, "fail_count": fail_count, "retries": 0}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except _ArgumentUsageError:
        print("capability smoke contract/usage rejected: invalid arguments", file=sys.stderr)
        return EXIT_USAGE
    except SystemExit as exception:
        return EXIT_SUCCESS if exception.code == 0 else EXIT_USAGE
    if not args.execute:
        if not args.authorization_receipt:
            print("capability smoke blocked: missing execution authorization", file=sys.stderr)
            return EXIT_BLOCKED
        print("capability smoke requires explicit --execute", file=sys.stderr)
        return EXIT_USAGE
    try:
        plan = load_capability_smoke_plan_v1(Path(args.plan))
        if not args.authorization_receipt:
            print("capability smoke blocked: missing execution authorization", file=sys.stderr)
            return EXIT_BLOCKED
        authorization = load_execution_authorization_v1(Path(args.authorization_receipt), plan)
        selected = _selected_attempts(plan, authorization)
        escalated = any(attempt.phase is AttemptPhase.ESCALATION for attempt in selected)
        if escalated:
            if not args.escalation_anomaly_receipt or not args.base_receipts:
                print("capability smoke blocked: missing escalation anomaly evidence", file=sys.stderr)
                return EXIT_BLOCKED
            anomaly, anomaly_raw = load_capability_anomaly_receipt_v1(
                Path(args.escalation_anomaly_receipt), plan
            )
            if hashlib.sha256(anomaly_raw).hexdigest() != authorization.escalation_anomaly_receipt_sha256:
                raise ValueError("escalation anomaly receipt hash mismatch")
            validate_escalation_anomaly_receipt_v1(anomaly, plan, selected)
            base_receipts, base_receipts_raw = load_canonical_jsonl_v1(
                Path(args.base_receipts), CapabilityAttemptReceiptV1, label="base receipts"
            )
            validate_escalation_anomaly_evidence_v1(
                anomaly, base_receipts, base_receipts_raw, plan, selected
            )
        elif (
            authorization.escalation_anomaly_receipt_sha256 is not None
            or args.escalation_anomaly_receipt
            or args.base_receipts
        ):
            raise ValueError("base-only authorization cannot carry escalation evidence")
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
        adapter, adapter_raw, adapter_source = _capture_python_adapter(Path(args.adapter_executable))
        if hashlib.sha256(adapter_raw).hexdigest() != authorization.adapter_sha256:
            raise ValueError("adapter source hash mismatch")
        _after_adapter_pinned(adapter, adapter)
    except (ValueError, OSError):
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    try:
        output, output_fd, parent_identity, output_identity = _reserve_output(Path(args.output))
    except FileExistsError:
        print("capability smoke rejected: output path is unavailable", file=sys.stderr)
        return EXIT_OUTPUT
    except (ValueError, OSError):
        print("capability smoke rejected: unsafe output path", file=sys.stderr)
        return EXIT_OUTPUT

    payload = b"".join(canonical_bytes(attempt) + b"\n" for attempt in selected)
    timeout_seconds = min(300, max(attempt.budget.timeout_seconds for attempt in selected))
    finalized = False
    try:
        _before_adapter_run(output)
        returncode, stdout, stderr = _run_python_adapter_bounded(
            adapter_source, payload, timeout_seconds
        )
        if returncode != 0 or stderr:
            raise _AdapterProtocolError
        adapter_results = _parse_adapter_stdout(stdout)
        receipts = _adapter_results_to_receipts(selected, adapter_results)
        validate_capability_attempt_receipts_v1(selected, receipts)
        receipt_raw = b"".join(canonical_bytes(receipt) + b"\n" for receipt in receipts)
        _finalize_reserved_output(output, output_fd, parent_identity, output_identity, receipt_raw)
        finalized = True
    except (OSError, ValueError, _AdapterProtocolError):
        if not finalized:
            _discard_reserved_output(output, output_fd, parent_identity, output_identity)
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    except Exception:
        if not finalized:
            _discard_reserved_output(output, output_fd, parent_identity, output_identity)
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    print(_summary(output, selected, receipts))
    return EXIT_SUCCESS if all(receipt.status is GateStatus.PASS for receipt in receipts) else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
