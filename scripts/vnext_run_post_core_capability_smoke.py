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
import tempfile
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


def _selected_attempts(plan: Any, authorization: Any) -> tuple[Any, ...]:
    allowed = set(authorization.authorized_call_ids)
    return tuple(attempt for attempt in plan.attempts if attempt.call_id in allowed)


def _pin_adapter(path: Path) -> tuple[Path, Path]:
    adapter = _regular_single_link(path, "adapter executable")
    raw = _read_regular_single_link(adapter, "adapter executable")
    private_dir = Path(tempfile.mkdtemp(prefix="mub-capability-smoke-"))
    os.chmod(private_dir, 0o700)
    private_path = private_dir / ("adapter.py" if adapter.suffix.lower() == ".py" else "adapter.bin")
    descriptor: int | None = None
    try:
        descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o700)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if _read_regular_single_link(private_path, "pinned adapter") != raw:
            raise ValueError("pinned adapter changed")
        return private_dir, private_path
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            private_path.unlink(missing_ok=True)
            private_dir.rmdir()
        except OSError:
            pass
        raise


def _after_adapter_pinned(original: Path, pinned: Path) -> None:
    return None


def _before_adapter_run(output: Path) -> None:
    return None


def _pinned_adapter_stable(
    private_dir: Path,
    private_path: Path,
    directory_identity: tuple[int, int],
    adapter_identity: tuple[int, int],
    adapter_sha256: str,
) -> bool:
    try:
        return (
            _identity(private_dir.stat()) == directory_identity
            and _identity(private_path.lstat()) == adapter_identity
            and hashlib.sha256(_read_regular_single_link(private_path, "pinned adapter")).hexdigest()
            == adapter_sha256
        )
    except (OSError, ValueError):
        return False


def _clean_pinned_adapter(private_dir: Path, private_path: Path) -> None:
    try:
        metadata = private_path.lstat()
        if stat.S_ISREG(metadata.st_mode) and getattr(metadata, "st_nlink", 1) == 1 and not _is_reparse(private_path):
            private_path.unlink()
        private_dir.rmdir()
    except OSError:
        pass


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
    except OSError:
        return False


def _discard_reserved_output(path: Path, descriptor: int, parent_identity: tuple[int, int], output_identity: tuple[int, int]) -> None:
    try:
        expected_size = os.fstat(descriptor).st_size
    except OSError:
        expected_size = -1
    stable = _reserved_output_stable(
        path, descriptor, parent_identity, output_identity, expected_size
    )
    try:
        os.close(descriptor)
    finally:
        if stable:
            try:
                metadata = path.lstat()
                if (
                    _identity(path.parent.stat()) == parent_identity
                    and _identity(metadata) == output_identity
                    and stat.S_ISREG(metadata.st_mode)
                    and not _is_reparse(path)
                    and getattr(metadata, "st_nlink", 1) == 1
                    and metadata.st_size == expected_size
                ):
                    path.unlink()
            except OSError:
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
            and result.response_model is not None
            and result.response_model.strip()
            and result.stop_reason is not None
            and result.stop_reason.strip()
            and result.usage_present is not None
        )
    return result.response_format == "LOCAL_TEXT" and result.latency_ms is not None


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


def _summary(output: Path, selected: tuple[Any, ...]) -> str:
    return json.dumps({"status": "SUCCESS", "output": str(output), "call_count": len(selected), "base_count": sum(item.phase is AttemptPhase.BASE for item in selected), "escalation_count": sum(item.phase is AttemptPhase.ESCALATION for item in selected), "retries": 0}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
        private_dir, pinned_adapter = _pin_adapter(Path(args.adapter_executable))
        pinned_dir_identity = _identity(private_dir.stat())
        pinned_adapter_identity = _identity(pinned_adapter.lstat())
        pinned_adapter_sha256 = hashlib.sha256(
            _read_regular_single_link(pinned_adapter, "pinned adapter")
        ).hexdigest()
        _after_adapter_pinned(Path(args.adapter_executable), pinned_adapter)
    except (ValueError, OSError):
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    try:
        output, output_fd, parent_identity, output_identity = _reserve_output(Path(args.output))
    except FileExistsError:
        _clean_pinned_adapter(private_dir, pinned_adapter)
        print("capability smoke rejected: output path is unavailable", file=sys.stderr)
        return EXIT_OUTPUT
    except (ValueError, OSError):
        _clean_pinned_adapter(private_dir, pinned_adapter)
        print("capability smoke rejected: unsafe output path", file=sys.stderr)
        return EXIT_OUTPUT

    command = [sys.executable, str(pinned_adapter), "--jsonl-protocol-v1"] if pinned_adapter.suffix.lower() == ".py" else [str(pinned_adapter), "--jsonl-protocol-v1"]
    payload = b"".join(canonical_bytes(attempt) + b"\n" for attempt in selected)
    timeout_seconds = min(300, max(attempt.budget.timeout_seconds for attempt in selected))
    finalized = False
    try:
        _before_adapter_run(output)
        if not _pinned_adapter_stable(
            private_dir,
            pinned_adapter,
            pinned_dir_identity,
            pinned_adapter_identity,
            pinned_adapter_sha256,
        ):
            raise _AdapterProtocolError
        completed = subprocess.run(command, input=payload, capture_output=True, timeout=timeout_seconds, shell=False, env=_adapter_environment())
        if completed.returncode != 0 or completed.stderr or len(completed.stdout) > _MAX_CAPTURE_BYTES or len(completed.stderr) > _MAX_CAPTURE_BYTES:
            raise _AdapterProtocolError
        adapter_results = _parse_adapter_stdout(completed.stdout)
        receipts = _adapter_results_to_receipts(selected, adapter_results)
        validate_capability_attempt_receipts_v1(selected, receipts)
        receipt_raw = b"".join(canonical_bytes(receipt) + b"\n" for receipt in receipts)
        _finalize_reserved_output(output, output_fd, parent_identity, output_identity, receipt_raw)
        finalized = True
    except (subprocess.TimeoutExpired, OSError, ValueError, _AdapterProtocolError):
        if not finalized:
            _discard_reserved_output(output, output_fd, parent_identity, output_identity)
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    except Exception:
        if not finalized:
            _discard_reserved_output(output, output_fd, parent_identity, output_identity)
        print("capability smoke adapter/runtime/protocol rejected", file=sys.stderr)
        return EXIT_ADAPTER
    finally:
        _clean_pinned_adapter(private_dir, pinned_adapter)
    print(_summary(output, selected))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
