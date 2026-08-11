from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
import queue
import subprocess
import threading
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.v3.common import (
    FrozenJsonObjectV3,
    StrictIdentifier,
)
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.external.visibility import validate_visible_payload
from mub.vnext.io import canonical_json_bytes


class BridgeError(RuntimeError):
    pass


class BridgeTimeoutError(BridgeError):
    pass


class BridgeProtocolError(BridgeError):
    pass


class BridgeProcessError(BridgeError):
    pass


class WorkerOperation(str, Enum):
    HEALTH = "health"
    RESET = "reset"
    INGEST_EVENT = "ingest_event"
    RETRIEVE = "retrieve"
    EXPORT_ENTRIES = "export_entries"
    EXPORT_RAW_STATE = "export_raw_state"
    EXPORT_VERSION_HISTORY = "export_version_history"
    CLOSE = "close"


class WorkerResponseStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class WorkerRequestV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.worker_request.v1"] = (
        "memupdatebench.external.worker_request.v1"
    )
    request_id: StrictIdentifier
    operation: WorkerOperation
    payload: FrozenJsonObjectV3 = Field(default_factory=dict)

    @model_validator(mode="after")
    def _visible_only(self) -> Self:
        validate_visible_payload(self.payload)
        return self


class WorkerResponseV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.worker_response.v1"] = (
        "memupdatebench.external.worker_response.v1"
    )
    request_id: StrictIdentifier
    status: WorkerResponseStatus
    payload: FrozenJsonObjectV3 = Field(default_factory=dict)
    error_code: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.status is WorkerResponseStatus.OK:
            if self.error_code is not None:
                raise ValueError("successful worker responses cannot carry errors")
        elif self.error_code is None or self.payload:
            raise ValueError(
                "failed worker responses require an error code and no payload"
            )
        return self


def _revalidate_request(request: WorkerRequestV1) -> WorkerRequestV1:
    if type(request) is not WorkerRequestV1:
        raise BridgeProtocolError(
            "worker request must be an exact WorkerRequestV1"
        )
    try:
        rebuilt = WorkerRequestV1.model_validate(
            {
                field_name: request.__dict__[field_name]
                for field_name in WorkerRequestV1.model_fields
            },
            strict=True,
        )
    except Exception as exc:
        raise BridgeProtocolError(
            "worker request fails trust-boundary validation"
        ) from exc
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(request):
        raise BridgeProtocolError("worker request serialization is unstable")
    return rebuilt


class JsonlSubprocessBridge:
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        cwd: str | Path,
        environment: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if (
            type(command) is not tuple
            or not command
            or any(type(part) is not str or not part for part in command)
        ):
            raise ValueError("bridge command must be a nonempty exact tuple")
        if scan_for_secrets(command):
            raise ValueError("bridge command security scan failed")
        executable = Path(command[0])
        if not executable.is_absolute():
            raise ValueError("bridge executable must use an absolute path")
        working_directory_input = Path(cwd).absolute()
        assert_no_reparse_components(working_directory_input)
        working_directory = working_directory_input.resolve(strict=True)
        if not working_directory.is_dir() or working_directory.is_symlink():
            raise ValueError("bridge working directory must be a real directory")
        if type(timeout_seconds) is not float or timeout_seconds <= 0:
            raise ValueError("bridge timeout must be a positive exact float")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("bridge response limit must be a positive integer")
        copied_environment: dict[str, str] = {}
        for name, value in environment.items():
            if type(name) is not str or type(value) is not str:
                raise ValueError("bridge environment must contain strings")
            copied_environment[name] = value

        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._responses: queue.Queue[bytes | None] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=copied_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="external-worker-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="external-worker-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            self._responses.put(None)
            return
        while True:
            line = stream.readline(self._max_response_bytes + 1)
            if not line:
                self._responses.put(None)
                return
            self._responses.put(line)

    def _drain_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        while stream.read(65536):
            pass

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

    def _ensure_worker_remains_available(self) -> None:
        observed_terminal = False
        try:
            queued = self._responses.get(timeout=0.01)
        except queue.Empty:
            queued = b""
        else:
            if queued is None:
                observed_terminal = True
            else:
                self._terminate()
                self._closed = True
                raise BridgeProtocolError(
                    "worker returned an unsolicited extra response"
                )
        exit_code = self._process.poll()
        if observed_terminal or exit_code is not None:
            if exit_code is None:
                try:
                    exit_code = self._process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    exit_code = None
            self._terminate()
            self._closed = True
            raise BridgeProcessError(
                "worker became unavailable after a response; "
                f"exit_code={exit_code}"
            )

    def request(self, request: WorkerRequestV1) -> WorkerResponseV1:
        request = _revalidate_request(request)
        with self._lock:
            if self._closed:
                raise BridgeProcessError("worker bridge is closed")
            stdin = self._process.stdin
            if stdin is None or self._process.poll() is not None:
                raise BridgeProcessError("worker process is unavailable")
            try:
                stdin.write(canonical_json_bytes(request) + b"\n")
                stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate()
                self._closed = True
                raise BridgeProcessError(
                    "worker process rejected a request"
                ) from exc
            try:
                line = self._responses.get(timeout=self._timeout_seconds)
            except queue.Empty as exc:
                self._terminate()
                self._closed = True
                raise BridgeTimeoutError(
                    "worker response deadline exceeded"
                ) from exc
            if line is None:
                exit_code = self._process.poll()
                self._closed = True
                raise BridgeProcessError(
                    "worker exited before returning a response; "
                    f"exit_code={exit_code}"
                )
            try:
                response = self._parse_response(line)
            except BridgeProtocolError:
                self._terminate()
                self._closed = True
                raise
            if response.request_id != request.request_id:
                self._terminate()
                self._closed = True
                raise BridgeProtocolError(
                    "worker response request ID does not match"
                )
            self._ensure_worker_remains_available()
            return response

    def _parse_response(self, line: bytes) -> WorkerResponseV1:
        if len(line) > self._max_response_bytes:
            raise BridgeProtocolError("worker response exceeds size limit")
        if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
            raise BridgeProtocolError("worker returned a noncanonical response")
        raw = line[:-1]
        if raw.endswith(b"\r"):
            raise BridgeProtocolError("worker returned a noncanonical response")
        try:
            response = WorkerResponseV1.model_validate_json(raw)
        except Exception:
            raise BridgeProtocolError(
                "worker returned an invalid canonical response"
            ) from None
        if canonical_json_bytes(response) != raw:
            raise BridgeProtocolError("worker returned a noncanonical response")
        return response

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                stdin = self._process.stdin
                if stdin is not None:
                    try:
                        stdin.close()
                    except OSError:
                        pass
            self._terminate()
            for stream in (
                self._process.stdin,
                self._process.stdout,
                self._process.stderr,
            ):
                if stream is None:
                    continue
                try:
                    stream.close()
                except OSError:
                    pass
        self._stdout_thread.join(timeout=1.0)
        self._stderr_thread.join(timeout=1.0)

    def __enter__(self) -> JsonlSubprocessBridge:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "BridgeError",
    "BridgeProcessError",
    "BridgeProtocolError",
    "BridgeTimeoutError",
    "JsonlSubprocessBridge",
    "WorkerOperation",
    "WorkerRequestV1",
    "WorkerResponseStatus",
    "WorkerResponseV1",
]
