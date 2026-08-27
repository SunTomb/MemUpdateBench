from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import re
import sys
from typing import Any, BinaryIO, Protocol

from mub.vnext.contracts.common import thaw_json
from mub.vnext.external.bridge import (
    WorkerOperation,
    WorkerRequestV1,
    WorkerResponseStatus,
    WorkerResponseV1,
)
from mub.vnext.external.providers.letta import (
    LETTA_PACKAGE_VERSION,
    LETTA_SOURCE_COMMIT,
    LettaAdapterConfigurationV1,
    compute_letta_configuration_hash,
)
from mub.vnext.external.providers.letta_protocol import (
    LettaWorkerCloseResultV1,
    LettaWorkerEntryListV1,
    LettaWorkerEntryV1,
    LettaWorkerHealthV1,
    LettaWorkerMutationResultV1,
    LettaWorkerResetResultV1,
    LettaWorkerRetrievalResultV1,
)
from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1
from mub.vnext.io import canonical_json_bytes

_MUTATION_PATTERN = re.compile(r"^(Add|Update) (.+) with value (.+)\.$")
_DELETE_PATTERN = re.compile(r"^Delete (.+)\.$")
_DELETE_METADATA_PATTERN = re.compile(r"\s+\[[^\[\]]+\]$")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class LettaDependencyUnavailable(RuntimeError):
    pass


class LettaWorkerProtocolError(RuntimeError):
    pass


class _InvalidRequestPayload(ValueError):
    pass


class LettaBlockClientV1(Protocol):
    def get_block(self, namespace: str, block_id: str) -> Mapping | None: ...

    def create_block(self, namespace: str, block_id: str, value: dict) -> None: ...

    def update_block(self, namespace: str, block_id: str, value: dict) -> None: ...

    def delete_block(self, namespace: str, block_id: str) -> None: ...

    def search_blocks(self, namespace: str) -> tuple[tuple[str, Mapping], ...]: ...


class LettaBackendV1(Protocol):
    def health(self) -> LettaWorkerHealthV1: ...

    def reset_namespace(self, namespace: str) -> None: ...

    def ingest_event(self, event: ProviderEventInputV1) -> LettaWorkerMutationResultV1: ...

    def retrieve(self, query: ProviderQueryInputV1) -> LettaWorkerRetrievalResultV1: ...

    def export_entries(self, namespace: str) -> tuple[LettaWorkerEntryV1, ...]: ...

    def close(self) -> None: ...


def _exact_payload(payload: object, expected_keys: set[str]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise _InvalidRequestPayload("worker payload must be a mapping")
    values = dict(payload)
    if set(values) != expected_keys:
        raise _InvalidRequestPayload("worker payload keys are invalid")
    return values


def _namespace(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise _InvalidRequestPayload("runtime namespace must be a nonblank string")
    return value


def _response_payload(model) -> dict[str, Any]:
    return model.model_dump(mode="json")


class LettaWorkerServiceV1:
    def __init__(self, backend: LettaBackendV1) -> None:
        self._backend = backend
        self._closed = False

    def _ok(self, request_id: str, payload: dict[str, Any]) -> WorkerResponseV1:
        return WorkerResponseV1(
            request_id=request_id, status=WorkerResponseStatus.OK, payload=payload
        )

    def _error(self, request_id: str, code: str) -> WorkerResponseV1:
        return WorkerResponseV1(
            request_id=request_id, status=WorkerResponseStatus.ERROR, error_code=code
        )

    def handle(self, request: WorkerRequestV1) -> WorkerResponseV1:
        if type(request) is not WorkerRequestV1:
            raise ValueError("Letta worker requires exact WorkerRequestV1")
        if self._closed:
            return self._error(request.request_id, "worker_closed")
        try:
            return self._dispatch(request)
        except _InvalidRequestPayload:
            return self._error(request.request_id, "invalid_request_payload")
        except Exception:
            return self._error(request.request_id, "worker_backend_error")

    def _dispatch(self, request: WorkerRequestV1) -> WorkerResponseV1:
        if request.operation is WorkerOperation.HEALTH:
            _exact_payload(request.payload, set())
            return self._ok(request.request_id, _response_payload(self._backend.health()))
        if request.operation is WorkerOperation.RESET:
            payload = _exact_payload(request.payload, {"runtime_namespace"})
            namespace = _namespace(payload["runtime_namespace"])
            self._backend.reset_namespace(namespace)
            return self._ok(
                request.request_id,
                _response_payload(LettaWorkerResetResultV1(namespace=namespace)),
            )
        if request.operation is WorkerOperation.INGEST_EVENT:
            payload = _exact_payload(request.payload, {"event"})
            try:
                event = ProviderEventInputV1.model_validate(
                    thaw_json(payload["event"]), strict=True
                )
            except Exception:
                raise _InvalidRequestPayload("event payload is invalid") from None
            return self._ok(request.request_id, _response_payload(self._backend.ingest_event(event)))
        if request.operation is WorkerOperation.RETRIEVE:
            payload = _exact_payload(request.payload, {"query"})
            try:
                query = ProviderQueryInputV1.model_validate(
                    thaw_json(payload["query"]), strict=True
                )
            except Exception:
                raise _InvalidRequestPayload("query payload is invalid") from None
            return self._ok(request.request_id, _response_payload(self._backend.retrieve(query)))
        if request.operation is WorkerOperation.EXPORT_ENTRIES:
            payload = _exact_payload(request.payload, {"runtime_namespace"})
            namespace = _namespace(payload["runtime_namespace"])
            return self._ok(
                request.request_id,
                _response_payload(LettaWorkerEntryListV1(entries=self._backend.export_entries(namespace))),
            )
        if request.operation in {WorkerOperation.EXPORT_RAW_STATE, WorkerOperation.EXPORT_VERSION_HISTORY}:
            return self._error(request.request_id, "not_supported")
        if request.operation is WorkerOperation.CLOSE:
            _exact_payload(request.payload, set())
            self._backend.close()
            self._closed = True
            return self._ok(request.request_id, _response_payload(LettaWorkerCloseResultV1()))
        return self._error(request.request_id, "not_supported")


class LettaBlockProfileBackendV1:
    """Protocol-test backend only; cannot authenticate an official Letta runtime."""

    def __init__(self, *, client: LettaBlockClientV1, configuration: LettaAdapterConfigurationV1) -> None:
        if type(configuration) is not LettaAdapterConfigurationV1:
            raise ValueError("Letta backend requires exact public configuration")
        self._client = client
        self._configuration = configuration

    def health(self) -> LettaWorkerHealthV1:
        return LettaWorkerHealthV1(
            package_name="letta-protocol-fake",
            package_version="0",
            source_commit="0" * 40,
            license_id="test-only",
            configuration_hash=compute_letta_configuration_hash(self._configuration),
        )

    def _native_namespace(self, runtime_namespace: str) -> str:
        return f"{self._configuration.namespace_root}/{runtime_namespace}"

    @staticmethod
    def _entry_id(canonical_object_id: str) -> str:
        digest = hashlib.sha256(canonical_object_id.encode("utf-8")).hexdigest()
        return f"letta-block-{digest[:32]}"

    def reset_namespace(self, namespace: str) -> None:
        native_namespace = self._native_namespace(namespace)
        for block_id, _ in self._client.search_blocks(native_namespace):
            if type(block_id) is not str or not block_id:
                raise ValueError("Letta block ID is invalid")
            self._client.delete_block(native_namespace, block_id)

    def _parse_event(self, event: ProviderEventInputV1) -> tuple[str, str | None, object | None]:
        text = event.raw_text.strip()
        if text == "No memory object changes.":
            return "noop", None, None
        mutation = _MUTATION_PATTERN.fullmatch(text)
        if mutation is not None:
            object_id = self._single_object_id(mutation.group(2))
            try:
                value = json.loads(mutation.group(3))
            except (TypeError, ValueError):
                raise ValueError("Letta visible value is not canonical JSON") from None
            if value is None:
                raise ValueError("Letta visible value cannot be null")
            return mutation.group(1).lower(), object_id, value
        deletion = _DELETE_PATTERN.fullmatch(text)
        if deletion is None:
            raise ValueError("Letta visible event has unsupported action surface")
        rendered = deletion.group(1)
        object_id = self._single_object_id(_DELETE_METADATA_PATTERN.sub("", rendered))
        if "scope=object" not in rendered:
            raise ValueError("Letta profile mode only supports object DELETE")
        return "delete", object_id, None

    @staticmethod
    def _single_object_id(rendered: str) -> str:
        values = tuple(part.strip() for part in rendered.split(","))
        if len(values) != 1 or not values[0]:
            raise ValueError("Letta profile mode requires one target object")
        return values[0]

    @staticmethod
    def _stored_value(event: ProviderEventInputV1, object_id: str, value: object) -> dict[str, object]:
        return {
            "canonical_object_id": object_id,
            "content": event.raw_text,
            "value": value,
            "source_event_id": event.event_id,
            "sequence_index": event.sequence_index,
        }

    def _entry(self, block_id: str, raw: Mapping) -> LettaWorkerEntryV1:
        canonical = raw.get("canonical_object_id")
        content = raw.get("content")
        source_event_id = raw.get("source_event_id")
        sequence_index = raw.get("sequence_index")
        if (
            type(block_id) is not str or not block_id or type(canonical) is not str
            or type(content) is not str or type(source_event_id) is not str
            or type(sequence_index) is not int or sequence_index < 0 or raw.get("value") is None
        ):
            raise ValueError("Letta native block value is invalid")
        return LettaWorkerEntryV1(
            entry_id=block_id,
            canonical_object_id=canonical,
            content=content,
            value=raw["value"],
            source_event_ids=(source_event_id,),
            sequence_index=sequence_index,
        )

    def ingest_event(self, event: ProviderEventInputV1) -> LettaWorkerMutationResultV1:
        operation, object_id, value = self._parse_event(event)
        if operation == "noop":
            return LettaWorkerMutationResultV1(event_id=event.event_id, effective_operation="noop")
        if object_id is None:
            raise ValueError("Letta mutation object ID is missing")
        block_id = self._entry_id(object_id)
        native_namespace = self._native_namespace(event.runtime_namespace)
        existing = self._client.get_block(native_namespace, block_id)
        if operation == "add":
            if existing is not None:
                return LettaWorkerMutationResultV1(event_id=event.event_id, effective_operation="noop")
            if self._client.search_blocks(native_namespace):
                raise ValueError("Letta profile mode rejects collection inserts")
            self._client.create_block(native_namespace, block_id, self._stored_value(event, object_id, value))
        elif operation == "update":
            if existing is None:
                return LettaWorkerMutationResultV1(event_id=event.event_id, effective_operation="noop")
            self._client.update_block(native_namespace, block_id, self._stored_value(event, object_id, value))
        elif operation == "delete":
            if existing is None:
                return LettaWorkerMutationResultV1(event_id=event.event_id, effective_operation="noop")
            self._client.delete_block(native_namespace, block_id)
        else:
            raise ValueError("Letta operation is invalid")
        return LettaWorkerMutationResultV1(event_id=event.event_id, effective_operation=operation, entry_id=block_id)

    def export_entries(self, namespace: str) -> tuple[LettaWorkerEntryV1, ...]:
        return tuple(
            self._entry(block_id, raw)
            for block_id, raw in self._client.search_blocks(self._native_namespace(namespace))
        )

    def retrieve(self, query: ProviderQueryInputV1) -> LettaWorkerRetrievalResultV1:
        ranked = sorted(
            ((_deterministic_score(query.query_text, entry), entry) for entry in self.export_entries(query.runtime_namespace)),
            key=lambda row: (-row[0], row[1].entry_id),
        )[:query.k]
        return LettaWorkerRetrievalResultV1(
            query_id=query.query_id,
            entries=tuple(entry for _, entry in ranked),
            scores=tuple(score for score, _ in ranked),
        )

    def close(self) -> None:
        return None


def _deterministic_score(query_text: str, entry: LettaWorkerEntryV1) -> float:
    query_tokens = set(_TOKEN_PATTERN.findall(query_text.casefold()))
    entry_tokens = set(_TOKEN_PATTERN.findall(
        f"{entry.canonical_object_id} {entry.content} {json.dumps(entry.value, ensure_ascii=False, sort_keys=True)}".casefold()
    ))
    if not query_tokens or not entry_tokens:
        return 0.0
    return len(query_tokens & entry_tokens) / len(query_tokens | entry_tokens)


def inspect_local_letta_package() -> dict[str, object]:
    """Inspect package metadata only; never imports Letta or starts its server."""
    try:
        distribution = importlib.metadata.distribution("letta")
    except importlib.metadata.PackageNotFoundError:
        return {"package_present": False, "identity_verified": False, "blocker": "letta_package_not_installed"}
    version = distribution.version
    license_text = distribution.metadata.get("License")
    return {
        "package_present": True,
        "package_version": version,
        "license_metadata": license_text if isinstance(license_text, str) else None,
        "identity_verified": False,
        "blocker": "local_source_commit_and_server_bootstrap_unverified",
    }


def build_official_letta_backend(configuration: LettaAdapterConfigurationV1) -> LettaBlockProfileBackendV1:
    inspection = inspect_local_letta_package()
    if inspection["identity_verified"] is not True:
        raise LettaDependencyUnavailable(str(inspection["blocker"]))
    raise LettaDependencyUnavailable("official Letta bootstrap adapter is not implemented")


def serve_letta_worker_jsonl(service: LettaWorkerServiceV1, *, input_stream: BinaryIO, output_stream: BinaryIO, max_request_bytes: int = 16 * 1024 * 1024) -> None:
    if type(service) is not LettaWorkerServiceV1:
        raise ValueError("Letta JSONL loop requires exact worker service")
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("Letta JSONL request limit must be positive")
    while True:
        line = input_stream.readline(max_request_bytes + 2)
        if line == b"":
            return
        if len(line) > max_request_bytes + 1 or not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
            raise LettaWorkerProtocolError("Letta worker request is invalid")
        raw = line[:-1]
        if raw.endswith(b"\r"):
            raise LettaWorkerProtocolError("Letta worker request is invalid")
        try:
            request = WorkerRequestV1.model_validate_json(raw, strict=True)
        except Exception:
            raise LettaWorkerProtocolError("Letta worker request is invalid") from None
        if canonical_json_bytes(request) != raw:
            raise LettaWorkerProtocolError("Letta worker request is noncanonical")
        response = service.handle(request)
        output_stream.write(canonical_json_bytes(response) + b"\n")
        output_stream.flush()
        if request.operation is WorkerOperation.CLOSE:
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated Letta 0.16.8 block profile JSONL worker.")
    parser.add_argument("--configuration-json", required=True)
    arguments = parser.parse_args(argv)
    try:
        configuration = LettaAdapterConfigurationV1.model_validate_json(arguments.configuration_json, strict=True)
        if canonical_json_bytes(configuration).decode("utf-8") != arguments.configuration_json:
            raise ValueError("Letta configuration must be canonical")
        backend = build_official_letta_backend(configuration)
        serve_letta_worker_jsonl(LettaWorkerServiceV1(backend), input_stream=sys.stdin.buffer, output_stream=sys.stdout.buffer)
    except Exception:
        return 2
    return 0


__all__ = [
    "LettaBackendV1",
    "LettaBlockClientV1",
    "LettaBlockProfileBackendV1",
    "LettaDependencyUnavailable",
    "LettaWorkerProtocolError",
    "LettaWorkerServiceV1",
    "build_official_letta_backend",
    "inspect_local_letta_package",
    "serve_letta_worker_jsonl",
]


if __name__ == "__main__":
    raise SystemExit(main())
