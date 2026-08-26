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
from mub.vnext.external.providers.langmem import (
    LANGMEM_INSTALLED_CONTENT_FILE_COUNT,
    LANGMEM_INSTALLED_CONTENT_SHA256,
    LANGMEM_PACKAGE_VERSION,
    LANGMEM_SOURCE_COMMIT,
    LangMemAdapterConfigurationV1,
    compute_langmem_configuration_hash,
)
from mub.vnext.external.providers.langmem_protocol import (
    LangMemWorkerCloseResultV1,
    LangMemWorkerEntryListV1,
    LangMemWorkerEntryV1,
    LangMemWorkerHealthV1,
    LangMemWorkerMutationResultV1,
    LangMemWorkerResetResultV1,
    LangMemWorkerRetrievalResultV1,
)
from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1
from mub.vnext.io import canonical_json_bytes

_MUTATION_PATTERN = re.compile(r"^(Add|Update) (.+) with value (.+)\.$")
_DELETE_PATTERN = re.compile(r"^Delete (.+)\.$")
_DELETE_METADATA_PATTERN = re.compile(r"\s+\[[^\[\]]+\]$")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class LangMemDependencyUnavailable(RuntimeError):
    pass


class LangMemWorkerProtocolError(RuntimeError):
    pass


class _InvalidRequestPayload(ValueError):
    pass


class _ProfileStore(Protocol):
    def get(self, namespace: tuple[str, ...], key: str): ...

    def put(self, namespace: tuple[str, ...], key: str, value: dict) -> None: ...

    def delete(self, namespace: tuple[str, ...], key: str) -> None: ...

    def search(self, namespace: tuple[str, ...], *, limit: int): ...


class LangMemBackendV1(Protocol):
    def health(self) -> LangMemWorkerHealthV1: ...

    def reset_namespace(self, namespace: str) -> None: ...

    def ingest_event(
        self, event: ProviderEventInputV1
    ) -> LangMemWorkerMutationResultV1: ...

    def retrieve(
        self, query: ProviderQueryInputV1
    ) -> LangMemWorkerRetrievalResultV1: ...

    def export_entries(self, namespace: str) -> tuple[LangMemWorkerEntryV1, ...]: ...

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


class LangMemWorkerServiceV1:
    def __init__(self, backend: LangMemBackendV1) -> None:
        self._backend = backend
        self._closed = False

    def _ok(self, request_id: str, payload: dict[str, Any]) -> WorkerResponseV1:
        return WorkerResponseV1(
            request_id=request_id,
            status=WorkerResponseStatus.OK,
            payload=payload,
        )

    def _error(self, request_id: str, code: str) -> WorkerResponseV1:
        return WorkerResponseV1(
            request_id=request_id,
            status=WorkerResponseStatus.ERROR,
            error_code=code,
        )

    def handle(self, request: WorkerRequestV1) -> WorkerResponseV1:
        if type(request) is not WorkerRequestV1:
            raise ValueError("LangMem worker requires exact WorkerRequestV1")
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
                _response_payload(LangMemWorkerResetResultV1(namespace=namespace)),
            )
        if request.operation is WorkerOperation.INGEST_EVENT:
            payload = _exact_payload(request.payload, {"event"})
            try:
                event = ProviderEventInputV1.model_validate(
                    thaw_json(payload["event"]), strict=True
                )
            except Exception:
                raise _InvalidRequestPayload("event payload is invalid") from None
            return self._ok(
                request.request_id,
                _response_payload(self._backend.ingest_event(event)),
            )
        if request.operation is WorkerOperation.RETRIEVE:
            payload = _exact_payload(request.payload, {"query"})
            try:
                query = ProviderQueryInputV1.model_validate(
                    thaw_json(payload["query"]), strict=True
                )
            except Exception:
                raise _InvalidRequestPayload("query payload is invalid") from None
            return self._ok(
                request.request_id,
                _response_payload(self._backend.retrieve(query)),
            )
        if request.operation is WorkerOperation.EXPORT_ENTRIES:
            payload = _exact_payload(request.payload, {"runtime_namespace"})
            namespace = _namespace(payload["runtime_namespace"])
            return self._ok(
                request.request_id,
                _response_payload(
                    LangMemWorkerEntryListV1(
                        entries=self._backend.export_entries(namespace)
                    )
                ),
            )
        if request.operation in {
            WorkerOperation.EXPORT_RAW_STATE,
            WorkerOperation.EXPORT_VERSION_HISTORY,
        }:
            return self._error(request.request_id, "not_supported")
        if request.operation is WorkerOperation.CLOSE:
            _exact_payload(request.payload, set())
            self._backend.close()
            self._closed = True
            return self._ok(
                request.request_id,
                _response_payload(LangMemWorkerCloseResultV1()),
            )
        return self._error(request.request_id, "not_supported")


class LangMemProfileBackendV1:
    def __init__(
        self,
        *,
        store: _ProfileStore,
        configuration: LangMemAdapterConfigurationV1,
    ) -> None:
        if type(configuration) is not LangMemAdapterConfigurationV1:
            raise ValueError("LangMem backend requires exact public configuration")
        self._store = store
        self._configuration = configuration

    def health(self) -> LangMemWorkerHealthV1:
        return LangMemWorkerHealthV1(
            package_name="langmem",
            package_version=LANGMEM_PACKAGE_VERSION,
            source_commit=LANGMEM_SOURCE_COMMIT,
            license_id="MIT",
            configuration_hash=compute_langmem_configuration_hash(
                self._configuration
            ),
        )

    def _namespace(self, runtime_namespace: str) -> tuple[str, ...]:
        return (self._configuration.namespace_root, runtime_namespace)

    def _entry_id(self, canonical_object_id: str) -> str:
        digest = hashlib.sha256(canonical_object_id.encode("utf-8")).hexdigest()
        return f"langmem-profile-{digest[:32]}"

    def _items(self, runtime_namespace: str) -> tuple[object, ...]:
        rows = self._store.search(self._namespace(runtime_namespace), limit=2)
        if not isinstance(rows, (list, tuple)):
            raise ValueError("LangMem native search returned an invalid result")
        if len(rows) > 1:
            raise ValueError("LangMem profile mode observed collection state")
        return tuple(rows)

    def reset_namespace(self, namespace: str) -> None:
        for item in self._items(namespace):
            key = getattr(item, "key", None)
            if type(key) is not str or not key:
                raise ValueError("LangMem native item key is invalid")
            self._store.delete(self._namespace(namespace), key)

    def _parse_event(
        self, event: ProviderEventInputV1
    ) -> tuple[str, str | None, object | None]:
        text = event.raw_text.strip()
        if text == "No memory object changes.":
            return "noop", None, None
        mutation = _MUTATION_PATTERN.fullmatch(text)
        if mutation is not None:
            canonical_object_id = self._single_object_id(mutation.group(2))
            try:
                value = json.loads(mutation.group(3))
            except (TypeError, ValueError):
                raise ValueError("LangMem visible value is not canonical JSON") from None
            if value is None:
                raise ValueError("LangMem visible value cannot be null")
            return mutation.group(1).lower(), canonical_object_id, value
        deletion = _DELETE_PATTERN.fullmatch(text)
        if deletion is None:
            raise ValueError("LangMem visible event has unsupported action surface")
        rendered = deletion.group(1)
        object_id = self._single_object_id(
            _DELETE_METADATA_PATTERN.sub("", rendered)
        )
        if "scope=object" not in rendered:
            raise ValueError("LangMem profile mode only supports object DELETE")
        return "delete", object_id, None

    @staticmethod
    def _single_object_id(rendered: str) -> str:
        values = tuple(part.strip() for part in rendered.split(","))
        if len(values) != 1 or not values[0]:
            raise ValueError("LangMem profile mode requires one target object")
        return values[0]

    def _entry(self, item: object) -> LangMemWorkerEntryV1:
        key = getattr(item, "key", None)
        raw = getattr(item, "value", None)
        if type(key) is not str or not key or not isinstance(raw, Mapping):
            raise ValueError("LangMem native item is invalid")
        canonical = raw.get("canonical_object_id")
        content = raw.get("content")
        source_event_id = raw.get("source_event_id")
        sequence_index = raw.get("sequence_index")
        if (
            type(canonical) is not str
            or type(content) is not str
            or type(source_event_id) is not str
            or type(sequence_index) is not int
            or sequence_index < 0
            or raw.get("value") is None
        ):
            raise ValueError("LangMem native item value is invalid")
        return LangMemWorkerEntryV1(
            entry_id=key,
            canonical_object_id=canonical,
            content=content,
            value=raw["value"],
            created_at=_optional_string(getattr(item, "created_at", None)),
            updated_at=_optional_string(getattr(item, "updated_at", None)),
            source_event_ids=(source_event_id,),
            sequence_index=sequence_index,
        )

    def ingest_event(
        self, event: ProviderEventInputV1
    ) -> LangMemWorkerMutationResultV1:
        operation, canonical_object_id, value = self._parse_event(event)
        if operation == "noop":
            return LangMemWorkerMutationResultV1(
                event_id=event.event_id,
                effective_operation="noop",
            )
        if canonical_object_id is None:
            raise ValueError("LangMem mutation object ID is missing")
        entry_id = self._entry_id(canonical_object_id)
        existing = self._store.get(self._namespace(event.runtime_namespace), entry_id)
        if operation == "add":
            if existing is not None:
                return LangMemWorkerMutationResultV1(
                    event_id=event.event_id,
                    effective_operation="noop",
                )
            if self._items(event.runtime_namespace):
                raise ValueError("LangMem profile mode rejects collection inserts")
            self._store.put(
                self._namespace(event.runtime_namespace),
                entry_id,
                self._stored_value(event, canonical_object_id, value),
            )
        elif operation == "update":
            if existing is None:
                return LangMemWorkerMutationResultV1(
                    event_id=event.event_id,
                    effective_operation="noop",
                )
            self._store.put(
                self._namespace(event.runtime_namespace),
                entry_id,
                self._stored_value(event, canonical_object_id, value),
            )
        elif operation == "delete":
            if existing is None:
                return LangMemWorkerMutationResultV1(
                    event_id=event.event_id,
                    effective_operation="noop",
                )
            self._store.delete(self._namespace(event.runtime_namespace), entry_id)
        else:
            raise ValueError("LangMem operation is invalid")
        return LangMemWorkerMutationResultV1(
            event_id=event.event_id,
            effective_operation=operation,
            entry_id=entry_id,
        )

    @staticmethod
    def _stored_value(
        event: ProviderEventInputV1,
        canonical_object_id: str,
        value: object,
    ) -> dict[str, object]:
        return {
            "canonical_object_id": canonical_object_id,
            "content": event.raw_text,
            "value": value,
            "source_event_id": event.event_id,
            "sequence_index": event.sequence_index,
        }

    def export_entries(self, namespace: str) -> tuple[LangMemWorkerEntryV1, ...]:
        return tuple(self._entry(item) for item in self._items(namespace))

    def retrieve(
        self, query: ProviderQueryInputV1
    ) -> LangMemWorkerRetrievalResultV1:
        entries = self.export_entries(query.runtime_namespace)
        ranked = sorted(
            (
                (_deterministic_score(query.query_text, entry), entry)
                for entry in entries
            ),
            key=lambda row: (-row[0], row[1].entry_id),
        )[: query.k]
        return LangMemWorkerRetrievalResultV1(
            query_id=query.query_id,
            entries=tuple(entry for _, entry in ranked),
            scores=tuple(score for score, _ in ranked),
        )

    def close(self) -> None:
        return None


def _optional_string(value: object) -> str | None:
    return value if type(value) is str else None


def _deterministic_score(query_text: str, entry: LangMemWorkerEntryV1) -> float:
    query_tokens = set(_TOKEN_PATTERN.findall(query_text.casefold()))
    entry_tokens = set(
        _TOKEN_PATTERN.findall(
            f"{entry.canonical_object_id} {entry.content} "
            f"{json.dumps(entry.value, ensure_ascii=False, sort_keys=True)}".casefold()
        )
    )
    if not query_tokens or not entry_tokens:
        return 0.0
    return len(query_tokens & entry_tokens) / len(query_tokens | entry_tokens)


def _installed_langmem_content_digest(distribution) -> tuple[str, int]:
    rows: list[tuple[str, bytes]] = []
    if distribution.files is None:
        raise LangMemDependencyUnavailable("LangMem installation manifest is unavailable")
    for item in distribution.files:
        name = str(item).replace(chr(92), "/")
        if name.endswith((".pyc", ".dist-info/RECORD")):
            continue
        if name.endswith((".dist-info/INSTALLER", ".dist-info/REQUESTED")):
            continue
        path = distribution.locate_file(item)
        if not path.is_file() or path.is_symlink():
            raise LangMemDependencyUnavailable("LangMem installed content is unavailable")
        rows.append((name, path.read_bytes()))
    digest = hashlib.sha256()
    for name, raw in sorted(rows):
        digest.update(name.encode("utf-8"))
        digest.update(bytes([0]))
        digest.update(hashlib.sha256(raw).digest())
        digest.update(bytes([10]))
    return digest.hexdigest(), len(rows)


class OfficialLangMemBackendV1(LangMemProfileBackendV1):
    def __init__(self, configuration: LangMemAdapterConfigurationV1) -> None:
        try:
            distribution = importlib.metadata.distribution("langmem")
            digest, count = _installed_langmem_content_digest(distribution)
            license_text = distribution.metadata.get("License")
            if (
                distribution.version != LANGMEM_PACKAGE_VERSION
                or digest != LANGMEM_INSTALLED_CONTENT_SHA256
                or count != LANGMEM_INSTALLED_CONTENT_FILE_COUNT
                or not isinstance(license_text, str)
                or "MIT License" not in license_text
            ):
                raise LangMemDependencyUnavailable(
                    "LangMem installation does not match the frozen identity"
                )
            from langgraph.store.memory import InMemoryStore
        except LangMemDependencyUnavailable:
            raise
        except Exception:
            raise LangMemDependencyUnavailable("LangMem dependency is unavailable") from None
        super().__init__(store=InMemoryStore(), configuration=configuration)


def serve_langmem_worker_jsonl(
    service: LangMemWorkerServiceV1,
    *,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    max_request_bytes: int = 16 * 1024 * 1024,
) -> None:
    if type(service) is not LangMemWorkerServiceV1:
        raise ValueError("LangMem JSONL loop requires exact worker service")
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("LangMem JSONL request limit must be positive")
    while True:
        line = input_stream.readline(max_request_bytes + 2)
        if line == b"":
            return
        if (
            len(line) > max_request_bytes + 1
            or not line.endswith(b"\n")
            or line in {b"\n", b"\r\n"}
        ):
            raise LangMemWorkerProtocolError("LangMem worker request is invalid")
        raw = line[:-1]
        if raw.endswith(b"\r"):
            raise LangMemWorkerProtocolError("LangMem worker request is invalid")
        try:
            request = WorkerRequestV1.model_validate_json(raw, strict=True)
        except Exception:
            raise LangMemWorkerProtocolError("LangMem worker request is invalid") from None
        if canonical_json_bytes(request) != raw:
            raise LangMemWorkerProtocolError("LangMem worker request is noncanonical")
        response = service.handle(request)
        output_stream.write(canonical_json_bytes(response) + b"\n")
        output_stream.flush()
        if request.operation is WorkerOperation.CLOSE:
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated LangMem 0.0.30 profile JSONL worker."
    )
    parser.add_argument("--configuration-json", required=True)
    arguments = parser.parse_args(argv)
    try:
        configuration = LangMemAdapterConfigurationV1.model_validate_json(
            arguments.configuration_json,
            strict=True,
        )
        if canonical_json_bytes(configuration).decode("utf-8") != arguments.configuration_json:
            raise ValueError("LangMem configuration must be canonical")
        backend = OfficialLangMemBackendV1(configuration)
        serve_langmem_worker_jsonl(
            LangMemWorkerServiceV1(backend),
            input_stream=sys.stdin.buffer,
            output_stream=sys.stdout.buffer,
        )
    except Exception:
        return 2
    return 0


__all__ = [
    "LangMemBackendV1",
    "LangMemDependencyUnavailable",
    "LangMemProfileBackendV1",
    "LangMemWorkerProtocolError",
    "LangMemWorkerServiceV1",
    "OfficialLangMemBackendV1",
    "serve_langmem_worker_jsonl",
]


if __name__ == "__main__":
    raise SystemExit(main())
