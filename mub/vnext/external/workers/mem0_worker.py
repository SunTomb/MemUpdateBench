from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
from typing import Any, BinaryIO, Protocol

from mub.vnext.contracts.common import thaw_json
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.bridge import (
    WorkerOperation,
    WorkerRequestV1,
    WorkerResponseStatus,
    WorkerResponseV1,
)
from mub.vnext.external.providers.mem0 import (
    MEM0_EXTRACTION_INSTRUCTIONS,
    MEM0_INSTALLED_CONTENT_FILE_COUNT,
    MEM0_INSTALLED_CONTENT_SHA256,
    MEM0_PACKAGE_VERSION,
    Mem0WorkerConfigurationV1,
    compute_mem0_configuration_hash,
    validate_mem0_worker_configuration,
)
from mub.vnext.external.providers.mem0_protocol import (
    Mem0WorkerCloseResultV1,
    Mem0WorkerEntryListV1,
    Mem0WorkerEntryV1,
    Mem0WorkerHealthV1,
    Mem0WorkerIngestResultV1,
    Mem0WorkerResetResultV1,
    Mem0WorkerRetrievalResultV1,
)
from mub.vnext.external.visibility import (
    ProviderEventInputV1,
    ProviderQueryInputV1,
)
from mub.vnext.io import canonical_json_bytes

class Mem0DependencyUnavailable(RuntimeError):
    pass


class Mem0WorkerProtocolError(RuntimeError):
    pass


class _InvalidRequestPayload(ValueError):
    pass


class Mem0BackendV1(Protocol):
    def health(self) -> Mem0WorkerHealthV1: ...

    def reset_namespace(self, namespace: str) -> None: ...

    def ingest_event(
        self,
        event: ProviderEventInputV1,
        *,
        infer: bool,
    ) -> Mem0WorkerIngestResultV1: ...

    def retrieve(
        self,
        query: ProviderQueryInputV1,
    ) -> Mem0WorkerRetrievalResultV1: ...

    def export_entries(
        self,
        namespace: str,
    ) -> tuple[Mem0WorkerEntryV1, ...]: ...

    def close(self) -> None: ...


class LocalQwenMem0Llm:
    def __init__(self, config) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            raise Mem0DependencyUnavailable(
                "local Qwen runtime dependencies are unavailable"
            ) from None
        model_path = config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self._max_tokens = min(int(config.max_tokens or 512), 512)

    def generate_response(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> str:
        if tools:
            raise ValueError("local Qwen extraction does not support tools")
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(
            self._model.device
        )
        import torch

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self._max_tokens,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        return self._tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


def build_mem0_memory_config(
    worker_configuration: Mem0WorkerConfigurationV1,
) -> dict[str, Any]:
    worker = validate_mem0_worker_configuration(worker_configuration)
    public = worker.public_configuration
    return {
        "llm": {
            "provider": public.llm_provider,
            "config": {
                "model": worker.qwen_local_path,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
            },
        },
        "embedder": {
            "provider": public.embedding_provider,
            "config": {
                "model": worker.minilm_local_path,
                "embedding_dims": public.embedding_dims,
                "model_kwargs": {
                    "device": "cpu",
                    "local_files_only": True,
                },
            },
        },
        "vector_store": {
            "provider": public.vector_store_provider,
            "config": {
                "collection_name": public.collection_name,
                "embedding_model_dims": public.embedding_dims,
                "path": worker.qdrant_path,
                "on_disk": True,
            },
        },
        "history_db_path": worker.history_db_path,
        "custom_instructions": MEM0_EXTRACTION_INSTRUCTIONS,
    }


def load_mem0_worker_configuration(
    path: str | Path,
) -> Mem0WorkerConfigurationV1:
    input_path = Path(path)
    if not input_path.is_absolute():
        raise ValueError("Mem0 worker configuration path must be absolute")
    assert_no_reparse_components(input_path)
    resolved = input_path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("Mem0 worker configuration must be a real file")
    before = resolved.stat()
    if before.st_size > 1024 * 1024:
        raise ValueError("Mem0 worker configuration is too large")
    raw = resolved.read_bytes()
    assert_no_reparse_components(input_path)
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ValueError("Mem0 worker configuration changed while reading")
    try:
        configuration = Mem0WorkerConfigurationV1.model_validate_json(
            raw,
            strict=True,
        )
    except Exception:
        raise ValueError("Mem0 worker configuration is invalid") from None
    if canonical_json_bytes(configuration) != raw:
        raise ValueError("Mem0 worker configuration must be canonical")
    return validate_mem0_worker_configuration(configuration)


def _payload_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _InvalidRequestPayload("worker payload must be a mapping")
    return dict(value)


def _exact_payload(
    payload: object,
    expected_keys: set[str],
) -> dict[str, Any]:
    values = _payload_dict(payload)
    if set(values) != expected_keys:
        raise _InvalidRequestPayload("worker payload keys are invalid")
    return values


def _response_payload(model) -> dict[str, Any]:
    return model.model_dump(mode="json")


class Mem0WorkerServiceV1:
    def __init__(self, backend: Mem0BackendV1) -> None:
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
            raise ValueError("Mem0 worker requires exact WorkerRequestV1")
        if self._closed:
            return self._error(request.request_id, "worker_closed")
        try:
            return self._dispatch(request)
        except _InvalidRequestPayload:
            return self._error(request.request_id, "invalid_request_payload")
        except Exception:
            return self._error(request.request_id, "worker_backend_error")

    def _dispatch(self, request: WorkerRequestV1) -> WorkerResponseV1:
        operation = request.operation
        if operation is WorkerOperation.HEALTH:
            _exact_payload(request.payload, set())
            return self._ok(
                request.request_id,
                _response_payload(self._backend.health()),
            )
        if operation is WorkerOperation.RESET:
            payload = _exact_payload(
                request.payload,
                {"runtime_namespace"},
            )
            namespace = _namespace(payload["runtime_namespace"])
            self._backend.reset_namespace(namespace)
            result = Mem0WorkerResetResultV1(
                namespace=namespace,
                success=True,
            )
            return self._ok(request.request_id, _response_payload(result))
        if operation is WorkerOperation.INGEST_EVENT:
            payload = _exact_payload(
                request.payload,
                {"event", "probe_mode"},
            )
            try:
                event = ProviderEventInputV1.model_validate(
                    thaw_json(payload["event"]),
                    strict=True,
                )
            except Exception:
                raise _InvalidRequestPayload("event payload is invalid") from None
            probe_mode = payload["probe_mode"]
            if probe_mode not in {"normal", "sentinel"}:
                raise _InvalidRequestPayload("unknown probe mode")
            result = self._backend.ingest_event(
                event,
                infer=probe_mode == "normal",
            )
            return self._ok(request.request_id, _response_payload(result))
        if operation is WorkerOperation.RETRIEVE:
            payload = _exact_payload(request.payload, {"query"})
            try:
                query = ProviderQueryInputV1.model_validate(
                    thaw_json(payload["query"]),
                    strict=True,
                )
            except Exception:
                raise _InvalidRequestPayload("query payload is invalid") from None
            result = self._backend.retrieve(query)
            return self._ok(request.request_id, _response_payload(result))
        if operation is WorkerOperation.EXPORT_ENTRIES:
            payload = _exact_payload(
                request.payload,
                {"runtime_namespace"},
            )
            namespace = _namespace(payload["runtime_namespace"])
            result = Mem0WorkerEntryListV1(
                entries=self._backend.export_entries(namespace)
            )
            return self._ok(request.request_id, _response_payload(result))
        if operation in {
            WorkerOperation.EXPORT_RAW_STATE,
            WorkerOperation.EXPORT_VERSION_HISTORY,
        }:
            return self._error(request.request_id, "not_supported")
        if operation is WorkerOperation.CLOSE:
            _exact_payload(request.payload, set())
            self._backend.close()
            self._closed = True
            return self._ok(
                request.request_id,
                _response_payload(Mem0WorkerCloseResultV1()),
            )
        return self._error(request.request_id, "not_supported")


def serve_mem0_worker_jsonl(
    service: Mem0WorkerServiceV1,
    *,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    max_request_bytes: int = 16 * 1024 * 1024,
) -> None:
    if type(service) is not Mem0WorkerServiceV1:
        raise ValueError("Mem0 JSONL loop requires exact worker service")
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("Mem0 JSONL request limit must be positive")
    while True:
        line = input_stream.readline(max_request_bytes + 2)
        if line == b"":
            return
        if (
            len(line) > max_request_bytes + 1
            or not line.endswith(b"\n")
            or line in {b"\n", b"\r\n"}
        ):
            raise Mem0WorkerProtocolError("Mem0 worker request is invalid")
        raw = line[:-1]
        if raw.endswith(b"\r"):
            raise Mem0WorkerProtocolError("Mem0 worker request is invalid")
        try:
            request = WorkerRequestV1.model_validate_json(raw, strict=True)
        except Exception:
            raise Mem0WorkerProtocolError(
                "Mem0 worker request is invalid"
            ) from None
        if canonical_json_bytes(request) != raw:
            raise Mem0WorkerProtocolError("Mem0 worker request is noncanonical")
        response = service.handle(request)
        output_stream.write(canonical_json_bytes(response) + b"\n")
        output_stream.flush()
        if request.operation is WorkerOperation.CLOSE:
            return


def _namespace(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise _InvalidRequestPayload(
            "runtime namespace must be a nonblank string"
        )
    return value


def _installed_mem0_content_digest(distribution) -> tuple[str, int]:
    rows: list[tuple[str, bytes]] = []
    files = distribution.files
    if files is None:
        raise Mem0DependencyUnavailable(
            "official Mem0 OSS installation manifest is unavailable"
        )
    for item in files:
        name = str(item).replace("\\", "/")
        if name.endswith(".pyc") or name.endswith(".dist-info/RECORD"):
            continue
        if name.endswith((".dist-info/INSTALLER", ".dist-info/REQUESTED")):
            continue
        path = Path(distribution.locate_file(item))
        if not path.is_file() or path.is_symlink():
            raise Mem0DependencyUnavailable(
                "official Mem0 OSS installed content is unavailable"
            )
        rows.append((name, path.read_bytes()))
    digest = hashlib.sha256()
    for name, raw in sorted(rows):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).digest())
        digest.update(b"\n")
    return digest.hexdigest(), len(rows)


class OfficialMem0BackendV1:
    def __init__(self, worker_configuration: Mem0WorkerConfigurationV1) -> None:
        self._configuration = validate_mem0_worker_configuration(
            worker_configuration
        )
        os.environ["MEM0_TELEMETRY"] = "false"
        try:
            distribution = importlib.metadata.distribution("mem0ai")
            installed_version = distribution.version
            installed_digest, installed_file_count = (
                _installed_mem0_content_digest(distribution)
            )
        except Exception:
            raise Mem0DependencyUnavailable(
                "official Mem0 OSS dependency is unavailable"
            ) from None
        if (
            installed_version != MEM0_PACKAGE_VERSION
            or installed_digest != MEM0_INSTALLED_CONTENT_SHA256
            or installed_file_count != MEM0_INSTALLED_CONTENT_FILE_COUNT
        ):
            raise Mem0DependencyUnavailable(
                "official Mem0 OSS dependency installation is unavailable"
            )
        try:
            from mem0 import Memory
            from mem0.configs.base import MemoryConfig
            from mem0.utils.factory import LlmFactory
        except Exception:
            raise Mem0DependencyUnavailable(
                "official Mem0 OSS dependency is unavailable"
            ) from None
        LlmFactory.register_provider(
            "mub_local_qwen_v1",
            "mub.vnext.external.workers.mem0_worker.LocalQwenMem0Llm",
        )
        memory_config = build_mem0_memory_config(self._configuration)
        llm_config = memory_config["llm"]
        validation_config = {
            **memory_config,
            "llm": {**llm_config, "provider": "openai"},
        }
        validated_config = MemoryConfig(**validation_config)
        # Mem0 2.0.17 exposes LlmFactory.register_provider but its LlmConfig
        # validator hard-codes built-in names. Validate the complete config with
        # a built-in placeholder, then restore the registered provider exactly.
        validated_config.llm.provider = llm_config["provider"]
        validated_config.llm.config = llm_config["config"]
        self._memory = Memory(validated_config)

    def health(self) -> Mem0WorkerHealthV1:
        public = self._configuration.public_configuration
        return Mem0WorkerHealthV1(
            package_name="mem0ai",
            package_version=MEM0_PACKAGE_VERSION,
            collection_name=public.collection_name,
            configuration_hash=compute_mem0_configuration_hash(public),
        )

    def reset_namespace(self, namespace: str) -> None:
        self._memory.delete_all(user_id=namespace)

    def _export(self, namespace: str) -> tuple[Mem0WorkerEntryV1, ...]:
        result = self._memory.get_all(
            filters={"user_id": namespace},
            top_k=10000,
        )
        rows = result.get("results", ())
        if not isinstance(rows, list):
            raise ValueError("Mem0 get_all returned invalid results")
        return tuple(_normalize_entry(row) for row in rows)

    def ingest_event(
        self,
        event: ProviderEventInputV1,
        *,
        infer: bool,
    ) -> Mem0WorkerIngestResultV1:
        before_ids = {
            entry.entry_id
            for entry in self._export(event.runtime_namespace)
        }
        metadata = {
            "mub_source_event_id": event.event_id,
            "mub_sequence_index": event.sequence_index,
        }
        if event.logical_time is not None:
            metadata["mub_logical_time"] = event.logical_time
        result = self._memory.add(
            [{"role": "user", "content": event.raw_text}],
            user_id=event.runtime_namespace,
            metadata=metadata,
            infer=infer,
        )
        if not isinstance(result, Mapping):
            raise ValueError("Mem0 add returned invalid result")
        after = self._export(event.runtime_namespace)
        after_ids = {entry.entry_id for entry in after}
        delta_ids = tuple(
            entry.entry_id for entry in after if entry.entry_id not in before_ids
        )
        reported_ids = tuple(
            entry_id
            for entry_id in _reported_result_ids(result)
            if entry_id in after_ids
        )
        affected = tuple(dict.fromkeys((*reported_ids, *delta_ids)))
        return Mem0WorkerIngestResultV1(
            event_id=event.event_id,
            effective_operation=("add" if affected else "noop"),
            affected_entry_ids=affected,
        )

    def retrieve(
        self,
        query: ProviderQueryInputV1,
    ) -> Mem0WorkerRetrievalResultV1:
        result = self._memory.search(
            query.query_text,
            top_k=query.k,
            filters={"user_id": query.runtime_namespace},
            threshold=0.0,
            rerank=False,
        )
        rows = result.get("results", ())
        if not isinstance(rows, list):
            raise ValueError("Mem0 search returned invalid results")
        entries = tuple(_normalize_entry(row) for row in rows)
        scores = tuple(_score(row) for row in rows)
        return Mem0WorkerRetrievalResultV1(
            query_id=query.query_id,
            entries=entries,
            scores=scores,
        )

    def export_entries(
        self,
        namespace: str,
    ) -> tuple[Mem0WorkerEntryV1, ...]:
        return self._export(namespace)

    def close(self) -> None:
        return None


def _reported_result_ids(result: Mapping) -> tuple[str, ...]:
    rows = result.get("results", ())
    if not isinstance(rows, list):
        return ()
    values: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = row.get("id") or row.get("memory_id")
        if type(value) is str and value and value not in values:
            values.append(value)
    return tuple(values)


def _normalize_entry(value: object) -> Mem0WorkerEntryV1:
    if not isinstance(value, Mapping):
        raise ValueError("Mem0 entry is not a mapping")
    entry_id = value.get("id") or value.get("memory_id")
    content = value.get("memory") or value.get("text")
    if type(entry_id) is not str or not entry_id:
        raise ValueError("Mem0 entry ID is invalid")
    if type(content) is not str:
        raise ValueError("Mem0 entry content is invalid")
    raw_metadata = value.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    source_event_id = metadata.get("mub_source_event_id")
    source_ids = (
        (source_event_id,)
        if type(source_event_id) is str and source_event_id
        else ()
    )
    normalized_metadata: dict[str, Any] = {}
    for key in ("mub_sequence_index", "mub_logical_time"):
        item = metadata.get(key)
        if type(item) in {str, int}:
            normalized_metadata[key] = item
    return Mem0WorkerEntryV1(
        entry_id=entry_id,
        content=content,
        created_at=_optional_string(value.get("created_at")),
        updated_at=_optional_string(value.get("updated_at")),
        source_event_ids=source_ids,
        native_metadata=normalized_metadata,
    )


def _optional_string(value: object) -> str | None:
    return value if type(value) is str else None


def _score(value: object) -> float:
    if not isinstance(value, Mapping):
        raise ValueError("Mem0 retrieval result is not a mapping")
    score = value.get("score")
    if type(score) not in {int, float} or isinstance(score, bool):
        raise ValueError("Mem0 retrieval score is invalid")
    return float(score)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the isolated official Mem0 OSS JSONL worker."
    )
    parser.add_argument(
        "--worker-configuration",
        required=True,
        help="Absolute path to a canonical private worker configuration.",
    )
    arguments = parser.parse_args(argv)
    backend: OfficialMem0BackendV1 | None = None
    try:
        configuration = load_mem0_worker_configuration(
            arguments.worker_configuration
        )
        backend = OfficialMem0BackendV1(configuration)
        serve_mem0_worker_jsonl(
            Mem0WorkerServiceV1(backend),
            input_stream=sys.stdin.buffer,
            output_stream=sys.stdout.buffer,
        )
    except Exception:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        return 2
    try:
        backend.close()
    except Exception:
        return 2
    return 0


__all__ = [
    "MEM0_EXTRACTION_INSTRUCTIONS",
    "LocalQwenMem0Llm",
    "Mem0BackendV1",
    "Mem0DependencyUnavailable",
    "Mem0WorkerProtocolError",
    "Mem0WorkerServiceV1",
    "OfficialMem0BackendV1",
    "build_mem0_memory_config",
    "load_mem0_worker_configuration",
    "main",
    "serve_mem0_worker_jsonl",
]


if __name__ == "__main__":
    raise SystemExit(main())
