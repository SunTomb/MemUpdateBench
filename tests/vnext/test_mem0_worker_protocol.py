from __future__ import annotations

import hashlib
import builtins
from io import BytesIO
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.external.bridge import (
    WorkerOperation,
    WorkerRequestV1,
    WorkerResponseStatus,
)
from mub.vnext.external.visibility import (
    ProviderEventInputV1,
    ProviderQueryInputV1,
)


MODEL_PROVENANCE_SHA256 = (
    "8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e"
)


def _worker_config(tmp_path: Path):
    from mub.vnext.external.providers.mem0 import (
        build_mem0_adapter_configuration,
        build_mem0_worker_configuration,
    )

    qwen = tmp_path / "Qwen2.5-7B-Instruct"
    minilm = tmp_path / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    qdrant = tmp_path / "qdrant"
    history = tmp_path / "history"
    for path in (qwen, minilm, qdrant, history):
        path.mkdir()
    public = build_mem0_adapter_configuration(
        run_id="task10-mem0-worker",
        model_provenance_ref=ArtifactRef(
            path="model_provenance.json",
            sha256=MODEL_PROVENANCE_SHA256,
            media_type="application/json",
            record_count=1,
        ),
    )
    return build_mem0_worker_configuration(
        public_configuration=public,
        qwen_local_path=qwen,
        minilm_local_path=minilm,
        qdrant_path=qdrant,
        history_directory=history,
    )


class FakeMem0Backend:
    def __init__(self) -> None:
        from mub.vnext.external.providers.mem0_protocol import (
            Mem0WorkerEntryV1,
            Mem0WorkerHealthV1,
            Mem0WorkerIngestResultV1,
            Mem0WorkerRetrievalResultV1,
        )

        self.health_result = Mem0WorkerHealthV1(
            package_name="mem0ai",
            package_version="2.0.17",
            collection_name="mub_mem0_fake",
            configuration_hash="1" * 64,
        )
        self.entry_type = Mem0WorkerEntryV1
        self.ingest_type = Mem0WorkerIngestResultV1
        self.retrieval_type = Mem0WorkerRetrievalResultV1
        self.entries: dict[str, list] = {}
        self.events: list[ProviderEventInputV1] = []
        self.queries: list[ProviderQueryInputV1] = []
        self.closed = False

    def health(self):
        return self.health_result

    def reset_namespace(self, namespace: str) -> None:
        self.entries[namespace] = []

    def ingest_event(
        self,
        event: ProviderEventInputV1,
        *,
        infer: bool,
    ):
        self.events.append(event)
        entry = self.entry_type(
            entry_id=f"entry-{event.event_id}",
            content=event.raw_text,
            created_at=event.logical_time,
            updated_at=event.logical_time,
            source_event_ids=(event.event_id,),
            native_metadata={"sequence_index": event.sequence_index},
        )
        self.entries.setdefault(event.runtime_namespace, []).append(entry)
        return self.ingest_type(
            event_id=event.event_id,
            effective_operation="add",
            affected_entry_ids=(entry.entry_id,),
        )

    def retrieve(self, query: ProviderQueryInputV1):
        self.queries.append(query)
        entries = tuple(
            self.entries.get(query.runtime_namespace, ())[: query.k]
        )
        return self.retrieval_type(
            query_id=query.query_id,
            entries=entries,
            scores=tuple(float(1.0 - index / 10) for index in range(len(entries))),
        )

    def export_entries(self, namespace: str):
        return tuple(self.entries.get(namespace, ()))

    def close(self) -> None:
        self.closed = True


def _request(
    request_id: str,
    operation: WorkerOperation,
    payload: dict | None = None,
) -> WorkerRequestV1:
    return WorkerRequestV1(
        request_id=request_id,
        operation=operation,
        payload={} if payload is None else payload,
    )


def test_mem0_worker_modules_import_no_optional_sdk() -> None:
    before = set(sys.modules)
    from mub.vnext.external.providers import mem0_protocol
    from mub.vnext.external.workers import mem0_worker

    assert mem0_protocol.Mem0WorkerEntryV1.__name__ == "Mem0WorkerEntryV1"
    assert mem0_worker.Mem0WorkerServiceV1.__name__ == "Mem0WorkerServiceV1"
    imported = set(sys.modules) - before
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imported
        for prefix in (
            "mem0",
            "qdrant_client",
            "sentence_transformers",
            "torch",
            "transformers",
        )
    )


def test_mem0_memory_config_uses_frozen_local_components(tmp_path: Path) -> None:
    from mub.vnext.external.workers.mem0_worker import (
        MEM0_EXTRACTION_INSTRUCTIONS,
        build_mem0_memory_config,
    )

    worker = _worker_config(tmp_path)
    config = build_mem0_memory_config(worker)

    assert config["llm"] == {
        "provider": "mub_local_qwen_v1",
        "config": {
            "model": worker.qwen_local_path,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
        },
    }
    assert config["embedder"] == {
        "provider": "huggingface",
        "config": {
            "model": worker.minilm_local_path,
            "embedding_dims": 384,
            "model_kwargs": {
                "device": "cpu",
                "local_files_only": True,
            },
        },
    }
    assert config["vector_store"] == {
        "provider": "qdrant",
        "config": {
            "collection_name": worker.public_configuration.collection_name,
            "embedding_model_dims": 384,
            "path": worker.qdrant_path,
            "on_disk": True,
        },
    }
    assert config["history_db_path"] == worker.history_db_path
    assert config["custom_instructions"] == MEM0_EXTRACTION_INSTRUCTIONS
    assert "default" not in str(config).casefold()


def test_mem0_worker_service_dispatches_visible_operations(tmp_path: Path) -> None:
    from mub.vnext.external.providers.mem0_protocol import (
        Mem0WorkerCloseResultV1,
        Mem0WorkerEntryListV1,
        Mem0WorkerIngestResultV1,
        Mem0WorkerRetrievalResultV1,
        Mem0WorkerResetResultV1,
    )
    from mub.vnext.external.workers.mem0_worker import Mem0WorkerServiceV1

    backend = FakeMem0Backend()
    service = Mem0WorkerServiceV1(backend)
    namespace = "task10-mem0-ns"

    health = service.handle(_request("health-1", WorkerOperation.HEALTH))
    assert health.status is WorkerResponseStatus.OK
    assert health.payload["package_version"] == "2.0.17"

    reset = service.handle(
        _request(
            "reset-1",
            WorkerOperation.RESET,
            {"runtime_namespace": namespace},
        )
    )
    assert Mem0WorkerResetResultV1.model_validate(reset.payload).success is True

    event = ProviderEventInputV1(
        event_id="event-1",
        sequence_index=0,
        logical_time="2026-01-01T00:00:00Z",
        raw_text="Add default|alice|city| with value \"Paris\".",
        runtime_namespace=namespace,
    )
    ingested = service.handle(
        _request(
            "ingest-1",
            WorkerOperation.INGEST_EVENT,
            {"event": event.model_dump(mode="json"), "probe_mode": "normal"},
        )
    )
    ingest_result = Mem0WorkerIngestResultV1.model_validate(ingested.payload)
    assert ingest_result.affected_entry_ids == ("entry-event-1",)
    assert backend.events == [event]

    exported = service.handle(
        _request(
            "export-1",
            WorkerOperation.EXPORT_ENTRIES,
            {"runtime_namespace": namespace},
        )
    )
    entries = Mem0WorkerEntryListV1.model_validate(exported.payload)
    assert tuple(entry.entry_id for entry in entries.entries) == (
        "entry-event-1",
    )

    query = ProviderQueryInputV1(
        query_id="query-1",
        query_text="Where does Alice live?",
        k=5,
        runtime_namespace=namespace,
    )
    retrieved = service.handle(
        _request(
            "retrieve-1",
            WorkerOperation.RETRIEVE,
            {"query": query.model_dump(mode="json")},
        )
    )
    retrieval = Mem0WorkerRetrievalResultV1.model_validate(retrieved.payload)
    assert retrieval.query_id == "query-1"
    assert retrieval.scores == (1.0,)
    assert backend.queries == [query]

    closed = service.handle(_request("close-1", WorkerOperation.CLOSE))
    assert closed.status is WorkerResponseStatus.OK
    assert Mem0WorkerCloseResultV1.model_validate(closed.payload).closed is True
    assert backend.closed is True


def test_mem0_worker_service_uses_infer_false_only_for_probe_sentinel(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.workers.mem0_worker import Mem0WorkerServiceV1

    backend = FakeMem0Backend()
    inferred: list[bool] = []
    original_ingest = backend.ingest_event

    def record_infer(event, *, infer: bool):
        inferred.append(infer)
        return original_ingest(event, infer=infer)

    backend.ingest_event = record_infer
    service = Mem0WorkerServiceV1(backend)
    event = ProviderEventInputV1(
        event_id="sentinel-1",
        sequence_index=0,
        logical_time=None,
        raw_text="MUB_RESET_SENTINEL_abc",
        runtime_namespace="reset-ns",
    )
    service.handle(
        _request(
            "normal",
            WorkerOperation.INGEST_EVENT,
            {"event": event.model_dump(mode="json"), "probe_mode": "normal"},
        )
    )
    service.handle(
        _request(
            "sentinel",
            WorkerOperation.INGEST_EVENT,
            {"event": event.model_dump(mode="json"), "probe_mode": "sentinel"},
        )
    )
    assert inferred == [True, False]


def test_mem0_worker_service_rejects_malformed_payload_without_leakage() -> None:
    from mub.vnext.external.workers.mem0_worker import Mem0WorkerServiceV1

    class FailingBackend(FakeMem0Backend):
        def reset_namespace(self, namespace: str) -> None:
            raise ValueError("client_secret=hunter2")

    service = Mem0WorkerServiceV1(FailingBackend())
    malformed = service.handle(
        _request(
            "bad-reset",
            WorkerOperation.RESET,
            {"runtime_namespace": "ns", "extra": "value"},
        )
    )
    assert malformed.status is WorkerResponseStatus.ERROR
    assert malformed.error_code == "invalid_request_payload"
    assert "hunter2" not in str(malformed)

    failed = service.handle(
        _request(
            "failed-reset",
            WorkerOperation.RESET,
            {"runtime_namespace": "ns"},
        )
    )
    assert failed.status is WorkerResponseStatus.ERROR
    assert failed.error_code == "worker_backend_error"
    assert "hunter2" not in str(failed)


def test_mem0_worker_service_marks_unsupported_exports() -> None:
    from mub.vnext.external.workers.mem0_worker import Mem0WorkerServiceV1

    service = Mem0WorkerServiceV1(FakeMem0Backend())
    for index, operation in enumerate(
        (
            WorkerOperation.EXPORT_RAW_STATE,
            WorkerOperation.EXPORT_VERSION_HISTORY,
        )
    ):
        response = service.handle(
            _request(
                f"unsupported-{index}",
                operation,
                {"runtime_namespace": "ns"},
            )
        )
        assert response.status is WorkerResponseStatus.ERROR
        assert response.error_code == "not_supported"


def test_mem0_worker_jsonl_loop_returns_canonical_typed_responses() -> None:
    from mub.vnext.external.bridge import WorkerResponseV1
    from mub.vnext.external.workers.mem0_worker import (
        Mem0WorkerServiceV1,
        serve_mem0_worker_jsonl,
    )
    from mub.vnext.io import canonical_json_bytes

    requests = (
        _request("health-jsonl", WorkerOperation.HEALTH),
        _request("close-jsonl", WorkerOperation.CLOSE),
    )
    source = BytesIO(
        b"".join(canonical_json_bytes(request) + b"\n" for request in requests)
    )
    destination = BytesIO()

    serve_mem0_worker_jsonl(
        Mem0WorkerServiceV1(FakeMem0Backend()),
        input_stream=source,
        output_stream=destination,
    )

    lines = destination.getvalue().splitlines(keepends=True)
    assert len(lines) == 2
    responses = tuple(
        WorkerResponseV1.model_validate_json(line[:-1]) for line in lines
    )
    assert tuple(response.request_id for response in responses) == (
        "health-jsonl",
        "close-jsonl",
    )
    assert all(
        canonical_json_bytes(response) + b"\n" == line
        for response, line in zip(responses, lines)
    )


def test_mem0_worker_jsonl_loop_rejects_noncanonical_input_without_echo() -> None:
    from mub.vnext.external.workers.mem0_worker import (
        Mem0WorkerProtocolError,
        Mem0WorkerServiceV1,
        serve_mem0_worker_jsonl,
    )

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    source = BytesIO(
        json.dumps(
            {
                "schema_version": "memupdatebench.external.worker_request.v1",
                "request_id": "bad-jsonl",
                "operation": "health",
                "payload": {"visible_text": secret},
            },
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    destination = BytesIO()

    with pytest.raises(Mem0WorkerProtocolError) as exc_info:
        serve_mem0_worker_jsonl(
            Mem0WorkerServiceV1(FakeMem0Backend()),
            input_stream=source,
            output_stream=destination,
        )
    assert secret not in str(exc_info.value)
    assert destination.getvalue() == b""


def test_mem0_worker_configuration_loader_requires_canonical_real_file(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.workers.mem0_worker import (
        load_mem0_worker_configuration,
    )
    from mub.vnext.io import canonical_json_bytes

    worker = _worker_config(tmp_path)
    configuration_path = tmp_path / "worker-configuration.json"
    configuration_path.write_bytes(canonical_json_bytes(worker))
    assert load_mem0_worker_configuration(configuration_path) == worker

    noncanonical_path = tmp_path / "noncanonical-worker-configuration.json"
    noncanonical_path.write_text(
        json.dumps(worker.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        load_mem0_worker_configuration(noncanonical_path)


def test_mem0_worker_module_cli_help_needs_no_optional_sdk() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "mub.vnext.external.workers.mem0_worker",
            "--help",
        ),
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--worker-configuration" in result.stdout
    assert "Traceback" not in result.stderr


def test_official_mem0_backend_uses_fixed_filtered_sdk_calls(tmp_path: Path) -> None:
    from mub.vnext.external.workers.mem0_worker import OfficialMem0BackendV1

    class FakeMemory:
        def __init__(self):
            self.rows: list[dict] = []
            self.add_calls: list[tuple] = []
            self.get_all_calls: list[dict] = []
            self.search_calls: list[tuple] = []
            self.delete_calls: list[dict] = []

        def delete_all(self, **kwargs):
            self.delete_calls.append(kwargs)
            self.rows = []

        def get_all(self, **kwargs):
            self.get_all_calls.append(kwargs)
            return {"results": list(self.rows)}

        def add(self, messages, **kwargs):
            self.add_calls.append((messages, kwargs))
            row = {
                "id": "native-entry-1",
                "memory": messages[0]["content"],
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "metadata": dict(kwargs["metadata"]),
            }
            self.rows = [row]
            return {
                "results": [
                    {"id": "deleted-entry"},
                    {"id": row["id"]},
                ]
            }

        def search(self, query, **kwargs):
            self.search_calls.append((query, kwargs))
            return {"results": [{**self.rows[0], "score": 0.75}]}

    backend = object.__new__(OfficialMem0BackendV1)
    backend._configuration = _worker_config(tmp_path)
    backend._memory = FakeMemory()
    namespace = "task10-official-sdk"
    backend.reset_namespace(namespace)
    event = ProviderEventInputV1(
        event_id="event-sdk-1",
        sequence_index=3,
        logical_time="2026-01-01T00:00:00Z",
        raw_text='Add default|alice|city| with value "Paris".',
        runtime_namespace=namespace,
    )
    ingested = backend.ingest_event(event, infer=True)
    query = ProviderQueryInputV1(
        query_id="query-sdk-1",
        query_text="Where does Alice live?",
        k=5,
        runtime_namespace=namespace,
    )
    retrieved = backend.retrieve(query)

    assert backend._memory.delete_calls == [{"user_id": namespace}]
    assert backend._memory.add_calls == [
        (
            [{"role": "user", "content": event.raw_text}],
            {
                "user_id": namespace,
                "metadata": {
                    "mub_source_event_id": "event-sdk-1",
                    "mub_sequence_index": 3,
                    "mub_logical_time": "2026-01-01T00:00:00Z",
                },
                "infer": True,
            },
        )
    ]
    assert all(
        call == {"filters": {"user_id": namespace}, "top_k": 10000}
        for call in backend._memory.get_all_calls
    )
    assert backend._memory.search_calls == [
        (
            query.query_text,
            {
                "top_k": 5,
                "filters": {"user_id": namespace},
                "threshold": 0.0,
                "rerank": False,
            },
        )
    ]
    assert ingested.affected_entry_ids == ("native-entry-1",)
    assert retrieved.scores == (0.75,)
    assert retrieved.entries[0].source_event_ids == ("event-sdk-1",)
    assert retrieved.entries[0].native_metadata == {
        "mub_sequence_index": 3,
        "mub_logical_time": "2026-01-01T00:00:00Z",
    }


def test_official_mem0_backend_disables_telemetry_before_sdk_import(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from mub.vnext.external.workers.mem0_worker import OfficialMem0BackendV1

    class DummyMemory:
        @classmethod
        def from_config(cls, config):
            return cls()

        def close(self):
            pass

    class DummyLlmFactory:
        @classmethod
        def register_provider(cls, name, path):
            pass

    mem0_module = types.ModuleType("mem0")
    mem0_module.Memory = DummyMemory
    utils_module = types.ModuleType("mem0.utils")
    factory_module = types.ModuleType("mem0.utils.factory")
    factory_module.LlmFactory = DummyLlmFactory
    monkeypatch.setitem(sys.modules, "mem0", mem0_module)
    monkeypatch.setitem(sys.modules, "mem0.utils", utils_module)
    monkeypatch.setitem(sys.modules, "mem0.utils.factory", factory_module)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: "2.0.17",
    )
    monkeypatch.setenv("MEM0_TELEMETRY", "true")
    observed: list[str | None] = []
    original_import = builtins.__import__

    def recording_import(name, *args, **kwargs):
        if name == "mem0":
            observed.append(os.environ.get("MEM0_TELEMETRY"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    OfficialMem0BackendV1(_worker_config(tmp_path))
    assert observed == ["false"]


def test_official_mem0_backend_fails_closed_without_dependency(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.workers.mem0_worker import (
        Mem0DependencyUnavailable,
        OfficialMem0BackendV1,
    )

    worker = _worker_config(tmp_path)
    with pytest.raises(Mem0DependencyUnavailable, match="unavailable") as exc_info:
        OfficialMem0BackendV1(worker)
    assert "ModuleNotFoundError" not in str(exc_info.value)
