from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN, StrictBool
from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, StrictIdentifier, object_identity, thaw_json
from mub.vnext.contracts.v3.native_multi_object import (
    NativeMultiObjectMutationRequestV3,
    NativeObjectMutationV3,
    NativeResetResultV3,
    NativeRetrievalRequestV3,
)
from mub.vnext.external.providers.qdrant_native_multi_object import (
    QDRANT_NATIVE_PROVIDER_VERSION,
    QdrantNativeMultiObjectConfigurationV1,
    QdrantNativeMultiObjectError,
    QdrantNativeMultiObjectProvider,
    QdrantNativeMultiObjectUnavailable,
    _hash_raw,
    _native_provenance,
    build_qdrant_native_multi_object_configuration,
    qdrant_client_availability,
)


QDRANT_NATIVE_MANAGER_VERSION = "memupdatebench-qdrant-native-manager-v1"
QDRANT_NATIVE_MANAGER_CONTRACT_VERSION = "memupdatebench.external.qdrant-native-manager.v1"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"


def _canonical_json(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        thaw_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _binding_payload(
    *,
    provider_configuration: QdrantNativeMultiObjectConfigurationV1,
    source_revision: str,
    runtime_revision: str,
    source_hash: str,
    runtime_hash: str,
) -> dict[str, Any]:
    return {
        "provider_configuration": provider_configuration.model_dump(mode="json", exclude_none=False),
        "source_revision": source_revision,
        "runtime_revision": runtime_revision,
        "source_hash": source_hash,
        "runtime_hash": runtime_hash,
    }


class QdrantNativeManagerConfigurationV1(ImmutableContractModel):
    schema_version: Literal[QDRANT_NATIVE_MANAGER_CONTRACT_VERSION] = QDRANT_NATIVE_MANAGER_CONTRACT_VERSION
    manager_version: Literal[QDRANT_NATIVE_MANAGER_VERSION] = QDRANT_NATIVE_MANAGER_VERSION
    provider_name: Literal["qdrant"] = "qdrant"
    provider_version: StrictIdentifier = QDRANT_NATIVE_PROVIDER_VERSION
    provider_configuration: QdrantNativeMultiObjectConfigurationV1
    source_revision: str = Field(pattern=_GIT_SHA_PATTERN, strict=True)
    runtime_revision: str = Field(pattern=_GIT_SHA_PATTERN, strict=True)
    source_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    runtime_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    configuration_hash: str = Field(pattern=SHA256_PATTERN, strict=True)

    @model_validator(mode="after")
    def _binding_is_self_consistent(self) -> Self:
        if self.source_revision == "0" * 40 or self.runtime_revision == "0" * 40:
            raise ValueError("Qdrant manager revisions cannot be placeholders")
        if self.source_hash == "0" * 64 or self.runtime_hash == "0" * 64:
            raise ValueError("Qdrant manager hashes cannot be placeholders")
        expected = _sha256_json(
            _binding_payload(
                provider_configuration=self.provider_configuration,
                source_revision=self.source_revision,
                runtime_revision=self.runtime_revision,
                source_hash=self.source_hash,
                runtime_hash=self.runtime_hash,
            )
        )
        if self.configuration_hash != expected:
            raise ValueError("Qdrant manager configuration hash does not match binding")
        return self

    @property
    def run_id(self) -> str:
        return self.provider_configuration.run_id

    @property
    def storage_path(self) -> str:
        return self.provider_configuration.storage_path

    @property
    def collection_name(self) -> str:
        return self.provider_configuration.collection_name


def build_qdrant_native_manager_configuration(
    *,
    run_id: str,
    path: str | Path,
    source_revision: str,
    runtime_revision: str,
    source_hash: str,
    runtime_hash: str,
) -> QdrantNativeManagerConfigurationV1:
    provider_configuration = build_qdrant_native_multi_object_configuration(run_id=run_id, path=path)
    binding = _binding_payload(
        provider_configuration=provider_configuration,
        source_revision=source_revision,
        runtime_revision=runtime_revision,
        source_hash=source_hash,
        runtime_hash=runtime_hash,
    )
    return QdrantNativeManagerConfigurationV1(
        provider_configuration=provider_configuration,
        source_revision=source_revision,
        runtime_revision=runtime_revision,
        source_hash=source_hash,
        runtime_hash=runtime_hash,
        configuration_hash=_sha256_json(binding),
    )


def build_qdrant_native_manager_factory(
    *,
    base_path: str | Path,
    source_revision: str,
    runtime_revision: str,
    source_hash: str,
    runtime_hash: str,
    provider_factory: Callable[..., QdrantNativeMultiObjectProvider] | None = None,
) -> Callable[[Any, Mapping[str, Any]], "QdrantNativeExternalManagerV1"]:
    """Build a production-bound dispatch seam without consulting task gold or targets."""
    base = Path(base_path)
    if not base.is_absolute():
        raise ValueError("Qdrant manager base path must be absolute")

    def factory(task: Any, cell: Mapping[str, Any] | None = None) -> "QdrantNativeExternalManagerV1":
        del cell
        task_id = getattr(task, "task_id", None)
        if type(task_id) is not str or not task_id.strip():
            raise ValueError("Qdrant manager dispatch requires a nonblank task_id")
        configuration = build_qdrant_native_manager_configuration(
            run_id=f"main-track-{task_id}",
            path=base,
            source_revision=source_revision,
            runtime_revision=runtime_revision,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )
        return QdrantNativeExternalManagerV1(
            configuration=configuration,
            provider_factory=provider_factory,
        )

    factory.production_bound = provider_factory is None
    return factory


class QdrantNativeExternalManagerV1:
    """ExternalManager seam backed only by direct Qdrant CRUD and retrieval.

    This adapter has no target selector, answer model, local state, cache, reranker,
    deduplicator, filter, or sort layer. The provider owns all authoritative state.
    """

    production_bound = True

    def __init__(
        self,
        *,
        configuration: QdrantNativeManagerConfigurationV1,
        provider_factory: Callable[..., QdrantNativeMultiObjectProvider] | None = None,
    ) -> None:
        if type(configuration) is not QdrantNativeManagerConfigurationV1:
            raise ValueError("Qdrant manager requires exact configuration")
        self.production_bound = provider_factory is None
        self.configuration = configuration
        self._runtime_namespace = f"qdrant_native_{configuration.run_id}"
        if provider_factory is None:
            availability = qdrant_client_availability()
            if availability.status != "AVAILABLE":
                raise QdrantNativeMultiObjectUnavailable(availability.blocker or "qdrant_client_unavailable")
            provider_factory = QdrantNativeMultiObjectProvider
        self._provider = provider_factory(
            configuration=configuration.provider_configuration,
            runtime_namespace=self._runtime_namespace,
        )
        self._closed = False

    @property
    def point_id_derivation(self) -> str:
        return self.configuration.provider_configuration.point_id_derivation

    def point_id(self, object_key: FrozenMemoryObjectKey) -> str:
        return self._provider.point_id(self._runtime_namespace, object_key)

    @property
    def provider_witness_hash(self) -> str:
        witnesses = []
        for witness in self._provider.observation_witnesses:
            witnesses.append(
                {
                    "method": witness.method,
                    "call_index": witness.call_index,
                    "raw_request_hash": witness.raw_request_hash,
                    "raw_response_hash": witness.raw_response_hash,
                    "success": witness.success,
                }
            )
        return _sha256_json(witnesses)

    @property
    def provider_witnesses(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                "method": witness.method,
                "call_index": witness.call_index,
                "raw_request_hash": witness.raw_request_hash,
                "raw_response_hash": witness.raw_response_hash,
                "success": witness.success,
            }
            for witness in self._provider.observation_witnesses
        )

    def capabilities(self):
        return self._provider.capabilities()

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "manager_id": "qdrant_native",
            "adapter_id": "qdrant_native",
            "adapter_version": QDRANT_NATIVE_MANAGER_VERSION,
            "system_name": "qdrant",
            "system_version": self.configuration.provider_version,
            "backend": "qdrant_native_direct_provider",
            "provider_configuration": self.configuration.provider_configuration.model_dump(mode="json"),
            "configuration_hash": self.configuration.configuration_hash,
            "source_revision": self.configuration.source_revision,
            "runtime_revision": self.configuration.runtime_revision,
            "source_hash": self.configuration.source_hash,
            "runtime_hash": self.configuration.runtime_hash,
            "runtime_namespace": self._runtime_namespace,
            "provider_witness_hash": self.provider_witness_hash,
            "direct_provider_crud": True,
            "provider_owned_collection": True,
            "uses_local_authoritative_store": False,
            "uses_local_reranking": False,
            "uses_local_deduplication": False,
            "uses_local_filtering": False,
            "hidden_target_injection": False,
            "supports_native_answer": False,
            "answer_layer": "separate",
            "provenance": "provider_owned_crud_and_retrieval; visible_event_inputs_only",
        }

    def reset(self, task: Any) -> None:
        del task
        result = self._provider.reset(self._runtime_namespace)
        if type(result) is not NativeResetResultV3:
            raise QdrantNativeMultiObjectError("Qdrant manager reset returned an invalid result")
        if result.namespace != self._runtime_namespace:
            raise QdrantNativeMultiObjectError("Qdrant manager reset namespace is inconsistent")
        if not result.success:
            raise QdrantNativeMultiObjectError(
                f"Qdrant manager reset failed: {result.error}"
            )

    @staticmethod
    def _event_id(event: Any) -> str:
        event_id = getattr(event, "event_id", None)
        if type(event_id) is not str or not event_id.strip():
            raise ValueError("Qdrant manager events require a nonblank event_id")
        return event_id

    @staticmethod
    def _sequence_index(event: Any) -> int:
        sequence_index = getattr(event, "sequence_index", None)
        if type(sequence_index) is not int or isinstance(sequence_index, bool) or sequence_index < 0:
            raise ValueError("Qdrant manager events require a nonnegative sequence_index")
        return sequence_index

    def ingest(self, event: Any, *, operation: str, value: Any, object_key: Any) -> Mapping[str, Any]:
        if type(object_key) is not FrozenMemoryObjectKey:
            object_key = FrozenMemoryObjectKey.model_validate(object_key, strict=True)
        operation_enum = Operation(operation.upper())
        event_id = self._event_id(event)
        request = NativeMultiObjectMutationRequestV3(
            request_id=f"qdrant-native-{event_id}",
            event_id=event_id,
            sequence_index=self._sequence_index(event),
            operation=operation_enum,
            objects=()
            if operation_enum is Operation.NOOP
            else (
                NativeObjectMutationV3(
                    object_key=object_key,
                )
                if operation_enum is Operation.DELETE
                else NativeObjectMutationV3(object_key=object_key, value=value),
            ),
            provenance=_native_provenance(),
        )
        result = self._provider.mutate(request)
        affected = [
            outcome.provider_entry_id
            for outcome in result.outcomes
            if outcome.provider_entry_id is not None
        ]
        effective = operation_enum.value.lower()
        if result.outcomes and all(outcome.status.value == "no_effect" for outcome in result.outcomes):
            effective = "noop"
        return {"effective_operation": effective, "affected_entry_ids": affected}

    def _scroll(self) -> list[Any]:
        points: list[Any] = []
        offset = None
        seen_cursors: set[tuple[str, str]] = set()
        while True:
            kwargs = {
                "collection_name": self._provider.collection_name,
                "scroll_filter": self._provider._filter_for_scope(
                    self._runtime_namespace, "namespace"
                ),
                "limit": 10000,
                "with_payload": True,
                "with_vectors": False,
            }
            if offset is not None:
                kwargs["offset"] = offset
            response = self._provider._provider_call("scroll", **kwargs)
            page, next_offset = self._provider._scroll_points(response)
            points.extend(page)
            if next_offset is None:
                return points
            cursor_key = (type(next_offset).__name__, _hash_raw(next_offset))
            if cursor_key in seen_cursors:
                raise QdrantNativeMultiObjectError(
                    "Qdrant scroll continuation cursor repeated"
                )
            seen_cursors.add(cursor_key)
            offset = next_offset

    def export_entries(self) -> Sequence[Mapping[str, Any]]:
        result: list[Mapping[str, Any]] = []
        for rank, point in enumerate(self._scroll()):
            point_id = str(self._provider._point_field(point, "id"))
            payload = self._provider._point_field(point, "payload")
            key, source_event_ids = self._provider._payload_key(
                payload,
                expected_runtime_namespace=self._runtime_namespace,
                expected_point_id=point_id,
            )
            result.append(
                {
                    "entry_id": point_id,
                    "object_key": key.model_dump(mode="json"),
                    "value": thaw_json(payload["value"]),
                    "content": payload["content"],
                    "source_event_ids": list(source_event_ids),
                    "score": 0.0,
                    "rank": rank + 1,
                    "version_metadata": thaw_json(payload["version_metadata"]),
                }
            )
        return result

    def retrieve(self, query: Any) -> Mapping[str, Any]:
        query_id = getattr(query, "query_id", None)
        query_text = getattr(query, "text", None)
        if type(query_id) is not str or not query_id.strip():
            raise ValueError("Qdrant retrieval requires a visible query_id")
        if type(query_text) is not str or not query_text.strip():
            raise ValueError("Qdrant retrieval requires visible query text")
        trace = self._provider.retrieve(
            NativeRetrievalRequestV3(
                query_id=query_id,
                query_text=query_text,
                k=16,
                namespace=self._runtime_namespace,
            )
        )
        entries = []
        for entry in trace.entries:
            payload = thaw_json(entry.provider_payload)
            entries.append(
                {
                    "entry_id": entry.provider_entry_id,
                    "object_key": entry.object_key.model_dump(mode="json"),
                    "value": thaw_json(payload["value"]),
                    "content": payload["content"],
                    "source_event_ids": list(entry.source_event_ids),
                    "score": entry.score,
                    "rank": entry.rank,
                    "version_metadata": thaw_json(payload["version_metadata"]),
                }
            )
        return {
            "entries": entries,
            "context_order": "provider_order",
            "version_metadata": {
                "provider": "qdrant",
                "provider_order_preserved": True,
                "provider_scores_preserved": True,
                "provider_entry_ids_preserved": True,
                "provider_payload_preserved": True,
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        self._provider.close()
        self._closed = True


QdrantNativeExternalManager = QdrantNativeExternalManagerV1
QdrantNativeManagerConfiguration = QdrantNativeManagerConfigurationV1


__all__ = [
    "QDRANT_NATIVE_MANAGER_CONTRACT_VERSION",
    "QDRANT_NATIVE_MANAGER_VERSION",
    "QdrantNativeExternalManager",
    "QdrantNativeExternalManagerV1",
    "QdrantNativeManagerConfiguration",
    "QdrantNativeManagerConfigurationV1",
    "build_qdrant_native_manager_configuration",
]
