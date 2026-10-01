from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Literal
import uuid

from pydantic import AliasChoices, Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, StrictBool, thaw_json
from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.common import (
    FrozenMemoryObjectKey,
    StrictIdentifier,
    object_identity,
)
from mub.vnext.contracts.v3.native_multi_object import (
    NativeAtomicityV3,
    NativeCapabilityDeclarationV3,
    NativeDataProvenanceV3,
    NativeMultiObjectMutationRequestV3,
    NativeMultiObjectMutationResultV3,
    NativeObjectOutcomeV3,
    NativeProvenanceV3,
    NativeResetResultV3,
    NativeRetrievalEntryV3,
    NativeRetrievalRequestV3,
    NativeRetrievalStatusV3,
    NativeRetrievalTraceV3,
)
from mub.vnext.external.native_multi_object import check_forbidden_metadata


QDRANT_NATIVE_PROVIDER_VERSION = "memupdatebench-qdrant-native-multi-object-v1"
QDRANT_NATIVE_CONTRACT_VERSION = "memupdatebench.external.qdrant-native-multi-object.v1"
QDRANT_VECTOR_SIZE = 384
QDRANT_DISTANCE = "Cosine"
QDRANT_POINT_ID_DERIVATION = "uuid5-adapter-v1"
QDRANT_POINT_NAMESPACE = uuid.UUID("4b4fdd45-c4d8-4f70-9cf8-6a90d43b1c8a")
_REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "runtime_namespace",
        "object_id",
        "namespace",
        "entity",
        "attribute",
        "subkey",
        "object_type",
        "value",
        "content",
        "source_event_ids",
        "sequence_index",
        "version",
        "version_metadata",
    }
)


class QdrantNativeMultiObjectError(RuntimeError):
    """Base error for the optional direct Qdrant provider."""


class QdrantNativeMultiObjectUnavailable(QdrantNativeMultiObjectError):
    status = "UNAVAILABLE"

    def __init__(self, blocker: str, detail: str | None = None) -> None:
        self.blocker = blocker
        self.detail = detail
        message = blocker if detail is None else f"{blocker}: {detail}"
        super().__init__(message)


class QdrantAvailabilityV1(ImmutableContractModel):
    schema_version: Literal[QDRANT_NATIVE_CONTRACT_VERSION] = QDRANT_NATIVE_CONTRACT_VERSION
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    blocker: StrictIdentifier | None = None
    package_name: Literal["qdrant-client"] = "qdrant-client"
    package_version: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.status == "AVAILABLE" and self.blocker is not None:
            raise ValueError("available Qdrant dependency cannot carry a blocker")
        if self.status == "UNAVAILABLE" and self.blocker is None:
            raise ValueError("unavailable Qdrant dependency requires a blocker")
        return self


class QdrantNativeMultiObjectConfigurationV1(ImmutableContractModel):
    schema_version: Literal[QDRANT_NATIVE_CONTRACT_VERSION] = QDRANT_NATIVE_CONTRACT_VERSION
    run_id: StrictIdentifier
    collection_name: StrictIdentifier
    storage_path: str = Field(
        validation_alias=AliasChoices("storage_path", "path", "qdrant_path"),
        strict=True,
        min_length=1,
    )
    vector_size: Literal[QDRANT_VECTOR_SIZE] = QDRANT_VECTOR_SIZE
    distance: Literal[QDRANT_DISTANCE] = QDRANT_DISTANCE
    point_id_derivation: Literal[QDRANT_POINT_ID_DERIVATION] = QDRANT_POINT_ID_DERIVATION

    @model_validator(mode="after")
    def _isolated(self) -> Self:
        path = Path(self.storage_path)
        if not path.is_absolute():
            raise ValueError("Qdrant storage path must be absolute")
        if self.collection_name.casefold() in {"default", "shared"}:
            raise ValueError("Qdrant collection must be run-isolated")
        if not self.collection_name.startswith("mub_qdrant_"):
            raise ValueError("Qdrant collection name must use the run-isolated prefix")
        path_text = self.storage_path.casefold().replace("\\", "/")
        if any(part in {"default", "shared"} for part in path_text.split("/")):
            raise ValueError("Qdrant storage path must not be default or shared")
        return self


class QdrantProviderObservationV1(ImmutableContractModel):
    """Hash-only witness of a direct provider request and response."""

    method: StrictIdentifier
    call_index: int = Field(strict=True, ge=0)
    raw_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    raw_response_hash: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    success: StrictBool = True
    provider_operation_id: StrictIdentifier | None = None

    @property
    def request_hash(self) -> str:
        return self.raw_request_hash

    @property
    def response_hash(self) -> str:
        return self.raw_response_hash


class QdrantDirectVerificationV1(ImmutableContractModel):
    schema_version: Literal[QDRANT_NATIVE_CONTRACT_VERSION] = QDRANT_NATIVE_CONTRACT_VERSION
    status: Literal["READY", "BLOCKED", "UNAVAILABLE"]
    direct_observed: StrictBool
    collection_schema_verified: StrictBool
    required_methods: tuple[StrictIdentifier, ...]
    missing_methods: tuple[StrictIdentifier, ...] = ()
    blocker: StrictIdentifier | None = None
    evidence_anchor: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", strict=True)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.status == "READY":
            if not self.direct_observed or not self.collection_schema_verified:
                raise ValueError("READY verification requires direct API and schema observation")
            if self.blocker is not None or self.evidence_anchor is None:
                raise ValueError("READY verification requires a SHA-256 evidence anchor")
        elif self.blocker is None:
            raise ValueError("blocked verification requires blocker")
        return self


class QdrantScopedDeleteResultV1(ImmutableContractModel):
    schema_version: Literal[QDRANT_NATIVE_CONTRACT_VERSION] = QDRANT_NATIVE_CONTRACT_VERSION
    runtime_namespace: StrictIdentifier
    scope: Literal["object", "attribute", "entity", "namespace"]
    provider_operation_id: StrictIdentifier | None = None
    observation_index: int = Field(strict=True, ge=0)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return repr(value)


def _hash_raw(value: Any) -> str:
    try:
        raw = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raw = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def _configuration_collection(run_id: str) -> str:
    digest = hashlib.sha256(f"qdrant-native-v1\x1f{run_id}".encode("utf-8")).hexdigest()
    return f"mub_qdrant_{digest[:32]}"


def _configuration_storage_path(base_path: Path, run_id: str) -> Path:
    digest = hashlib.sha256(f"qdrant-native-path-v1\x1f{run_id}".encode("utf-8")).hexdigest()
    return base_path / f"mub_qdrant_{digest[:24]}"


def build_qdrant_native_multi_object_configuration(
    *, run_id: str, path: str | Path
) -> QdrantNativeMultiObjectConfigurationV1:
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a nonblank exact string")
    storage_path = Path(path)
    if not storage_path.is_absolute():
        raise ValueError("Qdrant storage path must be absolute")
    return QdrantNativeMultiObjectConfigurationV1(
        run_id=run_id,
        collection_name=_configuration_collection(run_id),
        storage_path=str(_configuration_storage_path(storage_path, run_id)),
    )


def qdrant_client_availability() -> QdrantAvailabilityV1:
    try:
        module = importlib.import_module("qdrant_client")
    except ModuleNotFoundError as exc:
        return QdrantAvailabilityV1(
            status="UNAVAILABLE",
            blocker="qdrant_client_missing" if exc.name == "qdrant_client" else "qdrant_client_import_failed",
        )
    except ImportError:
        return QdrantAvailabilityV1(status="UNAVAILABLE", blocker="qdrant_client_import_failed")
    except BaseException:
        return QdrantAvailabilityV1(status="UNAVAILABLE", blocker="qdrant_client_import_failed")
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version("qdrant-client")
        except importlib.metadata.PackageNotFoundError:
            version = None
    if version is not None and type(version) is not str:
        version = str(version)
    return QdrantAvailabilityV1(status="AVAILABLE", package_version=version)


def _load_qdrant_client_module():
    try:
        module = importlib.import_module("qdrant_client")
    except ModuleNotFoundError as exc:
        blocker = "qdrant_client_missing" if exc.name == "qdrant_client" else "qdrant_client_import_failed"
        raise QdrantNativeMultiObjectUnavailable(blocker) from exc
    except BaseException as exc:
        raise QdrantNativeMultiObjectUnavailable("qdrant_client_import_failed") from exc
    if not callable(getattr(module, "QdrantClient", None)):
        raise QdrantNativeMultiObjectUnavailable("qdrant_client_api_missing")
    return module


def _native_provenance() -> NativeProvenanceV3:
    return NativeProvenanceV3(
        kind=NativeDataProvenanceV3.PROVIDER_NATIVE,
        provider_owned=True,
    )


class QdrantNativeMultiObjectProvider:
    """Thin, direct Qdrant collection provider with no local memory projection."""

    def __init__(
        self,
        *,
        configuration: QdrantNativeMultiObjectConfigurationV1,
        client: object | None = None,
        runtime_namespace: str | None = None,
    ) -> None:
        cleanup_client = client
        try:
            if type(configuration) is not QdrantNativeMultiObjectConfigurationV1:
                raise ValueError("Qdrant provider requires exact configuration")
            self.configuration = configuration
            self._closed = False
            self._runtime_namespace = runtime_namespace or configuration.run_id
            if type(self._runtime_namespace) is not str or not self._runtime_namespace.strip():
                raise ValueError("runtime_namespace must be a nonblank exact string")
            self._observation_witnesses: list[QdrantProviderObservationV1] = []
            self._observed_provider_methods: set[str] = set()
            self._models = None
            if client is None:
                module = _load_qdrant_client_module()
                self._models = getattr(module, "models", None)
                if self._models is None:
                    try:
                        self._models = importlib.import_module("qdrant_client.models")
                    except (ImportError, ModuleNotFoundError):
                        self._models = None
                try:
                    client = module.QdrantClient(path=configuration.storage_path)
                except Exception as exc:
                    raise QdrantNativeMultiObjectUnavailable("qdrant_client_runtime_unavailable") from exc
                cleanup_client = client
            else:
                self._models = getattr(client, "models", None)
            self._client = client
            self._schema_verified = False
            self._collection_created = False
            self._ensure_collection()
        except BaseException as exc:
            self._cleanup_failed_initialization(
                client=cleanup_client,
                collection_created=getattr(self, "_collection_created", False),
                primary_error=exc,
            )
            raise

    @staticmethod
    def _record_constructor_cleanup_failure(
        primary_error: BaseException,
        operation: str,
        cleanup_error: BaseException,
    ) -> None:
        try:
            detail = f"{type(cleanup_error).__name__}: {cleanup_error}"
        except BaseException:
            detail = type(cleanup_error).__name__
        message = f"Qdrant provider constructor cleanup {operation} failed: {detail}"
        try:
            existing = getattr(primary_error, "cleanup_details", ())
            if not isinstance(existing, tuple):
                existing = ()
            setattr(primary_error, "cleanup_details", (*existing, message))
        except BaseException:
            pass
        add_note = getattr(primary_error, "add_note", None)
        if not callable(add_note):
            return
        try:
            add_note(message)
        except BaseException:
            pass

    def _cleanup_failed_initialization(
        self,
        *,
        client: object | None,
        collection_created: bool,
        primary_error: BaseException,
    ) -> None:
        if client is None:
            return
        if collection_created:
            try:
                delete_method = getattr(client, "delete_collection", None)
                if not callable(delete_method):
                    raise QdrantNativeMultiObjectError(
                        "Qdrant provider constructor cleanup cannot delete collection"
                    )
                result = delete_method(collection_name=self.configuration.collection_name)
                if result is False:
                    raise QdrantNativeMultiObjectError(
                        "Qdrant provider constructor collection deletion returned false"
                    )
            except BaseException as cleanup_error:
                self._record_constructor_cleanup_failure(
                    primary_error, "delete_collection", cleanup_error
                )
        try:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                close_method()
        except BaseException as cleanup_error:
            self._record_constructor_cleanup_failure(primary_error, "close", cleanup_error)
        finally:
            self._closed = True

    @property
    def collection_name(self) -> str:
        return self.configuration.collection_name

    @property
    def runtime_namespace(self) -> str:
        return self._runtime_namespace

    @property
    def observation_witnesses(self) -> tuple[QdrantProviderObservationV1, ...]:
        return tuple(self._observation_witnesses)

    def _ensure_open(self) -> None:
        if self._closed:
            raise QdrantNativeMultiObjectError("Qdrant native provider is closed")

    def _ensure_collection(self) -> None:
        exists_method = getattr(self._client, "collection_exists", None)
        get_method = getattr(self._client, "get_collection", None)
        create_method = getattr(self._client, "create_collection", None)
        if not callable(exists_method) and not callable(get_method):
            raise QdrantNativeMultiObjectUnavailable("qdrant_collection_api_missing")
        try:
            if callable(exists_method):
                exists = bool(
                    self._provider_call(
                        "collection_exists", collection_name=self.collection_name
                    )
                )
            else:
                try:
                    self._provider_call("get_collection", collection_name=self.collection_name)
                    exists = True
                except QdrantNativeMultiObjectError:
                    exists = False
            if not exists:
                if not callable(create_method):
                    raise QdrantNativeMultiObjectUnavailable("qdrant_collection_create_missing")
                vectors_config = self._vector_config()
                self._provider_call(
                    "create_collection",
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                )
                self._collection_created = True
            self._schema_verified = self._verify_collection_schema()
            if not self._schema_verified:
                raise QdrantNativeMultiObjectUnavailable("qdrant_collection_schema_mismatch")
        except QdrantNativeMultiObjectUnavailable:
            raise
        except Exception as exc:
            raise QdrantNativeMultiObjectUnavailable("qdrant_collection_setup_failed") from exc

    def _verify_collection_schema(self) -> bool:
        get_method = getattr(self._client, "get_collection", None)
        if not callable(get_method):
            return False
        try:
            info = self._provider_call("get_collection", collection_name=self.collection_name)
        except QdrantNativeMultiObjectError:
            return False
        config = getattr(info, "config", None)
        if config is None and isinstance(info, Mapping):
            config = info.get("config")
        if config is None:
            return False
        params = getattr(config, "params", None)
        if params is None and isinstance(config, Mapping):
            params = config.get("params")
        if params is None:
            return False
        vectors = getattr(params, "vectors", None)
        if vectors is None and isinstance(params, Mapping):
            vectors = params.get("vectors")
        if vectors is None:
            return False
        size = getattr(vectors, "size", None)
        if size is None and isinstance(vectors, Mapping):
            size = vectors.get("size")
        distance = getattr(vectors, "distance", None)
        if distance is None and isinstance(vectors, Mapping):
            distance = vectors.get("distance")
        if size is None or distance is None:
            return False
        if hasattr(distance, "value"):
            distance = distance.value
        return size == QDRANT_VECTOR_SIZE and str(distance).casefold() == QDRANT_DISTANCE.casefold()

    def _vector_config(self) -> Any:
        if self._models is not None:
            vector_params = getattr(self._models, "VectorParams", None)
            distance_enum = getattr(self._models, "Distance", None)
            if callable(vector_params):
                distance = getattr(distance_enum, "COSINE", QDRANT_DISTANCE)
                return vector_params(size=QDRANT_VECTOR_SIZE, distance=distance)
        return {"size": QDRANT_VECTOR_SIZE, "distance": QDRANT_DISTANCE}

    def _provider_call(self, method_name: str, **kwargs: Any) -> Any:
        self._ensure_open()
        method = getattr(self._client, method_name, None)
        if not callable(method):
            raise QdrantNativeMultiObjectUnavailable(f"qdrant_method_missing_{method_name}")
        raw_request = {"method": method_name, "kwargs": kwargs}
        try:
            response = method(**kwargs)
        except Exception as exc:
            response_hash = _hash_raw({"error_type": type(exc).__name__})
            self._observation_witnesses.append(
                QdrantProviderObservationV1(
                    method=method_name,
                    call_index=len(self._observation_witnesses),
                    raw_request_hash=_hash_raw(raw_request),
                    raw_response_hash=response_hash,
                    success=False,
                )
            )
            raise QdrantNativeMultiObjectError(
                f"direct Qdrant {method_name} call failed"
            ) from exc
        operation_id = self._operation_id(response)
        self._observation_witnesses.append(
            QdrantProviderObservationV1(
                method=method_name,
                call_index=len(self._observation_witnesses),
                raw_request_hash=_hash_raw(raw_request),
                raw_response_hash=_hash_raw(response),
                provider_operation_id=operation_id,
            )
        )
        return response

    @staticmethod
    def _operation_id(response: Any) -> str | None:
        value = None
        if isinstance(response, Mapping):
            value = response.get("operation_id") or response.get("id")
        else:
            value = getattr(response, "operation_id", None)
        if value is None:
            return None
        return str(value)

    @staticmethod
    def point_id(runtime_namespace: str, object_key: FrozenMemoryObjectKey) -> str:
        if type(runtime_namespace) is not str or not runtime_namespace.strip():
            raise ValueError("runtime_namespace must be a nonblank exact string")
        if type(object_key) is not FrozenMemoryObjectKey:
            raise ValueError("point_id requires exact FrozenMemoryObjectKey")
        identity = object_identity(object_key)
        seed = json.dumps(
            [runtime_namespace, *identity],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, seed))

    def _point_payload(
        self,
        runtime_namespace: str,
        object_key: FrozenMemoryObjectKey,
        value: Any,
        request: NativeMultiObjectMutationRequestV3,
    ) -> dict[str, Any]:
        thawed_value = thaw_json(value)
        return {
            "runtime_namespace": runtime_namespace,
            "object_id": object_key.canonical_id,
            "namespace": object_key.namespace,
            "entity": object_key.entity,
            "attribute": object_key.attribute,
            "subkey": object_key.subkey,
            "object_type": object_key.object_type,
            "value": thawed_value,
            "content": json.dumps(
                thawed_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
            "source_event_ids": list(request.source_event_ids),
            "sequence_index": request.sequence_index,
            "version": request.sequence_index,
            "version_metadata": {
                "point_id_derivation": QDRANT_POINT_ID_DERIVATION,
                "version_index": request.sequence_index,
            },
        }

    def _point_struct(self, point_id: str, payload: Mapping[str, Any]) -> Any:
        if self._models is not None:
            point_struct = getattr(self._models, "PointStruct", None)
            if callable(point_struct):
                return point_struct(
                    id=point_id,
                    vector=self._vector_for_seed(payload["object_id"]),
                    payload=dict(payload),
                )
        return {
            "id": point_id,
            "vector": self._vector_for_seed(payload["object_id"]),
            "payload": dict(payload),
        }

    @staticmethod
    def _vector_for_seed(seed: str) -> list[float]:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        values = []
        for index in range(QDRANT_VECTOR_SIZE):
            value = digest[index % len(digest)]
            values.append((value / 127.5) - 1.0)
        return values

    @staticmethod
    def _point_field(point: Any, name: str, default: Any = None) -> Any:
        if isinstance(point, Mapping):
            return point.get(name, default)
        return getattr(point, name, default)

    def _payload_key(
        self,
        payload: Any,
        *,
        expected_runtime_namespace: str,
        expected_point_id: str | None = None,
    ) -> tuple[FrozenMemoryObjectKey, tuple[str, ...]]:
        if not isinstance(payload, Mapping) or not _REQUIRED_PAYLOAD_FIELDS.issubset(payload):
            raise QdrantNativeMultiObjectError("Qdrant provider payload is malformed")
        runtime_namespace = payload.get("runtime_namespace")
        object_id = payload.get("object_id")
        if type(runtime_namespace) is not str or runtime_namespace != expected_runtime_namespace:
            raise QdrantNativeMultiObjectError("Qdrant provider payload runtime namespace mismatches request")
        if type(object_id) is not str or not object_id.strip():
            raise QdrantNativeMultiObjectError("Qdrant provider payload object ID is malformed")
        if expected_point_id is not None and expected_point_id != self.point_id(
            expected_runtime_namespace,
            FrozenMemoryObjectKey(
                object_type=str(payload.get("object_type")),
                namespace=str(payload.get("namespace")),
                entity=str(payload.get("entity")),
                attribute=str(payload.get("attribute")),
                subkey=payload.get("subkey"),
            ),
        ):
            raise QdrantNativeMultiObjectError("Qdrant provider point ID mismatches payload identity")
        key = FrozenMemoryObjectKey(
            object_type=payload.get("object_type"),
            namespace=payload.get("namespace"),
            entity=payload.get("entity"),
            attribute=payload.get("attribute"),
            subkey=payload.get("subkey"),
        )
        if key.canonical_id != object_id:
            raise QdrantNativeMultiObjectError("Qdrant provider payload object ID mismatches identity fields")
        source_event_ids = payload.get("source_event_ids")
        if not isinstance(source_event_ids, (list, tuple)) or not source_event_ids:
            raise QdrantNativeMultiObjectError("Qdrant provider payload source event linkage is malformed")
        if any(type(value) is not str or not value.strip() for value in source_event_ids):
            raise QdrantNativeMultiObjectError("Qdrant provider payload source event linkage is malformed")
        if len(source_event_ids) != len(set(source_event_ids)):
            raise QdrantNativeMultiObjectError("Qdrant provider payload source event linkage is duplicated")
        for name in ("sequence_index", "version"):
            value = payload.get(name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise QdrantNativeMultiObjectError(f"Qdrant provider payload {name} is malformed")
        version_metadata = payload.get("version_metadata")
        if not isinstance(version_metadata, Mapping):
            raise QdrantNativeMultiObjectError("Qdrant provider payload version metadata is malformed")
        if version_metadata.get("point_id_derivation") != QDRANT_POINT_ID_DERIVATION:
            raise QdrantNativeMultiObjectError(
                "Qdrant provider payload version metadata is malformed"
            )
        if version_metadata.get("version_index") != payload.get("version"):
            raise QdrantNativeMultiObjectError(
                "Qdrant provider payload version metadata is malformed"
            )
        if type(payload.get("content")) is not str:
            raise QdrantNativeMultiObjectError("Qdrant provider payload content is malformed")
        return key, tuple(source_event_ids)

    def _retrieve_point(self, point_id: str, runtime_namespace: str) -> Any | None:
        response = self._provider_call(
            "retrieve",
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not isinstance(response, Sequence) or isinstance(response, (str, bytes, bytearray)):
            raise QdrantNativeMultiObjectError("Qdrant retrieve response is malformed")
        points = list(response)
        if len(points) > 1:
            raise QdrantNativeMultiObjectError("Qdrant retrieve returned duplicate point IDs")
        if not points:
            return None
        point = points[0]
        payload = self._point_field(point, "payload")
        self._payload_key(payload, expected_runtime_namespace=runtime_namespace, expected_point_id=point_id)
        return point

    def mutate(
        self, request: NativeMultiObjectMutationRequestV3
    ) -> NativeMultiObjectMutationResultV3:
        self._ensure_open()
        if type(request) is not NativeMultiObjectMutationRequestV3:
            raise ValueError("Qdrant mutation requires exact NativeMultiObjectMutationRequestV3")
        runtime_namespace = self._runtime_namespace
        if request.operation is Operation.NOOP:
            return NativeMultiObjectMutationResultV3(
                request_id=request.request_id,
                event_id=request.event_id,
                operation=request.operation,
                outcomes=(),
                atomicity=NativeAtomicityV3.NOT_APPLICABLE,
                provenance=_native_provenance(),
            )
        outcomes: list[NativeObjectOutcomeV3] = []
        provider_operation_id: str | None = None
        for item in request.objects:
            point_id = self.point_id(runtime_namespace, item.object_key)
            existing = self._retrieve_point(point_id, runtime_namespace)
            if request.operation is Operation.ADD and existing is not None:
                outcomes.append(
                    NativeObjectOutcomeV3(
                        event_id=request.event_id,
                        object_key=item.object_key,
                        operation=request.operation,
                        status="no_effect",
                        source_event_ids=request.source_event_ids,
                        reason="already_present",
                        provenance=_native_provenance(),
                    )
                )
                continue
            if request.operation is Operation.UPDATE and existing is None:
                outcomes.append(
                    NativeObjectOutcomeV3(
                        event_id=request.event_id,
                        object_key=item.object_key,
                        operation=request.operation,
                        status="no_effect",
                        source_event_ids=request.source_event_ids,
                        reason="absent_update",
                        provenance=_native_provenance(),
                    )
                )
                continue
            if request.operation is Operation.DELETE:
                response = self._provider_call(
                    "delete",
                    collection_name=self.collection_name,
                    points_selector=self._points_selector(
                        self._filter_for_scope(
                            runtime_namespace, "object", item.object_key
                        )
                    ),
                    wait=True,
                )
                provider_operation_id = self._operation_id(response) or provider_operation_id
                outcomes.append(
                    NativeObjectOutcomeV3(
                        event_id=request.event_id,
                        object_key=item.object_key,
                        operation=request.operation,
                        status="executed" if existing is not None else "no_effect",
                        provider_entry_id=point_id if existing is not None else None,
                        source_event_ids=request.source_event_ids,
                        write_count=1 if existing is not None else 0,
                        reason=None if existing is not None else "absent_delete",
                        provenance=_native_provenance(),
                    )
                )
                continue
            payload = self._point_payload(runtime_namespace, item.object_key, item.value, request)
            response = self._provider_call(
                "upsert",
                collection_name=self.collection_name,
                points=[self._point_struct(point_id, payload)],
                wait=True,
            )
            provider_operation_id = self._operation_id(response) or provider_operation_id
            outcomes.append(
                NativeObjectOutcomeV3(
                    event_id=request.event_id,
                    object_key=item.object_key,
                    operation=request.operation,
                    status="executed",
                    provider_entry_id=point_id,
                    source_event_ids=request.source_event_ids,
                    write_count=1,
                    provenance=_native_provenance(),
                )
            )
        statuses = {item.status for item in outcomes}
        atomicity = (
            NativeAtomicityV3.PARTIAL
            if len(statuses) > 1
            else NativeAtomicityV3.UNKNOWN
        )
        return NativeMultiObjectMutationResultV3(
            request_id=request.request_id,
            event_id=request.event_id,
            operation=request.operation,
            outcomes=tuple(outcomes),
            atomicity=atomicity,
            provider_operation_id=provider_operation_id,
            provenance=_native_provenance(),
        )

    def _condition(self, key: str, value: Any) -> Any:
        if self._models is not None:
            field_condition = getattr(self._models, "FieldCondition", None)
            match_value = getattr(self._models, "MatchValue", None)
            if callable(field_condition) and callable(match_value):
                return field_condition(key=key, match=match_value(value=value))
        return {"key": key, "match": {"value": value}}

    def _filter_for_scope(
        self,
        runtime_namespace: str,
        scope: Literal["object", "attribute", "entity", "namespace"],
        object_key: FrozenMemoryObjectKey | None = None,
    ) -> Any:
        conditions: list[Any] = [self._condition("runtime_namespace", runtime_namespace)]
        if scope == "object":
            if type(object_key) is not FrozenMemoryObjectKey:
                raise ValueError("object-scoped delete requires exact object key")
            conditions.append(self._condition("object_id", object_key.canonical_id))
        elif scope == "attribute":
            if type(object_key) is not FrozenMemoryObjectKey:
                raise ValueError("attribute-scoped delete requires exact object key")
            conditions.extend(
                [
                    self._condition("namespace", object_key.namespace),
                    self._condition("entity", object_key.entity),
                    self._condition("attribute", object_key.attribute),
                ]
            )
        elif scope == "entity":
            if type(object_key) is not FrozenMemoryObjectKey:
                raise ValueError("entity-scoped delete requires exact object key")
            conditions.extend(
                [
                    self._condition("namespace", object_key.namespace),
                    self._condition("entity", object_key.entity),
                ]
            )
        elif scope == "namespace":
            if type(object_key) is FrozenMemoryObjectKey:
                conditions.append(self._condition("namespace", object_key.namespace))
        else:
            raise ValueError("Qdrant scoped delete requires object, attribute, entity, or namespace")
        if self._models is not None:
            filter_type = getattr(self._models, "Filter", None)
            if callable(filter_type):
                return filter_type(must=conditions)
        return {"must": conditions}

    def _points_selector(self, filter_value: Any) -> Any:
        if self._models is not None:
            selector_type = getattr(self._models, "FilterSelector", None)
            if callable(selector_type):
                return selector_type(filter=filter_value)
        return filter_value

    def delete_scoped(
        self,
        runtime_namespace: str,
        scope: Literal["object", "attribute", "entity", "namespace"],
        object_key: FrozenMemoryObjectKey | None = None,
    ) -> QdrantScopedDeleteResultV1:
        self._ensure_open()
        if type(runtime_namespace) is not str or not runtime_namespace.strip():
            raise ValueError("runtime_namespace must be a nonblank exact string")
        response = self._provider_call(
            "delete",
            collection_name=self.collection_name,
            points_selector=self._points_selector(
                self._filter_for_scope(runtime_namespace, scope, object_key)
            ),
            wait=True,
        )
        return QdrantScopedDeleteResultV1(
            runtime_namespace=runtime_namespace,
            scope=scope,
            provider_operation_id=self._operation_id(response),
            observation_index=len(self._observation_witnesses) - 1,
        )

    def _provider_filter(self, request: NativeRetrievalRequestV3) -> Any:
        try:
            check_forbidden_metadata(request.filters, "Qdrant retrieval filters")
            check_forbidden_metadata(request.options, "Qdrant retrieval options")
            check_forbidden_metadata(request.query_text, "Qdrant retrieval query")
        except Exception as exc:
            raise QdrantNativeMultiObjectError("Qdrant retrieval request contains forbidden metadata") from exc
        supplied = thaw_json(request.filters)
        if not isinstance(supplied, Mapping):
            supplied = {}
        combined = dict(supplied)
        supplied_conditions = combined.get("must", ())
        if isinstance(supplied_conditions, Sequence) and not isinstance(
            supplied_conditions, (str, bytes, bytearray)
        ):
            conditions = []
            for condition in supplied_conditions:
                if not isinstance(condition, Mapping):
                    raise QdrantNativeMultiObjectError(
                        "Qdrant retrieval filter conditions must be mappings"
                    )
                match = condition.get("match")
                if not isinstance(match, Mapping) or set(match) != {"value"}:
                    raise QdrantNativeMultiObjectError(
                        "Qdrant retrieval filters must use exact match values"
                    )
                key = condition.get("key")
                if type(key) is not str or not key.strip():
                    raise QdrantNativeMultiObjectError(
                        "Qdrant retrieval filter condition key is malformed"
                    )
                conditions.append(self._condition(key, match["value"]))
        elif "must" in combined:
            raise QdrantNativeMultiObjectError(
                "Qdrant retrieval filter must contain a sequence-valued must clause"
            )
        else:
            conditions = []
        if set(combined) - {"must"}:
            raise QdrantNativeMultiObjectError(
                "Qdrant retrieval filters may contain only a must clause"
            )
        conditions.insert(
            0,
            self._condition("runtime_namespace", request.namespace),
        )
        if self._models is not None:
            filter_type = getattr(self._models, "Filter", None)
            if callable(filter_type):
                return filter_type(must=conditions)
        return {"must": conditions}

    @staticmethod
    def _query_points(response: Any) -> list[Any]:
        points = getattr(response, "points", None)
        if points is None and isinstance(response, Mapping):
            points = response.get("points")
        if points is None and isinstance(response, (list, tuple)):
            points = response
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes, bytearray)):
            raise QdrantNativeMultiObjectError("Qdrant query_points response is malformed")
        return list(points)

    def retrieve(self, request: NativeRetrievalRequestV3) -> NativeRetrievalTraceV3:
        self._ensure_open()
        if type(request) is not NativeRetrievalRequestV3:
            raise ValueError("Qdrant retrieval requires exact NativeRetrievalRequestV3")
        response = self._provider_call(
            "query_points",
            collection_name=self.collection_name,
            query=self._vector_for_seed(request.query_text),
            limit=request.k,
            query_filter=self._provider_filter(request),
            with_payload=True,
            with_vectors=False,
        )
        points = self._query_points(response)
        entries: list[NativeRetrievalEntryV3] = []
        for order_index, point in enumerate(points):
            provider_id = self._point_field(point, "id")
            score = self._point_field(point, "score")
            payload = self._point_field(point, "payload")
            if provider_id is None or score is None:
                raise QdrantNativeMultiObjectError("Qdrant query_points returned malformed point")
            provider_id = str(provider_id)
            key, source_event_ids = self._payload_key(
                payload,
                expected_runtime_namespace=request.namespace,
                expected_point_id=provider_id,
            )
            entries.append(
                NativeRetrievalEntryV3(
                    provider_entry_id=provider_id,
                    object_key=key,
                    provider_payload=payload,
                    rank=order_index + 1,
                    order_index=order_index,
                    score=float(score),
                    source_event_ids=source_event_ids,
                    provenance=_native_provenance(),
                )
            )
        return NativeRetrievalTraceV3(
            query=request,
            entries=tuple(entries),
            status=NativeRetrievalStatusV3.RETURNED if entries else NativeRetrievalStatusV3.EMPTY,
            provenance=_native_provenance(),
        )

    @staticmethod
    def _scroll_points(response: Any) -> tuple[list[Any], Any | None]:
        cursor = None
        if isinstance(response, tuple):
            if len(response) != 2:
                raise QdrantNativeMultiObjectError("Qdrant scroll response is malformed")
            points, cursor = response
        elif isinstance(response, Mapping):
            points = response.get("points")
            cursor = response.get("next_page_offset")
        else:
            points = getattr(response, "points", None)
            if points is None and isinstance(response, Sequence):
                points = response
            elif points is not None:
                cursor = getattr(response, "next_page_offset", None)
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes, bytearray)):
            raise QdrantNativeMultiObjectError("Qdrant scroll response is malformed")
        if cursor is not None and (
            type(cursor) is bool
            or not isinstance(cursor, (int, str, uuid.UUID))
            or (type(cursor) is int and cursor < 0)
            or (type(cursor) is str and not cursor.strip())
        ):
            raise QdrantNativeMultiObjectError(
                "Qdrant scroll continuation cursor is malformed"
            )
        return list(points), cursor

    def reset(self, namespace: str) -> NativeResetResultV3:
        self._ensure_open()
        if type(namespace) is not str or not namespace.strip():
            raise ValueError("Qdrant reset namespace must be a nonblank exact string")
        response = self._provider_call(
            "delete",
            collection_name=self.collection_name,
            points_selector=self._points_selector(
                self._filter_for_scope(namespace, "namespace")
            ),
            wait=True,
        )
        verify = self._provider_call(
            "scroll",
            collection_name=self.collection_name,
            scroll_filter=self._filter_for_scope(namespace, "namespace"),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        remaining, _ = self._scroll_points(verify)
        operation_id = self._operation_id(response)
        if remaining:
            return NativeResetResultV3(
                namespace=namespace,
                success=False,
                provider_operation_id=operation_id,
                provenance=_native_provenance(),
                error="namespace_not_empty",
            )
        self._runtime_namespace = namespace
        return NativeResetResultV3(
            namespace=namespace,
            success=True,
            provider_operation_id=operation_id,
            provenance=_native_provenance(),
        )

    def verify_api_runtime(self, evidence_anchor: str | None = None) -> QdrantDirectVerificationV1:
        required = ("upsert", "retrieve", "delete", "query_points", "scroll")
        missing = tuple(name for name in required if not callable(getattr(self._client, name, None)))
        if missing:
            return QdrantDirectVerificationV1(
                status="BLOCKED",
                direct_observed=False,
                collection_schema_verified=self._schema_verified,
                required_methods=required,
                missing_methods=missing,
                blocker="qdrant_api_methods_missing",
                evidence_anchor=evidence_anchor,
            )
        if not self._schema_verified:
            return QdrantDirectVerificationV1(
                status="BLOCKED",
                direct_observed=False,
                collection_schema_verified=False,
                required_methods=required,
                blocker="qdrant_schema_not_verified",
                evidence_anchor=evidence_anchor,
            )
        observed = {item.method for item in self._observation_witnesses if item.success}
        unobserved = tuple(name for name in required if name not in observed)
        if unobserved:
            return QdrantDirectVerificationV1(
                status="BLOCKED",
                direct_observed=False,
                collection_schema_verified=True,
                required_methods=required,
                missing_methods=unobserved,
                blocker="qdrant_api_runtime_not_observed",
                evidence_anchor=evidence_anchor,
            )
        if type(evidence_anchor) is not str or not evidence_anchor.strip():
            return QdrantDirectVerificationV1(
                status="BLOCKED",
                direct_observed=True,
                collection_schema_verified=True,
                required_methods=required,
                blocker="missing_evidence_anchor",
            )
        return QdrantDirectVerificationV1(
            status="READY",
            direct_observed=True,
            collection_schema_verified=True,
            required_methods=required,
            evidence_anchor=evidence_anchor.strip(),
        )

    def capabilities(self) -> NativeCapabilityDeclarationV3:
        return NativeCapabilityDeclarationV3(
            direct_provider_crud=True,
            provider_owned_collection=True,
            supports_multi_object_mutation=True,
            supports_multi_object_retrieval=True,
            supports_add=True,
            supports_update=True,
            supports_noop=True,
            supports_delete=True,
            supports_atomic_mutation=False,
            supports_isolated_reset=True,
            exports_provider_entry_ids=True,
            exports_provider_payloads=True,
            exports_provider_order=True,
            exports_provider_scores=True,
            exports_source_event_linkage=True,
            visible_only=True,
        )

    def observed_capabilities(self) -> NativeCapabilityDeclarationV3:
        if self._schema_verified:
            return self.capabilities()
        return NativeCapabilityDeclarationV3()

    def close(self) -> None:
        if self._closed:
            return
        try:
            close_method = getattr(self._client, "close", None)
            if callable(close_method):
                close_method()
        except Exception as exc:
            self._closed = True
            raise QdrantNativeMultiObjectError("Qdrant provider close failed") from exc
        self._closed = True


QdrantNativeMultiObjectProviderV1 = QdrantNativeMultiObjectProvider
QdrantNativeMultiObjectConfiguration = QdrantNativeMultiObjectConfigurationV1
QdrantNativeMultiObjectProviderError = QdrantNativeMultiObjectError
QdrantProviderObservation = QdrantProviderObservationV1


__all__ = [
    "QDRANT_DISTANCE",
    "QDRANT_NATIVE_CONTRACT_VERSION",
    "QDRANT_NATIVE_PROVIDER_VERSION",
    "QDRANT_POINT_ID_DERIVATION",
    "QDRANT_VECTOR_SIZE",
    "QdrantAvailabilityV1",
    "QdrantDirectVerificationV1",
    "QdrantNativeMultiObjectConfiguration",
    "QdrantNativeMultiObjectConfigurationV1",
    "QdrantNativeMultiObjectError",
    "QdrantNativeMultiObjectProvider",
    "QdrantNativeMultiObjectProviderV1",
    "QdrantNativeMultiObjectProviderError",
    "QdrantNativeMultiObjectUnavailable",
    "QdrantProviderObservation",
    "QdrantProviderObservationV1",
    "QdrantScopedDeleteResultV1",
    "build_qdrant_native_multi_object_configuration",
    "qdrant_client_availability",
]
