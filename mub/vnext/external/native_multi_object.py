from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, StrictBool
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, StrictIdentifier, object_identity
from mub.vnext.contracts.v3.native_multi_object import (
    NativeCapabilityDeclarationV3,
    NativeDataProvenanceV3,
    NativeMultiObjectMutationRequestV3,
    NativeMultiObjectMutationResultV3,
    NativeProvenanceV3,
    NativeResetResultV3,
    NativeRetrievalRequestV3,
    NativeRetrievalTraceV3,
)


import re

FORBIDDEN_EXACT_KEYS = {
    "target_object_keys",
    "target_objects",
    "target_ids",
    "gold_answer",
    "answer",
    "selector",
    "raw_prompt",
    "raw_output",
    "reasoning",
}

FORBIDDEN_CREDENTIAL_PATTERNS = (
    "token=",
    "api_key=",
    "api-key=",
    "secret=",
    "password=",
    "bearer ",
    "bearer=",
    "authorization:",
    "private_key",
    "access_token",
)

FORBIDDEN_PRIVATE_SCHEMES = ("file://", "s3://", "gcs://", "ftp://", "ws://", "wss://", "unix://", "ipc://")
PRIVATE_HOST_REGEX = re.compile(
    r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|\[::1\])[/\:\?#]?",
    re.IGNORECASE,
)
CREDENTIAL_URL_REGEX = re.compile(r"https?://[^/\s:@]+:[^/\s:@]+@", re.IGNORECASE)


def _is_forbidden_key(key: str) -> bool:
    k = key.lower()
    if k in FORBIDDEN_EXACT_KEYS:
        return True
    if k.startswith(("expected_", "hidden_", "gold_")):
        return True
    return False


def _is_forbidden_value(val: str) -> bool:
    v = val.lower()
    for pat in FORBIDDEN_CREDENTIAL_PATTERNS:
        if pat in v:
            return True
    for scheme in FORBIDDEN_PRIVATE_SCHEMES:
        if scheme in v:
            return True
    if PRIVATE_HOST_REGEX.search(val):
        return True
    if CREDENTIAL_URL_REGEX.search(val):
        return True
    return False


def check_forbidden_metadata(data: object, label: str = "payload") -> None:
    if isinstance(data, Mapping):
        for k, v in data.items():
            if isinstance(k, str) and _is_forbidden_key(k):
                raise NativeMultiObjectAdapterError(f"forbidden key '{k}' found in {label}")
            check_forbidden_metadata(v, label)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            check_forbidden_metadata(item, label)
    elif isinstance(data, str):
        if _is_forbidden_value(data):
            raise NativeMultiObjectAdapterError(f"forbidden URL or credential value in {label}")


class NativeMultiObjectAdapterError(RuntimeError):
    pass


@runtime_checkable
class NativeMultiObjectProvider(Protocol):
    """Direct provider-owned CRUD/retrieval boundary.

    Implementations must return provider-owned IDs, payloads, ranks, order, scores,
    and event linkage. This protocol intentionally exposes no answer method and no
    benchmark target list to retrieval.
    """

    def mutate(
        self, request: NativeMultiObjectMutationRequestV3
    ) -> NativeMultiObjectMutationResultV3: ...

    def retrieve(self, request: NativeRetrievalRequestV3) -> NativeRetrievalTraceV3: ...

    def reset(self, namespace: str) -> NativeResetResultV3 | None: ...

    def close(self) -> None: ...


class NativeCapabilityValidationV3(ImmutableContractModel):
    declared: NativeCapabilityDeclarationV3
    observed: NativeCapabilityDeclarationV3
    passed: StrictBool
    status: str = Field(pattern=r"^(validated|overclaimed|invalid)$", strict=True)
    overclaimed_fields: tuple[StrictIdentifier, ...] = ()
    undeclared_observed_fields: tuple[StrictIdentifier, ...] = ()
    provenance: NativeProvenanceV3

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.passed and (self.status != "validated" or self.overclaimed_fields or self.undeclared_observed_fields):
            raise ValueError("passed capability validation must be validated and discrepancy-free")
        if not self.passed and self.status == "validated":
            raise ValueError("failed capability validation cannot have validated status")
        return self


def _revalidate_capabilities(value: NativeCapabilityDeclarationV3 | Mapping) -> NativeCapabilityDeclarationV3:
    try:
        if isinstance(value, Mapping):
            return NativeCapabilityDeclarationV3.model_validate(dict(value), strict=True)
        if type(value) is not NativeCapabilityDeclarationV3:
            raise ValueError("native capabilities require exact NativeCapabilityDeclarationV3")
        payload = {
            field_name: value.__dict__[field_name]
            for field_name in NativeCapabilityDeclarationV3.model_fields
        }
        return NativeCapabilityDeclarationV3.model_validate(payload, strict=True)
    except Exception as exc:
        raise ValueError("native capability declaration is invalid") from exc


def validate_native_capabilities(
    declared: NativeCapabilityDeclarationV3,
    observed: NativeCapabilityDeclarationV3,
) -> NativeCapabilityValidationV3:
    declared = _revalidate_capabilities(declared)
    observed = _revalidate_capabilities(observed)
    fields = tuple(NativeCapabilityDeclarationV3.model_fields)
    overclaimed = tuple(
        name for name in fields
        if getattr(declared, name) is True and getattr(observed, name) is False
    )
    undeclared = tuple(
        name for name in fields
        if getattr(declared, name) is False and getattr(observed, name) is True
    )
    discrepancies = tuple(sorted(set(overclaimed + undeclared)))
    passed = not discrepancies
    return NativeCapabilityValidationV3(
        declared=declared,
        observed=observed,
        passed=passed,
        status="validated" if passed else "overclaimed",
        overclaimed_fields=tuple(sorted(overclaimed)),
        undeclared_observed_fields=tuple(sorted(undeclared)),
        provenance=NativeProvenanceV3(
            kind=NativeDataProvenanceV3.OBSERVED,
            provider_owned=True,
        ),
    )


def _revalidate_native_contract(value, model_type, label: str):
    if type(value) is not model_type:
        raise NativeMultiObjectAdapterError(f"native provider returned an invalid {label}")
    try:
        payload = value.model_dump(mode="python", exclude_unset=True)
        return model_type.model_validate(payload, strict=True)
    except Exception as exc:
        raise NativeMultiObjectAdapterError(f"native provider returned an invalid {label}") from exc


def _require_provider_method(provider: object, method: str) -> None:
    if not callable(getattr(provider, method, None)):
        raise NativeMultiObjectAdapterError(f"native provider is missing direct {method} method")


def _validate_boundary_declaration(provider: object) -> None:
    declaration = getattr(provider, "native_boundary", None)
    if declaration is None:
        return
    if not isinstance(declaration, Mapping):
        raise NativeMultiObjectAdapterError("native boundary declaration is invalid")
    forbidden = (
        "uses_local_authoritative_store",
        "local_authoritative_store",
        "uses_local_reranking",
        "local_reranking",
        "uses_local_deduplication",
        "local_deduplication",
        "uses_local_filtering",
        "local_filtering",
        "hidden_target_injection",
    )
    active = tuple(name for name in forbidden if declaration.get(name) is True)
    if active:
        label = active[0].replace("uses_", "").replace("local_", "local ").replace("_", " ")
        raise NativeMultiObjectAdapterError(f"native boundary forbids {label}")


class NativeMultiObjectAdapter:
    """Thin provider-neutral adapter with no authoritative local memory state."""

    def __init__(
        self,
        *,
        provider: NativeMultiObjectProvider,
        target_objects: tuple[FrozenMemoryObjectKey, ...] = (),
        declared_capabilities: NativeCapabilityDeclarationV3 | None = None,
    ) -> None:
        _validate_boundary_declaration(provider)
        for method in ("mutate", "retrieve", "reset", "close"):
            _require_provider_method(provider, method)
        if type(target_objects) is not tuple or any(type(item) is not FrozenMemoryObjectKey for item in target_objects):
            raise ValueError("native target_objects require an exact tuple of FrozenMemoryObjectKey")
        identities = tuple(object_identity(item) for item in target_objects)
        if len(identities) != len(set(identities)):
            raise ValueError("native target object identities must be unique")
        self._provider = provider
        self._target_objects = target_objects
        self._closed = False
        self._capability_validation: NativeCapabilityValidationV3 | None = None
        if declared_capabilities is not None:
            observed = getattr(provider, "observed_capabilities", None)
            if callable(observed):
                observed = observed()
            if observed is None:
                raise NativeMultiObjectAdapterError(
                    "declared native capabilities require explicit observed capabilities"
                )
            self._capability_validation = validate_native_capabilities(declared_capabilities, observed)
            if not self._capability_validation.passed:
                raise NativeMultiObjectAdapterError("declared native capabilities overclaim observed capabilities")

    @property
    def capability_validation(self) -> NativeCapabilityValidationV3 | None:
        return self._capability_validation

    @property
    def target_objects(self) -> tuple[FrozenMemoryObjectKey, ...]:
        return self._target_objects

    def _ensure_open(self) -> None:
        if self._closed:
            raise NativeMultiObjectAdapterError("native adapter is closed")

    @staticmethod
    def _require_native_provenance(value: NativeProvenanceV3, label: str) -> None:
        if value.kind is not NativeDataProvenanceV3.PROVIDER_NATIVE or not value.provider_owned:
            raise NativeMultiObjectAdapterError(f"{label} is not provider-native")

    def mutate(self, request: NativeMultiObjectMutationRequestV3) -> NativeMultiObjectMutationResultV3:
        self._ensure_open()
        if type(request) is not NativeMultiObjectMutationRequestV3:
            raise ValueError("native mutate requires exact NativeMultiObjectMutationRequestV3")
        request = _revalidate_native_contract(
            request,
            NativeMultiObjectMutationRequestV3,
            "mutation request",
        )
        expected = tuple(object_identity(item.object_key) for item in request.objects)
        if self._target_objects:
            allowed = {object_identity(item) for item in self._target_objects}
            if any(identity not in allowed for identity in expected):
                raise NativeMultiObjectAdapterError("native mutation object is outside adapter target scope")
        try:
            result = self._provider.mutate(request)
        except NativeMultiObjectAdapterError:
            raise
        except Exception as exc:
            raise NativeMultiObjectAdapterError("direct native provider mutation failed") from exc
        if type(result) is not NativeMultiObjectMutationResultV3:
            raise NativeMultiObjectAdapterError("native provider returned an invalid mutation result")
        result = _revalidate_native_contract(
            result,
            NativeMultiObjectMutationResultV3,
            "mutation result",
        )
        if result.request_id != request.request_id:
            raise NativeMultiObjectAdapterError("native mutation result request ID is inconsistent")
        if result.event_id != request.event_id or result.operation is not request.operation:
            raise NativeMultiObjectAdapterError("native mutation result event/operation identity is inconsistent")
        observed = tuple(object_identity(item.object_key) for item in result.outcomes)
        if expected != observed:
            raise NativeMultiObjectAdapterError("native mutation result object-key linkage is inconsistent")
        self._require_native_provenance(result.provenance, "native mutation result")
        for outcome in result.outcomes:
            self._require_native_provenance(outcome.provenance, "native object outcome")
        return result

    def retrieve(self, request: NativeRetrievalRequestV3 | Mapping) -> NativeRetrievalTraceV3:
        self._ensure_open()
        if isinstance(request, Mapping):
            check_forbidden_metadata(request, "retrieval request mapping")
            try:
                request = NativeRetrievalRequestV3.model_validate(dict(request), strict=True)
            except Exception as exc:
                if isinstance(exc, NativeMultiObjectAdapterError):
                    raise
                raise NativeMultiObjectAdapterError("native retrieval request is invalid") from exc
        if type(request) is not NativeRetrievalRequestV3:
            raise ValueError("native retrieve requires exact NativeRetrievalRequestV3")
        request = _revalidate_native_contract(
            request,
            NativeRetrievalRequestV3,
            "retrieval request",
        )
        check_forbidden_metadata(request.filters, "retrieval request filters")
        check_forbidden_metadata(request.options, "retrieval request options")
        check_forbidden_metadata(request.query_text, "retrieval query text")
        try:
            trace = self._provider.retrieve(request)
        except NativeMultiObjectAdapterError:
            raise
        except Exception as exc:
            raise NativeMultiObjectAdapterError("direct native provider retrieval failed") from exc
        if type(trace) is not NativeRetrievalTraceV3:
            raise NativeMultiObjectAdapterError("native provider returned an invalid retrieval trace")
        trace = _revalidate_native_contract(trace, NativeRetrievalTraceV3, "retrieval trace")
        if trace.query != request:
            raise NativeMultiObjectAdapterError("native retrieval trace query identity is inconsistent")
        self._require_native_provenance(trace.provenance, "native retrieval trace")
        for entry in trace.entries:
            self._require_native_provenance(entry.provenance, "native retrieval entry")
            check_forbidden_metadata(entry.provider_payload, "provider payload")
        return trace

    def reset(self, namespace: str) -> NativeResetResultV3:
        self._ensure_open()
        if type(namespace) is not str or not namespace.strip():
            raise ValueError("native reset namespace must be a nonblank exact string")
        try:
            result = self._provider.reset(namespace)
        except NativeMultiObjectAdapterError:
            raise
        except Exception as exc:
            raise NativeMultiObjectAdapterError("direct native provider reset failed") from exc
        if result is None:
            raise NativeMultiObjectAdapterError("native reset provider returned None")
        if type(result) is not NativeResetResultV3:
            raise NativeMultiObjectAdapterError("native provider returned an invalid reset result")
        result = _revalidate_native_contract(result, NativeResetResultV3, "reset result")
        if result.namespace != namespace:
            raise NativeMultiObjectAdapterError("native reset namespace identity is inconsistent")
        self._require_native_provenance(result.provenance, "native reset result")
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._provider.close()
        except Exception as exc:
            self._closed = True
            raise NativeMultiObjectAdapterError("direct native provider close failed") from exc
        self._closed = True


NativeMultiObjectAdapterV3 = NativeMultiObjectAdapter
NativeMultiObjectProviderV3 = NativeMultiObjectProvider
NativeCapabilityValidation = NativeCapabilityValidationV3


__all__ = [
    "NativeCapabilityValidation",
    "NativeCapabilityValidationV3",
    "NativeMultiObjectAdapter",
    "NativeMultiObjectAdapterV3",
    "NativeMultiObjectAdapterError",
    "NativeMultiObjectProvider",
    "NativeMultiObjectProviderV3",
    "validate_native_capabilities",
]
