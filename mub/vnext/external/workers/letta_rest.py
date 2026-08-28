from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from mub.vnext.external.providers.letta import LettaAdapterConfigurationV1

LETTA_NATIVE_API_BASE_URL_ENV = "LETTA_NATIVE_API_BASE_URL"
_NAMESPACE_MARKER_KEY = "memupdatebench_namespace_marker_v1"
_NAMESPACE_MARKER_TAG = "mub-namespace-v1-"
_PAGE_LIMIT = 100
_MAX_PAGES = 16


class LettaRestDependencyUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class _StdlibResponse:
    status: int
    payload: object


RequestFunction = Callable[..., object]


def _valid_loopback_base_url(value: object) -> str:
    if type(value) is not str or not value or any(ord(char) < 32 for char in value):
        raise LettaRestDependencyUnavailable("letta_rest_base_url_invalid")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise ValueError
        if parsed.username is not None or parsed.password is not None:
            raise ValueError
        if parsed.query or parsed.fragment:
            raise ValueError
        if hostname.casefold() == "localhost":
            loopback = True
        else:
            loopback = ipaddress.ip_address(hostname).is_loopback
        if not loopback:
            raise ValueError
        if parsed.port is not None and not (1 <= parsed.port <= 65535):
            raise ValueError
        path = parsed.path.rstrip("/") or ""
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    except (ValueError, TypeError):
        raise LettaRestDependencyUnavailable("letta_rest_base_url_invalid") from None


def _namespace_marker(configuration: LettaAdapterConfigurationV1, namespace: str) -> str:
    if type(namespace) is not str or not namespace:
        raise LettaRestDependencyUnavailable("letta_rest_namespace_invalid")
    raw = f"{configuration.namespace_root}\0{namespace}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _entry_id(canonical_object_id: str) -> str:
    if type(canonical_object_id) is not str or not canonical_object_id:
        raise LettaRestDependencyUnavailable("letta_rest_block_id_invalid")
    digest = hashlib.sha256(canonical_object_id.encode("utf-8")).hexdigest()
    return f"letta-block-{digest[:32]}"


def _canonical_value(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        raise LettaRestDependencyUnavailable("letta_rest_value_invalid") from None


class LettaRestBlockClientV1:
    """Small, dependency-lazy client for Letta's native block REST endpoints."""

    def __init__(
        self,
        *,
        configuration: LettaAdapterConfigurationV1,
        base_url: str,
        request_fn: RequestFunction | None = None,
    ) -> None:
        if type(configuration) is not LettaAdapterConfigurationV1:
            raise ValueError("Letta REST client requires exact public configuration")
        self._configuration = configuration
        self._base_url = _valid_loopback_base_url(base_url)
        self._request_fn = _stdlib_request if request_fn is None else request_fn
        if not callable(self._request_fn):
            raise LettaRestDependencyUnavailable("letta_rest_request_function_invalid")

    def _url(self, path: str, params: Mapping[str, object] | None = None) -> str:
        url = f"{self._base_url}{path}"
        if params:
            url += "?" + urlencode(params)
        return url

    def _request(self, method: str, path: str, *, body: Mapping[str, object] | None = None, params=None) -> object:
        try:
            response = self._request_fn(
                method,
                self._url(path, params),
                headers={"accept": "application/json", "content-type": "application/json"},
                body=None if body is None else dict(body),
                timeout=10.0,
            )
        except (HTTPError, URLError, TimeoutError, OSError):
            raise LettaRestDependencyUnavailable("letta_rest_transport_unavailable") from None
        except LettaRestDependencyUnavailable:
            raise
        except Exception:
            raise LettaRestDependencyUnavailable("letta_rest_transport_unavailable") from None
        status = getattr(response, "status", None)
        payload = getattr(response, "payload", None)
        if type(status) is not int or not 100 <= status <= 599:
            raise LettaRestDependencyUnavailable("letta_rest_response_invalid")
        if status >= 500:
            raise LettaRestDependencyUnavailable("letta_rest_http_5xx")
        if status == 429:
            raise LettaRestDependencyUnavailable("letta_rest_http_429")
        if status in {401, 403}:
            raise LettaRestDependencyUnavailable("letta_rest_http_auth")
        if status == 404:
            raise LettaRestDependencyUnavailable("letta_rest_http_404")
        if status == 409:
            raise LettaRestDependencyUnavailable("letta_rest_http_409")
        if status < 200 or status >= 300:
            raise LettaRestDependencyUnavailable("letta_rest_http_error")
        return payload

    def _payload(self, namespace: str, value: Mapping[str, object]) -> dict[str, object]:
        marker = _namespace_marker(self._configuration, namespace)
        return {
            "label": self._configuration.block_label,
            "value": _canonical_value(value),
            "description": f"MemUpdateBench namespace marker v1:{marker}",
            "metadata": {_NAMESPACE_MARKER_KEY: marker},
            "tags": [f"{_NAMESPACE_MARKER_TAG}{marker}"],
        }

    @staticmethod
    def _native_id(block: object) -> str:
        if not isinstance(block, Mapping):
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        native_id = block.get("id", block.get("block_id"))
        if type(native_id) is not str or not native_id:
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        return native_id

    def _is_marked(self, block: Mapping, namespace: str) -> bool:
        metadata = block.get("metadata")
        marker = _namespace_marker(self._configuration, namespace)
        return isinstance(metadata, Mapping) and metadata.get(_NAMESPACE_MARKER_KEY) == marker

    def _raw_value(self, block: Mapping, *, expected_id: str | None = None) -> dict[str, object]:
        try:
            raw = json.loads(block.get("value"))
        except (TypeError, ValueError):
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed") from None
        if not isinstance(raw, Mapping):
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        if _canonical_value(raw) != block.get("value"):
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        expected_keys = {"canonical_object_id", "content", "value", "source_event_id", "sequence_index"}
        if set(raw) != expected_keys:
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        canonical = raw.get("canonical_object_id")
        if (
            type(canonical) is not str
            or not canonical
            or type(raw.get("content")) is not str
            or type(raw.get("source_event_id")) is not str
            or type(raw.get("sequence_index")) is not int
            or raw.get("sequence_index") < 0
            or raw.get("value") is None
        ):
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        if expected_id is not None and _entry_id(canonical) != expected_id:
            raise LettaRestDependencyUnavailable("letta_rest_block_foreign")
        return dict(raw)

    def _list_page(self, *, after: str | None = None) -> tuple[list[Mapping], str | None]:
        payload = self._request("GET", "/v1/blocks/", params={"limit": _PAGE_LIMIT, **({"after": after} if after else {})})
        if isinstance(payload, list):
            rows = payload
            next_cursor = None
        elif isinstance(payload, Mapping):
            rows = payload.get("blocks", payload.get("data"))
            next_cursor = payload.get("next_cursor", payload.get("next_page"))
            if next_cursor is not None and type(next_cursor) is not str:
                raise LettaRestDependencyUnavailable("letta_rest_pagination_invalid")
        else:
            raise LettaRestDependencyUnavailable("letta_rest_response_invalid")
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise LettaRestDependencyUnavailable("letta_rest_response_invalid")
        if len(rows) >= _PAGE_LIMIT and not next_cursor:
            raise LettaRestDependencyUnavailable("letta_rest_pagination_incomplete")
        return list(rows), next_cursor

    def _marked_blocks(self, namespace: str) -> list[tuple[str, Mapping, dict[str, object]]]:
        rows: list[Mapping] = []
        cursor = None
        for page in range(_MAX_PAGES):
            current, cursor = self._list_page(after=cursor)
            rows.extend(current)
            if not cursor:
                break
            if page == _MAX_PAGES - 1:
                raise LettaRestDependencyUnavailable("letta_rest_pagination_incomplete")
        result = []
        for block in rows:
            if not self._is_marked(block, namespace):
                continue
            native_id = self._native_id(block)
            raw = self._raw_value(block)
            deterministic_id = _entry_id(raw["canonical_object_id"])
            result.append((deterministic_id, block, raw))
        result.sort(key=lambda row: row[0])
        return result

    def _resolve(self, namespace: str, block_id: str) -> tuple[str, dict[str, object]] | None:
        for deterministic_id, block, raw in self._marked_blocks(namespace):
            if deterministic_id == block_id:
                return self._native_id(block), raw
        return None

    def get_block(self, namespace: str, block_id: str) -> Mapping | None:
        resolved = self._resolve(namespace, block_id)
        if resolved is None:
            return None
        native_id, _ = resolved
        block = self._request("GET", f"/v1/blocks/{quote(native_id, safe='')}")
        if not isinstance(block, Mapping) or not self._is_marked(block, namespace):
            raise LettaRestDependencyUnavailable("letta_rest_block_foreign")
        raw = self._raw_value(block, expected_id=block_id)
        return raw

    def create_block(self, namespace: str, block_id: str, value: dict) -> None:
        if _entry_id(value.get("canonical_object_id")) != block_id:
            raise LettaRestDependencyUnavailable("letta_rest_block_id_invalid")
        response = self._request("POST", "/v1/blocks/", body=self._payload(namespace, value))
        if not isinstance(response, Mapping) or not self._is_marked(response, namespace):
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        self._raw_value(response, expected_id=block_id)

    def update_block(self, namespace: str, block_id: str, value: dict) -> None:
        resolved = self._resolve(namespace, block_id)
        if resolved is None:
            raise LettaRestDependencyUnavailable("letta_rest_http_404")
        native_id, _ = resolved
        if _entry_id(value.get("canonical_object_id")) != block_id:
            raise LettaRestDependencyUnavailable("letta_rest_block_id_invalid")
        response = self._request("PATCH", f"/v1/blocks/{quote(native_id, safe='')}", body=self._payload(namespace, value))
        if not isinstance(response, Mapping) or not self._is_marked(response, namespace):
            raise LettaRestDependencyUnavailable("letta_rest_block_malformed")
        self._raw_value(response, expected_id=block_id)

    def delete_block(self, namespace: str, block_id: str) -> None:
        resolved = self._resolve(namespace, block_id)
        if resolved is None:
            return
        native_id, _ = resolved
        self._request("DELETE", f"/v1/blocks/{quote(native_id, safe='')}")

    def search_blocks(self, namespace: str) -> tuple[tuple[str, Mapping], ...]:
        return tuple((deterministic_id, raw) for deterministic_id, _, raw in self._marked_blocks(namespace))

    def close(self) -> None:
        return None


def _stdlib_request(method: str, url: str, *, headers: Mapping[str, str], body: Mapping[str, object] | None, timeout: float) -> _StdlibResponse:
    raw = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=raw, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response.read()
            payload = None if not content else json.loads(content.decode("utf-8"))
            return _StdlibResponse(response.status, payload)
    except HTTPError as exc:
        return _StdlibResponse(exc.code, None)


def build_rest_letta_block_client(
    configuration: LettaAdapterConfigurationV1,
    *,
    base_url: str | None = None,
    request_fn: RequestFunction | None = None,
) -> LettaRestBlockClientV1:
    if type(configuration) is not LettaAdapterConfigurationV1:
        raise ValueError("Letta REST client requires exact public configuration")
    selected = base_url if base_url is not None else os.environ.get(LETTA_NATIVE_API_BASE_URL_ENV)
    if selected is None:
        raise LettaRestDependencyUnavailable("letta_native_api_unverified")
    return LettaRestBlockClientV1(configuration=configuration, base_url=selected, request_fn=request_fn)


__all__ = [
    "LETTA_NATIVE_API_BASE_URL_ENV",
    "LettaRestBlockClientV1",
    "LettaRestDependencyUnavailable",
    "build_rest_letta_block_client",
]
