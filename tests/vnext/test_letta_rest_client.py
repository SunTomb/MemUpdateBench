from __future__ import annotations

import hashlib
import json

import pytest

from mub.vnext.external.providers.letta import build_letta_adapter_configuration
from mub.vnext.external.workers.letta_rest import (
    LettaRestBlockClientV1,
    LettaRestDependencyUnavailable,
    build_rest_letta_block_client,
)


class FakeResponse:
    def __init__(self, status: int, payload):
        self.status = status
        self.payload = payload


class FakeTransport:
    def __init__(self):
        self.blocks = {}
        self.calls = []

    def __call__(self, method, url, *, headers, body, timeout):
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        path = parsed.path
        self.calls.append((method, path, body))
        if method == "GET" and path == "/v1/blocks/":
            return FakeResponse(200, list(self.blocks.values()))
        if method == "POST" and path == "/v1/blocks/":
            native_id = f"native-{len(self.blocks) + 1}"
            block = dict(body)
            block["id"] = native_id
            self.blocks[native_id] = block
            return FakeResponse(201, block)
        native_id = path.rsplit("/", 1)[-1]
        if native_id not in self.blocks:
            return FakeResponse(404, {"detail": "not found"})
        if method == "GET":
            return FakeResponse(200, self.blocks[native_id])
        if method == "PATCH":
            self.blocks[native_id].update(body)
            return FakeResponse(200, self.blocks[native_id])
        if method == "DELETE":
            del self.blocks[native_id]
            return FakeResponse(204, None)
        raise AssertionError((method, path))


def _client(transport=None):
    configuration = build_letta_adapter_configuration(run_id="rest-client-test")
    return build_rest_letta_block_client(
        configuration, base_url="http://127.0.0.1:8283", request_fn=transport
    )


def test_factory_rejects_non_loopback_or_credentialed_urls():
    configuration = build_letta_adapter_configuration(run_id="url-test")
    for url in (
        "https://example.test",
        "http://user:pass@127.0.0.1:8283",
        "ftp://127.0.0.1:8283",
        "http://127.0.0.1:8283/?token=secret",
    ):
        with pytest.raises(LettaRestDependencyUnavailable) as exc_info:
            build_rest_letta_block_client(configuration, base_url=url)
        assert str(exc_info.value) == "letta_rest_base_url_invalid"


def test_crud_uses_native_ids_but_exposes_stable_memupdatebench_id():
    transport = FakeTransport()
    client = _client(transport)
    value = {
        "canonical_object_id": "default|alice|city|",
        "content": "Add default|alice|city| with value \"Paris\".",
        "value": "Paris",
        "source_event_id": "event-1",
        "sequence_index": 0,
    }
    block_id = "letta-block-" + hashlib.sha256(b"default|alice|city|").hexdigest()[:32]
    client.create_block("memupdatebench/ns-a", block_id, value)
    rows = client.search_blocks("memupdatebench/ns-a")
    assert rows == ((block_id, value),)
    assert client.get_block("memupdatebench/ns-a", rows[0][0]) == value
    updated = dict(value, value="Lyon", source_event_id="event-2", sequence_index=1)
    client.update_block("memupdatebench/ns-a", rows[0][0], updated)
    assert client.get_block("memupdatebench/ns-a", rows[0][0]) == updated
    client.delete_block("memupdatebench/ns-a", rows[0][0])
    assert client.search_blocks("memupdatebench/ns-a") == ()
    assert [call[0] for call in transport.calls] == ["POST", "GET", "GET", "GET", "GET", "PATCH", "GET", "GET", "GET", "DELETE", "GET"]


def test_namespace_filtering_and_malformed_marked_blocks_fail_closed():
    transport = FakeTransport()
    client = _client(transport)
    good = {
        "canonical_object_id": "default|alice|city|",
        "content": "x",
        "value": "Paris",
        "source_event_id": "e",
        "sequence_index": 0,
    }
    block_id = "letta-block-" + hashlib.sha256(b"default|alice|city|").hexdigest()[:32]
    client.create_block("memupdatebench/ns-a", block_id, good)
    other = client._payload("memupdatebench/ns-b", good)
    other["id"] = "foreign"
    transport.blocks["foreign"] = other
    assert len(client.search_blocks("memupdatebench/ns-a")) == 1
    transport.blocks["bad"] = dict(client._payload("memupdatebench/ns-a", good), id="bad", value="not-json")
    with pytest.raises(LettaRestDependencyUnavailable) as exc_info:
        client.search_blocks("memupdatebench/ns-a")
    assert str(exc_info.value) == "letta_rest_block_malformed"


def test_http_errors_are_stable_and_redacted():
    configuration = build_letta_adapter_configuration(run_id="error-test")

    def failing(*args, **kwargs):
        return FakeResponse(503, {"token": "secret", "detail": "private"})

    client = build_rest_letta_block_client(configuration, base_url="http://localhost:8283", request_fn=failing)
    with pytest.raises(LettaRestDependencyUnavailable) as exc_info:
        client.search_blocks("memupdatebench/ns")
    assert str(exc_info.value) == "letta_rest_http_5xx"
    assert "secret" not in str(exc_info.value)


def test_default_worker_factory_still_fails_closed_without_explicit_endpoint(monkeypatch):
    from mub.vnext.external.workers import letta_worker

    configuration = build_letta_adapter_configuration(run_id="default-factory-test")
    evidence = {
        "identity_verified": True,
        "installed_content_sha256": "a" * 64,
        "installed_content_file_count": 1,
        "installed_content_verified": True,
        "source_binding_verified": True,
    }
    monkeypatch.setattr(letta_worker, "inspect_local_letta_package", lambda: evidence)
    monkeypatch.delenv("LETTA_NATIVE_API_BASE_URL", raising=False)
    with pytest.raises(letta_worker.LettaDependencyUnavailable) as exc_info:
        letta_worker.build_official_letta_backend(configuration)
    assert str(exc_info.value) == "letta_native_api_unverified"
