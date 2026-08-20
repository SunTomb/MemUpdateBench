from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import mub.vnext.post_core.provenance_v1 as provenance_v1
from mub.vnext.post_core.provenance_v1 import redacted_command, validate_secret_free


def test_secret_scanner_rejects_keys_values_but_never_reads_environment(monkeypatch) -> None:
    with pytest.raises(ValueError, match="secret-like key"):
        validate_secret_free({"api_key": "redacted"})
    with pytest.raises(ValueError, match="secret-like value"):
        validate_secret_free({"value": "sk-abcdefghijklmnop"})

    class TrapEnvironment:
        def items(self):
            raise AssertionError("Phase 0 must not inspect environment values")

        def get(self, key, default=None):
            return default

        def __getitem__(self, key):
            raise KeyError(key)

        def __contains__(self, key):
            return False

    monkeypatch.setattr(provenance_v1, "os", SimpleNamespace(environ=TrapEnvironment()), raising=False)
    validate_secret_free({"value": "environment-secret-value"})
    validate_secret_free({"credential_env_var": "ANTHROPIC_API_KEY"})


def test_nested_credential_environment_names_are_allowlisted() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        validate_secret_free({"rows": [{"credential_env_var": "EVIL_SECRET"}]})


def test_command_rejects_separated_and_inline_credential_flags() -> None:
    forbidden = (
        "--api-key", "--api-key=secret", "--token", "--token=secret",
        "--authorization", "--authorization=secret", "--password",
        "--password=secret", "--secret", "--secret=secret",
        "--private-key", "--private-key=secret", "--bearer", "--bearer=secret",
    )
    for flag in forbidden:
        with pytest.raises(ValueError, match="credential flags"):
            redacted_command(["tool", flag, "secret"])
    assert redacted_command(["tool", "--registry", "registry.json"]) == (
        "tool", "--registry", "registry.json"
    )


def test_model_identity_tokenizer_metadata_is_not_secret_like() -> None:
    validate_secret_free(
        {
            "tokenizer_identity": "tokenizer@revision",
            "prompt_token_cap": 128,
            "output_token_cap": 32,
        }
    )


def test_credential_headers_are_rejected_without_relying_on_token_shape() -> None:
    commands = (
        ["curl", "-H", "X-Api-Key: opaque"],
        ["curl", "-HX-Api-Key: opaque"],
        ["curl", "-HAuthorization: opaque"],
        ["curl", "--header", "Api-Key: opaque"],
        ["curl", "--header=proxy-authorization: opaque"],
        ["curl", "X-Access-Token: opaque"],
    )
    for command in commands:
        with pytest.raises(ValueError, match="credential|header|authorization"):
            redacted_command(command)

    values = (
        {"X-Api-Key": "opaque"},
        {"headers": {"Api-Key": "opaque"}},
        {"proxy-authorization": "opaque"},
        "X-Api-Key: opaque",
        "Authorization=opaque",
    )
    for value in values:
        with pytest.raises(ValueError, match="credential|header|authorization"):
            validate_secret_free(value)
