from __future__ import annotations

import os

import pytest

from mub.vnext.post_core.provenance_v1 import redacted_command, validate_secret_free


def test_secret_scanner_rejects_keys_values_and_environment_values(monkeypatch) -> None:
    with pytest.raises(ValueError, match="secret-like key"):
        validate_secret_free({"api_key": "redacted"})
    with pytest.raises(ValueError, match="secret-like value"):
        validate_secret_free({"value": "sk-abcdefghijklmnop"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-secret-value")
    with pytest.raises(ValueError, match="secret-like value"):
        validate_secret_free({"value": "environment-secret-value"})
    validate_secret_free({"credential_env_var": "ANTHROPIC_API_KEY"})


def test_command_rejects_credential_flags() -> None:
    with pytest.raises(ValueError, match="credential flags"):
        redacted_command(["tool", "--api-key", "secret"])
    assert redacted_command(["tool", "--registry", "registry.json"]) == (
        "tool", "--registry", "registry.json"
    )
