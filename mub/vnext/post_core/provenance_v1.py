from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from pydantic import Field, StrictInt, StrictStr, field_validator

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.post_core.contracts_v1 import ALLOWED_CREDENTIAL_ENV_NAMES, canonical_hash


_CREDENTIAL_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "x_api_key",
        "authorization",
        "proxy_authorization",
        "bearer",
        "password",
        "secret",
        "private_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "x_auth_token",
        "x_access_token",
        "access_token",
        "x_api_token",
        "x_goog_api_key",
        "x_amz_security_token",
        "www_authenticate",
        "proxy_authenticate",
        "cookie",
        "set_cookie",
        "token",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "google_api_key",
        "xai_api_key",
    }
)
_CREDENTIAL_HEADER_NAMES = (
    "x-api-key",
    "api-key",
    "apikey",
    "authorization",
    "proxy-authorization",
    "x-auth-token",
    "x-access-token",
    "x-api-token",
    "x-goog-api-key",
    "x-amz-security-token",
    "access-token",
    "www-authenticate",
    "proxy-authenticate",
    "cookie",
    "set-cookie",
)
_CREDENTIAL_HEADER = re.compile(
    r"(?:^|[=;\s])(?:" + "|".join(_CREDENTIAL_HEADER_NAMES) + r")\s*:",
    re.IGNORECASE,
)
_SECRET_VALUES = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+\S+|-----BEGIN [A-Z ]+PRIVATE KEY-----)", re.I)


def _normalized_key(value: str) -> str:
    return re.sub(r"[-\s]+", "_", value.strip()).lower()


def _is_credential_key(value: str) -> bool:
    return _normalized_key(value) in _CREDENTIAL_KEY_NAMES


class ProvenanceRecordV1(ImmutableContractModel):
    schema_version: str = "memupdatebench.post-core.provenance.v1"
    registry_key: StrictStr
    identity_status: StrictStr
    evidence_type: StrictStr
    artifact_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: StrictInt = Field(ge=0)
    source_location: StrictStr
    credential_env_var: StrictStr | None = None
    git_revision: StrictStr | None = None
    runtime: Mapping[str, StrictStr] = {}

    @field_validator("credential_env_var")
    @classmethod
    def _credential_env_allowlist(cls, value: str | None) -> str | None:
        if value is not None and value not in ALLOWED_CREDENTIAL_ENV_NAMES:
            raise ValueError("credential environment variable name is not allowlisted")
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(path: Path) -> tuple[str, int, int]:
    root = Path(path).resolve(strict=True)
    rows = []
    total = 0
    for member in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix().encode("utf-8")):
        raw = member.read_bytes()
        total += len(raw)
        rows.append({"path": member.relative_to(root).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return canonical_hash({"entries": rows}), len(rows), total


def _scan(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text == "credential_env_var" and item is not None and item not in ALLOWED_CREDENTIAL_ENV_NAMES:
                raise ValueError("credential environment variable name is not allowlisted")
            if _CREDENTIAL_HEADER.search(key_text):
                raise ValueError("credential header rejected")
            if _is_credential_key(key_text) and key_text != "credential_env_var":
                raise ValueError("secret-like key rejected: credential header/key")
            _scan(item, f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan(item, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _CREDENTIAL_HEADER.search(value):
            raise ValueError("credential header rejected")
        if "=" in value:
            assignment_key = value.split("=", 1)[0].strip()
            if _is_credential_key(assignment_key):
                raise ValueError("credential assignment rejected")
        if _SECRET_VALUES.search(value):
            raise ValueError("secret-like value rejected")


def validate_secret_free(value: Any, *, read_environment: bool = False) -> None:
    # The parameter remains for source compatibility, but Phase 0 never reads
    # credential values from the process environment.
    _scan(value)


_FORBIDDEN_COMMAND_FLAG = re.compile(
    r"^(?:-H|--header|--(?:api[-_]?key|authorization|password|secret|private[-_]?key|bearer|token))(?:=.*)?$",
    re.IGNORECASE,
)


def redacted_command(argv: list[str]) -> tuple[str, ...]:
    if any(_FORBIDDEN_COMMAND_FLAG.fullmatch(part) for part in argv):
        raise ValueError("commands may not contain credential flags")
    validate_secret_free(argv)
    return tuple(argv)


__all__ = ["ProvenanceRecordV1", "redacted_command", "sha256_file", "snapshot_tree", "validate_secret_free"]
