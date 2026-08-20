from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from pydantic import Field, StrictInt, StrictStr

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.post_core.contracts_v1 import canonical_hash


_SECRET_KEYS = re.compile(r"(?:api[_-]?key|authorization|bearer|secret|password|private[_-]?key|token)", re.I)
_SECRET_VALUES = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+\S+|-----BEGIN [A-Z ]+PRIVATE KEY-----)", re.I)
_ALLOWED_CREDENTIAL_ENV_NAMES = {
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY",
}


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


def _scan(value: Any, path: str = "$", *, env_values: set[str] | None = None) -> None:
    env_values = env_values or set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEYS.search(key_text) and key_text not in {"credential_env_var", "prompt_token_cap", "output_token_cap"}:
                raise ValueError(f"secret-like key rejected at {path}.{key_text}")
            _scan(item, f"{path}.{key_text}", env_values=env_values)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan(item, f"{path}[{index}]", env_values=env_values)
        return
    if isinstance(value, str):
        if value in env_values or _SECRET_VALUES.search(value):
            raise ValueError(f"secret-like value rejected at {path}")


def validate_secret_free(value: Any, *, read_environment: bool = True) -> None:
    if read_environment:
        env_values = {
            value
            for name, value in os.environ.items()
            if name in _ALLOWED_CREDENTIAL_ENV_NAMES and value
        }
    else:
        env_values = set()
    _scan(value, env_values=env_values)
    if isinstance(value, Mapping):
        env_name = value.get("credential_env_var")
        if env_name is not None and env_name not in _ALLOWED_CREDENTIAL_ENV_NAMES:
            raise ValueError("credential environment variable name is not allowlisted")


def redacted_command(argv: list[str]) -> tuple[str, ...]:
    forbidden = {"--api-key", "--token", "--authorization", "--password"}
    if any(part.lower() in forbidden for part in argv):
        raise ValueError("commands may not contain credential flags")
    validate_secret_free(argv)
    return tuple(argv)


__all__ = ["ProvenanceRecordV1", "redacted_command", "sha256_file", "snapshot_tree", "validate_secret_free"]
