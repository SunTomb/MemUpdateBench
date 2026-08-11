from __future__ import annotations

from collections.abc import Mapping
import re
from types import MappingProxyType
from typing import Any
import unicodedata

from pydantic import Field

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.external.artifacts import RawPayloadLicenseStatus


_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "apikey",
        "authorization",
        "authtoken",
        "accesstoken",
        "bearertoken",
        "credential",
        "credentials",
        "idtoken",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "token",
    }
)
_FIELD_NONALNUM = re.compile(r"[^0-9a-z]+")
_SECRET_PATTERNS = (
    (
        "credential_assignment",
        re.compile(
            r"\b(?:access[_-]?token|api[_-]?key|auth[_-]?token|authorization|"
            r"client[_-]?secret|id[_-]?token|password|private[_-]?key|"
            r"refresh[_-]?token|secret|token)"
            r"\s*[:=]\s*(?:Bearer\s+)?[^\s,;]+",
            re.IGNORECASE,
        ),
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    ),
    (
        "bearer_token",
        re.compile(
            r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    ),
    (
        "google_api_key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
)


class SecretScanFindingV1(ImmutableContractModel):
    location: str = Field(strict=True, min_length=1)
    rule: str = Field(strict=True, min_length=1)


def _location(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "$"


def _sensitive_field_name(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = _FIELD_NONALNUM.sub("", normalized)
    return compact in _SENSITIVE_FIELD_NAMES or compact.endswith(
        ("apikey", "credential", "password", "privatekey", "secret")
    )


def scan_for_secrets(value: Any) -> tuple[SecretScanFindingV1, ...]:
    findings: list[SecretScanFindingV1] = []
    seen: set[tuple[str, str]] = set()

    def add(path: tuple[str, ...], rule: str) -> None:
        key = (_location(path), rule)
        if key in seen:
            return
        seen.add(key)
        findings.append(
            SecretScanFindingV1(location=key[0], rule=rule)
        )

    def visit(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if type(key) is not str:
                    add((*path, "<non-string-key>"), "non_string_key")
                    continue
                child_path = (*path, key)
                if _sensitive_field_name(key):
                    add(child_path, "sensitive_field")
                visit(nested, child_path)
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, (*path, str(index)))
        elif type(item) is str:
            for rule, pattern in _SECRET_PATTERNS:
                if pattern.search(item):
                    add(path, rule)
        elif isinstance(item, (bytes, bytearray, memoryview)):
            add(path, "binary_payload")

    visit(value, ())
    return tuple(
        sorted(findings, key=lambda finding: (finding.location, finding.rule))
    )


def redact_sensitive_text(value: str) -> str:
    if type(value) is not str:
        raise ValueError("redaction input must be an exact built-in string")
    redacted = value
    for _, pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def require_redistributable_payload(
    value: Any,
    *,
    license_status: RawPayloadLicenseStatus | None = None,
) -> Any:
    if (
        type(license_status) is not RawPayloadLicenseStatus
        or license_status is not RawPayloadLicenseStatus.REDISTRIBUTABLE
    ):
        raise ValueError(
            "redistributable payload requires an explicit compatible license"
        )
    findings = scan_for_secrets(value)
    if findings:
        rules = ",".join(sorted({finding.rule for finding in findings}))
        raise ValueError(
            "redistributable payload failed security scan: "
            f"{len(findings)} finding(s); rules={rules}"
        )
    return value


def build_worker_environment(
    source_environment: Mapping[str, str],
    *,
    allowed_names: tuple[str, ...],
    required_names: tuple[str, ...] = (),
) -> Mapping[str, str]:
    if type(allowed_names) is not tuple or type(required_names) is not tuple:
        raise ValueError("worker environment names must use exact tuples")
    if len(allowed_names) != len(set(allowed_names)):
        raise ValueError("allowed worker environment names must be unique")
    allowed = set(allowed_names)
    if not set(required_names) <= allowed:
        raise ValueError(
            "required worker environment names must be allowed"
        )
    for name in (*allowed_names, *required_names):
        if type(name) is not str or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError("invalid environment variable name")
    selected: dict[str, str] = {}
    for name in sorted(allowed):
        if name not in source_environment:
            continue
        value = source_environment[name]
        if type(value) is not str:
            raise ValueError("worker environment values must be strings")
        selected[name] = value
    missing = tuple(
        name
        for name in required_names
        if name not in selected or not selected[name]
    )
    if missing:
        raise ValueError(
            "required worker environment variables are unavailable: "
            + ",".join(missing)
        )
    return MappingProxyType(selected)


__all__ = [
    "SecretScanFindingV1",
    "build_worker_environment",
    "redact_sensitive_text",
    "require_redistributable_payload",
    "scan_for_secrets",
]
