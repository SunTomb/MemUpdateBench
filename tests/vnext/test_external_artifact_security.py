from __future__ import annotations

import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMMUTABLE_CORE_ROOT = PROJECT_ROOT / "data" / "vnext" / "core" / "v3"


def test_secret_scan_reports_rules_without_copying_secret_values():
    from mub.vnext.external.security import scan_for_secrets

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    findings = scan_for_secrets(
        {
            "nested": {
                "api_key": secret,
                "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            },
            "private": "-----BEGIN PRIVATE KEY-----\nabc\n",
        }
    )

    assert {finding.rule for finding in findings} >= {
        "sensitive_field",
        "bearer_token",
        "private_key",
        "openai_key",
    }
    rendered = " ".join(
        f"{finding.location}:{finding.rule}" for finding in findings
    )
    assert secret not in rendered
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered


def test_public_payload_guard_rejects_secrets_and_redacts_error_text():
    from mub.vnext.external.artifacts import RawPayloadLicenseStatus
    from mub.vnext.external.security import (
        redact_sensitive_text,
        require_redistributable_payload,
    )

    secret = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"
    with pytest.raises(ValueError, match="security scan") as exc_info:
        require_redistributable_payload(
            {"error": f"provider said {secret}"},
            license_status=RawPayloadLicenseStatus.REDISTRIBUTABLE,
        )
    assert secret not in str(exc_info.value)

    composite_credentials = (
        "client_secret=clientvalue; access_token=opaquevalue"
    )
    with pytest.raises(ValueError, match="security scan"):
        require_redistributable_payload(
            {"error": composite_credentials},
            license_status=RawPayloadLicenseStatus.REDISTRIBUTABLE,
        )

    redacted = redact_sensitive_text(
        "Authorization: Bearer abcdefghijklmnop; "
        f"key={secret}; password=hunter2; "
        "client_secret=clientvalue; access_token=opaquevalue"
    )
    assert secret not in redacted
    assert "abcdefghijklmnop" not in redacted
    assert "hunter2" not in redacted
    assert "clientvalue" not in redacted
    assert "opaquevalue" not in redacted
    assert "[REDACTED]" in redacted


def test_public_payload_guard_allows_noncredential_usage_metadata():
    from mub.vnext.external.artifacts import RawPayloadLicenseStatus
    from mub.vnext.external.security import require_redistributable_payload

    payload = {
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 4,
        },
        "configuration_hash": "a" * 64,
    }
    assert require_redistributable_payload(
        payload,
        license_status=RawPayloadLicenseStatus.REDISTRIBUTABLE,
    ) is payload


@pytest.mark.parametrize(
    "license_status",
    [
        None,
        "private",
        "license_uncertain",
    ],
)
def test_redistributable_payload_requires_explicit_compatible_license(
    license_status,
):
    from mub.vnext.external.artifacts import RawPayloadLicenseStatus
    from mub.vnext.external.security import require_redistributable_payload

    if license_status is not None:
        license_status = RawPayloadLicenseStatus(license_status)
    with pytest.raises(ValueError, match="license"):
        require_redistributable_payload(
            {"payload": "normalized provider record"},
            license_status=license_status,
        )


def test_worker_environment_is_explicit_allowlist_and_never_mutates_source():
    from mub.vnext.external.security import build_worker_environment

    source = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "OPENAI_API_KEY": "environment-secret",
        "UNRELATED_SECRET": "must-not-pass",
    }
    before = dict(source)
    worker_environment = build_worker_environment(
        source,
        allowed_names=("PATH", "SYSTEMROOT", "OPENAI_API_KEY"),
        required_names=("OPENAI_API_KEY",),
    )

    assert dict(worker_environment) == {
        "OPENAI_API_KEY": "environment-secret",
        "PATH": source["PATH"],
        "SYSTEMROOT": source["SYSTEMROOT"],
    }
    assert "UNRELATED_SECRET" not in worker_environment
    assert source == before
    with pytest.raises(TypeError):
        worker_environment["NEW_SECRET"] = "forged"


def test_worker_environment_rejects_missing_or_unsafe_names():
    from mub.vnext.external.security import build_worker_environment

    with pytest.raises(ValueError, match="required worker environment"):
        build_worker_environment(
            {},
            allowed_names=("OPENAI_API_KEY",),
            required_names=("OPENAI_API_KEY",),
        )
    with pytest.raises(ValueError, match="environment variable name"):
        build_worker_environment(
            {"BAD-NAME": "value"},
            allowed_names=("BAD-NAME",),
        )


def test_external_artifact_contract_separates_private_raw_from_public_paths():
    from mub.vnext.external.artifacts import (
        NormalizedArtifactRefV1,
        PrivateRawArtifactRefV1,
    )

    private_ref = PrivateRawArtifactRefV1(
        sha256="1" * 64,
        size_bytes=123,
        media_type="application/json",
    )
    assert "path" not in type(private_ref).model_fields
    normalized_ref = NormalizedArtifactRefV1(
        path="normalized/run.jsonl",
        sha256="2" * 64,
        media_type="application/x-ndjson",
        record_count=64,
        private_raw_hashes=(private_ref.sha256,),
        redaction_version="external-redaction-v1",
    )
    assert normalized_ref.private_raw_hashes == ("1" * 64,)

    with pytest.raises(ValueError, match="at least one"):
        NormalizedArtifactRefV1(
            path="normalized/run.jsonl",
            sha256="2" * 64,
            media_type="application/x-ndjson",
            record_count=64,
            private_raw_hashes=(),
            redaction_version="external-redaction-v1",
        )

    with pytest.raises(ValueError, match="unique"):
        NormalizedArtifactRefV1(
            path="normalized/run.jsonl",
            sha256="2" * 64,
            media_type="application/x-ndjson",
            record_count=64,
            private_raw_hashes=(private_ref.sha256, private_ref.sha256),
            redaction_version="external-redaction-v1",
        )


def test_artifact_roots_reject_overlap_and_reparse_components(tmp_path):
    from mub.vnext.external.artifacts import validate_external_artifact_roots

    private_root = tmp_path / "private"
    normalized_root = tmp_path / "normalized"
    private_root.mkdir()
    normalized_root.mkdir()
    validated = validate_external_artifact_roots(
        private_root,
        normalized_root,
    )
    assert validated.private_raw_root == private_root.resolve()
    assert validated.normalized_root == normalized_root.resolve()

    nested = private_root / "public"
    nested.mkdir()
    with pytest.raises(ValueError, match="must not overlap"):
        validate_external_artifact_roots(private_root, nested)

    with pytest.raises(ValueError, match="outside immutable Core"):
        validate_external_artifact_roots(
            IMMUTABLE_CORE_ROOT.parent,
            normalized_root,
        )


def test_artifact_roots_reject_reparse_components(tmp_path):
    from mub.vnext.external.artifacts import validate_external_artifact_roots

    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "private-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    with pytest.raises(ValueError, match="reparse"):
        validate_external_artifact_roots(link, normalized_root)
