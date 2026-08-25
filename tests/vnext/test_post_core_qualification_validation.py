from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.qualification_receipts_v1 import ProviderCapabilityAttestationV1
from tests.vnext.qualification_fixtures import (
    SOURCE_BINDING_IDS,
    failed_ssh_setup_event,
    provider_attestations,
)


def _replace_row(
    rows: tuple[ProviderCapabilityAttestationV1, ...], index: int, **changes: object
) -> tuple[ProviderCapabilityAttestationV1, ...]:
    replacement = rows[index].model_copy(update=changes)
    return (*rows[:index], replacement, *rows[index + 1 :])


def _replace_observation(
    row: ProviderCapabilityAttestationV1, index: int, **changes: object
) -> ProviderCapabilityAttestationV1:
    observations = list(row.observations)
    observations[index] = observations[index].model_copy(update=changes)
    return row.model_copy(update={"observations": tuple(observations)})


def test_provider_attestations_have_exact_counts_and_total() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    rows = validate_provider_attestations_v1(provider_attestations())

    assert {row.registry_key: row.provider_call_count for row in rows} == {
        "claude_sonnet_4_6": 2,
        "claude_opus_4_8": 2,
        "gemini_3_6_flash": 2,
        "grok_4_5": 2,
        "gpt_5_5": 4,
    }
    assert sum(row.provider_call_count for row in rows) == 12
    assert all(row.retry_count == 0 and row.benchmark_generation_count == 0 for row in rows)


@pytest.mark.parametrize(
    "changes",
    [
        {"request_name": "gemini-3.6-flash"},
        {"canonical_model_identity": "Gemini 3.6 Flash (Low)"},
        {"reasoning_tier": "High"},
    ],
)
def test_gemini_identity_triplet_is_exact(changes: dict[str, object]) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    with pytest.raises(ValueError):
        validate_provider_attestations_v1(_replace_row(provider_attestations(), 2, **changes))


def test_gpt_format_and_observation_order_are_exact() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    rows = provider_attestations()
    gpt = rows[-1]
    assert tuple(item.response_format for item in gpt.observations) == (
        "SSE", "SSE", "SSE", "ANTHROPIC_MESSAGE_JSON"
    )
    assert tuple(item.observation_id for item in gpt.observations) == (
        "LOCAL_INITIAL_SSE", "LOCAL_EXPLICIT_FALSE_SSE", "TANG2_PREFIX_SSE", "TANG2_POSTFIX_JSON"
    )
    with pytest.raises(ValueError, match="GPT"):
        validate_provider_attestations_v1(
            _replace_row(
                rows,
                4,
                observations=_replace_observation(
                    gpt, 1, observation_id="LOCAL_MERGED_SSE"
                ).observations,
            )
        )
    with pytest.raises(ValueError, match="GPT"):
        validate_provider_attestations_v1(
            _replace_row(rows, 4, observations=tuple(reversed(gpt.observations)))
        )
    with pytest.raises(ValueError, match="GPT"):
        validate_provider_attestations_v1(
            _replace_row(
                rows,
                4,
                observations=_replace_observation(
                    gpt, 2, response_format="ANTHROPIC_MESSAGE_JSON"
                ).observations,
            )
        )


def test_failed_ssh_setup_event_has_zero_provider_calls_and_setup_stage() -> None:
    event = failed_ssh_setup_event()

    assert event.stage == "PRE_PROVIDER_SETUP"
    assert event.status == "FAILED"
    assert event.provider_call_count == 0
    assert event.reason_class == "command_quoting"
    assert event.source_binding_ids == SOURCE_BINDING_IDS


@pytest.mark.parametrize(
    "value",
    [
        {"Authorization": "Bearer opaque-value"},
        {"value": "api_key=opaque-value"},
        {"value": "-----BEGIN PRIVATE KEY-----\nopaque\n-----END PRIVATE KEY-----"},
        {"value": "AIzaSyA123456789012345678901234567890"},
        {"value": "AKIAIOSFODNN7EXAMPLE"},
        {"endpoint": "https://user:pass@example.test/api"},
        {"endpoint": "http://example.test/api"},
        {"endpoint": "not-a-url"},
        {"source-url": "https://example.test/data?token=opaque"},
        {"endpoint url": "https://example.test/data#fragment"},
        {"credential_env_var": "UNALLOWLISTED_SECRET"},
    ],
)
def test_secret_and_url_scan_rejects_synthetic_sensitive_payloads(value: object) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    with pytest.raises(ValueError):
        validate_qualification_secret_free(value)


def test_secret_scan_allows_plain_https_endpoint_and_never_reads_environment(monkeypatch) -> None:
    import mub.vnext.post_core.provenance_v1 as provenance_v1
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    class TrapEnvironment:
        def __getattribute__(self, name: str):
            raise AssertionError(f"environment access is forbidden: {name}")

    monkeypatch.setattr(provenance_v1, "os", SimpleNamespace(environ=TrapEnvironment()), raising=False)
    validate_qualification_secret_free({"endpoint": "https://example.test/api"})
    validate_qualification_secret_free({"credential_env_var": "ANTHROPIC_API_KEY"})


def test_load_canonical_jsonl_roundtrip_and_rejects_invalid_rows(tmp_path: Path) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import load_canonical_jsonl_v1

    row = provider_attestations()[0]
    raw = canonical_bytes(row) + b"\n"
    path = tmp_path / "attestations.jsonl"
    path.write_bytes(raw)
    loaded, observed_raw = load_canonical_jsonl_v1(path, ProviderCapabilityAttestationV1, label="attestations")
    assert loaded == (row,)
    assert observed_raw == raw

    cases = {
        "pretty.jsonl": json.dumps(row.model_dump(mode="json"), indent=2).encode() + b"\n",
        "missing-lf.jsonl": canonical_bytes(row),
        "empty-row.jsonl": raw + b"\n",
        "secret.jsonl": b'{"api_key":"opaque"}\n',
        "empty.jsonl": b"",
        "typed-invalid.jsonl": b'{"registry_key":"missing-contract-fields"}\n',
    }
    for name, contents in cases.items():
        candidate = tmp_path / name
        candidate.write_bytes(contents)
        with pytest.raises(ValueError) as exc_info:
            load_canonical_jsonl_v1(candidate, ProviderCapabilityAttestationV1, label="attestations")
        assert "opaque" not in str(exc_info.value)


def test_load_canonical_jsonl_rejects_symlink_when_host_permits(tmp_path: Path) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import load_canonical_jsonl_v1

    target = tmp_path / "target.jsonl"
    target.write_bytes(canonical_bytes(provider_attestations()[0]) + b"\n")
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(ValueError, match="regular single-link|reparse"):
        load_canonical_jsonl_v1(link, ProviderCapabilityAttestationV1, label="attestations")


@pytest.mark.parametrize(
    "rows",
    [
        lambda base: tuple(reversed(base)),
        lambda base: base[:-1],
        lambda base: (*base[:-1], base[0]),
        lambda base: _replace_row(base, 0, provider_call_count=1),
        lambda base: _replace_row(base, 0, raw_response_persisted=True),
        lambda base: _replace_row(base, 0, retry_count=1),
        lambda base: _replace_row(base, 0, benchmark_generation_count=1),
        lambda base: _replace_row(base, 0, source_binding_ids=()),
        lambda base: _replace_row(base, 0, source_binding_ids=("workflow_source", "workflow_source")),
        lambda base: _replace_row(base, 0, observations=(_replace_observation(base[0], 0, response_model="wrong").observations[0], base[0].observations[1])),
        lambda base: _replace_row(base, 0, observations=(_replace_observation(base[0], 0, retry_count=1).observations[0], base[0].observations[1])),
        lambda base: _replace_row(base, 1, observations=(_replace_observation(base[1], 0, observation_id=base[0].observations[0].observation_id).observations[0], base[1].observations[1])),
    ],
)
def test_provider_attestation_validation_rejects_structural_mutations(rows) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    with pytest.raises(ValueError):
        validate_provider_attestations_v1(rows(provider_attestations()))


def _unsafe_observation(observation, **changes: object):
    payload = observation.model_dump(mode="python")
    payload.update(changes)
    return type(observation).model_construct(**payload)


def _unsafe_row(row, **changes: object):
    payload = row.model_dump(mode="python")
    payload.update(changes)
    return type(row).model_construct(**payload)


def test_secret_scan_delegates_first_with_environment_reading_disabled(monkeypatch) -> None:
    import mub.vnext.post_core.qualification_validation_v1 as validation

    calls = []

    def delegate(value, *, read_environment: bool) -> None:
        calls.append((value, read_environment))

    monkeypatch.setattr(validation, "validate_secret_free", delegate)
    with pytest.raises(ValueError, match="credential"):
        validation.validate_qualification_secret_free({"backupToken": "opaque"})
    assert calls == [({"backupToken": "opaque"}, False)]


@pytest.mark.parametrize(
    "value",
    [
        {"backup_token": "opaque"},
        {"backupToken": "opaque"},
        {"backup-token": "opaque"},
        {"backup token": "opaque"},
        {"value": "backup_token=opaque"},
        {"endpointUrl": "https://example.test/?key=opaque"},
        {"sourceUrl": "https://example.test/?sig=opaque"},
        {"endpoint_url": "https://example.test/?signature=opaque"},
        {"source_url": "https://example.test/?X-Amz-Signature=opaque"},
        {"endpoint": "https://example.test/?X-Goog-Signature=opaque"},
    ],
)
def test_secret_scan_rejects_normalized_credential_components_and_query_keys(value: object) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    with pytest.raises(ValueError):
        validate_qualification_secret_free(value)


def test_provider_identity_metadata_is_exact_and_complete() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    rows = provider_attestations()
    sonnet, opus, gemini, grok, gpt = rows
    assert (sonnet.canonical_model_identity, sonnet.reasoning_tier, sonnet.identity_caveat) == (
        "claude-sonnet-4-6", None, None
    )
    assert (opus.canonical_model_identity, opus.reasoning_tier, opus.identity_caveat) == (
        "claude-opus-4-8", None, None
    )
    assert (gemini.canonical_model_identity, gemini.reasoning_tier, gemini.identity_caveat) == (
        "gemini-3.6-flash", "Low", None
    )
    assert (grok.canonical_model_identity, grok.reasoning_tier, grok.identity_caveat) == (
        None, None, "explicitly mutable transfer alias"
    )
    assert (gpt.canonical_model_identity, gpt.reasoning_tier, gpt.identity_caveat) == (
        None, None, "unverified official upstream identity"
    )
    for index, changes in (
        (0, {"reasoning_tier": "Low"}),
        (0, {"identity_caveat": "unexpected"}),
        (1, {"reasoning_tier": "Low"}),
        (1, {"identity_caveat": "unexpected"}),
        (2, {"identity_caveat": "unexpected"}),
        (3, {"canonical_model_identity": "grok-4.5"}),
        (3, {"reasoning_tier": "Low"}),
        (3, {"identity_caveat": "mutable alias"}),
        (4, {"reasoning_tier": "Low"}),
        (4, {"identity_caveat": "unverified"}),
    ):
        with pytest.raises(ValueError):
            validate_provider_attestations_v1(_replace_row(rows, index, **changes))


@pytest.mark.parametrize("row_index", [0, 4])
@pytest.mark.parametrize("changes", [{"stop_reason": None}, {"usage_present": False}])
def test_all_retained_observations_require_end_turn_and_usage(row_index: int, changes: dict[str, object]) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    rows = provider_attestations()
    row = rows[row_index]
    with pytest.raises(ValueError):
        validate_provider_attestations_v1(
            _replace_row(rows, row_index, observations=_replace_observation(row, 0, **changes).observations)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"http_status": 500},
        {"exact_ok": False},
        {"provider_call_count": 0},
        {"retry_count": 1},
        {"response_model": "wrong"},
        {"response_format": "SSE"},
    ],
)
def test_observation_contract_mutations_are_rejected(changes: dict[str, object]) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    rows = provider_attestations()
    changed = _unsafe_observation(rows[0].observations[0], **changes)
    invalid_row = _unsafe_row(rows[0], observations=(changed, rows[0].observations[1]))
    with pytest.raises(ValueError):
        validate_provider_attestations_v1((invalid_row, *rows[1:]))


def test_load_canonical_jsonl_rejects_hardlink_when_host_permits(tmp_path: Path) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import load_canonical_jsonl_v1

    target = tmp_path / "target.jsonl"
    target.write_bytes(canonical_bytes(provider_attestations()[0]) + b"\n")
    linked = tmp_path / "linked.jsonl"
    try:
        os.link(target, linked)
    except OSError:
        pytest.skip("host does not permit hardlink creation")
    with pytest.raises(ValueError, match="regular single-link"):
        load_canonical_jsonl_v1(linked, ProviderCapabilityAttestationV1, label="attestations")


def test_load_canonical_jsonl_rejects_replacement_before_reading_swapped_bytes(tmp_path: Path, monkeypatch) -> None:
    import mub.vnext.post_core.qualification_validation_v1 as validation

    path = tmp_path / "attestations.jsonl"
    path.write_bytes(canonical_bytes(provider_attestations()[0]) + b"\n")
    attacker = tmp_path / "attacker.jsonl"
    attacker.write_bytes(b"ATTACKER_BYTES_MUST_NOT_BE_READ")
    original_open = os.open
    read_attempts = []

    def open_after_swap(name, flags, *args):
        os.replace(attacker, path)
        return original_open(name, flags, *args)

    def trap_read(*args):
        read_attempts.append(args)
        raise AssertionError("swapped-in bytes must not be read")

    monkeypatch.setattr(validation.os, "open", open_after_swap)
    monkeypatch.setattr(validation, "_read_fd_all", trap_read, raising=False)
    with pytest.raises(ValueError, match="changed while being read"):
        validation.load_canonical_jsonl_v1(path, ProviderCapabilityAttestationV1, label="attestations")
    assert read_attempts == []


def test_qualification_validation_exports_only_the_contract_functions() -> None:
    import mub.vnext.post_core.qualification_validation_v1 as validation

    assert validation.__all__ == [
        "load_capability_anomaly_receipt_v1",
        "load_canonical_jsonl_v1",
        "load_execution_authorization_v1",
        "validate_capability_attempt_receipts_v1",
        "validate_canonical_capability_smoke_plan_v1",
        "validate_escalation_anomaly_evidence_v1",
        "validate_escalation_anomaly_receipt_v1",
        "validate_provider_attestations_v1",
        "validate_qualification_secret_free",
        "validate_runtime_receipts_v1",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "credential",
        "backup_token",
        "APIKey",
        "XAPIKey",
        "accessToken",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\nopaque",
        {"value": "XAPIKey=opaque"},
    ],
)
def test_post_scan_rejects_standalone_sensitive_identifier_values(value: object) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    with pytest.raises(ValueError):
        validate_qualification_secret_free(value)


@pytest.mark.parametrize(
    "value",
    [
        "the credential is redacted in prose",
        "this APIKey label appears in a sentence",
        {"api_version": "v1"},
        {"key": "public-label"},
        {"key_count": 2},
    ],
)
def test_post_scan_allows_benign_prose_and_generic_mapping_keys(value: object) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    validate_qualification_secret_free(value)


@pytest.mark.parametrize(
    "url",
    [
        "https://exa mple.test/path",
        "https://example.test/%ZZ",
        "https:///missing-host",
        "https://-leading.test/path",
        "https://trailing-.test/path",
        "https://double..label.test/path",
        "https://example!.test/path",
    ],
)
def test_url_validation_rejects_whitespace_bad_escapes_and_invalid_host_syntax(url: str) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    with pytest.raises(ValueError):
        validate_qualification_secret_free({"endpoint": url})


@pytest.mark.parametrize(
    "query_key",
    ["key", "sig", "signature", "private", "APIKey", "XAPIKey", "X-Amz-Signature", "X-Goog-Signature", "access_token"],
)
def test_url_query_validation_rejects_generic_and_compact_credential_keys(query_key: str) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    with pytest.raises(ValueError):
        validate_qualification_secret_free({"endpoint": f"https://example.test/?{query_key}=opaque"})


def test_url_query_validation_rejects_percent_encoded_private_key_armor() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    encoded_armor = "-----BEGIN%20PGP%20PRIVATE%20KEY%20BLOCK-----"
    with pytest.raises(ValueError):
        validate_qualification_secret_free({"endpoint": f"https://example.test/?payload={encoded_armor}"})


def test_url_validation_rejects_overlong_idna_dns_host() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    overlong_host = ".".join(["a" * 63] * 4)
    assert len(overlong_host.encode("ascii")) > 253
    with pytest.raises(ValueError):
        validate_qualification_secret_free({"endpoint": f"https://{overlong_host}/path"})


@pytest.mark.parametrize("row_index", [0, 4])
@pytest.mark.parametrize(
    "changes",
    [
        {"stop_reason": None},
        {"stop_reason": "other"},
        {"usage_present": False},
        {"usage_present": None},
    ],
)
def test_all_observations_reject_missing_or_substituted_retention_metadata(
    row_index: int, changes: dict[str, object]
) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_provider_attestations_v1

    rows = provider_attestations()
    row = rows[row_index]
    changed = _unsafe_observation(row.observations[0], **changes)
    invalid_row = _unsafe_row(row, observations=(changed, *row.observations[1:]))
    with pytest.raises(ValueError):
        validate_provider_attestations_v1((*rows[:row_index], invalid_row, *rows[row_index + 1:]))


def test_load_canonical_jsonl_rejects_same_inode_same_size_timestamp_mutation(tmp_path: Path, monkeypatch) -> None:
    import mub.vnext.post_core.qualification_validation_v1 as validation

    path = tmp_path / "attestations.jsonl"
    path.write_bytes(canonical_bytes(provider_attestations()[0]) + b"\n")
    original_read = validation._read_fd_all

    def read_then_mutate(descriptor: int) -> bytes:
        raw = original_read(descriptor)
        current = path.stat()
        os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000))
        return raw

    monkeypatch.setattr(validation, "_read_fd_all", read_then_mutate)
    with pytest.raises(ValueError, match="changed while being read"):
        validation.load_canonical_jsonl_v1(path, ProviderCapabilityAttestationV1, label="attestations")


@pytest.mark.parametrize(
    "value",
    [
        "backup_private_key",
        "privateKeyBackup",
        "private-key-backup",
        {"private key backup": "opaque"},
        "AWSAccessKeyId",
        "AWSAccessKeyID",
        "AWSSecretAccessKey",
        "GCPServiceAccountKey",
        {"AWSAccessKeyId": "opaque"},
        {"AWSAccessKeyID": "opaque"},
        {"AWSSecretAccessKey": "opaque"},
        {"GCPServiceAccountKey": "opaque"},
    ],
)
def test_post_scan_rejects_private_key_sequences_and_acronym_credentials(value: object) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    with pytest.raises(ValueError):
        validate_qualification_secret_free(value)


@pytest.mark.parametrize(
    "value",
    [
        "private",
        "key",
        "authored_by",
        "tokenizer",
        {"private": "label"},
        {"key": "label"},
        {"api_version": "v1"},
        {"key_count": 2},
        {"authored_by": "writer"},
        {"tokenizer": "identity"},
    ],
)
def test_post_scan_allows_noncredential_private_key_and_word_prefix_counterexamples(value: object) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_qualification_secret_free

    validate_qualification_secret_free(value)
