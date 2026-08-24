from __future__ import annotations

import json
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
