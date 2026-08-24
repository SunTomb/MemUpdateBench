from __future__ import annotations

from mub.vnext.post_core.qualification_receipts_v1 import (
    ProviderCapabilityAttestationV1,
    ProviderObservationV1,
    ProviderSetupEventV1,
)


SOURCE_BINDING_IDS = ("workflow_source", "handoff_source")


def _observation(
    location: str,
    observation_id: str,
    request_name: str,
    response_format: str = "ANTHROPIC_MESSAGE_JSON",
) -> ProviderObservationV1:
    return ProviderObservationV1(
        location=location,
        observation_id=observation_id,
        response_format=response_format,
        response_model=request_name,
        stop_reason="end_turn",
        usage_present=True,
    )


def _paired_observations(request_name: str, prefix: str) -> tuple[ProviderObservationV1, ...]:
    return (
        _observation("LOCAL", f"{prefix}_LOCAL", request_name),
        _observation("TANG2", f"{prefix}_TANG2", request_name),
    )


def provider_attestations() -> tuple[ProviderCapabilityAttestationV1, ...]:
    claude_sonnet = "claude-sonnet-4-6"
    claude_opus = "claude-opus-4-8"
    gemini = "Gemini 3.6 Flash (Low)"
    grok = "grok-4.5"
    gpt = "gpt-5.5"
    return (
        ProviderCapabilityAttestationV1(
            registry_key="claude_sonnet_4_6",
            request_name=claude_sonnet,
            canonical_model_identity=claude_sonnet,
            observations=_paired_observations(claude_sonnet, "CLAUDE_SONNET"),
            provider_call_count=2,
            source_binding_ids=SOURCE_BINDING_IDS,
        ),
        ProviderCapabilityAttestationV1(
            registry_key="claude_opus_4_8",
            request_name=claude_opus,
            canonical_model_identity=claude_opus,
            observations=_paired_observations(claude_opus, "CLAUDE_OPUS"),
            provider_call_count=2,
            source_binding_ids=SOURCE_BINDING_IDS,
        ),
        ProviderCapabilityAttestationV1(
            registry_key="gemini_3_6_flash",
            request_name=gemini,
            canonical_model_identity="gemini-3.6-flash",
            reasoning_tier="Low",
            observations=_paired_observations(gemini, "GEMINI_FLASH"),
            provider_call_count=2,
            source_binding_ids=SOURCE_BINDING_IDS,
        ),
        ProviderCapabilityAttestationV1(
            registry_key="grok_4_5",
            request_name=grok,
            identity_caveat="explicitly mutable transfer alias",
            observations=_paired_observations(grok, "GROK"),
            provider_call_count=2,
            source_binding_ids=SOURCE_BINDING_IDS,
        ),
        ProviderCapabilityAttestationV1(
            registry_key="gpt_5_5",
            request_name=gpt,
            identity_caveat="unverified official upstream identity",
            observations=(
                _observation("LOCAL", "LOCAL_INITIAL_SSE", gpt, "SSE"),
                _observation("LOCAL", "LOCAL_EXPLICIT_FALSE_SSE", gpt, "SSE"),
                _observation("TANG2", "TANG2_PREFIX_SSE", gpt, "SSE"),
                _observation("TANG2", "TANG2_POSTFIX_JSON", gpt),
            ),
            provider_call_count=4,
            source_binding_ids=SOURCE_BINDING_IDS,
        ),
    )


def failed_ssh_setup_event() -> ProviderSetupEventV1:
    return ProviderSetupEventV1(
        event_id="TANG2_SSH_SETUP_FAILED",
        reason_class="command_quoting",
        source_binding_ids=SOURCE_BINDING_IDS,
    )


__all__ = ["SOURCE_BINDING_IDS", "failed_ssh_setup_event", "provider_attestations"]
