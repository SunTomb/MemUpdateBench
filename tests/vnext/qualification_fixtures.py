from __future__ import annotations

from mub.vnext.post_core.qualification_receipts_v1 import (
    GateStatus,
    OpenRuntimeReceiptV1,
    ProviderCapabilityAttestationV1,
    ProviderObservationV1,
    ProviderSetupEventV1,
    RuntimeManifestV1,
)


SOURCE_BINDING_IDS = ("workflow_source", "handoff_source", "provider_attestations")

RUNTIME_SOURCE_BINDINGS = {
    "qwen35_9b_bf16": ("open_snapshot_closure_receipt", "qwen_load_receipt", "runtime_receipts"),
    "meta_muse_glimmer_30b_int4": ("open_snapshot_closure_receipt", "runtime_receipts"),
    "meta_muse_glimmer_30b_bf16": ("open_snapshot_closure_receipt", "runtime_receipts"),
}

_RUNTIME_TREE_HASHES = {
    "qwen35_9b_bf16": "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db",
    "meta_muse_glimmer_30b_int4": "55357aa0a0a9dfe738725f864eb4183e9aa2a0a84da1245b13c47bd85ce9f90f",
    "meta_muse_glimmer_30b_bf16": "7a90420d22f8c98737f15bc31473bbe8a3579ee95f9bf2237172679709877782",
}


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




def _runtime_manifest(
    *,
    engine: str,
    engine_version: str,
    device_name: str = "A40",
    engine_commit: str | None = None,
    binary_sha256: str | None = None,
    build_options_sha256: str | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    transformers_version: str | None = None,
    accelerate_version: str | None = None,
    cuda_version: str | None = None,
    driver_version: str | None = None,
    trust_remote_code: bool,
    compute_dtype: str,
    attention_implementation: str,
    seed: int,
    sampling_mode: str,
    timeout_seconds: int,
    engine_args_sha256: str,
) -> RuntimeManifestV1:
    return RuntimeManifestV1(
        engine=engine,
        engine_version=engine_version,
        engine_commit=engine_commit,
        binary_sha256=binary_sha256,
        python_version=python_version,
        torch_version=torch_version,
        transformers_version=transformers_version,
        accelerate_version=accelerate_version,
        cuda_version=cuda_version,
        driver_version=driver_version,
        device_name=device_name,
        context_tokens=4096,
        max_output_tokens=128,
        build_options_sha256=build_options_sha256,
        trust_remote_code=trust_remote_code,
        compute_dtype=compute_dtype,
        attention_implementation=attention_implementation,
        seed=seed,
        sampling_mode=sampling_mode,
        timeout_seconds=timeout_seconds,
        engine_args_sha256=engine_args_sha256,
    )


def open_runtime_receipts() -> tuple[OpenRuntimeReceiptV1, ...]:
    return (
        OpenRuntimeReceiptV1(
            registry_key="qwen35_9b_bf16",
            revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            snapshot_tree_sha256=_RUNTIME_TREE_HASHES["qwen35_9b_bf16"],
            runtime=_runtime_manifest(
                engine="transformers",
                engine_version="4.57.1",
                python_version="3.11.9",
                torch_version="2.7.1",
                transformers_version="4.57.1",
                accelerate_version="1.10.1",
                cuda_version="12.8",
                driver_version="570.124.06",
                trust_remote_code=False,
                compute_dtype="bf16",
                attention_implementation="sdpa",
                seed=0,
                sampling_mode="greedy",
                timeout_seconds=60,
                engine_args_sha256="a" * 64,
            ),
            load_status=GateStatus.PASS,
            generation_status=GateStatus.NOT_RUN,
            determinism_status=GateStatus.NOT_RUN,
            unload_status=GateStatus.PASS,
            source_binding_ids=RUNTIME_SOURCE_BINDINGS["qwen35_9b_bf16"],
        ),
        OpenRuntimeReceiptV1(
            registry_key="meta_muse_glimmer_30b_int4",
            revision="70bf1b61ac09f91b24d39038091b41c582bc5d7a",
            snapshot_tree_sha256=_RUNTIME_TREE_HASHES["meta_muse_glimmer_30b_int4"],
            runtime=_runtime_manifest(
                engine="llama.cpp",
                engine_version="b1",
                engine_commit="d" * 40,
                binary_sha256="e" * 64,
                build_options_sha256="f" * 64,
                device_name="A40",
                trust_remote_code=False,
                compute_dtype="int4",
                attention_implementation="llama-cuda",
                seed=0,
                sampling_mode="greedy",
                timeout_seconds=60,
                engine_args_sha256="b" * 64,
            ),
            speculative_decoding="off",
            load_status=GateStatus.NOT_RUN,
            generation_status=GateStatus.NOT_RUN,
            determinism_status=GateStatus.NOT_RUN,
            unload_status=GateStatus.NOT_RUN,
            source_binding_ids=RUNTIME_SOURCE_BINDINGS["meta_muse_glimmer_30b_int4"],
        ),
        OpenRuntimeReceiptV1(
            registry_key="meta_muse_glimmer_30b_bf16",
            revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
            snapshot_tree_sha256=_RUNTIME_TREE_HASHES["meta_muse_glimmer_30b_bf16"],
            runtime=_runtime_manifest(
                engine="transformers",
                engine_version="4.57.1",
                python_version="3.11.9",
                torch_version="2.7.1",
                transformers_version="4.57.1",
                accelerate_version="1.10.1",
                cuda_version="12.8",
                driver_version="570.124.06",
                trust_remote_code=False,
                compute_dtype="bf16",
                attention_implementation="eager",
                seed=0,
                sampling_mode="greedy",
                timeout_seconds=60,
                engine_args_sha256="c" * 64,
            ),
            load_status=GateStatus.BLOCKED,
            generation_status=GateStatus.NOT_RUN,
            determinism_status=GateStatus.NOT_RUN,
            unload_status=GateStatus.NOT_RUN,
            blocked_reasons=("resource/runtime unavailable",),
            source_binding_ids=RUNTIME_SOURCE_BINDINGS["meta_muse_glimmer_30b_bf16"],
        ),
    )


__all__ = [
    "RUNTIME_SOURCE_BINDINGS",
    "SOURCE_BINDING_IDS",
    "failed_ssh_setup_event",
    "open_runtime_receipts",
    "provider_attestations",
]
