from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from mub.vnext.post_core.contracts_v1 import CandidateIdentityState, ModelCandidateV1


_EXPECTED_KEYS = (
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5",
)


def build_initial_model_registry_v1() -> Mapping[str, ModelCandidateV1]:
    candidates = (
        ModelCandidateV1(registry_key="qwen35_9b_bf16", role="modern_open_anchor", state=CandidateIdentityState.PENDING_OFFICIAL_IDENTITY, identity=None, scopes=("full",)),
        ModelCandidateV1(registry_key="meta_muse_glimmer_30b_int4", role="large_open_anchor", state=CandidateIdentityState.PENDING_OFFICIAL_IDENTITY, identity=None, scopes=("full",)),
        ModelCandidateV1(registry_key="meta_muse_glimmer_30b_bf16", role="quantization_control", state=CandidateIdentityState.PENDING_OFFICIAL_IDENTITY, identity=None, scopes=("k16_subset",)),
        ModelCandidateV1(registry_key="claude_sonnet_4_6", role="closed_full", state=CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION, identity=None, scopes=("full",), credential_env_var="ANTHROPIC_API_KEY"),
        ModelCandidateV1(registry_key="claude_opus_4_8", role="closed_expensive_hard", state=CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION, identity=None, scopes=("hard_subset",), credential_env_var="ANTHROPIC_API_KEY"),
        ModelCandidateV1(registry_key="gemini_3_6_flash", role="closed_full", state=CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION, identity=None, scopes=("full",), credential_env_var="GEMINI_API_KEY"),
        ModelCandidateV1(registry_key="grok_4_5", role="closed_full", state=CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION, identity=None, scopes=("full",), credential_env_var="XAI_API_KEY"),
        ModelCandidateV1(registry_key="gpt_5_5", role="closed_proposed", state=CandidateIdentityState.PENDING_OFFICIAL_IDENTITY, identity=None, scopes=("none",)),
    )
    registry = {item.registry_key: item for item in candidates}
    if tuple(registry) != _EXPECTED_KEYS:
        raise AssertionError("post-Core registry order drift")
    return MappingProxyType(registry)


def validate_model_registry_v1(registry: Mapping[str, ModelCandidateV1]) -> None:
    if tuple(registry) != _EXPECTED_KEYS or len(set(registry)) != len(_EXPECTED_KEYS):
        raise ValueError("post-Core registry requires the exact frozen candidate keys")
    for key, candidate in registry.items():
        if key != candidate.registry_key or type(candidate) is not ModelCandidateV1:
            raise ValueError("post-Core registry candidate binding mismatch")
    expected = build_initial_model_registry_v1()
    for key, candidate in registry.items():
        if candidate != expected[key]:
            raise ValueError("post-Core registry candidate differs from frozen initial semantics")
    if registry["qwen35_9b_bf16"].scopes != ("full",):
        raise ValueError("Qwen3.5-9B must remain the BF16 full-matrix intent")
    if registry["meta_muse_glimmer_30b_bf16"].scopes != ("k16_subset",):
        raise ValueError("Glimmer BF16 must remain k16-only")
    if registry["claude_opus_4_8"].scopes != ("hard_subset",):
        raise ValueError("Claude Opus must remain hard-subset-only")
    if registry["gpt_5_5"].scopes != ("none",):
        raise ValueError("unverified GPT-5.5 cannot enter an execution scope")


__all__ = ["build_initial_model_registry_v1", "validate_model_registry_v1"]
