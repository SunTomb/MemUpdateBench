from __future__ import annotations

import pytest
from pydantic import ValidationError

from mub.vnext.post_core.contracts_v1 import CandidateIdentityState
from mub.vnext.post_core.model_registry_v1 import (
    build_initial_model_registry_v1,
    validate_model_registry_v1,
)


def test_initial_registry_has_exact_pending_candidates_and_roles() -> None:
    registry = build_initial_model_registry_v1()
    assert tuple(registry) == (
        "qwen35_9b_bf16",
        "meta_muse_glimmer_30b_int4",
        "meta_muse_glimmer_30b_bf16",
        "claude_sonnet_4_6",
        "claude_opus_4_8",
        "gemini_3_6_flash",
        "grok_4_5",
        "gpt_5_5",
    )
    validate_model_registry_v1(registry)
    assert all(candidate.identity is None for candidate in registry.values())
    assert registry["gpt_5_5"].state is CandidateIdentityState.PENDING_OFFICIAL_IDENTITY
    assert registry["gpt_5_5"].scopes == ("none",)
    assert registry["claude_opus_4_8"].scopes == ("hard_subset",)
    assert registry["meta_muse_glimmer_30b_bf16"].scopes == ("k16_subset",)


def test_registry_cannot_invent_pending_identity() -> None:
    registry = build_initial_model_registry_v1()
    qwen = registry["qwen35_9b_bf16"]
    with pytest.raises(ValidationError, match="pending identity"):
        qwen.validated_replace(
            identity={
                "official_model_id": "guessed/model",
                "revision": "guessed",
                "license_id": "unknown",
                "architecture": "unknown",
                "weights_uri": None,
                "tokenizer_identity": None,
                "endpoint": None,
                "resolved_upstream_identity": None,
            }
        )
