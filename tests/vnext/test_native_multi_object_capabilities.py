from __future__ import annotations

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.v3.native_multi_object import (
    NativeCapabilityDeclarationV3,
    NativeDataProvenanceV3,
)
from mub.vnext.external.native_multi_object import (
    NativeCapabilityValidationV3,
    validate_native_capabilities,
)


def caps(**changes):
    values = {
        "direct_provider_crud": True,
        "provider_owned_collection": True,
        "supports_multi_object_mutation": True,
        "supports_multi_object_retrieval": True,
        "supports_atomic_mutation": True,
        "supports_isolated_reset": True,
        "exports_provider_entry_ids": True,
        "exports_provider_payloads": True,
        "exports_provider_order": True,
        "exports_provider_scores": True,
        "exports_source_event_linkage": True,
        "visible_only": True,
    }
    values.update(changes)
    return NativeCapabilityDeclarationV3(**values)




def test_native_capability_declaration_does_not_promote_per_operation_flags():
    declaration = caps()
    assert declaration.supports_multi_object_mutation is True
    assert declaration.supports_add is False
    assert declaration.supports_update is False
    assert declaration.supports_noop is False
    assert declaration.supports_delete is False


def test_native_capability_declaration_forbids_native_answer_claim_and_local_substitutes():
    declaration = caps()
    assert declaration.supports_native_answer is False
    assert declaration.uses_local_authoritative_store is False
    assert declaration.uses_local_reranking is False
    assert declaration.uses_local_deduplication is False
    assert declaration.uses_local_filtering is False
    assert declaration.hidden_target_injection is False
    with pytest.raises(ValidationError):
        caps(supports_native_answer=True)
    with pytest.raises(ValidationError, match="local authoritative"):
        caps(uses_local_authoritative_store=True)
    with pytest.raises(ValidationError, match="hidden target"):
        caps(hidden_target_injection=True)


def test_capability_validation_rejects_declared_overclaim_and_accepts_equal_observation():
    declared = caps()
    observed = caps(exports_provider_scores=False)
    validation = validate_native_capabilities(declared, observed)
    assert isinstance(validation, NativeCapabilityValidationV3)
    assert validation.passed is False
    assert validation.overclaimed_fields == ("exports_provider_scores",)
    assert validation.provenance.kind is NativeDataProvenanceV3.OBSERVED

    equal = validate_native_capabilities(declared, declared)
    assert equal.passed is True
    assert equal.overclaimed_fields == ()


def test_observed_capabilities_cannot_be_inflated_or_untyped():
    with pytest.raises(ValidationError):
        NativeCapabilityDeclarationV3.model_validate({"direct_provider_crud": "true"}, strict=True)
    declared = caps()
    forged = NativeCapabilityDeclarationV3.model_construct(
        **{**declared.__dict__, "exports_provider_order": "true"}
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_native_capabilities(declared, forged)


def test_capability_validation_rejects_non_native_boundary_observations():
    declared = caps()
    observed = caps(
        direct_provider_crud=False,
        provider_owned_collection=False,
        supports_multi_object_mutation=False,
        supports_multi_object_retrieval=False,
        supports_atomic_mutation=False,
    ).model_copy(update={"visible_only": False})
    validation = validate_native_capabilities(declared, observed)
    assert validation.passed is False
    assert set(validation.overclaimed_fields) >= {"provider_owned_collection", "visible_only"}
    assert validation.status == "overclaimed"
