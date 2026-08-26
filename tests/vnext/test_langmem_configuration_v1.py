from __future__ import annotations

from mub.vnext.external.providers.langmem import (
    LANGMEM_PACKAGE_VERSION,
    LANGMEM_SOURCE_COMMIT,
    LangMemAdapterConfigurationV1,
    build_langmem_adapter_configuration,
    fixed_langmem_package_provenance,
)


def test_langmem_frozen_package_identity_is_exact() -> None:
    provenance = fixed_langmem_package_provenance()

    assert provenance.package_name == "langmem"
    assert provenance.package_version == "0.0.30"
    assert provenance.source_commit == (
        "29cbe41e58528f92e9efa773c12e15c47be3808c"
    )
    assert provenance.license_id == "MIT"
    assert LANGMEM_PACKAGE_VERSION == provenance.package_version
    assert LANGMEM_SOURCE_COMMIT == provenance.source_commit


def test_langmem_profile_configuration_explicitly_excludes_collection_mode() -> None:
    configuration = build_langmem_adapter_configuration(
        run_id="langmem-profile-config",
    )

    assert type(configuration) is LangMemAdapterConfigurationV1
    assert configuration.mode == "profile_single_record"
    assert configuration.collection_mode_supported is False
    assert configuration.llm_required is False
    assert configuration.network_required is False
    assert configuration.credentials_required is False
