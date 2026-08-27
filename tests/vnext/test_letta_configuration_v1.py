from __future__ import annotations

import pytest


def test_letta_configuration_freezes_release_source_and_apache_license() -> None:
    from mub.vnext.external.providers.letta import (
        LETTA_PACKAGE_VERSION,
        LETTA_SOURCE_COMMIT,
        build_letta_adapter_configuration,
        fixed_letta_package_provenance,
    )

    configuration = build_letta_adapter_configuration(run_id="letta-config-test")
    provenance = fixed_letta_package_provenance()

    assert LETTA_PACKAGE_VERSION == "0.16.8"
    assert LETTA_SOURCE_COMMIT == "1131535716e8a31c9a437f8695e25ac98f203a24"
    assert provenance.package_name == "letta"
    assert provenance.package_version == "0.16.8"
    assert provenance.release_tag == "0.16.8"
    assert provenance.source_commit == LETTA_SOURCE_COMMIT
    assert provenance.license_id == "Apache-2.0"
    assert provenance.wheel_url == (
        "https://files.pythonhosted.org/packages/10/20/"
        "eedf6bd8b55e97edf9cbcaea8f575b83157737d6483ddba6b304babc0a4a/"
        "letta-0.16.8-py3-none-any.whl"
    )
    assert provenance.python_requires == ">=3.11, <3.14"
    assert configuration.mode == "direct_block_profile"
    assert configuration.llm_required is False
    assert configuration.network_required is False
    assert configuration.credentials_required is False


def test_letta_configuration_rejects_release_identity_drift() -> None:
    from mub.vnext.external.providers.letta import fixed_letta_package_provenance

    with pytest.raises(ValueError, match="Apache-2.0"):
        fixed_letta_package_provenance().model_copy(update={"license_id": "MIT"})
