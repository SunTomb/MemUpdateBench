from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mub.vnext.external.providers.letta import build_letta_adapter_configuration
from mub.vnext.external.providers.letta_protocol import LettaWorkerHealthV2
from mub.vnext.external.workers.letta_worker import (
    LettaBlockProfileBackendV1,
    _installed_letta_content_digest,
    inspect_local_letta_package,
    verify_letta_source_binding,
)


def _distribution(root: Path, *, version: str = "0.16.8", license_text: str = "Apache-2.0", commit: str | None = None):
    package = root / "letta"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"stable")
    direct_url = package / "direct_url.json"
    if commit is not None:
        direct_url.write_text(json.dumps({"vcs_info": {"vcs": "git", "commit_id": commit}}))
    # Distribution.files entries need Path-like names for locate_file.
    item = Path("letta/__init__.py")
    return SimpleNamespace(
        version=version,
        metadata={"License": license_text},
        files=(item,),
        locate_file=lambda value: root / value,
        read_text=lambda name: direct_url.read_text() if name == "direct_url.json" else None,
    )


def test_installed_content_digest_matches_and_changes_with_content(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path)
    digest, count = _installed_letta_content_digest(distribution)
    assert count == 1
    assert digest
    (tmp_path / "letta" / "__init__.py").write_bytes(b"changed")
    changed, changed_count = _installed_letta_content_digest(distribution)
    assert changed_count == count
    assert changed != digest


def test_installed_content_digest_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "letta"
    root.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"secret")
    link = root / "__init__.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    distribution = SimpleNamespace(files=(Path("letta/__init__.py"),), locate_file=lambda value: root / "__init__.py")
    with pytest.raises(Exception, match="letta_installed_content_unavailable"):
        _installed_letta_content_digest(distribution)


def test_source_binding_is_explicitly_verified_or_blocked(tmp_path: Path) -> None:
    distribution = _distribution(tmp_path, commit="1131535716e8a31c9a437f8695e25ac98f203a24")
    assert verify_letta_source_binding(distribution)
    unbound = _distribution(tmp_path / "other")
    assert not verify_letta_source_binding(unbound)


def test_inspection_distinguishes_version_and_license(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    distribution = _distribution(tmp_path, version="0.1.0", license_text="MIT")
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: distribution)
    result = inspect_local_letta_package(expected_digest="a" * 64, expected_file_count=1)
    assert result["package_present"] is True
    assert result["version_verified"] is False
    assert result["license_verified"] is False
    assert result["identity_verified"] is False
    assert result["blocker"] == "letta_package_version_mismatch"


def test_fake_backend_health_rejects_official_health_contract() -> None:
    configuration = build_letta_adapter_configuration(run_id="letta-health-test")
    backend = LettaBlockProfileBackendV1(client=SimpleNamespace(), configuration=configuration)
    with pytest.raises(Exception):
        backend.health()


def test_worker_failure_is_typed_and_redacted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from mub.vnext.external.workers import letta_worker

    monkeypatch.setattr(letta_worker, "build_official_letta_backend", lambda _: (_ for _ in ()).throw(letta_worker.LettaDependencyUnavailable("letta_source_binding_unverified")))
    config = build_letta_adapter_configuration(run_id="letta-cli-test")
    from mub.vnext.io import canonical_json_bytes

    assert letta_worker.main(["--configuration-json", canonical_json_bytes(config).decode("utf-8")]) == 2
    stderr = capsys.readouterr().err
    assert "letta_source_binding_unverified" in stderr
    assert "token" not in stderr.casefold()
    assert "api_key" not in stderr.casefold()


def test_health_v2_blocks_unverified_identity() -> None:
    with pytest.raises(Exception):
        LettaWorkerHealthV2(
            package_name="letta",
            package_version="0.16.8",
            source_commit="1131535716e8a31c9a437f8695e25ac98f203a24",
            license_id="Apache-2.0",
            installed_content_sha256="a" * 64,
            installed_content_file_count=1,
            installed_content_verified=False,
            source_binding_status="blocked",
            configuration_hash="b" * 64,
            identity_verified=True,
        )


def test_official_backend_default_factory_fails_closed_on_unverified_native_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mub.vnext.external.workers import letta_worker

    configuration = build_letta_adapter_configuration(run_id="letta-official-boundary")
    evidence = {
        "identity_verified": True,
        "installed_content_sha256": "a" * 64,
        "installed_content_file_count": 1,
        "installed_content_verified": True,
        "source_binding_verified": True,
    }
    monkeypatch.setattr(letta_worker, "inspect_local_letta_package", lambda: evidence)
    with pytest.raises(letta_worker.LettaDependencyUnavailable) as exc_info:
        letta_worker.build_official_letta_backend(configuration)
    assert str(exc_info.value) == "letta_native_api_unverified"


def test_official_backend_uses_injected_narrow_client_and_v2_health() -> None:
    from mub.vnext.external.workers.letta_worker import OfficialLettaBackendV1
    from tests.vnext.test_letta_worker_protocol import DirectBlockClientFake

    configuration = build_letta_adapter_configuration(run_id="letta-official-health")
    client = DirectBlockClientFake()
    evidence = {
        "identity_verified": True,
        "installed_content_sha256": "a" * 64,
        "installed_content_file_count": 1,
        "installed_content_verified": True,
        "source_binding_verified": True,
    }
    backend = OfficialLettaBackendV1(
        client=client, configuration=configuration, inspection=evidence
    )
    health = backend.health()
    assert isinstance(health, LettaWorkerHealthV2)
    assert health.identity_verified is True
    assert health.source_binding_status == "verified"
    assert health.configuration_hash


def test_official_backend_rejects_non_client_without_leaking_details() -> None:
    from mub.vnext.external.workers.letta_worker import (
        LettaDependencyUnavailable,
        OfficialLettaBackendV1,
    )

    configuration = build_letta_adapter_configuration(run_id="letta-official-invalid")
    with pytest.raises(LettaDependencyUnavailable) as exc_info:
        OfficialLettaBackendV1(
            client=SimpleNamespace(token="secret-token"),
            configuration=configuration,
            inspection={"identity_verified": True},
        )
    assert str(exc_info.value) == "letta_native_client_interface_invalid"
    assert "token" not in str(exc_info.value)
