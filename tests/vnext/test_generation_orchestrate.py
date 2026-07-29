from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

import mub.vnext.generation as generation


orchestrate = import_module("mub.vnext.generation.orchestrate")
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"


class RevisionString(str):
    pass


@pytest.fixture(scope="module")
def config():
    return generation.load_pilot_config(CONFIG_PATH)


def test_build_pilot_is_exported_from_generation_package() -> None:
    assert generation.build_pilot is orchestrate.build_pilot


@pytest.mark.parametrize("overwrite", [False, True])
def test_build_pilot_composes_stages_in_order_with_exact_arguments(
    tmp_path, monkeypatch, config, overwrite
) -> None:
    output_dir = tmp_path / "pilot"
    compiled = object()
    bundle = object()
    published = object()
    calls = []

    def fake_compile(config_arg, *, code_revision):
        calls.append(("compile", config_arg, code_revision))
        return compiled

    def fake_bundle(compiled_arg, config_arg):
        calls.append(("bundle", compiled_arg, config_arg))
        return bundle

    def fake_publish(bundle_arg, output_dir_arg, *, overwrite):
        calls.append(("publish", bundle_arg, output_dir_arg, overwrite))
        return published

    monkeypatch.setattr(orchestrate, "compile_pilot_tasks", fake_compile)
    monkeypatch.setattr(orchestrate, "build_pilot_artifact_bundle", fake_bundle)
    monkeypatch.setattr(orchestrate, "publish_pilot_artifact_bundle", fake_publish)

    result = orchestrate.build_pilot(
        config,
        output_dir,
        code_revision="  exact-revision  ",
        overwrite=overwrite,
    )

    assert result is published
    assert calls == [
        ("compile", config, "  exact-revision  "),
        ("bundle", compiled, config),
        ("publish", bundle, output_dir, overwrite),
    ]


@pytest.mark.parametrize("failing_stage", ["compile", "bundle"])
def test_prepublication_stage_failure_is_unmodified_and_leaves_output_absent(
    tmp_path, monkeypatch, config, failing_stage
) -> None:
    output_dir = tmp_path / "pilot"
    failure = RuntimeError(f"{failing_stage} failed")
    calls = []
    compiled = object()

    def fake_compile(config_arg, *, code_revision):
        calls.append("compile")
        if failing_stage == "compile":
            raise failure
        return compiled

    def fake_bundle(compiled_arg, config_arg):
        calls.append("bundle")
        if failing_stage == "bundle":
            raise failure
        return object()

    def forbidden_publish(bundle_arg, output_dir_arg, *, overwrite):
        calls.append("publish")
        output_dir_arg.mkdir()
        return object()

    monkeypatch.setattr(orchestrate, "compile_pilot_tasks", fake_compile)
    monkeypatch.setattr(orchestrate, "build_pilot_artifact_bundle", fake_bundle)
    monkeypatch.setattr(
        orchestrate, "publish_pilot_artifact_bundle", forbidden_publish
    )

    with pytest.raises(RuntimeError) as caught:
        orchestrate.build_pilot(
            config,
            output_dir,
            code_revision="failure-revision",
        )

    assert caught.value is failure
    expected_calls = (
        ["compile"]
        if failing_stage == "compile"
        else ["compile", "bundle"]
    )
    assert calls == expected_calls
    assert not output_dir.exists()


def _forbid_stages(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("stage must not be called")

    monkeypatch.setattr(orchestrate, "compile_pilot_tasks", forbidden)
    monkeypatch.setattr(orchestrate, "build_pilot_artifact_bundle", forbidden)
    monkeypatch.setattr(orchestrate, "publish_pilot_artifact_bundle", forbidden)


def test_build_pilot_accepts_pilot_config_subclass_like_stages(
    tmp_path, monkeypatch, config
) -> None:
    class DerivedPilotConfig(generation.PilotConfig):
        pass

    derived = DerivedPilotConfig.model_validate(config.model_dump(mode="python"))
    compiled = object()
    bundle = object()
    published = object()
    monkeypatch.setattr(
        orchestrate,
        "compile_pilot_tasks",
        lambda config_arg, *, code_revision: compiled,
    )
    monkeypatch.setattr(
        orchestrate,
        "build_pilot_artifact_bundle",
        lambda compiled_arg, config_arg: bundle,
    )
    monkeypatch.setattr(
        orchestrate,
        "publish_pilot_artifact_bundle",
        lambda bundle_arg, output_dir_arg, *, overwrite: published,
    )

    result = orchestrate.build_pilot(
        derived,
        tmp_path / "pilot",
        code_revision="revision",
    )

    assert result is published


def test_build_pilot_requires_pilot_config(tmp_path, monkeypatch) -> None:
    _forbid_stages(monkeypatch)

    with pytest.raises(TypeError, match="config must be a PilotConfig"):
        orchestrate.build_pilot(
            object(),
            tmp_path / "pilot",
            code_revision="revision",
        )


def test_build_pilot_requires_path_and_exact_bool(
    tmp_path, monkeypatch, config
) -> None:
    _forbid_stages(monkeypatch)

    with pytest.raises(TypeError, match="output_dir must be a Path"):
        orchestrate.build_pilot(
            config,
            str(tmp_path / "pilot"),
            code_revision="revision",
        )
    for invalid in (0, 1, None):
        with pytest.raises(TypeError, match="overwrite must be a bool"):
            orchestrate.build_pilot(
                config,
                tmp_path / "pilot",
                code_revision="revision",
                overwrite=invalid,
            )


@pytest.mark.parametrize(
    "invalid", [None, 1, b"revision", RevisionString("revision")]
)
def test_build_pilot_requires_exact_string_revision(
    tmp_path, monkeypatch, config, invalid
) -> None:
    _forbid_stages(monkeypatch)

    with pytest.raises(TypeError, match="code_revision must be a string"):
        orchestrate.build_pilot(
            config,
            tmp_path / "pilot",
            code_revision=invalid,
        )


@pytest.mark.parametrize("invalid", ["", " ", "\t\r\n"])
def test_build_pilot_rejects_blank_revision(
    tmp_path, monkeypatch, config, invalid
) -> None:
    _forbid_stages(monkeypatch)

    with pytest.raises(ValueError, match="code_revision must not be blank"):
        orchestrate.build_pilot(
            config,
            tmp_path / "pilot",
            code_revision=invalid,
        )
