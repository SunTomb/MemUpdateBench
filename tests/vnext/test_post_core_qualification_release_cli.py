from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.qualification_release_v1 import (
    REQUIRED_SOURCE_IDS,
    load_qualification_release_config_v1,
    verify_qualification_release_v1,
)
from tests.vnext.qualification_fixtures import open_runtime_receipts, provider_attestations
from scripts import vnext_prepare_post_core_qualification_release as command


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDENTITY_SOURCE = (
    PROJECT_ROOT / "configs" / "vnext" / "post_core" / "official_identity_evidence_v1.json"
)


def _write_jsonl(path: Path, rows: object) -> None:
    path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))


def _inputs(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    for index, source_id in enumerate(REQUIRED_SOURCE_IDS):
        path = tmp_path / f"{source_id}.blob"
        raw = IDENTITY_SOURCE.read_bytes() if source_id == "identity_evidence" else f"source-{index}".encode()
        path.write_bytes(raw)
        source_paths[source_id] = path
        source_hashes[source_id] = hashlib.sha256(raw).hexdigest()
    config_path = tmp_path / "qualification-config.json"
    config_path.write_bytes(canonical_bytes({
        "base_attempts_per_role": 8,
        "base_commit": "a56857431023d2af1a392c75c5575316a916c174",
        "escalation_attempts_per_role": 8,
        "max_retries": 0,
        "publisher_network_allowed": False,
        "registry_keys": [
            "qwen35_9b_bf16", "meta_muse_glimmer_30b_int4", "meta_muse_glimmer_30b_bf16",
            "claude_sonnet_4_6", "claude_opus_4_8", "gemini_3_6_flash", "grok_4_5", "gpt_5_5",
        ],
        "release_id": "memupdatebench.post-core.qualification.v1",
        "required_source_sha256": source_hashes,
        "schema_version": "memupdatebench.post-core.qualification-config.v1",
        "scientific_execution_allowed": False,
    }))
    provider_path = tmp_path / "provider.jsonl"
    runtime_path = tmp_path / "runtime.jsonl"
    _write_jsonl(provider_path, provider_attestations())
    _write_jsonl(runtime_path, open_runtime_receipts())
    arguments = [
        "--config", str(config_path),
        "--core-manifest", str(source_paths["core_manifest"]),
        "--task14-index", str(source_paths["task14_index"]),
        "--phase0-index", str(source_paths["phase0_index"]),
        "--identity-evidence", str(source_paths["identity_evidence"]),
        "--workflow-source", str(source_paths["workflow_source"]),
        "--handoff-source", str(source_paths["handoff_source"]),
        "--open-snapshot-closure-receipt", str(source_paths["open_snapshot_closure_receipt"]),
        "--open-snapshot-audit-receipt", str(source_paths["open_snapshot_audit_receipt"]),
        "--qwen-load-receipt", str(source_paths["qwen_load_receipt"]),
        "--provider-attestations", str(provider_path),
        "--runtime-receipts", str(runtime_path),
        "--output-root", str(tmp_path / "published"),
    ]
    return arguments, {**source_paths, "config": config_path, "provider": provider_path, "runtime": runtime_path}


def _release_inputs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "config": load_qualification_release_config_v1(paths["config"]),
        "source_paths": {source_id: paths[source_id] for source_id in REQUIRED_SOURCE_IDS},
        "provider_attestations_path": paths["provider"],
        "runtime_receipts_path": paths["runtime"],
    }


def test_prepare_cli_has_exact_safe_surface_and_no_runtime_capability_access() -> None:
    parser = command.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert options == {
        "--config", "--core-manifest", "--task14-index", "--phase0-index", "--identity-evidence",
        "--workflow-source", "--handoff-source", "--open-snapshot-closure-receipt",
        "--open-snapshot-audit-receipt", "--qwen-load-receipt", "--provider-attestations",
        "--runtime-receipts", "--output-root", "--execute",
    }
    help_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "vnext_prepare_post_core_qualification_release.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert help_result.stderr == ""
    help_text = help_result.stdout.lower()
    for forbidden in ("--api-key", "--token", "--authorization", "--endpoint", "--allow-network"):
        assert forbidden not in help_text
    source = inspect.getsource(command)
    for forbidden in ("os.environ", "subprocess", "socket", "urllib", "requests", "http.client"):
        assert forbidden not in source


def test_prepare_cli_requires_execute(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments, _ = _inputs(tmp_path)
    assert command.main(arguments) == command.EXIT_USAGE
    assert capsys.readouterr().err == "qualification release requires explicit --execute\n"


def test_prepare_cli_publishes_reopenable_fixture_root_with_canonical_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, paths = _inputs(tmp_path)
    assert command.main([*arguments, "--execute"]) == command.EXIT_SUCCESS
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    summary = json.loads(captured.out)
    output = Path(summary["output_root"])
    assert output == (tmp_path / "published").resolve()
    assert summary == {
        "benchmark_generations": 0,
        "credential_reads_during_publication": 0,
        "decision_counts": {"BLOCKED": 26, "READY": 6, "UNSUPPORTED": 0},
        "index_sha256": summary["index_sha256"],
        "model_loads_during_publication": 0,
        "network_calls_during_publication": 0,
        "output_root": str(output),
        "provider_calls_during_publication": 0,
        "release_id": "memupdatebench.post-core.qualification.v1",
        "status": "SUCCESS_WITH_BLOCKERS",
    }
    reopened = verify_qualification_release_v1(output, **_release_inputs(paths))
    assert reopened.output_root == output
    assert reopened.index_sha256 == summary["index_sha256"]


def test_prepare_cli_rejects_existing_root_with_fixed_publication_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, _ = _inputs(tmp_path)
    (tmp_path / "published").mkdir()
    assert command.main([*arguments, "--execute"]) == command.EXIT_PUBLICATION
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification publication rejected: output root is unavailable\n"


def test_prepare_cli_maps_source_mutation_to_stale_source_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, paths = _inputs(tmp_path)
    original = command.publish_qualification_release_v1

    def mutate_before_commit(output_root: Path, config: object, **inputs: object):
        return original(
            output_root,
            config,
            before_commit=lambda: paths["workflow_source"].write_bytes(b"changed"),
            **inputs,
        )

    monkeypatch.setattr(command, "publish_qualification_release_v1", mutate_before_commit)
    assert command.main([*arguments, "--execute"]) == command.EXIT_STALE_SOURCE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification stale source rejected: source/config mismatch\n"
    assert not (tmp_path / "published").exists()


def test_prepare_cli_maps_valid_source_hash_mismatch_to_stale_source_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, paths = _inputs(tmp_path)
    paths["workflow_source"].write_bytes(b"changed-source")
    assert command.main([*arguments, "--execute"]) == command.EXIT_STALE_SOURCE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification stale source rejected: source/config mismatch\n"
    assert not (tmp_path / "published").exists()


def test_prepare_cli_maps_valid_config_snapshot_mutation_to_stale_source_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, paths = _inputs(tmp_path)
    original = command.publish_qualification_release_v1

    def mutate_before_commit(output_root: Path, config: object, **inputs: object):
        return original(
            output_root,
            config,
            before_commit=lambda: paths["config"].write_bytes(paths["config"].read_bytes() + b" "),
            **inputs,
        )

    monkeypatch.setattr(command, "publish_qualification_release_v1", mutate_before_commit)
    assert command.main([*arguments, "--execute"]) == command.EXIT_STALE_SOURCE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification stale source rejected: source/config mismatch\n"
    assert not (tmp_path / "published").exists()


def test_prepare_cli_rejects_secret_input_without_leaking_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, paths = _inputs(tmp_path)
    paths["provider"].write_bytes(b'{"api_key":"DO_NOT_LEAK_9e5ba"}\n')
    assert command.main([*arguments, "--execute"]) == command.EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification contract/usage rejected: invalid source input\n"
    assert "DO_NOT_LEAK_9e5ba" not in captured.err
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("config_case", ("short_hash", "invalid_json", "wrong_schema", "pretty"))
def test_prepare_cli_rejects_malformed_production_style_config_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], config_case: str
) -> None:
    arguments, paths = _inputs(tmp_path)
    production_config = PROJECT_ROOT / "configs" / "vnext" / "post_core" / "qualification_release_v1.json"
    payload = json.loads(production_config.read_bytes())
    if config_case == "short_hash":
        payload["required_source_sha256"]["workflow_source"] = "a" * 63
        paths["config"].write_bytes(canonical_bytes(payload))
    elif config_case == "invalid_json":
        paths["config"].write_bytes(b"{")
    elif config_case == "wrong_schema":
        payload["schema_version"] = "memupdatebench.invalid-config.v1"
        paths["config"].write_bytes(canonical_bytes(payload))
    else:
        paths["config"].write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="")
    assert command.main([*arguments, "--execute"]) == command.EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification contract/usage rejected: invalid source input\n"
    assert not (tmp_path / "published").exists()


def test_prepare_cli_emits_utf8_canonical_summary_for_non_ascii_output_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, _ = _inputs(tmp_path)
    output_index = arguments.index("--output-root") + 1
    arguments[output_index] = str(tmp_path / "发布")
    assert command.main([*arguments, "--execute"]) == command.EXIT_SUCCESS
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert "发布" in captured.out
    assert "\\u" not in captured.out
    assert captured.out == json.dumps(
        summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"


def test_prepare_cli_maps_unsafe_value_error_to_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments, _ = _inputs(tmp_path)

    def unsafe_path(*_args: object, **_kwargs: object) -> object:
        raise ValueError("source path contains a link or reparse component")

    monkeypatch.setattr(command, "publish_qualification_release_v1", unsafe_path)
    assert command.main([*arguments, "--execute"]) == command.EXIT_PUBLICATION
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification publication rejected: unsafe publication path\n"
    assert not (tmp_path / "published").exists()
