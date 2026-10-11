from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.maintenance import run_test_shards as shards


def test_parse_collected_nodeids_ignores_pytest_summary_and_deduplicates():
    lines = [
        "tests/vnext/test_a.py::test_one",
        "tests/vnext/test_a.py::test_one",
        "tests/maintenance/test_b.py::test_two[param]",
        "========================= 2 tests collected in 0.01s =========================",
        "",
    ]
    assert shards.parse_collected_nodeids(lines) == [
        "tests/maintenance/test_b.py::test_two[param]",
        "tests/vnext/test_a.py::test_one",
    ]


def test_build_shards_is_sorted_balanced_and_exact():
    nodeids = [f"tests/vnext/test_{index}.py::test_case" for index in range(7)]
    manifest = shards.build_shard_manifest(nodeids, shard_count=3)
    assert manifest["schema"] == "memupdatebench.test-shards.v1"
    assert manifest["test_count"] == 7
    assert [len(shard["nodeids"]) for shard in manifest["shards"]] == [3, 2, 2]
    assert shards.validate_shard_manifest(manifest) == nodeids


def test_validate_rejects_duplicate_or_missing_nodeids():
    manifest = shards.build_shard_manifest(["a::one", "b::two", "c::three"], shard_count=2)
    manifest["shards"][1]["nodeids"].append("a::one")
    with pytest.raises(ValueError, match="duplicate"):
        shards.validate_shard_manifest(manifest)
    manifest = shards.build_shard_manifest(["a::one", "b::two", "c::three"], shard_count=2)
    manifest["shards"][0]["nodeids"].pop()
    with pytest.raises(ValueError, match="coverage"):
        shards.validate_shard_manifest(manifest)


def test_validate_rejects_shard_count_and_empty_nodes():
    with pytest.raises(ValueError, match="shard_count"):
        shards.build_shard_manifest([], shard_count=0)
    with pytest.raises(ValueError, match="nodeids"):
        shards.build_shard_manifest(["a::one", "a::one"], shard_count=2)


def test_publish_manifest_is_no_replace_and_reopenable(tmp_path: Path):
    manifest = shards.build_shard_manifest(["a::one", "b::two"], shard_count=2)
    output = tmp_path / "manifest.json"
    shards.publish_manifest(manifest, output)
    assert shards.load_manifest(output)["test_count"] == 2
    with pytest.raises(FileExistsError):
        shards.publish_manifest(manifest, output)


def test_build_pytest_args_is_cache_free_and_exact():
    manifest = shards.build_shard_manifest(["tests/a.py::test_one", "tests/b.py::test_two"], shard_count=2)
    arguments = shards.build_pytest_args(manifest, 1)
    assert arguments == ["-q", "-p", "no:cacheprovider", "tests/b.py::test_two"]


def test_build_pytest_args_rejects_invalid_index():
    manifest = shards.build_shard_manifest(["a::one"], shard_count=1)
    with pytest.raises(ValueError, match="shard index"):
        shards.build_pytest_args(manifest, 1)


def test_run_shard_uses_in_process_pytest_instead_of_long_argv(monkeypatch, tmp_path):
    manifest = shards.build_shard_manifest(
        [f"tests/vnext/test_{index}.py::test_case" for index in range(700)],
        shard_count=1,
    )
    calls = []

    def fake_pytest_main(arguments):
        calls.append(arguments)
        return 0

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("run_shard must not expand nodeids into subprocess argv")

    monkeypatch.setattr(shards.pytest, "main", fake_pytest_main)
    monkeypatch.setattr(shards.subprocess, "run", forbidden_subprocess)

    result = shards.run_shard(manifest, 0, project_root=tmp_path)

    assert result["returncode"] == 0
    assert result["nodeid_count"] == 700
    assert len(calls) == 1
    assert calls[0] == ["-q", "-p", "no:cacheprovider", *manifest["shards"][0]["nodeids"]]
