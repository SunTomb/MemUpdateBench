"""Build and validate deterministic offline pytest shard manifests."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import pytest


NODEID = re.compile(r"^(?P<node>[^\s:]+(?:/[^\s:]+)*::[^\s]+(?:\[[^\]]*\])?)$")


def parse_collected_nodeids(lines: Iterable[str]) -> list[str]:
    values = set()
    for raw in lines:
        value = raw.strip()
        if not value or value.startswith("=") or value.startswith("<"):
            continue
        match = NODEID.fullmatch(value)
        if match:
            values.add(match.group("node"))
    return sorted(values)


def build_pytest_args(manifest: dict, shard_index: int) -> list[str]:
    validate_shard_manifest(manifest)
    if type(shard_index) is not int or shard_index < 0 or shard_index >= manifest["shard_count"]:
        raise ValueError("shard index is out of range")
    return ["-q", "-p", "no:cacheprovider", *manifest["shards"][shard_index]["nodeids"]]


@contextmanager
def _project_environment(project_root: Path):
    root = Path(project_root).resolve()
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_pythonpath = os.environ.get("PYTHONPATH")
    old_no_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath
        if old_no_bytecode is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = old_no_bytecode


def run_shard(manifest: dict, shard_index: int, *, project_root: Path) -> dict:
    arguments = build_pytest_args(manifest, shard_index)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with _project_environment(project_root):
        with redirect_stdout(stdout), redirect_stderr(stderr):
            returncode = int(pytest.main(arguments))
    return {
        "shard_index": shard_index,
        "returncode": returncode,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "nodeid_count": len(manifest["shards"][shard_index]["nodeids"]),
    }


def run_all_shards(manifest: dict, *, project_root: Path) -> list[dict]:
    return [run_shard(manifest, index, project_root=project_root) for index in range(manifest["shard_count"])]


def build_shard_manifest(nodeids: Iterable[str], *, shard_count: int) -> dict:
    values = sorted(nodeids)
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not values or len(values) != len(set(values)):
        raise ValueError("nodeids must be nonempty and unique")
    shard_count = min(shard_count, len(values))
    shards = [{"index": index, "nodeids": []} for index in range(shard_count)]
    for index, nodeid in enumerate(values):
        shards[index % shard_count]["nodeids"].append(nodeid)
    return {
        "schema": "memupdatebench.test-shards.v1",
        "test_count": len(values),
        "shard_count": shard_count,
        "shards": shards,
    }


def validate_shard_manifest(manifest: dict) -> list[str]:
    if manifest.get("schema") != "memupdatebench.test-shards.v1":
        raise ValueError("shard manifest schema mismatch")
    shard_count = manifest.get("shard_count")
    shards = manifest.get("shards")
    if type(shard_count) is not int or shard_count <= 0 or not isinstance(shards, list) or len(shards) != shard_count:
        raise ValueError("shard_count or shards is invalid")
    values: list[str] = []
    for index, shard in enumerate(shards):
        if shard.get("index") != index or not isinstance(shard.get("nodeids"), list) or not shard["nodeids"]:
            raise ValueError("shard nodeids are invalid")
        values.extend(shard["nodeids"])
    if len(values) != len(set(values)):
        raise ValueError("duplicate nodeid coverage")
    expected = sorted(values)
    if manifest.get("test_count") != len(expected):
        raise ValueError("test_count coverage mismatch")
    return expected


def collect_nodeids(project_root: Path, roots: Iterable[str]) -> list[str]:
    project_root = Path(project_root).resolve()
    command = [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", *roots]
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(project_root)}
    result = subprocess.run(command, cwd=project_root, env={**os.environ, **env}, capture_output=True, text=True, check=False, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"pytest collection failed: {result.stderr[-2000:]}")
    return parse_collected_nodeids(result.stdout.splitlines())


def publish_manifest(manifest: dict, output: Path) -> None:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_shard_manifest(manifest)
    output.write_bytes((json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode())


def load_manifest(path: Path) -> dict:
    value = json.loads(Path(path).read_bytes())
    validate_shard_manifest(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("roots", nargs="*", default=["tests/vnext", "tests/maintenance"])
    args = parser.parse_args(argv)
    if args.run_manifest:
        if args.shard_index is None:
            raise ValueError("--shard-index is required with --run-manifest")
        manifest = load_manifest(args.run_manifest)
        result = run_shard(manifest, args.shard_index, project_root=args.project_root)
        if args.run_output:
            output = Path(args.run_output)
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes((json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode())
        print(json.dumps({"shard_index": args.shard_index, "returncode": result["returncode"], "nodeid_count": result["nodeid_count"]}, sort_keys=True))
        return result["returncode"]
    nodeids = collect_nodeids(args.project_root, args.roots)
    manifest = build_shard_manifest(nodeids, shard_count=args.shards)
    publish_manifest(manifest, args.output)
    print(json.dumps({"output": str(args.output), "test_count": manifest["test_count"], "shard_count": manifest["shard_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
