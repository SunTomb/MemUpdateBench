from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.validation.cross_domain_qdrant_state import validate_evidence

_ARTIFACTS = (
    ("evidence_summary.json", "results/summary.json"),
    ("evidence_rows.jsonl", "results/rows.jsonl"),
    ("input_binding.json", "results/input_binding.json"),
    ("source_artifact_index.json", "results/artifact_index.json"),
    ("collection_audit.json", "collection_audit.json"),
)
_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|bearer|password|secret|credential|private[_-]?key|raw_prompt|raw_output|reasoning)", re.I)
_ABSOLUTE_PRIVATE = re.compile(r"^(?:[A-Za-z]:[\\/]|/NAS/|/tmp/|/private/|~/|\\\\)")
_CREDENTIAL_VALUE = re.compile(r"(?:bearer\s+|api[_-]?key\s*[:=]|password\s*[:=]|secret\s*[:=])", re.I)
_PROTECTED = (
    PROJECT_ROOT / "data/vnext/core",
    PROJECT_ROOT / "data/vnext/pilot",
    PROJECT_ROOT / "data/vnext/family_h_cross_domain_noaa",
    PROJECT_ROOT / "data/vnext/family_h_cross_domain_bea_gdp",
)


def _assert_no_reparse(path: Path) -> Path:
    path = path.absolute()
    for parent in (path, *path.parents):
        try:
            metadata = parent.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError("path contains a symlink or reparse component")
    return path


def _assert_safe_path(path: Path) -> Path:
    path = _assert_no_reparse(path)
    resolved = path.resolve(strict=False)
    for protected in _PROTECTED:
        protected = protected.resolve(strict=False)
        if resolved == protected or protected in resolved.parents:
            raise ValueError("output overlaps a protected release root")
    return path


def _snapshot_tree(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    result = {}
    for path in sorted(root.rglob("*")):
        _assert_no_reparse(path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha(path.read_bytes())
    return result


def _scan_public(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or _SECRET_KEY.search(key):
                raise ValueError(f"{label} contains a private field")
            _scan_public(child, label)
    elif isinstance(value, list):
        for child in value:
            _scan_public(child, label)
    elif isinstance(value, str):
        if _ABSOLUTE_PRIVATE.match(value) or _CREDENTIAL_VALUE.search(value):
            raise ValueError(f"{label} contains a private path or credential-like value")


def _public_bytes(raw: bytes, label: str) -> bytes:
    try:
        if label.endswith(".json"):
            _scan_public(json.loads(raw), label)
        elif label.endswith(".jsonl"):
            for line in raw.splitlines():
                _scan_public(json.loads(line), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    return raw


def _input_snapshots(collected: Path, bea: Path, noaa: Path) -> dict[str, dict[str, str]]:
    return {
        "collected": _snapshot_tree(collected),
        "bea_release": _snapshot_tree(bea),
        "noaa_release": _snapshot_tree(noaa),
    }


def _require_snapshot_unchanged(snapshots: dict[str, dict[str, str]], collected: Path,
                                bea: Path, noaa: Path) -> None:
    current = _input_snapshots(collected, bea, noaa)
    if current != snapshots:
        raise ValueError("evidence inputs changed during packaging")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def package_evidence(collected_root: Path | str, bea_root: Path | str,
                     noaa_root: Path | str, output_root: Path | str) -> dict[str, Any]:
    collected = _assert_no_reparse(Path(collected_root))
    bea = _assert_no_reparse(Path(bea_root))
    noaa = _assert_no_reparse(Path(noaa_root))
    collected = collected.resolve(strict=True)
    bea = bea.resolve(strict=True)
    noaa = noaa.resolve(strict=True)
    output = _assert_safe_path(Path(output_root))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    snapshots = _input_snapshots(collected, bea, noaa)
    validation = validate_evidence(collected, bea, noaa)
    if validation.get("status") != "VALID":
        raise ValueError("evidence validation is not approved")
    payload: dict[Path, bytes] = {}
    sources = []
    for published, relative in _ARTIFACTS:
        source = collected / relative
        raw = _public_bytes(source.read_bytes(), published)
        payload[output / published] = raw
        sources.append({"path": published, "bytes": len(raw), "sha256": _sha(raw)})
    _require_snapshot_unchanged(snapshots, collected, bea, noaa)
    manifest = {
        "schema": "memupdatebench.cross-domain.qdrant-state-evidence-package.v1",
        "status": "FINAL_APPROVED_BOUNDED_EVIDENCE",
        "evidence_class": "cross_domain_external_qdrant_state_retrieval",
        "scientific_release_allowed": False,
        "benchmark_accuracy_claimed": False,
        "answer_metrics": None,
        "claim_boundary": validation["metrics"],
        "cluster_unit": "source_trajectory",
        "cluster_n": 2,
        "artifact_index_sha256": validation["artifact_index_sha256"],
        "collection_audit_sha256": validation["collection_audit_sha256"],
        "source_manifest_sha256": validation["source_manifest_sha256"],
        "input_bindings": {
            "artifact_index_sha256": validation["artifact_index_sha256"],
            "collection_audit_sha256": validation["collection_audit_sha256"],
            "source_manifest_sha256": validation["source_manifest_sha256"],
            "bea_release_index_sha256": validation["release_bindings"]["bea"],
            "noaa_release_index_sha256": validation["release_bindings"]["noaa"],
        },
        "metrics": validation["metrics"],
        "base_events": validation["base_events"],
        "executed_mutations": validation["executed_mutations"],
        "retrieval_requests": validation["retrieval_requests"],
        "repetitions_equal": validation["repetitions_equal"],
        "artifacts": sources,
    }
    payload[output / "package_manifest.json"] = _canonical(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    def pre_publish() -> None:
        _require_snapshot_unchanged(snapshots, collected, bea, noaa)
        if validate_evidence(collected, bea, noaa).get("status") != "VALID":
            raise ValueError("evidence validation changed before publication")

    publish_files_atomically(payload, overwrite=False,
                             source_paths=tuple(collected / relative for _, relative in _ARTIFACTS),
                             pre_publish=pre_publish)
    return {"status": "PUBLISHED", "output_root": str(output),
            "package_manifest_sha256": _sha(payload[output / "package_manifest.json"]),
            "artifact_count": len(payload), "validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--bea-root", type=Path, required=True)
    parser.add_argument("--noaa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = package_evidence(args.collected_root, args.bea_root, args.noaa_root, args.output_root)
    print(json.dumps({k: v for k, v in result.items() if k != "validation"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
