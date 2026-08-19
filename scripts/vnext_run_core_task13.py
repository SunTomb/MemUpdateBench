from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from mub.vnext.io import canonical_json_bytes
from mub.vnext.statistics.contracts_v3 import Task13BootstrapConfigV1
from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1
from mub.vnext.statistics import task13_v3 as task13_publication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish authenticated Core Task 13 statistics artifacts."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--core-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--matrix-bundle-manifest", required=True)
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--matrix-integrity-audit", required=True)
    parser.add_argument("--statistics-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> Task13BootstrapConfigV1:
    raw = path.read_bytes()
    try:
        config = Task13BootstrapConfigV1.model_validate_json(raw)
    except Exception as exc:
        raise ValueError("statistics config is not a valid typed Task 13 config") from exc
    if canonical_json_bytes(config) != raw:
        raise ValueError("statistics config is not canonical JSON")
    return config


def _preflight_output_root(output_root: Path, protected_roots: tuple[Path, ...]) -> None:
    candidate = output_root.resolve(strict=False)
    if output_root.exists():
        raise FileExistsError("Task 13 output root must not already exist")
    for protected in protected_roots:
        root = protected.resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError:
            try:
                root.relative_to(candidate)
            except ValueError:
                continue
        raise ValueError("Task 13 output root overlaps an input root")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if not args.execute:
            print("Task 13 publication requires explicit --execute", file=sys.stderr)
            return 2
        repository_root = Path(__file__).resolve().parents[1]
        paths = {
            "manifest": Path(args.manifest),
            "plan": Path(args.plan),
            "core_root": Path(args.core_root),
            "evidence_root": Path(args.evidence_root),
            "matrix_root": Path(args.matrix_root),
            "matrix_manifest": Path(args.matrix_bundle_manifest),
            "matrix_summary": Path(args.matrix_summary),
            "integrity_audit": Path(args.matrix_integrity_audit),
            "statistics_config": Path(args.statistics_config),
        }
        _preflight_output_root(
            Path(args.output_root),
            (paths["core_root"], paths["evidence_root"], paths["matrix_root"], repository_root),
        )
        source_paths = (
            paths["manifest"], paths["plan"], paths["matrix_manifest"],
            paths["matrix_summary"], paths["integrity_audit"], paths["statistics_config"],
        )
        source_roots = (paths["core_root"], paths["evidence_root"], paths["matrix_root"])
        source_snapshot = task13_publication.capture_task13_source_snapshot_v3(
            source_paths, source_roots
        )
        task13_publication._revalidate_source_snapshot(source_snapshot)
        hashes = {
            key: task13_publication.source_snapshot_sha256_v3(source_snapshot, path)
            for key, path in paths.items()
            if key not in {"core_root", "evidence_root", "matrix_root"}
        }
        matrix = load_task13_authenticated_matrix_v1(
            preparation_manifest_path=paths["manifest"],
            plan_path=paths["plan"],
            core_root=paths["core_root"],
            evidence_root=paths["evidence_root"],
            matrix_root=paths["matrix_root"],
            matrix_manifest_path=paths["matrix_manifest"],
            matrix_summary_path=paths["matrix_summary"],
            integrity_audit_path=paths["integrity_audit"],
            repository_root=repository_root,
            expected_preparation_manifest_sha256=hashes["manifest"],
            expected_plan_sha256=hashes["plan"],
            expected_matrix_manifest_sha256=hashes["matrix_manifest"],
            expected_matrix_summary_sha256=hashes["matrix_summary"],
            expected_integrity_audit_sha256=hashes["integrity_audit"],
        )
        task13_publication._revalidate_source_snapshot(source_snapshot)
        config = _load_config(paths["statistics_config"])
        task13_publication._revalidate_source_snapshot(source_snapshot)
        runtime = task13_publication.current_clean_task13_runtime_v3(repository_root)
        publication = task13_publication.build_task13_publication_v3(
            matrix=matrix,
            bootstrap_config=config,
            statistics_config_sha256=hashes["statistics_config"],
            runtime=runtime,
            source_hashes={
                "preparation_manifest": matrix.input_hashes["task12_preparation_manifest"],
                "plan": matrix.input_hashes["task12_plan"],
                "matrix_manifest": matrix.input_hashes["task12_matrix_manifest"],
                "matrix_summary": matrix.input_hashes["task12_matrix_summary"],
                "integrity_audit": matrix.input_hashes["task12_integrity_audit"],
                "core_tasks": matrix.input_hashes["core_tasks"],
                "core_task_manifest": matrix.input_hashes["core_task_manifest"],
            },
        )
        result = task13_publication.publish_task13_artifacts_v3(
            publication,
            matrix=matrix,
            output_root=Path(args.output_root),
            source_snapshot=source_snapshot,
            repository_root=repository_root,
        )
        index_path = result.output_root / "task13_artifact_index.json"
        print(f"task13_artifact_index_sha256={_sha256(index_path)}")
        print(f"output_root={result.output_root}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Task 13 publication rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
