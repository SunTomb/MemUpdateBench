from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.external.artifacts import (
    PrivateRawArtifactRefV1,
    RawPayloadLicenseStatus,
)
from mub.vnext.external.canaries_v3 import (
    CanarySetManifestV1,
    authenticate_core_release,
)
from mub.vnext.external.model_provenance import (
    CORE_TASK10_CANARY_SET_MANIFEST_SHA256,
    CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SHA256,
    CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SIZE,
    CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SHA256,
    CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SIZE,
    CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SHA256,
    CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SIZE,
    CORE_TASK10_SOURCE_TASK_MANIFEST_SHA256,
    build_task10_model_provenance,
    publish_model_provenance,
    verify_model_input_artifact,
)
from mub.vnext.io import canonical_json_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish authenticated offline model provenance for Core Task 10."
        )
    )
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--canary-set-manifest", required=True)
    parser.add_argument("--snapshot-tree-evidence", required=True)
    parser.add_argument("--offline-probe-evidence", required=True)
    parser.add_argument("--package-versions-evidence", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _report_output(
    output: Path,
    *,
    evaluation_hash: str,
    provenance_hash: str,
) -> None:
    report = {
        "evaluation_configuration_sha256": evaluation_hash,
        "model_provenance_sha256": provenance_hash,
        "output_root": str(output),
    }
    sys.stdout.write(
        json.dumps(report, ensure_ascii=True, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    release = authenticate_core_release(args.release_root)
    if release.task_manifest_ref.sha256 != (
        CORE_TASK10_SOURCE_TASK_MANIFEST_SHA256
    ):
        raise ValueError("authenticated source task manifest hash is invalid")

    canary_bytes = verify_model_input_artifact(
        args.canary_set_manifest,
        CORE_TASK10_CANARY_SET_MANIFEST_SHA256,
    )
    try:
        canary_manifest = CanarySetManifestV1.model_validate_json(
            canary_bytes,
            strict=True,
        )
    except Exception:
        raise ValueError("canary set manifest is not strict canonical v1") from None
    if canonical_json_bytes(canary_manifest) != canary_bytes:
        raise ValueError("canary set manifest is not canonical")
    if (
        canary_manifest.source_release_manifest_hash
        != release.release_manifest_hash
    ):
        raise ValueError("canary set is not bound to the Core release")

    tree_evidence_bytes = verify_model_input_artifact(
        args.snapshot_tree_evidence,
        CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SHA256,
    )
    probe_evidence_bytes = verify_model_input_artifact(
        args.offline_probe_evidence,
        CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SHA256,
    )
    package_evidence_bytes = verify_model_input_artifact(
        args.package_versions_evidence,
        CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SHA256,
    )
    if len(tree_evidence_bytes) != CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SIZE:
        raise ValueError("snapshot tree evidence size is invalid")
    if len(probe_evidence_bytes) != CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SIZE:
        raise ValueError("offline probe evidence size is invalid")
    if (
        len(package_evidence_bytes)
        != CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SIZE
    ):
        raise ValueError("package versions evidence size is invalid")

    bundle = build_task10_model_provenance(
        source_task_manifest_ref=release.task_manifest_ref,
        canary_set_manifest_ref=ArtifactRef(
            path="canaries/canary_set_manifest.json",
            sha256=CORE_TASK10_CANARY_SET_MANIFEST_SHA256,
            media_type="application/json",
            record_count=1,
        ),
        snapshot_tree_raw_evidence=PrivateRawArtifactRefV1(
            sha256=CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SHA256,
            size_bytes=CORE_TASK10_SNAPSHOT_TREE_EVIDENCE_SIZE,
            media_type="text/plain; charset=utf-8",
            license_status=RawPayloadLicenseStatus.PRIVATE,
        ),
        offline_probe_raw_evidence=PrivateRawArtifactRefV1(
            sha256=CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SHA256,
            size_bytes=CORE_TASK10_OFFLINE_PROBE_EVIDENCE_SIZE,
            media_type="text/plain; charset=utf-8",
            license_status=RawPayloadLicenseStatus.PRIVATE,
        ),
        package_versions_raw_evidence=PrivateRawArtifactRefV1(
            sha256=CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SHA256,
            size_bytes=CORE_TASK10_PACKAGE_VERSIONS_EVIDENCE_SIZE,
            media_type="text/plain; charset=utf-8",
            license_status=RawPayloadLicenseStatus.PRIVATE,
        ),
    )
    output = publish_model_provenance(bundle, args.output_root)
    _report_output(
        output,
        evaluation_hash=(
            bundle.model_provenance.evaluation_configuration_hash
        ),
        provenance_hash=bundle.model_provenance_ref.sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
