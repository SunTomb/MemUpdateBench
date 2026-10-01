from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import Field
from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN, StrictBool
from mub.vnext.contracts.enums import StringEnum
from mub.vnext.contracts.v3.native_multi_object import NativeCapabilityDeclarationV3
from mub.vnext.external.providers.qdrant_native_multi_object import (
    QDRANT_NATIVE_CONTRACT_VERSION,
    build_qdrant_native_multi_object_configuration,
    qdrant_client_availability,
)
from mub.vnext.io.atomic import publish_files_atomically


class QdrantQualificationStatusV1(StringEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"


class QdrantQualificationReportV1(ImmutableContractModel):
    schema_version: Literal[QDRANT_NATIVE_CONTRACT_VERSION] = QDRANT_NATIVE_CONTRACT_VERSION
    status: QdrantQualificationStatusV1
    availability: Literal["AVAILABLE", "UNAVAILABLE"]
    direct_observed: StrictBool
    evidence_anchor: str | None = Field(default=None, pattern=SHA256_PATTERN, strict=True)
    blocker: str | None = Field(default=None, min_length=1, strict=True)
    configuration_hash: str
    capabilities: NativeCapabilityDeclarationV3
    scientific_evidence: Literal[False] = False
    benchmark_accuracy_claimed: Literal[False] = False
    provider_call_count: Literal[0] = 0

    @classmethod
    def blocked(
        cls,
        *,
        availability: Literal["AVAILABLE", "UNAVAILABLE"],
        direct_observed: bool,
        evidence_anchor: str | None,
        blocker: str,
        configuration_hash: str,
    ) -> "QdrantQualificationReportV1":
        return cls(
            status=QdrantQualificationStatusV1.BLOCKED,
            availability=availability,
            direct_observed=direct_observed,
            evidence_anchor=evidence_anchor,
            blocker=blocker,
            configuration_hash=configuration_hash,
            capabilities=NativeCapabilityDeclarationV3(),
        )


def qualify_qdrant_native_multi_object(
    *,
    run_id: str,
    path: str,
    direct_observed: bool = False,
    evidence_anchor: str | None = None,
) -> QdrantQualificationReportV1:
    """Return a fail-closed setup report; this function never runs a provider."""
    configuration = build_qdrant_native_multi_object_configuration(
        run_id=run_id,
        path=path,
    )
    availability = qdrant_client_availability()
    if type(direct_observed) is not bool:
        raise ValueError("direct_observed must be an exact boolean")
    if availability.status == "UNAVAILABLE":
        return QdrantQualificationReportV1.blocked(
            availability="UNAVAILABLE",
            direct_observed=False,
            evidence_anchor=None,
            blocker="qdrant_client_missing",
            configuration_hash=_configuration_hash(configuration),
        )
    return QdrantQualificationReportV1.blocked(
        availability="AVAILABLE",
        direct_observed=False,
        evidence_anchor=None,
        blocker="api_runtime_not_observed",
        configuration_hash=_configuration_hash(configuration),
    )


def _configuration_hash(configuration) -> str:
    raw = json.dumps(
        configuration.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report direct Qdrant native multi-object qualification without executing a provider."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--direct-observed", action="store_true")
    parser.add_argument("--evidence-anchor")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = qualify_qdrant_native_multi_object(
        run_id=args.run_id,
        path=args.path,
        direct_observed=args.direct_observed,
        evidence_anchor=args.evidence_anchor,
    )
    serialized = json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            raise ValueError("output path must be absolute")
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        publish_files_atomically(
            {output: (serialized + "\n").encode("utf-8")},
            overwrite=False,
        )
    print(serialized)
    return 0 if report.status is QdrantQualificationStatusV1.READY else 1


QdrantNativeMultiObjectQualificationReportV1 = QdrantQualificationReportV1
QdrantNativeMultiObjectQualificationStatusV1 = QdrantQualificationStatusV1


__all__ = [
    "QdrantNativeMultiObjectQualificationReportV1",
    "QdrantNativeMultiObjectQualificationStatusV1",
    "QdrantQualificationReportV1",
    "QdrantQualificationStatusV1",
    "qualify_qdrant_native_multi_object",
]


if __name__ == "__main__":
    raise SystemExit(main())
