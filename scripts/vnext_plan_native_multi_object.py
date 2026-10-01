from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN
from mub.vnext.contracts.v3.common import FrozenJsonObjectV3, StrictIdentifier
from mub.vnext.io.atomic import publish_files_atomically

_GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"
_ZERO_GIT_SHA = "0" * 40
_ZERO_SHA256 = "0" * 64
_FORBIDDEN_BINDINGS = (
    "factorial",
    "six_cell",
    "six-cell",
    "update_frequency_p63",
    "results/update_frequency_p63",
)
_FORBIDDEN_CANDIDATE_IDS = (
    "mem0_oss",
    "langgraph_store_custom_adapter",
    "langgraph_store_extract_then_store",
    "letta_profile",
    "letta_0_16_8_block_profile",
    "six_cell_factorial",
)
_FORBIDDEN_PREFIXES = (
    "mem0",
    "langgraph",
    "letta",
    "reference",
    "six_cell",
    "six-cell",
    "factorial",
    "frozen",
    "p63",
    "p83",
    "p84",
    "core",
    "pilot",
)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _config_hash(configuration: Mapping) -> str:
    raw = json.dumps(_jsonable(configuration), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _contains_forbidden_binding(value: object) -> bool:
    try:
        normalized = _jsonable(value)
        raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).casefold()
    except (TypeError, ValueError):
        return True
    if any(marker in raw for marker in _FORBIDDEN_BINDINGS):
        return True
    if all(f'"cell-{index}"' in raw for index in range(1, 7)):
        return True
    if isinstance(normalized, Mapping):
        cells = normalized.get("run_condition_ids")
        if isinstance(cells, list) and len(cells) == 6 and all(
            isinstance(item, str) and item.casefold().startswith("cell-") for item in cells
        ):
            return True
    return False


class NativePlanV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.native_multi_object.plan.v1"] = "memupdatebench.native_multi_object.plan.v1"
    candidate_id: StrictIdentifier
    provider_name: StrictIdentifier
    provider_version: StrictIdentifier
    source_revision: str = Field(pattern=_GIT_SHA_PATTERN, strict=True)
    runtime_revision: str = Field(pattern=_GIT_SHA_PATTERN, strict=True)
    source_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    runtime_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    configuration: FrozenJsonObjectV3
    configuration_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    factorial_binding: Literal[False] = False

    @field_validator("source_revision", "runtime_revision")
    @classmethod
    def _reject_zero_revisions(cls, value: str) -> str:
        if value == _ZERO_GIT_SHA:
            raise ValueError("native plan source/runtime revisions cannot be all-zero placeholders")
        return value

    @field_validator("source_hash", "runtime_hash")
    @classmethod
    def _reject_zero_hashes(cls, value: str) -> str:
        if value == _ZERO_SHA256:
            raise ValueError("native plan source/runtime hashes cannot be all-zero placeholders")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        candidate_fold = self.candidate_id.casefold()
        if candidate_fold in _FORBIDDEN_CANDIDATE_IDS or any(candidate_fold.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
            raise ValueError("native plan requires a new candidate identity")
        if _contains_forbidden_binding(self.configuration):
            raise ValueError("native plan cannot reuse the six-cell factorial or frozen update-frequency root")
        expected_configuration_hash = _config_hash(self.configuration)
        if self.configuration_hash != expected_configuration_hash:
            raise ValueError("native plan configuration hash does not match configuration")
        return self


def plan_native_multi_object(
    *,
    candidate_id: str,
    provider_name: str,
    provider_version: str,
    source_revision: str,
    runtime_revision: str,
    configuration: Mapping,
    source_hash: str | None = None,
    runtime_hash: str | None = None,
) -> NativePlanV1:
    if not isinstance(configuration, Mapping):
        raise ValueError("native plan configuration must be a mapping")
    if not source_hash or type(source_hash) is not str:
        raise ValueError("native plan requires an explicit actual source_hash")
    if not runtime_hash or type(runtime_hash) is not str:
        raise ValueError("native plan requires an explicit actual runtime_hash")
    return NativePlanV1(
        candidate_id=candidate_id,
        provider_name=provider_name,
        provider_version=provider_version,
        source_revision=source_revision,
        runtime_revision=runtime_revision,
        source_hash=source_hash,
        runtime_hash=runtime_hash,
        configuration=dict(configuration),
        configuration_hash=_config_hash(configuration),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a provisional native multi-object candidate without executing it.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--provider-version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--runtime-hash", required=True)
    parser.add_argument("--configuration-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    configuration = json.loads(args.configuration_json)
    output = Path(args.output)
    if not output.is_absolute():
        raise ValueError("output path must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    plan = plan_native_multi_object(
        candidate_id=args.candidate_id,
        provider_name=args.provider_name,
        provider_version=args.provider_version,
        source_revision=args.source_revision,
        runtime_revision=args.runtime_revision,
        source_hash=args.source_hash,
        runtime_hash=args.runtime_hash,
        configuration=configuration,
    )
    publish_files_atomically(
        {output: json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"},
        overwrite=False,
    )
    return 0


NativeMultiObjectPlanV1 = NativePlanV1
build_native_multi_object_plan = plan_native_multi_object


__all__ = [
    "NativeMultiObjectPlanV1",
    "NativePlanV1",
    "build_native_multi_object_plan",
    "plan_native_multi_object",
]


if __name__ == "__main__":
    raise SystemExit(main())
