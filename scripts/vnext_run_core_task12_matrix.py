from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_project_root_text = str(_PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _project_root_text]
sys.path.insert(0, _project_root_text)

from mub.vnext.preparation.task12 import (
    Task12DryRunPlanV1,
    Task12PreparationManifestV1,
)
from mub.vnext.runtime.answer_model_v3 import (
    AnswerModelSlotV3,
    DeterministicDecodeConfigV3,
    OfflinePromptedAnswerModelV3,
)
from mub.vnext.runtime.run_v3 import ExternalRunConfigV1
from mub.vnext.runtime.task12_bundle_v3 import _binding_for_slot
from mub.vnext.runtime.task12_execution_v3 import (
    load_task12_control_json_v3,
    task12_runtime_code_binding_v3,
)
from mub.vnext.runtime.task12_matrix_v3 import (
    Task12MatrixBundleManifestV1,
    execute_task12_matrix_bundles_v3,
)


def _decoding_from_config(
    configuration: ExternalRunConfigV1,
) -> DeterministicDecodeConfigV3:
    fields = {
        key: configuration.decoding_config[key]
        for key in ("do_sample", "num_beams", "max_new_tokens", "seed")
        if key in configuration.decoding_config
    }
    return DeterministicDecodeConfigV3.model_validate(fields)


def _model_for_slot(
    *,
    manifest: Task12PreparationManifestV1,
    matrix_root: Path,
    matrix_manifest: Task12MatrixBundleManifestV1,
    slot_id: str,
    snapshot_path: Path,
    device: str,
) -> OfflinePromptedAnswerModelV3:
    binding = _binding_for_slot(manifest, slot_id)
    refs = tuple(
        ref
        for ref in matrix_manifest.run_bundles
        if ref.answer_model_slot == slot_id
    )
    if not refs:
        raise ValueError(f"matrix bundle manifest has no run for {slot_id}")
    config = load_task12_control_json_v3(
        matrix_root / refs[0].bundle_leaf / "run_config.json",
        ExternalRunConfigV1,
    )
    for ref in refs:
        other = load_task12_control_json_v3(
            matrix_root / ref.bundle_leaf / "run_config.json",
            ExternalRunConfigV1,
        )
        if (
            other.model_name != binding.model_id
            or other.model_revision != binding.revision
            or other.answer_model_slot != binding.slot_id
        ):
            raise ValueError(f"run config differs from frozen model binding for {slot_id}")
        if other.decoding_config != config.decoding_config:
            raise ValueError(f"inconsistent decoding config for {slot_id}")
    slot = AnswerModelSlotV3(
        slot_id=binding.slot_id,
        model_id=binding.model_id,
        revision=binding.revision,
        snapshot_path=str(snapshot_path),
        license_id=binding.license_id,
        tree_manifest_sha256=binding.tree_manifest_sha256,
    )
    return OfflinePromptedAnswerModelV3(
        slot=slot,
        decoding=_decoding_from_config(config),
        device=device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute and score all 18 authenticated Task 12 matrix bundles"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--matrix-bundle-manifest", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--model-a-snapshot", type=Path, required=True)
    parser.add_argument("--model-b-snapshot", type=Path, required=True)
    parser.add_argument("--device-a", default="cpu")
    parser.add_argument("--device-b", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing to execute matrix without --execute")
    manifest = load_task12_control_json_v3(
        args.manifest,
        Task12PreparationManifestV1,
    )
    plan = load_task12_control_json_v3(
        args.plan,
        Task12DryRunPlanV1,
        allow_trailing_lf=True,
    )
    matrix_manifest = load_task12_control_json_v3(
        args.matrix_bundle_manifest,
        Task12MatrixBundleManifestV1,
    )
    runtime_code_binding = task12_runtime_code_binding_v3(_PROJECT_ROOT)
    from mub.vnext.adapters.core_v3 import RawAppendAdapterV3

    models = {
        "answer_model_a": _model_for_slot(
            manifest=manifest,
            matrix_root=args.matrix_root,
            matrix_manifest=matrix_manifest,
            slot_id="answer_model_a",
            snapshot_path=args.model_a_snapshot,
            device=args.device_a,
        ),
        "answer_model_b": _model_for_slot(
            manifest=manifest,
            matrix_root=args.matrix_root,
            matrix_manifest=matrix_manifest,
            slot_id="answer_model_b",
            snapshot_path=args.model_b_snapshot,
            device=args.device_b,
        ),
    }
    result = execute_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        matrix_bundle_manifest=matrix_manifest,
        matrix_root=args.matrix_root,
        core_root=args.core_root,
        evidence_root=args.evidence_root,
        repository_root=_PROJECT_ROOT,
        runtime_code_binding=runtime_code_binding,
        adapter_factory=lambda task: RawAppendAdapterV3(
            task,
            retrieval_policy="normal_topk",
        ),
        prompted_answer_models=models,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "run_count": len(result.summary.completed_runs),
                "total_task_rows": result.summary.total_task_rows,
                "total_score_rows": result.summary.total_score_rows,
                "matrix_root": str(result.matrix_root),
                "summary_path": str(result.summary_path),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
