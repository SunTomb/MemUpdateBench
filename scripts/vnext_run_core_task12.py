from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from mub.vnext.io import canonical_json_bytes
from mub.vnext.preparation.task12 import (
    Task12DryRunPlanV1,
    Task12PreparationManifestV1,
)
from mub.vnext.runtime.answer_model_v3 import (
    AnswerModelSlotV3,
    DeterministicDecodeConfigV3,
    OfflinePromptedAnswerModelV3,
)
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3
from mub.vnext.runtime.run_v3 import ExternalRunConfigV1
from mub.vnext.runtime.task12_bundle_v3 import (
    _binding_for_slot,
    validate_task12_run_bundle_v3,
)
from mub.vnext.runtime.task12_matrix_v3 import task12_cell_runtime_v3
from mub.vnext.runtime.task12_execution_v3 import (
    run_task12_cell_v3,
    task12_runtime_code_binding_v3,
)


def _load_canonical(path: Path, model_type):
    raw = path.read_bytes()
    model = model_type.model_validate_json(raw)
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"noncanonical artifact: {path}")
    return model


def _decoding_from_config(
    configuration: ExternalRunConfigV1,
) -> DeterministicDecodeConfigV3:
    fields = {
        key: configuration.decoding_config[key]
        for key in ("do_sample", "num_beams", "max_new_tokens", "seed")
        if key in configuration.decoding_config
    }
    return DeterministicDecodeConfigV3.model_validate(fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute and score one authenticated Task 12 run bundle"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing to execute without --execute")

    plan = _load_canonical(args.plan, Task12DryRunPlanV1)
    manifest = _load_canonical(
        args.preparation_manifest,
        Task12PreparationManifestV1,
    )
    runtime_code_binding = task12_runtime_code_binding_v3(_PROJECT_ROOT)
    bundle = validate_task12_run_bundle_v3(
        manifest=manifest,
        plan=plan,
        core_root=args.core_root,
        evidence_root=args.evidence_root,
        repository_root=_PROJECT_ROOT,
        runtime_code_binding=runtime_code_binding,
        bundle_root=args.bundle_root,
    )
    binding = _binding_for_slot(
        manifest,
        bundle.authorization.answer_model_slot,
    )
    slot = AnswerModelSlotV3(
        slot_id=binding.slot_id,
        model_id=binding.model_id,
        revision=binding.revision,
        snapshot_path=str(args.model_snapshot),
        license_id=binding.license_id,
        tree_manifest_sha256=binding.tree_manifest_sha256,
    )
    decoding = _decoding_from_config(bundle.run_configuration)
    context_order, context_annotation, retrieval_k = task12_cell_runtime_v3(
        manifest,
        bundle.authorization.cell_id,
    )
    runtime_config = RuntimeConfigV3(
        run_id=bundle.run_configuration.run_id,
        retrieval_policy=bundle.run_configuration.retrieval_policy,
        answer_mode=bundle.run_configuration.answer_mode,
        retrieval_k=retrieval_k,
        capture_snapshots=False,
    )
    from mub.vnext.adapters.core_v3 import RawAppendAdapterV3

    model = OfflinePromptedAnswerModelV3(
        slot=slot,
        decoding=decoding,
        device=args.device,
    )
    try:
        model.load()
        result = run_task12_cell_v3(
            bundle.tasks,
            adapter_factory=lambda task: RawAppendAdapterV3(
                task,
                retrieval_policy="normal_topk",
            ),
            run_configuration=bundle.run_configuration,
            runtime_config=runtime_config,
            prompted_answer_model=model,
            context_order=context_order,
            context_annotation=context_annotation,
            frozen_trajectories=bundle.frozen_trajectories,
            output_root=bundle.execution_output_root,
            task_manifest=bundle.task_manifest,
            run_manifest_artifact=None,
            task_artifact=bundle.run_configuration.task_view_ref,
            authenticated_task_manifest_sha256=(
                bundle.authorization.task_manifest_sha256
            ),
            resume=args.resume,
        )
    finally:
        model.close()
    run_manifest, rows, scores, receipt = result
    print(
        json.dumps(
            {
                "status": "completed",
                "cell_id": bundle.authorization.cell_id,
                "answer_model_slot": bundle.authorization.answer_model_slot,
                "task_count": len(rows),
                "score_count": len(scores),
                "run_manifest_sha256": hashlib.sha256(
                    canonical_json_bytes(run_manifest)
                ).hexdigest(),
                "score_artifact_sha256": receipt["score_artifact_sha256"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
