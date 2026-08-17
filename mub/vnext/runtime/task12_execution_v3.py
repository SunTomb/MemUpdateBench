from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Literal

from pydantic import Field, field_validator

from mub.vnext.contracts.common import ArtifactRef, ImmutableContractModel, thaw_json
from mub.vnext.contracts.v3.adapter import RetrievalRequestV3, RetrievalResultV3
from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3, RetrievalTraceV3, TaskRunRecordV3
from mub.vnext.contracts.v3.score import ScoreRecordV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.registry import _validate_portable_path
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.preparation.task12 import (
    RawAppendTrajectoryV1,
    Task12AdmittedAnswerRunV1,
    Task12AdmittedCellV1,
)
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3
from mub.vnext.runtime.run_v3 import (
    ExternalRunConfigV1,
    ExternalRunIdentityV1,
    ExternalRunProgressV1,
    ExternalRunWriterV1,
    _validate_public_row,
    compute_external_run_identity,
)
from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3


ContextOrder = Literal["chronological", "reverse_chronological"]
ContextAnnotation = Literal["none", "latest_outdated_label"]
_APPROVED = {
    ("chronological", "none"),
    ("reverse_chronological", "none"),
    ("reverse_chronological", "latest_outdated_label"),
}


def read_task12_regular_file_v3(path: str | Path) -> bytes:
    selected = Path(path)
    assert_no_reparse_components(selected)
    metadata = selected.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or selected.is_symlink()
        or getattr(metadata, "st_file_attributes", 0) & 0x400
        or getattr(metadata, "st_nlink", 1) != 1
    ):
        raise ValueError("Task 12 local artifacts must be single-link regular files")
    return selected.read_bytes()


class Task12RuntimeCodeBindingV1(ImmutableContractModel):
    code_revision: str = Field(strict=True, pattern=r"^[0-9a-f]{40}$")
    code_tree_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")


def task12_runtime_code_binding_v3(
    repository_root: str | Path,
) -> Task12RuntimeCodeBindingV1:
    root = Path(repository_root).resolve(strict=True)
    outputs: list[bytes] = []
    for command in (
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=normal"),
        ("ls-tree", "-r", "-z", "HEAD"),
    ):
        completed = subprocess.run(
            ("git", "-C", str(root), *command),
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise ValueError("Task 12 cannot determine runtime repository identity")
        outputs.append(completed.stdout)
    if outputs[1].strip():
        raise ValueError("Task 12 runtime requires a clean repository worktree")
    return Task12RuntimeCodeBindingV1(
        code_revision=outputs[0].decode("ascii").strip(),
        code_tree_sha256=hashlib.sha256(outputs[2]).hexdigest(),
    )


def validate_task12_runtime_code_binding_v3(
    expected: Task12RuntimeCodeBindingV1,
    observed: Task12RuntimeCodeBindingV1,
) -> Task12RuntimeCodeBindingV1:
    if observed != expected:
        raise ValueError("Task 12 runtime code binding differs from authorization")
    return observed


def task12_runtime_configuration_sha256_v3(
    runtime_config: RuntimeConfigV3,
    *,
    context_order: ContextOrder,
    context_annotation: ContextAnnotation,
) -> str:
    payload = {
        "answer_mode": runtime_config.answer_mode,
        "retrieval": {
            "retrieval_policy": runtime_config.retrieval_policy,
            "retrieval_k": runtime_config.retrieval_k,
        },
        "context_intervention": {
            "context_order": context_order,
            "context_annotation": context_annotation,
        },
        "capture_snapshots": runtime_config.capture_snapshots,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def validate_task12_runtime_configuration_v3(
    expected_sha256: str,
    *,
    expected_run_id: str,
    runtime_config: RuntimeConfigV3,
    context_order: ContextOrder,
    context_annotation: ContextAnnotation,
) -> None:
    observed_sha256 = task12_runtime_configuration_sha256_v3(
        runtime_config,
        context_order=context_order,
        context_annotation=context_annotation,
    )
    if (
        runtime_config.run_id != expected_run_id
        or observed_sha256 != expected_sha256
        or runtime_config.capture_snapshots
    ):
        raise ValueError("Task 12 runtime configuration differs from run binding")


class Task12ExecutionAuthorizationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task12-execution-authorization.v1"] = (
        "memupdatebench.core-task12-execution-authorization.v1"
    )
    preparation_manifest_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    plan_fingerprint_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    runtime_code_binding: Task12RuntimeCodeBindingV1
    cell_id: str = Field(strict=True, min_length=1)
    answer_model_slot: Literal["answer_model_a", "answer_model_b"]
    cell_binding_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    answer_model_binding_sha256: str = Field(
        strict=True,
        pattern=r"^[0-9a-f]{64}$",
    )
    canonical_run_binding_sha256: str = Field(
        strict=True,
        pattern=r"^[0-9a-f]{64}$",
    )
    task_manifest_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    task_view_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    run_config_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    expected_task_count: Literal[80] = 80
    execution_authorized: Literal[True] = True
    output_leaf: str = Field(strict=True, min_length=1)

    @field_validator("output_leaf")
    @classmethod
    def _single_leaf(cls, value: str) -> str:
        validated = _validate_portable_path(value)
        if "/" in validated:
            raise ValueError("execution output leaf must be one path component")
        return validated


def find_admitted_answer_run_v3(
    *,
    cell_id: str,
    answer_model_slot: Literal["answer_model_a", "answer_model_b"],
    admitted_cells: Sequence[Task12AdmittedCellV1],
    admitted_answer_runs: Sequence[Task12AdmittedAnswerRunV1],
) -> Task12AdmittedAnswerRunV1:
    cells = [cell for cell in admitted_cells if cell.cell_id == cell_id]
    if len(cells) != 1:
        raise ValueError("selection must identify exactly one admitted cell")
    matches = [
        run
        for run in admitted_answer_runs
        if run.cell_id == cell_id
        and run.answer_model_slot == answer_model_slot
        and run.cell_binding_sha256 == cells[0].canonical_binding_sha256
    ]
    if len(matches) != 1:
        raise ValueError("selection must identify exactly one admitted answer run")
    return matches[0]


def select_admitted_answer_run_v3(
    authorization: Task12ExecutionAuthorizationV1,
    *,
    preparation_manifest_sha256: str,
    plan_fingerprint_sha256: str,
    admitted_cells: Sequence[Task12AdmittedCellV1],
    admitted_answer_runs: Sequence[Task12AdmittedAnswerRunV1],
) -> Task12AdmittedAnswerRunV1:
    if authorization.preparation_manifest_sha256 != preparation_manifest_sha256:
        raise ValueError("preparation manifest hash does not match authorization")
    if authorization.plan_fingerprint_sha256 != plan_fingerprint_sha256:
        raise ValueError("plan fingerprint does not match authorization")
    selected = find_admitted_answer_run_v3(
        cell_id=authorization.cell_id,
        answer_model_slot=authorization.answer_model_slot,
        admitted_cells=admitted_cells,
        admitted_answer_runs=admitted_answer_runs,
    )
    if (
        authorization.cell_binding_sha256 != selected.cell_binding_sha256
        or authorization.answer_model_binding_sha256
        != selected.answer_model_binding_sha256
        or authorization.canonical_run_binding_sha256
        != selected.canonical_run_binding_sha256
    ):
        raise ValueError("authorization binding hashes differ from admitted answer run")
    return selected


def _event_index(entry: MemoryEntryRecordV3) -> int:
    value = entry.raw_metadata.get("event_index", entry.raw_metadata.get("sequence_index"))
    if type(value) is not int or value < 0:
        raise ValueError("Task 12 entries require nonnegative event_index metadata")
    return value


def _frozen_trajectory_labels(
    entries: tuple[MemoryEntryRecordV3, ...],
    receipt: RawAppendTrajectoryV1,
) -> dict[str, str]:
    rows: list[dict[str, object]] = []
    live_to_frozen: dict[str, str] = {}
    for entry in entries:
        if entry.object_key_candidate is None or entry.version_index is None:
            raise ValueError("Task 12 trajectory entries require object and version metadata")
        if len(entry.source_event_ids) != 1:
            raise ValueError("Task 12 trajectory entries require one source event")
        object_id = entry.object_key_candidate.canonical_id
        frozen_id = f"{entry.source_event_ids[0]}:{object_id}:{entry.version_index}"
        rows.append(
            {
                "entry_id": frozen_id,
                "event_index": _event_index(entry),
                "version_index": entry.version_index,
                "object_key": object_id,
                "value": thaw_json(entry.value_candidate),
            }
        )
        live_to_frozen[entry.entry_id] = frozen_id
    observed_hash = hashlib.sha256(
        json.dumps(
            rows,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if (
        tuple(row["entry_id"] for row in rows) != receipt.entry_ids
        or tuple(row["object_key"] for row in rows) != receipt.object_ids
        or tuple(row["event_index"] for row in rows) != receipt.event_indices
        or tuple(row["version_index"] for row in rows) != receipt.version_indices
        or observed_hash != receipt.trajectory_sha256
    ):
        raise ValueError("live raw trajectory differs from frozen Task 12 receipt")
    latest = set(receipt.latest_entry_ids)
    return {
        live_id: "latest" if frozen_id in latest else "outdated"
        for live_id, frozen_id in live_to_frozen.items()
    }


def transform_retrieval_trace_v3(
    trace: RetrievalTraceV3,
    *,
    context_order: ContextOrder,
    context_annotation: ContextAnnotation,
    full_trajectory: Sequence[MemoryEntryRecordV3],
    frozen_trajectory: RawAppendTrajectoryV1,
) -> RetrievalTraceV3:
    """Apply the frozen Task 12 presentation transform after normal_topk."""
    if (context_order, context_annotation) not in _APPROVED:
        raise ValueError("unsupported Task 12 context condition")
    reference = tuple(full_trajectory)
    reference_ids = {entry.entry_id for entry in reference}
    if len(reference_ids) != len(reference):
        raise ValueError("full trajectory contains duplicate entry IDs")
    retrieved_ids = {entry.entry_id for entry in trace.retrieved_entries}
    if not retrieved_ids <= reference_ids:
        raise ValueError("retrieved entries must belong to the full trajectory")

    labels = _frozen_trajectory_labels(reference, frozen_trajectory)
    indexed = list(enumerate(trace.retrieved_entries))
    indexed.sort(
        key=lambda item: (
            _event_index(item[1]),
            item[1].version_index if item[1].version_index is not None else -1,
            item[1].entry_id,
        ),
        reverse=context_order == "reverse_chronological",
    )
    entries = tuple(entry for _, entry in indexed)
    scores = tuple(trace.scores[index] for index, _ in indexed) if trace.scores else ()
    metadata = dict(trace.version_metadata)
    if context_annotation == "latest_outdated_label":
        metadata["labels"] = {entry_id: labels[entry_id] for entry_id in retrieved_ids}
        metadata.update({entry_id: label for entry_id, label in labels.items()})
        entries = tuple(
            entry.model_copy(
                update={
                    "raw_metadata": {
                        **dict(entry.raw_metadata),
                        "version_label": labels[entry.entry_id],
                    }
                }
            )
            for entry in entries
        )
    metadata.update(
        {
            "context_annotation": context_annotation,
            "label_reference": "full_trajectory",
        }
    )
    return trace.model_copy(
        update={
            "retrieved_entries": entries,
            "scores": scores,
            "ranks": tuple(range(1, len(entries) + 1)),
            "context_order": context_order,
            "version_metadata": metadata,
        }
    )


class Task12PresentationAdapterV3:
    """Delegate adapter state while transforming only post-retrieval presentation."""

    def __init__(
        self,
        adapter,
        *,
        context_order: ContextOrder,
        context_annotation: ContextAnnotation,
        frozen_trajectory: RawAppendTrajectoryV1,
    ) -> None:
        self._adapter = adapter
        self._context_order = context_order
        self._context_annotation = context_annotation
        self._frozen_trajectory = frozen_trajectory
        self.adapter_id = adapter.adapter_id
        self.append_only_observation = getattr(adapter, "append_only_observation", False)
        self.retrieval_policy = getattr(adapter, "retrieval_policy", None)

    def retrieve(self, request: RetrievalRequestV3) -> RetrievalResultV3:
        result = self._adapter.retrieve(request)
        full = self._adapter.export_entries().entries
        trace = transform_retrieval_trace_v3(
            result.trace,
            context_order=self._context_order,
            context_annotation=self._context_annotation,
            full_trajectory=full,
            frozen_trajectory=self._frozen_trajectory,
        )
        return result.model_copy(update={"trace": trace})

    def __getattr__(self, name: str):
        return getattr(self._adapter, name)


def canonical_task12_action_id_adapter_factory_v3(adapter_factory):
    class _CanonicalTask12ActionIdAdapter:
        def __init__(self, adapter, task: MemUpdateTaskV3) -> None:
            self._adapter = adapter
            self._gold_action_by_event = {}
            for event in task.events:
                if len(event.gold_action_ids) != 1:
                    raise ValueError("Task 12 requires exactly one gold action per event")
                self._gold_action_by_event[event.event_id] = event.gold_action_ids[0]
            self.adapter_id = adapter.adapter_id
            self.append_only_observation = getattr(
                adapter,
                "append_only_observation",
                False,
            )
            self.retrieval_policy = getattr(adapter, "retrieval_policy", None)

        def ingest_event(self, event):
            result = self._adapter.ingest_event(event)
            raw_result = (
                dict(result.raw_result)
                if isinstance(result.raw_result, Mapping)
                else {}
            )
            expected_observed_id = f"observed_action:{event.event_id}"
            if raw_result.get("parsed_action_id") != expected_observed_id:
                raise ValueError("Task 12 adapter returned an unexpected observed action ID")
            try:
                gold_action_id = self._gold_action_by_event[event.event_id]
            except KeyError as exc:
                raise ValueError("Task 12 adapter received an unbound event") from exc
            raw_result["parsed_action_id"] = gold_action_id
            return result.model_copy(update={"raw_result": raw_result})

        def __getattr__(self, name: str):
            return getattr(self._adapter, name)

    def _build(task: MemUpdateTaskV3):
        return _CanonicalTask12ActionIdAdapter(adapter_factory(task), task)

    return _build


def execute_task12_task_v3(
    task: MemUpdateTaskV3,
    adapter,
    run_config: RuntimeConfigV3,
    *,
    prompted_answer_model: Any,
    context_order: ContextOrder,
    context_annotation: ContextAnnotation,
    frozen_trajectory: RawAppendTrajectoryV1,
) -> TaskRunRecordV3:
    if run_config.answer_mode != "slot_prompt":
        raise ValueError("Task 12 execution requires slot_prompt answer mode")
    if run_config.retrieval_policy != "normal_topk":
        raise ValueError("Task 12 execution requires normal_topk retrieval")
    if frozen_trajectory.task_id != task.task_id:
        raise ValueError("Task 12 frozen trajectory is bound to another task")
    presented = Task12PresentationAdapterV3(
        adapter,
        context_order=context_order,
        context_annotation=context_annotation,
        frozen_trajectory=frozen_trajectory,
    )
    return execute_task_v3(
        task,
        presented,
        run_config,
        prompted_answer_model=prompted_answer_model,
    )


def persist_task12_rows_v3(
    output_root: str | Path,
    configuration: ExternalRunConfigV1,
    rows: Sequence[TaskRunRecordV3],
    *,
    resume: bool = False,
) -> RunManifestV3:
    """Persist ordered public rows and finalize one authenticated v3 run."""
    writer = (
        ExternalRunWriterV1.resume(output_root, configuration)
        if resume
        else ExternalRunWriterV1.create(output_root, configuration)
    )
    ordered = tuple(rows)
    if tuple(row.task_id for row in ordered) != configuration.expected_task_ids:
        raise ValueError("Task 12 rows must cover expected task IDs in order")
    if ordered[: len(writer.rows)] != writer.rows:
        raise ValueError("Task 12 resume rows differ from persisted prefix")
    for row in ordered[len(writer.rows) :]:
        writer.append(row)
    return writer.finalize()


def score_task12_rows_v3(
    tasks: Sequence[MemUpdateTaskV3],
    rows: Sequence[TaskRunRecordV3],
    *,
    task_manifest: TaskManifestV3,
    run_manifest: RunManifestV3,
    task_artifact: ArtifactRef,
    run_artifact: ArtifactRef,
    authenticated_task_manifest_sha256: str,
    authenticated_run_manifest_sha256: str,
) -> tuple[ScoreRecordV3, ...]:
    """Score persisted rows only through the authenticated v3 scorer path."""
    ordered_tasks = tuple(tasks)
    ordered_rows = tuple(rows)
    task_ids = tuple(task.task_id for task in ordered_tasks)
    row_ids = tuple(row.task_id for row in ordered_rows)
    if (
        task_ids != row_ids
        or len(task_ids) != len(set(task_ids))
        or len(row_ids) != len(set(row_ids))
    ):
        raise ValueError(
            "Task 12 scoring requires unique ordered task/run row coverage"
        )
    scores = []
    for task, row in zip(ordered_tasks, ordered_rows):
        context = VerifiedScoringContextV3.from_authenticated_manifests(
            task=task,
            run=row,
            task_manifest=task_manifest,
            run_manifest=run_manifest,
            task_artifact=task_artifact,
            run_artifact=run_artifact,
            authenticated_task_manifest_sha256=authenticated_task_manifest_sha256,
            authenticated_run_manifest_sha256=authenticated_run_manifest_sha256,
        )
        scores.append(score_task_v3(task, row, context))
    return tuple(scores)


def persist_task12_scores_v3(
    output_root: str | Path,
    scores: Sequence[ScoreRecordV3],
    *,
    run_manifest_sha256: str,
    task_manifest_sha256: str,
) -> dict[str, object]:
    """Write canonical score rows with a receipt bound to finalized manifests."""
    root = Path(output_root)
    normalized = tuple(scores)
    if any(type(score) is not ScoreRecordV3 for score in normalized):
        raise TypeError("Task 12 scores must be exact ScoreRecordV3 values")
    raw = b"".join(canonical_json_bytes(score) + b"\n" for score in normalized)
    score_path = root / "scores.jsonl"
    receipt = {
        "schema_version": "memupdatebench.core-task12-score-receipt.v1",
        "run_manifest_sha256": run_manifest_sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "score_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "score_count": len(normalized),
        "score_record_hashes": {
            score.task_id: hashlib.sha256(canonical_json_bytes(score)).hexdigest()
            for score in normalized
        },
    }
    receipt_raw = json.dumps(
        receipt,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    publish_files_atomically(
        {
            score_path: raw,
            root / "score_receipt.json": receipt_raw,
        },
        overwrite=False,
    )
    return receipt


def verify_task12_score_artifact_v3(
    output_root: str | Path,
    *,
    expected_task_ids: tuple[str, ...],
    run_manifest_sha256: str,
    task_manifest_sha256: str,
) -> tuple[tuple[ScoreRecordV3, ...], dict[str, object]]:
    root = Path(output_root)
    score_raw = read_task12_regular_file_v3(root / "scores.jsonl")
    if not score_raw.endswith(b"\n"):
        raise ValueError("Task 12 score artifact must end with a newline")
    score_lines = score_raw.splitlines()
    scores = tuple(ScoreRecordV3.model_validate_json(line) for line in score_lines)
    if any(
        canonical_json_bytes(score) != line
        for score, line in zip(scores, score_lines)
    ):
        raise ValueError("Task 12 score artifact contains a noncanonical row")
    if tuple(score.task_id for score in scores) != expected_task_ids:
        raise ValueError("Task 12 score artifact does not cover expected tasks in order")
    receipt_raw = read_task12_regular_file_v3(root / "score_receipt.json")
    receipt = json.loads(receipt_raw)
    canonical_receipt = json.dumps(
        receipt,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if canonical_receipt != receipt_raw:
        raise ValueError("Task 12 score receipt is not canonical")
    expected_receipt = {
        "schema_version": "memupdatebench.core-task12-score-receipt.v1",
        "run_manifest_sha256": run_manifest_sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "score_artifact_sha256": hashlib.sha256(score_raw).hexdigest(),
        "score_count": len(scores),
        "score_record_hashes": {
            score.task_id: hashlib.sha256(canonical_json_bytes(score)).hexdigest()
            for score in scores
        },
    }
    if receipt != expected_receipt:
        if receipt.get("score_artifact_sha256") != expected_receipt["score_artifact_sha256"]:
            raise ValueError("Task 12 score artifact hash differs from receipt")
        raise ValueError("Task 12 score receipt binding mismatch")
    return scores, receipt


def load_finalized_task12_run_v3(
    output_root: str | Path,
    configuration: ExternalRunConfigV1,
) -> tuple[RunManifestV3, tuple[TaskRunRecordV3, ...]]:
    root = Path(output_root)
    identity_raw = read_task12_regular_file_v3(root / "run_identity.json")
    expected_identity = ExternalRunIdentityV1(
        run_identity=compute_external_run_identity(configuration),
        configuration=configuration,
    )
    if identity_raw != canonical_json_bytes(expected_identity):
        raise ValueError("finalized Task 12 run identity differs from configuration")
    rows_raw = read_task12_regular_file_v3(root / "task_runs.jsonl")
    if not rows_raw.endswith(b"\n"):
        raise ValueError("finalized Task 12 task rows must end with a newline")
    row_lines = rows_raw.splitlines()
    rows = tuple(TaskRunRecordV3.model_validate_json(line) for line in row_lines)
    if any(canonical_json_bytes(row) != line for row, line in zip(rows, row_lines)):
        raise ValueError("finalized Task 12 run contains a noncanonical row")
    if tuple(_validate_public_row(row, configuration) for row in rows) != rows:
        raise ValueError("finalized Task 12 rows fail public-row validation")
    if tuple(row.task_id for row in rows) != configuration.expected_task_ids:
        raise ValueError("finalized Task 12 run has incomplete task coverage")
    if any(
        row.completion_status.value in {"failed", "partial"}
        for row in rows
    ):
        raise ValueError("finalized Task 12 run contains failed or partial rows")
    row_hashes = {row.task_id: sha256_model(row) for row in rows}
    status_ids = {
        status: tuple(
            row.task_id
            for row in rows
            if row.completion_status.value == status
        )
        for status in ("completed", "failed", "partial", "not_supported")
    }
    expected_progress = ExternalRunProgressV1(
        run_identity=expected_identity.run_identity,
        expected_task_ids=configuration.expected_task_ids,
        completed_task_ids=status_ids["completed"],
        failed_task_ids=status_ids["failed"],
        partial_task_ids=status_ids["partial"],
        not_supported_task_ids=status_ids["not_supported"],
        run_record_hashes=row_hashes,
        finalized=True,
    )
    progress_raw = read_task12_regular_file_v3(root / "progress.json")
    if progress_raw != canonical_json_bytes(expected_progress):
        raise ValueError("finalized Task 12 progress differs from task rows")
    manifest_raw = read_task12_regular_file_v3(root / "run_manifest.json")
    run_manifest = RunManifestV3.model_validate_json(manifest_raw)
    if canonical_json_bytes(run_manifest) != manifest_raw:
        raise ValueError("finalized Task 12 run manifest is not canonical")
    runtime_artifacts = tuple(
        artifact
        for artifact in run_manifest.normalized_runtime_artifacts
        if artifact.path == "task_runs.jsonl"
    )
    expected_runtime_artifact = ArtifactRef(
        path="task_runs.jsonl",
        sha256=hashlib.sha256(rows_raw).hexdigest(),
        media_type="application/x-ndjson",
        record_count=len(rows),
    )
    checks = {
        "run ID": run_manifest.run_id == configuration.run_id,
        "code revision": run_manifest.code_revision == configuration.code_revision,
        "dirty state": run_manifest.dirty_state == configuration.dirty_state,
        "task manifest": run_manifest.task_manifest
        == configuration.source_task_manifest_ref,
        "capability verification": run_manifest.capability_verification_artifact
        == configuration.capability_verification_ref,
        "scorer config": run_manifest.scorer_config == configuration.scorer_config,
        "adapter info": run_manifest.adapter_info == configuration.adapter_info,
        "adapter capabilities": run_manifest.adapter_capabilities
        == configuration.adapter_capabilities,
        "model name": run_manifest.model_name == configuration.model_name,
        "provider": run_manifest.provider == configuration.provider,
        "model revision": run_manifest.model_revision
        == configuration.model_revision,
        "prompt config": run_manifest.prompt_config == configuration.prompt_config,
        "decode config": run_manifest.decoding_config
        == configuration.decoding_config,
        "seed": run_manifest.seed_information == configuration.seed_information,
        "action parser": run_manifest.action_parser_version
        == configuration.action_parser_version,
        "answer parser": run_manifest.answer_parser_version
        == configuration.answer_parser_version,
        "entry extractor": run_manifest.memory_entry_extractor_version
        == configuration.memory_entry_extractor_version,
        "object extractor": run_manifest.object_value_extractor_config_hash
        == configuration.object_value_extractor_config_hash,
        "redaction policy": run_manifest.redaction_policy_version
        == configuration.redaction_policy_version,
        "environment": run_manifest.environment_summary
        == configuration.environment_summary,
        "packages": run_manifest.package_summary == configuration.package_summary,
        "expected count": run_manifest.expected_task_count == len(rows),
        "completed count": run_manifest.completed_task_count
        == len(status_ids["completed"]),
        "failed count": run_manifest.failed_task_count == 0,
        "not-supported count": run_manifest.not_supported_task_count
        == len(status_ids["not_supported"]),
        "row hashes": dict(run_manifest.run_record_hashes) == row_hashes,
        "runtime artifact": runtime_artifacts == (expected_runtime_artifact,),
    }
    failed = tuple(label for label, valid in checks.items() if not valid)
    if failed:
        raise ValueError(f"finalized Task 12 run manifest mismatch: {failed[0]}")
    return run_manifest, rows


def run_task12_cell_v3(
    tasks: Sequence[MemUpdateTaskV3],
    *,
    adapter_factory,
    run_configuration: ExternalRunConfigV1,
    runtime_config: RuntimeConfigV3,
    prompted_answer_model: Any,
    context_order: ContextOrder,
    context_annotation: ContextAnnotation,
    frozen_trajectories: Mapping[str, RawAppendTrajectoryV1],
    output_root: str | Path,
    task_manifest: TaskManifestV3,
    run_manifest_artifact: ArtifactRef | None,
    task_artifact: ArtifactRef,
    authenticated_task_manifest_sha256: str,
    resume: bool = False,
) -> tuple[RunManifestV3, tuple[TaskRunRecordV3, ...], tuple[ScoreRecordV3, ...], dict[str, object]]:
    """Execute, finalize, reload, and authenticated-score one cell/slot run."""
    ordered_tasks = tuple(tasks)
    if set(frozen_trajectories) != {task.task_id for task in ordered_tasks}:
        raise ValueError("Task 12 frozen trajectories must cover the run task set exactly")
    def validated_adapter_factory(task: MemUpdateTaskV3):
        adapter = adapter_factory(task)
        if (
            adapter.adapter_id != run_configuration.adapter_info.adapter_id
            or getattr(adapter, "retrieval_policy", None)
            != run_configuration.retrieval_policy
        ):
            raise ValueError("Task 12 adapter differs from run configuration")
        return adapter

    canonical_adapter_factory = canonical_task12_action_id_adapter_factory_v3(
        validated_adapter_factory
    )
    if tuple(task.task_id for task in ordered_tasks) != run_configuration.expected_task_ids:
        raise ValueError("Task 12 tasks must match run configuration order")
    validate_task12_runtime_configuration_v3(
        run_configuration.runtime_configuration_hash,
        expected_run_id=run_configuration.run_id,
        runtime_config=runtime_config,
        context_order=context_order,
        context_annotation=context_annotation,
    )
    output = Path(output_root)
    if (output / "run_manifest.json").is_file():
        if not resume:
            raise FileExistsError("Task 12 run is already finalized")
        run_manifest, persisted_rows = load_finalized_task12_run_v3(
            output,
            run_configuration,
        )
    else:
        existing_rows: tuple[TaskRunRecordV3, ...] = ()
        resume_incomplete = resume and output.exists()
        if resume_incomplete:
            writer = ExternalRunWriterV1.resume(output, run_configuration)
            existing_rows = writer.rows
        remaining_tasks = ordered_tasks[len(existing_rows) :]
        rows = existing_rows + tuple(
            execute_task12_task_v3(
                task,
                canonical_adapter_factory(task),
                runtime_config,
                prompted_answer_model=prompted_answer_model,
                context_order=context_order,
                context_annotation=context_annotation,
                frozen_trajectory=frozen_trajectories[task.task_id],
            )
            for task in remaining_tasks
        )
        run_manifest = persist_task12_rows_v3(
            output,
            run_configuration,
            rows,
            resume=resume_incomplete,
        )
        reloaded_manifest, persisted_rows = load_finalized_task12_run_v3(
            output,
            run_configuration,
        )
        if reloaded_manifest != run_manifest or persisted_rows != rows:
            raise ValueError("reloaded Task 12 run differs from finalized rows")
        run_manifest = reloaded_manifest
    run_manifest_sha256 = hashlib.sha256(canonical_json_bytes(run_manifest)).hexdigest()
    if run_manifest_artifact is None:
        candidates = tuple(
            artifact
            for artifact in run_manifest.normalized_runtime_artifacts
            if artifact.path == "task_runs.jsonl"
        )
        if len(candidates) != 1:
            raise ValueError("finalized run must authenticate exactly one task-runs artifact")
        run_manifest_artifact = candidates[0]
    scores = score_task12_rows_v3(
        ordered_tasks,
        persisted_rows,
        task_manifest=task_manifest,
        run_manifest=run_manifest,
        task_artifact=task_artifact,
        run_artifact=run_manifest_artifact,
        authenticated_task_manifest_sha256=authenticated_task_manifest_sha256,
        authenticated_run_manifest_sha256=run_manifest_sha256,
    )
    score_root = output / "scores"
    if score_root.exists():
        if not resume:
            raise FileExistsError("Task 12 score artifacts already exist")
        verified_scores, verified_receipt = verify_task12_score_artifact_v3(
            score_root,
            expected_task_ids=run_configuration.expected_task_ids,
            run_manifest_sha256=run_manifest_sha256,
            task_manifest_sha256=authenticated_task_manifest_sha256,
        )
        if verified_scores != scores:
            raise ValueError("existing Task 12 scores differ from recomputed scores")
        receipt = verified_receipt
    else:
        receipt = persist_task12_scores_v3(
            score_root,
            scores,
            run_manifest_sha256=run_manifest_sha256,
            task_manifest_sha256=authenticated_task_manifest_sha256,
        )
        verified_scores, verified_receipt = verify_task12_score_artifact_v3(
            score_root,
            expected_task_ids=run_configuration.expected_task_ids,
            run_manifest_sha256=run_manifest_sha256,
            task_manifest_sha256=authenticated_task_manifest_sha256,
        )
        if verified_scores != scores or verified_receipt != receipt:
            raise ValueError("reloaded Task 12 scores differ from scored rows")
    return run_manifest, persisted_rows, scores, receipt


__all__ = [
    "Task12ExecutionAuthorizationV1",
    "Task12PresentationAdapterV3",
    "Task12RuntimeCodeBindingV1",
    "canonical_task12_action_id_adapter_factory_v3",
    "execute_task12_task_v3",
    "find_admitted_answer_run_v3",
    "load_finalized_task12_run_v3",
    "persist_task12_rows_v3",
    "persist_task12_scores_v3",
    "read_task12_regular_file_v3",
    "score_task12_rows_v3",
    "run_task12_cell_v3",
    "select_admitted_answer_run_v3",
    "task12_runtime_code_binding_v3",
    "task12_runtime_configuration_sha256_v3",
    "transform_retrieval_trace_v3",
    "validate_task12_runtime_code_binding_v3",
    "validate_task12_runtime_configuration_v3",
    "verify_task12_score_artifact_v3",
]
