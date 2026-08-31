from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.io.atomic import publish_files_atomically


EXPECTED_TASK_COUNT = 720
TEST_SCOPE = "test720"
EVIDENCE_CLASS = "answer_layer_panel_comparison"
RELEASE_ID = "memupdatebench.main-track.answer-panel.v1"
SCHEMA_VERSION = "memupdatebench.main-track.answer-panel.v1"
INDEX_SCHEMA_VERSION = "memupdatebench.main-track.answer-panel.artifact-index.v1"

_REQUIRED_ARTIFACTS = ("rows.jsonl", "summary.json", "artifact_index.json")
_REQUIRED_ROW_FIELDS = (
    "task_id",
    "family",
    "domain",
    "language",
    "status",
    "expected_disposition",
    "answer_disposition",
    "answer_format_valid",
    "parsed_answer",
    "answer_outcome",
    "exact_match",
    "normalized_match",
    "typed_match",
    "typed_exact_match",
    "answer_f1",
)
_OUTCOMES = frozenset(
    {"CORRECT", "WRONG", "FORMAT_INVALID", "UNAVAILABLE", "CORRECT_ABSTENTION", "WRONG_ABSTENTION"}
)
_DISPOSITION_VALUES = frozenset({"answered", "abstained"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    root: Path
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    index: dict[str, Any]
    rows_bytes: bytes
    summary_bytes: bytes
    index_bytes: bytes
    task_ids: tuple[str, ...]
    candidate_artifact_hashes: dict[str, str]
    audit_attestation_sha256: str
    model_binding: dict[str, Any]


def canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file(path, label)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if canonical_bytes(value) != raw:
        raise ValueError(f"{label} must use canonical JSON")
    return value, raw


def _contains_raw_field(value: Any, path: str = "root") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                return f"{path} has a non-string field name"
            lowered = key.casefold()
            is_hash = lowered.endswith("_sha256")
            if not is_hash and (
                lowered in {"prompt", "raw_prompt", "rendered_prompt", "rendered_chat_prompt", "output", "raw_output", "generated_text", "reasoning", "reasoning_content", "raw_reasoning"}
                or lowered.startswith("raw_prompt_")
                or lowered.startswith("raw_output_")
                or lowered.startswith("raw_reasoning_")
            ):
                return f"{path}.{key} is a raw prompt/output/reasoning field"
            found = _contains_raw_field(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _contains_raw_field(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _require_hash_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty hash map")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str) or _HEX64.fullmatch(item) is None:
            raise ValueError(f"{label} contains an invalid SHA-256")
        result[key] = item
    return result


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_rows(path: Path) -> tuple[tuple[dict[str, Any], ...], bytes]:
    raw = _read_regular_file(path, "rows.jsonl")
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n"):
            raise ValueError(f"rows.jsonl line {line_number} must be canonical JSONL with LF")
        payload = line[:-1]
        if not payload:
            raise ValueError(f"rows.jsonl line {line_number} is blank")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"rows.jsonl line {line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"rows.jsonl line {line_number} must contain an object")
        if canonical_bytes(value) != payload:
            raise ValueError(f"rows.jsonl line {line_number} must use canonical JSON")
        found = _contains_raw_field(value, f"rows.jsonl line {line_number}")
        if found:
            raise ValueError(found)
        rows.append(value)
    if len(rows) != EXPECTED_TASK_COUNT:
        raise ValueError(f"rows.jsonl must contain exactly {EXPECTED_TASK_COUNT} rows")
    return tuple(rows), raw


def _validate_row(row: Mapping[str, Any], number: int, *, candidate_hashes: dict[str, str], audit_sha: str, model_binding: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"row {number} is missing answer outcome fields: {', '.join(missing)}")
    if row["status"] != "PASS":
        raise ValueError(f"row {number} status must be PASS")
    for field in ("task_id", "family", "domain", "language"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"row {number} {field} must be a nonblank string")
    if row["expected_disposition"] not in _DISPOSITION_VALUES:
        raise ValueError(f"row {number} expected_disposition is invalid")
    if row["answer_disposition"] is not None and row["answer_disposition"] not in _DISPOSITION_VALUES:
        raise ValueError(f"row {number} answer_disposition is invalid")
    if row["answer_outcome"] not in _OUTCOMES:
        raise ValueError(f"row {number} answer_outcome is invalid")
    if row["answer_format_valid"] is not None and type(row["answer_format_valid"]) is not bool:
        raise ValueError(f"row {number} answer_format_valid must be boolean or null")
    for field in ("exact_match", "normalized_match", "typed_match", "typed_exact_match"):
        if type(row[field]) is not bool:
            raise ValueError(f"row {number} {field} must be boolean")
    if type(row["answer_f1"]) not in (int, float) or isinstance(row["answer_f1"], bool) or not 0 <= float(row["answer_f1"]) <= 1:
        raise ValueError(f"row {number} answer_f1 must be a number in [0, 1]")
    if row.get("candidate_artifact_hashes") != candidate_hashes:
        raise ValueError(f"row {number} candidate_artifact_hashes do not match summary")
    if row.get("audit_attestation_sha256") != audit_sha:
        raise ValueError(f"row {number} audit_attestation_sha256 does not match summary")
    if row.get("model_binding") != model_binding:
        raise ValueError(f"row {number} model_binding does not match summary")


def _validate_artifact_metadata(index: Mapping[str, Any], filename: str, raw: bytes, record_count: int) -> None:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"rows.jsonl", "summary.json"}:
        raise ValueError("artifact_index.json artifacts must bind rows.jsonl and summary.json exactly")
    metadata = artifacts.get(filename)
    if not isinstance(metadata, dict):
        raise ValueError(f"artifact_index.json is missing {filename} metadata")
    if metadata.get("sha256") != sha256_bytes(raw):
        raise ValueError(f"artifact_index.json {filename} hash does not match bytes")
    if metadata.get("bytes") != len(raw) or metadata.get("record_count") != record_count:
        raise ValueError(f"artifact_index.json {filename} size/count does not match bytes")


def load_result_root(root: str | Path) -> ResultSnapshot:
    result_root = Path(root).resolve(strict=True)
    if result_root.is_symlink() or not result_root.is_dir():
        raise ValueError("result root must be a regular directory")
    for name in _REQUIRED_ARTIFACTS:
        if not (result_root / name).exists():
            raise FileNotFoundError(result_root / name)

    rows, rows_bytes = _read_rows(result_root / "rows.jsonl")
    summary, summary_bytes = _read_canonical_json(result_root / "summary.json", "summary.json")
    index, index_bytes = _read_canonical_json(result_root / "artifact_index.json", "artifact_index.json")
    for label, value in (("summary.json", summary), ("artifact_index.json", index)):
        found = _contains_raw_field(value, label)
        if found:
            raise ValueError(found)

    if summary.get("scope") != TEST_SCOPE or index.get("scope") != TEST_SCOPE:
        raise ValueError("answer panel inputs must use scope test720")
    if summary.get("rows") != EXPECTED_TASK_COUNT:
        raise ValueError(f"summary.json rows must be exactly {EXPECTED_TASK_COUNT}")
    if index.get("release_id") != summary.get("release_id"):
        raise ValueError("artifact_index.json release_id does not match summary.json")
    _validate_artifact_metadata(index, "rows.jsonl", rows_bytes, EXPECTED_TASK_COUNT)
    _validate_artifact_metadata(index, "summary.json", summary_bytes, 1)
    rows_sha = sha256_bytes(rows_bytes)
    if summary.get("rows_sha256") != rows_sha:
        raise ValueError("summary.json rows_sha256 does not match rows.jsonl")

    candidate_hashes = _require_hash_map(summary.get("candidate_artifact_hashes"), "summary.json candidate_artifact_hashes")
    if index.get("candidate_artifact_hashes") != candidate_hashes:
        raise ValueError("artifact_index.json candidate_artifact_hashes do not match summary.json")
    audit_sha = _require_sha(summary.get("audit_attestation_sha256"), "summary.json audit_attestation_sha256")
    if index.get("audit_attestation_sha256") != audit_sha:
        raise ValueError("artifact_index.json audit_attestation_sha256 does not match summary.json")
    model_binding = summary.get("model_binding")
    if not isinstance(model_binding, dict) or not model_binding:
        raise ValueError("summary.json model_binding must be a non-empty object")
    if index.get("model_binding") != model_binding:
        raise ValueError("artifact_index.json model_binding does not match summary.json")

    task_ids: list[str] = []
    for number, row in enumerate(rows, start=1):
        _validate_row(row, number, candidate_hashes=candidate_hashes, audit_sha=audit_sha, model_binding=model_binding)
        task_ids.append(row["task_id"])
    if len(set(task_ids)) != EXPECTED_TASK_COUNT:
        raise ValueError("rows.jsonl task_id values must be unique")
    return ResultSnapshot(
        root=result_root,
        rows=rows,
        summary=summary,
        index=index,
        rows_bytes=rows_bytes,
        summary_bytes=summary_bytes,
        index_bytes=index_bytes,
        task_ids=tuple(task_ids),
        candidate_artifact_hashes=candidate_hashes,
        audit_attestation_sha256=audit_sha,
        model_binding=model_binding,
    )


def _outcome_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(row["answer_outcome"] for row in rows)
    return {name: counts.get(name, 0) for name in sorted(_OUTCOMES)}


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def _axis_summary(rows: Sequence[Mapping[str, Any]], axis: str) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[axis]), []).append(row)
    em = {key: _mean(group, "exact_match") for key, group in sorted(groups.items())}
    details = {
        key: {
            "rows": len(group),
            "exact_match": _mean(group, "exact_match"),
            "answer_em": _mean(group, "exact_match"),
            "normalized_em": _mean(group, "normalized_match"),
            "typed_em": _mean(group, "typed_match"),
            "answer_f1": _mean(group, "answer_f1"),
            "answer_outcome_counts": _outcome_counts(group),
        }
        for key, group in sorted(groups.items())
    }
    return em, details


def _model_summary(snapshot: ResultSnapshot) -> dict[str, Any]:
    rows = snapshot.rows
    family_em, family_details = _axis_summary(rows, "family")
    language_em, language_details = _axis_summary(rows, "language")
    domain_em, domain_details = _axis_summary(rows, "domain")
    metrics = {
        "exact_match": _mean(rows, "exact_match"),
        "normalized_match": _mean(rows, "normalized_match"),
        "typed_match": _mean(rows, "typed_match"),
        "typed_exact_match": _mean(rows, "typed_exact_match"),
        "answer_f1": _mean(rows, "answer_f1"),
    }
    outcomes = _outcome_counts(rows)
    denominators = {
        "attempted": len(rows),
        "evaluable": sum(row["answer_outcome"] is not None for row in rows),
        "answerable": sum(row["expected_disposition"] == "answered" for row in rows),
        "abstention": sum(row["expected_disposition"] == "abstained" for row in rows),
    }
    return {
        "model_id": snapshot.model_binding.get("model_id"),
        "model_binding": snapshot.model_binding,
        "candidate_artifact_hashes": snapshot.candidate_artifact_hashes,
        "audit_attestation_sha256": snapshot.audit_attestation_sha256,
        "rows": len(rows),
        "attempted_denominator": denominators["attempted"],
        "evaluable_denominator": denominators["evaluable"],
        "answerable_denominator": denominators["answerable"],
        "abstention_denominator": denominators["abstention"],
        "metrics": metrics,
        "answer_em": metrics["exact_match"],
        "answer_normalized_em": metrics["normalized_match"],
        "answer_typed_em": metrics["typed_match"],
        "answer_f1": metrics["answer_f1"],
        "answer_outcome_counts": outcomes,
        "per_family_em": family_em,
        "per_language_em": language_em,
        "per_domain_em": domain_em,
        "family_em": family_em,
        "language_em": language_em,
        "domain_em": domain_em,
        "per_family": family_details,
        "per_language": language_details,
        "per_domain": domain_details,
        "by_family": family_details,
        "by_language": language_details,
        "by_domain": domain_details,
    }


def _answer_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("answer_disposition"),
        row.get("answer_format_valid"),
        row.get("answer_outcome"),
        canonical_bytes(row.get("parsed_answer")),
        row.get("exact_match"),
        row.get("normalized_match"),
        row.get("typed_match"),
        row.get("typed_exact_match"),
    )


def _disagreement_category(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> str:
    outcomes = {row_a.get("answer_outcome"), row_b.get("answer_outcome")}
    if outcomes.intersection({"FORMAT_INVALID", "UNAVAILABLE"}) or None in {row_a.get("answer_disposition"), row_b.get("answer_disposition")}:
        return "format_unavailable"
    if "abstained" in {row_a.get("answer_disposition"), row_b.get("answer_disposition")} or outcomes.intersection({"CORRECT_ABSTENTION", "WRONG_ABSTENTION"}):
        return "abstention"
    return "answer"


def build_panel_summary(model_a: ResultSnapshot, model_b: ResultSnapshot) -> dict[str, Any]:
    if model_a.task_ids != model_b.task_ids:
        raise ValueError("model A and model B task order does not match")
    if model_a.candidate_artifact_hashes != model_b.candidate_artifact_hashes:
        raise ValueError("model A and model B candidate_artifact_hashes do not match")
    if model_a.audit_attestation_sha256 != model_b.audit_attestation_sha256:
        raise ValueError("model A and model B audit_attestation_sha256 do not match")

    agreement = 0
    categories = Counter()
    shared_task_fields = (
        "task_id",
        "core_id",
        "semantic_core_id",
        "family",
        "domain",
        "attribute",
        "language",
        "split",
        "expected_disposition",
        "gold_answer",
        "task_sha256",
    )
    for row_a, row_b in zip(model_a.rows, model_b.rows):
        for field in shared_task_fields:
            if row_a.get(field) != row_b.get(field):
                raise ValueError(f"paired task metadata field {field} does not match")
        if _answer_signature(row_a) == _answer_signature(row_b):
            agreement += 1
        else:
            categories[_disagreement_category(row_a, row_b)] += 1
    disagreement = EXPECTED_TASK_COUNT - agreement
    if sum(categories.values()) != disagreement:
        raise ValueError("paired disagreement accounting does not cover all tasks")
    category_counts = {name: categories.get(name, 0) for name in ("answer", "abstention", "format_unavailable")}
    claim_boundary = "Paired fixed-reference answer-layer comparison only; no external-manager evidence and no statistical significance claim."
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": claim_boundary,
        "scope": TEST_SCOPE,
        "paired_task_count": EXPECTED_TASK_COUNT,
        "candidate_artifact_hashes": model_a.candidate_artifact_hashes,
        "audit_attestation_sha256": model_a.audit_attestation_sha256,
        "models": {"model_a": _model_summary(model_a), "model_b": _model_summary(model_b)},
        "paired_comparison": {
            "task_count": EXPECTED_TASK_COUNT,
            "agreement": agreement,
            "disagreement": disagreement,
            "agreement_rate": agreement / EXPECTED_TASK_COUNT,
            "disagreement_rate": disagreement / EXPECTED_TASK_COUNT,
        },
        "paired_agreement_count": agreement,
        "paired_disagreement_count": disagreement,
        "paired_agreement": {"count": agreement, "rate": agreement / EXPECTED_TASK_COUNT},
        "paired_disagreement": {"count": disagreement, "rate": disagreement / EXPECTED_TASK_COUNT},
        "disagreement_categories": category_counts,
    }


def _validate_output_root(output: Path, model_a: ResultSnapshot, model_b: ResultSnapshot) -> None:
    if output.is_symlink():
        raise ValueError("panel output root must not be a symlink")
    resolved = output.resolve(strict=False)
    for source in (model_a.root, model_b.root):
        if resolved == source or source in resolved.parents:
            raise ValueError("panel output root must not overlap an input result root")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("panel output root must be empty for no-replace publication")


def publish_panel(model_a_root: str | Path, model_b_root: str | Path, output_root: str | Path) -> Path:
    model_a = load_result_root(model_a_root)
    model_b = load_result_root(model_b_root)
    _validate_output_root(Path(output_root), model_a, model_b)
    summary = build_panel_summary(model_a, model_b)
    summary_bytes = canonical_bytes(summary)
    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "scope": TEST_SCOPE,
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": summary["claim_boundary"],
        "paired_task_count": EXPECTED_TASK_COUNT,
        "candidate_artifact_hashes": summary["candidate_artifact_hashes"],
        "audit_attestation_sha256": summary["audit_attestation_sha256"],
        "model_bindings": {
            "model_a": model_a.model_binding,
            "model_b": model_b.model_binding,
        },
        "artifacts": {
            "panel_summary.json": {
                "sha256": sha256_bytes(summary_bytes),
                "bytes": len(summary_bytes),
                "record_count": 1,
            }
        },
    }
    index_bytes = canonical_bytes(index)
    destinations = {
        Path(output_root) / "panel_summary.json": summary_bytes,
        Path(output_root) / "panel_index.json": index_bytes,
    }
    validators = {
        path: (lambda staged, expected=payload: None if staged.read_bytes() == expected else (_ for _ in ()).throw(ValueError("staged panel artifact bytes changed")))
        for path, payload in destinations.items()
    }
    publish_files_atomically(destinations, overwrite=False, validators=validators)
    return Path(output_root)


def run(model_a_root: str | Path, model_b_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    publish_panel(model_a_root, model_b_root, output_root)
    return json.loads((Path(output_root) / "panel_summary.json").read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two paired main-track answer-layer result roots")
    parser.add_argument("--model-a-root", "--model-a", dest="model_a_root", type=Path, required=True)
    parser.add_argument("--model-b-root", "--model-b", dest="model_b_root", type=Path, required=True)
    parser.add_argument("--output-root", "--output", dest="output_root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args.model_a_root, args.model_b_root, args.output_root)
    except Exception as exc:
        message = re.sub(r"[^a-zA-Z0-9_. -]", "", str(exc))
        print(f"main-track answer panel failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
