from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.vnext_promote_family_h_cross_domain_noaa import (
    CANDIDATE_INDEX_SHA256,
    CAPTURE_INDEX_SHA256,
    CAPTURE_MANIFEST_SHA256,
    REVIEW_INDEX_SHA256,
    publish_release,
    read_release,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "results/vnext/family_h_cross_domain_noaa_candidate_20260928_v3"
CAPTURE = ROOT / "external/family_h_cross_domain_noaa_20260928_v5"
REVIEW = ROOT / "results/vnext/family_h_cross_domain_noaa_review_completed_20260928_v1"
DECISIONS = ROOT / "results/vnext/family_h_cross_domain_noaa_review_20260928_v3/decisions.json"
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in (CANDIDATE, CAPTURE, REVIEW, DECISIONS)),
    reason="requires the separately retained original NOAA capture/candidate/review inputs; not bundled with source",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "release"
    result = publish_release(CANDIDATE, REVIEW, DECISIONS, CAPTURE, output)
    return output, result


def test_promotes_exact_candidate_and_review_bytes_without_raw_capture(tmp_path: Path) -> None:
    before = {path.name: path.read_bytes() for path in CANDIDATE.iterdir() if path.is_file()}
    output, result = _publish(tmp_path)

    assert result["manifest"]["status"] == "FINAL_APPROVED"
    assert {path.name for path in output.iterdir()} == {
        "manifest.json", "normalized_records.jsonl", "semantic_cores.jsonl", "tasks.jsonl",
        "audit_manifest.json", "decisions_template.json", "reference_sanity.json", "candidate_index.json",
        "decisions.json", "review_completion_receipt.json", "human_review_index.json",
        "release_manifest.json", "release_index.json",
    }
    assert (output / "tasks.jsonl").read_bytes() == before["tasks.jsonl"]
    assert (output / "candidate_index.json").read_bytes() == before["index.json"]
    assert (output / "decisions.json").read_bytes() == DECISIONS.read_bytes()
    assert not list(output.rglob("*.shtml"))
    assert not (output / "raw").exists()
    assert result["index_sha256"] == _sha(output / "release_index.json")


def test_release_reader_returns_stable_shape_and_does_not_write(tmp_path: Path) -> None:
    output, published = _publish(tmp_path)
    names_before = sorted(path.name for path in output.iterdir())
    result = read_release(output, published["index_sha256"])
    assert list(result) == ["manifest", "tasks", "index", "root", "index_sha256"]
    assert result["root"] == output.resolve()
    assert result["index_sha256"] == published["index_sha256"]
    assert sorted(path.name for path in output.iterdir()) == names_before


def test_release_hash_is_deterministic_without_timestamp(tmp_path: Path) -> None:
    first, one = _publish(tmp_path / "one")
    second, two = _publish(tmp_path / "two")
    assert one["index_sha256"] == two["index_sha256"]
    assert (first / "release_manifest.json").read_bytes() == (second / "release_manifest.json").read_bytes()


def test_scientific_boundary_and_lineage_metadata(tmp_path: Path) -> None:
    output, _ = _publish(tmp_path)
    manifest = json.loads((output / "release_manifest.json").read_bytes())
    assert manifest["status"] == "FINAL_APPROVED"
    assert manifest["formal_task_release"] is True
    assert manifest["scientific_release_allowed"] is False
    assert manifest["benchmark_accuracy_claimed"] is False
    assert manifest["model_backend_status"] == "NOT_RUN"
    assert manifest["runtime_state_metrics"] is None
    assert manifest["runtime_retrieval_metrics"] is None
    assert manifest["answer_metrics"] is None
    assert manifest["source_count"] == 1
    assert manifest["source_count_semantics"] == "one upstream storm trajectory, not statistically independent samples"
    assert manifest["source_domain"] == "weather"
    assert manifest["source_type"] == "public_advisory"
    assert manifest["revision_only"] is False
    assert manifest["counts"] == {"records": 7, "cores": 1, "tasks": 1, "events": 7, "decisions": 9}
    assert manifest["update_counts"] == {"ADD": 1, "UPDATE": 6, "equal_value_updates": 1}
    assert "NWS public-domain" in manifest["source_boundary"]
    assert "third-party" in manifest["source_boundary"]
    assert "endorsement" in manifest["source_boundary"]


def test_all_nine_review_decisions_are_bound(tmp_path: Path) -> None:
    output, _ = _publish(tmp_path)
    decisions = json.loads((output / "decisions.json").read_bytes())
    receipt = json.loads((output / "review_completion_receipt.json").read_bytes())
    assert len(decisions["decisions"]) == 9
    assert {row["decision"] for row in decisions["decisions"]} == {"RELEASE_READY"}
    assert receipt["decision_count"] == 9
    assert receipt["counts"] == {"RELEASE_READY": 9}


@pytest.mark.parametrize("kind", ["candidate", "review", "capture"])
def test_tampered_inputs_are_rejected(tmp_path: Path, kind: str) -> None:
    roots = {"candidate": CANDIDATE, "review": REVIEW, "capture": CAPTURE}
    source = roots[kind]
    copied = tmp_path / kind
    copied.mkdir()
    for path in source.rglob("*"):
        if path.is_file():
            destination = copied / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    tampered = next(path for path in copied.rglob("*") if path.is_file() and path.name != "index.json")
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    with pytest.raises((ValueError, FileNotFoundError)):
        publish_release(
            copied if kind == "candidate" else CANDIDATE,
            copied if kind == "review" else REVIEW,
            DECISIONS,
            copied if kind == "capture" else CAPTURE,
            tmp_path / "out",
        )


def test_no_replace_and_review_root_binding(tmp_path: Path) -> None:
    output, _ = _publish(tmp_path)
    with pytest.raises(FileExistsError):
        _publish(tmp_path)
    wrong_review = tmp_path / "wrong-review"
    wrong_review.mkdir()
    for path in REVIEW.iterdir():
        (wrong_review / path.name).write_bytes(path.read_bytes())
    raw = json.loads((wrong_review / "decisions.json").read_bytes())
    raw["candidate_index_sha256"] = "0" * 64
    (wrong_review / "decisions.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        publish_release(CANDIDATE, wrong_review, DECISIONS, CAPTURE, tmp_path / "wrong-out")


def test_expected_input_hash_constants_are_frozen() -> None:
    assert _sha(CANDIDATE / "index.json") == CANDIDATE_INDEX_SHA256
    assert _sha(REVIEW / "index.json") == REVIEW_INDEX_SHA256
    assert _sha(CAPTURE / "capture_manifest.json") == CAPTURE_MANIFEST_SHA256
    assert _sha(CAPTURE / "capture_index.json") == CAPTURE_INDEX_SHA256
    assert _sha(DECISIONS) == "a0075eaeb2d81de1d3b039b525cfdd31243eb93f92999ab9ee13004f6f02d247"
