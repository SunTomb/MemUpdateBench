from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import vnext_promote_family_h_cross_domain_bea as promotion
from scripts.vnext_prepare_family_h_bea_candidate import _make_task


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CANDIDATE_INDEX = "2a12b35dd2972f98d6b90f8cb9bfea46a60924e910ac94cab37c79756a6f86cd"
EXPECTED_CAPTURE_MANIFEST = "9395b50167b4b47f64e19b9ebcbbdafe6c3ba70e1084ebb9a273bc3ba179eb1b"
EXPECTED_CAPTURE_INDEX = "17aa9bd3a576effa7b30b79f87e6090e07edf58839594518f808049ac11e6313"
EXPECTED_POLICY = "e49a712b7e0ec2e8946e256315f4af675af72c29c8d9e019c46626fa56b5a79e"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _records() -> list[dict]:
    base = {
        "source_group_id": "family-h-cross-domain-bea-gdp-2024q4",
        "source_document_id": "bea-gdp-2024q4-revision-trajectory",
        "reference_period": "2024-Q4", "unit": "percent", "rate_basis": "annualized",
        "table_label": "Real GDP", "xlsx_url": "https://example.invalid/gdp.xlsx",
        "page_url": "https://example.invalid/gdp", "table_url": "https://example.invalid/gdp.xlsx",
        "source_anchor": {"worksheet": "Table 1", "row": 4, "column": 3},
    }
    rows = []
    for stage, date, value in (("advance", "2025-01-30", 2.3), ("second", "2025-02-27", 2.3), ("third", "2025-03-27", 2.4)):
        rows.append({**base, "record_id": f"bea-gdp-2024q4-{stage}", "release_date": date,
                     "estimate_stage": stage, "value": value, "source_file_sha256": hashlib.sha256(stage.encode()).hexdigest()})
    return rows


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    capture = tmp_path / "capture"
    capture.mkdir()
    policy = tmp_path / "policy"
    policy.mkdir()
    policy_html = ("<p>BEA information is in the public domain and may be used or reproduced without "
                   "specific permission. A citation such as \"Source: U.S. Bureau of Economic Analysis\" would be appreciated.</p>"
                   "<p>must not contain information that suggests such an endorsement</p>"
                   "<p>cannot authorize the use of copyrighted materials</p>").encode()
    (policy / "linking.html").write_bytes(policy_html)
    records = _records()
    policy_evidence = {"status": "POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY", "source_url": "https://www.bea.gov/about/policies-and-information/linking", "source_sha256": _sha(policy_html), "official_citation_requirement": "appreciated, not stated as mandatory", "quotes": {"reuse": "reuse", "citation": "citation", "endorsement": "endorsement", "external_copyright": "copyright"}, "raw_source_redistribution": False, "logos_included": False, "derivation_notice": "Derived records are not an official BEA product"}
    task_core, task = _make_task(records, "a" * 64, "b" * 64, policy_evidence["status"], policy_evidence["source_url"], policy_evidence)
    task_raw = _canonical(task.model_dump(mode="json"))
    core_raw = _canonical(task_core.model_dump(mode="json"))
    normalized_raw = b"".join(_canonical(record) for record in records)
    audit_items = []
    for record in records:
        material = {field: record[field] for field in ("record_id", "release_date", "estimate_stage", "reference_period", "unit", "rate_basis", "table_label", "value", "xlsx_url", "page_url", "table_url", "source_file_sha256", "source_anchor")}
        audit_items.append({"audit_id": "snapshot-" + record["estimate_stage"], "kind": "source_snapshot", "binding_sha256": _sha(_canonical(material)), "material": material})
    source_material = {"source_group_id": "family-h-cross-domain-bea-gdp-2024q4", "capture_manifest_sha256": "c" * 64, "capture_index_sha256": "d" * 64, "policy_url": policy_evidence["source_url"], "policy_status": policy_evidence["status"], "policy_evidence": policy_evidence, "source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "release_count": 3}
    audit_items.insert(0, {"audit_id": "source-bea-gdp-2024q4", "kind": "source_admission", "binding_sha256": _sha(_canonical(source_material)), "material": source_material})
    surface_material = {"task_id": task.task_id, "canonical_task_sha256": _sha(task_raw), "semantic_task_sha256": promotion.semantic_task_hash_v3(task), "event_count": 3, "query": task.queries[0].text}
    audit_items.append({"audit_id": "surface-bea-gdp-2024q4", "kind": "generated_surface", "binding_sha256": _sha(_canonical(surface_material)), "material": surface_material})
    audit = {"schema": "family-h-cross-domain-bea-candidate-audit-v1", "status": "NOT_STARTED", "items": audit_items, "item_count": 5}
    candidate_manifest = {"schema": "family-h-cross-domain-bea-candidate-v1-manifest-v1", "status": "CANDIDATE_PENDING_HUMAN_REVIEW", "source_group_id": "family-h-cross-domain-bea-gdp-2024q4", "source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "language": "en", "revision_only": True, "independent_samples": False, "semantic_core_count": 1, "task_count": 1, "snapshot_count": 3, "event_count": 3, "update_events": 2, "value_changing_updates": 1, "equal_value_update_events": 1, "policy_url": policy_evidence["source_url"], "policy_status": policy_evidence["status"], "policy_evidence": policy_evidence, "source_audit_status": "NOT_STARTED", "generated_surface_audit_status": "NOT_STARTED", "human_approved_sources": 0, "formal_task_release": False, "scientific_release_allowed": False, "answer_metrics": None, "model_loads": 0, "generations": 0, "provider_calls": 0, "capture_manifest_sha256": "c" * 64, "capture_index_sha256": "d" * 64, "reference_sanity": {"status": "PASS", "events": 3, "model_loads": 0, "generations": 0, "provider_calls": 0}}
    candidate_files = {"SOURCE_ATTRIBUTION.txt": b"Source: U.S. Bureau of Economic Analysis.\nThis is not an official BEA product.\n", "manifest.json": _canonical(candidate_manifest), "normalized_records.jsonl": normalized_raw, "semantic_cores.jsonl": core_raw, "tasks.jsonl": task_raw, "audit_manifest.json": _canonical(audit), "decisions_template.json": _canonical({"schema": "family-h-cross-domain-bea-candidate-human-decisions-v1", "status": "NOT_STARTED", "reviewer": None, "candidate_index_sha256": None, "decisions": [{"audit_id": item["audit_id"], "binding_sha256": item["binding_sha256"], "decision": None, "rationale": ""} for item in audit_items]}), "reference_sanity.json": _canonical(candidate_manifest["reference_sanity"])}
    candidate_index = {"schema": "family-h-cross-domain-bea-candidate-v1-index-v1", "status": "CANDIDATE_PENDING_HUMAN_REVIEW", "scientific_release_allowed": False, "artifacts": [{"path": name, "bytes": len(raw), "sha256": _sha(raw)} for name, raw in sorted(candidate_files.items())]}
    candidate_files["index.json"] = _canonical(candidate_index)
    candidate = tmp_path / "candidate"; candidate.mkdir()
    for name, raw in candidate_files.items(): (candidate / name).write_bytes(raw)
    capture_manifest = {"schema": "memupdatebench.family-h.bea-source-capture.v1", "status": "CAPTURED_PENDING_HUMAN_REVIEW", "source_group_id": "family-h-cross-domain-bea-gdp-2024q4", "source_document_id": "bea-gdp-2024q4-revision-trajectory", "source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "language": "en", "revision_only": True, "independent_samples": False, "policy_status": policy_evidence["status"], "policy": {"path": "policy/linking.html", "url": policy_evidence["source_url"], "sha256": _sha(policy_html)}, "records": [{"stage": record["estimate_stage"], "release_date": record["release_date"], "xlsx_url": record["xlsx_url"], "page_url": record["page_url"], "normalized": record} for record in records], "formal_task_release": False, "scientific_release_allowed": False}
    capture_manifest_raw = _canonical(capture_manifest); (capture / "capture_manifest.json").write_bytes(capture_manifest_raw)
    (capture / "policy").mkdir(); shutil.copy2(policy / "linking.html", capture / "policy/linking.html")
    capture_index = {"schema": "memupdatebench.family-h.bea-source-capture-index.v1", "status": "CAPTURED_PENDING_HUMAN_REVIEW", "scientific_release_allowed": False, "artifacts": [{"path": "policy/linking.html", "bytes": len(policy_html), "sha256": _sha(policy_html)}]}
    capture_index_raw = _canonical(capture_index); (capture / "capture_index.json").write_bytes(capture_index_raw)
    capture_manifest_sha = _sha(capture_manifest_raw); capture_index_sha = _sha(capture_index_raw)
    manifest_disk = json.loads((candidate / "manifest.json").read_bytes())
    manifest_disk["capture_manifest_sha256"] = capture_manifest_sha; manifest_disk["capture_index_sha256"] = capture_index_sha
    (candidate / "manifest.json").write_bytes(_canonical(manifest_disk))
    audit_disk = json.loads((candidate / "audit_manifest.json").read_bytes())
    source_item = next(item for item in audit_disk["items"] if item["kind"] == "source_admission")
    source_item["material"]["capture_manifest_sha256"] = capture_manifest_sha; source_item["material"]["capture_index_sha256"] = capture_index_sha
    source_item["binding_sha256"] = _sha(_canonical(source_item["material"]))
    (candidate / "audit_manifest.json").write_bytes(_canonical(audit_disk))
    index_disk = json.loads((candidate / "index.json").read_bytes())
    for item in index_disk["artifacts"]:
        raw = (candidate / item["path"]).read_bytes(); item.update(bytes=len(raw), sha256=_sha(raw))
    (candidate / "index.json").write_bytes(_canonical(index_disk))
    # Make fixture hashes authoritative for this test while module constants remain exact production pins.
    candidate_sha = _sha((candidate / "index.json").read_bytes())
    monkeypatch.setattr(promotion, "CANDIDATE_INDEX_SHA256", candidate_sha)
    monkeypatch.setattr(promotion, "CAPTURE_MANIFEST_SHA256", capture_manifest_sha)
    monkeypatch.setattr(promotion, "CAPTURE_INDEX_SHA256", capture_index_sha)
    monkeypatch.setattr(promotion, "POLICY_SHA256", _sha(policy_html))
    review = tmp_path / "review"; review.mkdir()
    decisions = {"schema": "family-h-cross-domain-bea-human-decisions-v1", "status": "COMPLETED", "reviewer": "reviewer", "candidate_index_sha256": candidate_sha, "decisions": [{"audit_id": item["audit_id"], "binding_sha256": item["binding_sha256"], "decision": "RELEASE_READY", "rationale": "checked"} for item in audit_disk["items"]]}
    decisions_raw = _canonical(decisions); (review / "decisions.json").write_bytes(decisions_raw)
    monkeypatch.setattr(promotion, "DECISIONS_SHA256", _sha(decisions_raw))
    receipt = {"candidate_index_sha256": candidate_sha, "decisions_sha256": _sha(decisions_raw), "decision_count": 5, "counts": {"RELEASE_READY": 5}, "status": "HUMAN_REVIEW_COMPLETE", "formal_task_release": False, "scientific_release_allowed": False, "reviewer": "reviewer"}
    receipt_raw = _canonical(receipt); (review / "review_completion_receipt.json").write_bytes(receipt_raw)
    review_index = {"schema": "family-h-cross-domain-bea-human-review-index.v1", "status": "HUMAN_REVIEW_COMPLETE", "formal_task_release": False, "scientific_release_allowed": False, "artifacts": [{"path": "decisions.json", "bytes": len(decisions_raw), "sha256": _sha(decisions_raw)}, {"path": "review_completion_receipt.json", "bytes": len(receipt_raw), "sha256": _sha(receipt_raw)}]}
    (review / "index.json").write_bytes(_canonical(review_index))
    return candidate, review, review / "decisions.json", capture


def test_promotes_verified_bea_candidate_with_bounded_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, review, decisions, capture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "release"
    result = promotion.publish_release(candidate, review, decisions, capture, tmp_path / "policy", output)
    assert result["manifest"]["formal_task_release"] is True
    assert result["manifest"]["scientific_release_allowed"] is False
    assert result["manifest"]["policy_status"] == "VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY"
    assert result["manifest"]["counts"] == {"records": 3, "cores": 1, "tasks": 1, "events": 3, "decisions": 5}
    assert result["manifest"]["update_counts"] == {"ADD": 1, "UPDATE": 2, "equal_value_updates": 1}
    assert result["manifest"]["answer_schema"] == "number"
    assert result["manifest"]["runtime_state_metrics"] is None
    assert result["manifest"]["answer_metrics"] is None
    assert set(path.name for path in output.iterdir()) == set(promotion.RELEASE_NAMES)
    assert not list(output.rglob("*.xlsx"))
    assert not list(output.rglob("*.pdf"))


def test_no_replace_reader_and_deterministic_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, review, decisions, capture = _fixture(tmp_path, monkeypatch)
    one = promotion.publish_release(candidate, review, decisions, capture, tmp_path / "policy", tmp_path / "one")
    two = promotion.publish_release(candidate, review, decisions, capture, tmp_path / "policy", tmp_path / "two")
    assert one["index_sha256"] == two["index_sha256"]
    with pytest.raises(FileExistsError):
        promotion.publish_release(candidate, review, decisions, capture, tmp_path / "policy", tmp_path / "one")
    loaded = promotion.read_release(tmp_path / "one", one["index_sha256"])
    assert list(loaded) == ["manifest", "tasks", "index", "root", "index_sha256"]


def test_decisions_only_review_is_completed_without_rebinding_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, review, decisions, capture = _fixture(tmp_path, monkeypatch)
    minimal_review = tmp_path / "review-only-decisions"; minimal_review.mkdir()
    shutil.copy2(decisions, minimal_review / "decisions.json")
    output = tmp_path / "decisions-only-release"
    result = promotion.publish_release(candidate, minimal_review, decisions, capture, tmp_path / "policy", output)
    assert result["manifest"]["reviewed_audit_item_count"] == 5
    assert (output / "human_review_index.json").is_file()


def test_rejects_wrong_hashes_policy_tamper_and_scientific_rebound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, review, decisions, capture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(promotion, "CANDIDATE_INDEX_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        promotion.publish_release(candidate, review, decisions, capture, tmp_path / "policy", tmp_path / "bad")
    monkeypatch.setattr(promotion, "CANDIDATE_INDEX_SHA256", _sha((candidate / "index.json").read_bytes()))
    (tmp_path / "policy/linking.html").write_bytes((tmp_path / "policy/linking.html").read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        promotion.publish_release(candidate, review, decisions, capture, tmp_path / "policy", tmp_path / "bad2")
    (tmp_path / "policy/linking.html").write_bytes((capture / "policy/linking.html").read_bytes())
    clean = tmp_path / "clean"
    result = promotion.publish_release(candidate, review, decisions, capture, tmp_path / "policy", clean)
    manifest = json.loads((clean / "release_manifest.json").read_bytes())
    manifest["policy_evidence"]["exact_boundary"] = "false policy claim"
    manifest_raw = _canonical(manifest); (clean / "release_manifest.json").write_bytes(manifest_raw)
    index = json.loads((clean / "release_index.json").read_bytes())
    next(row for row in index["artifacts"] if row["path"] == "release_manifest.json").update(bytes=len(manifest_raw), sha256=_sha(manifest_raw))
    index_raw = _canonical(index); (clean / "release_index.json").write_bytes(index_raw)
    with pytest.raises(ValueError):
        promotion.read_release(clean, _sha(index_raw))


def test_exact_production_hash_pins_are_frozen() -> None:
    assert promotion.CANDIDATE_INDEX_SHA256 == EXPECTED_CANDIDATE_INDEX
    assert promotion.CAPTURE_MANIFEST_SHA256 == EXPECTED_CAPTURE_MANIFEST
    assert promotion.CAPTURE_INDEX_SHA256 == EXPECTED_CAPTURE_INDEX
    assert promotion.POLICY_SHA256 == EXPECTED_POLICY
    assert promotion.DECISIONS_SHA256 == "43c8e8ad78aa427ff16b479b5eb63e9c79e2fa27bc7843d5f7eae260dce28b94"
