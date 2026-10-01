"""TDD coverage for the bounded BEA Family H cross-domain candidate."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path

import openpyxl
import pytest

from scripts import vnext_capture_family_h_bea_candidate as capture
from scripts import vnext_prepare_family_h_bea_candidate as prepare
from mub.vnext.validation.replay_v3 import replay_task_v3

PROJECT = Path(__file__).resolve().parents[2]


def _xlsx(*, sheet="Table 1", row_label="Real GDP", quarter="Q4 2024",
          title="Percent change from Q3 to Q4 in real GDP",
          unit="Percent change from Q3 to Q4", rate="Seasonally adjusted at annual rates",
          value=2.3, duplicate=False):
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = sheet
    worksheet.append([title])
    worksheet.append([rate])
    worksheet.append([unit])
    worksheet.append(["Line", quarter])
    worksheet.append([row_label, value])
    if duplicate:
        worksheet.append([row_label, value])
    stream = io.BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def _record(stage, date, value, xlsx_url=None, page_url=None):
    return {
        "stage": stage,
        "release_date": date,
        "reference_period": "2024-Q4",
        "unit": "percent",
        "rate_basis": "annualized",
        "table_label": "Real GDP",
        "value": value,
        "xlsx_url": xlsx_url or f"https://www.bea.gov/sites/default/files/{date[:7]}/gdp4q24-{stage}.xlsx",
        "page_url": page_url or f"https://www.bea.gov/news/{date[:4]}/gdp-{stage}",
        "worksheet": "Table 1",
        "row": 5,
        "column": 2,
        "source_file_sha256": hashlib.sha256(_xlsx(value=value)).hexdigest(),
    }


def _capture_tree(tmp_path, *, records=None):
    records = records or [
        _record("advance", "2025-01-30", 2.3,
                "https://www.bea.gov/sites/default/files/2025-01/gdp4q24-adv.xlsx",
                "https://www.bea.gov/news/2025/gross-domestic-product-fourth-quarter-and-year-2024-advance-estimate"),
        _record("second", "2025-02-27", 2.3,
                "https://www.bea.gov/sites/default/files/2025-02/gdp4q24-2nd.xlsx",
                "https://www.bea.gov/news/2025/gross-domestic-product-4th-quarter-and-year-2024-second-estimate"),
        _record("third", "2025-03-27", 2.4,
                "https://www.bea.gov/sites/default/files/2025-03/gdp4q24-3rd.xlsx",
                capture.RELEASES[2][3]),
    ]
    root = tmp_path / "capture"
    raw = []
    for record in records:
        workbook = _xlsx(value=record["value"])
        name = f"{record['stage']}.xlsx"
        path = root / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(workbook)
        raw.append({"stage": record["stage"], "release_date": record["release_date"], "xlsx_url": record["xlsx_url"],
                    "page_url": record["page_url"], "path": f"raw/{name}", "bytes": len(workbook),
                    "sha256": hashlib.sha256(workbook).hexdigest(), "normalized": {**record, "source_file_sha256": hashlib.sha256(workbook).hexdigest(),
                    "worksheet": "Table 1", "row": 5, "column": 2}})
    policy = b"BEA policies fixture; reuse status unresolved."
    (root / "raw/policy.html").write_bytes(policy)
    manifest = {
        "schema": capture.CAPTURE_SCHEMA,
        "status": "CAPTURED_PENDING_HUMAN_REVIEW",
        "source_group_id": capture.SOURCE_GROUP_ID,
        "records": raw,
        "policy": {"url": capture.POLICY_URL, "path": "raw/policy.html", "bytes": len(policy), "sha256": hashlib.sha256(policy).hexdigest()},
        "policy_status": "POLICY_PENDING",
        "schedule_url": capture.SCHEDULE_URL,
        "formal_task_release": False,
        "scientific_release_allowed": False,
    }
    (root / "capture_manifest.json").write_bytes(capture.canonical(manifest))
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"capture_index.json"}:
            artifacts.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    index = {"schema": capture.INDEX_SCHEMA, "status": manifest["status"], "artifacts": artifacts,
             "scientific_release_allowed": False}
    (root / "capture_index.json").write_bytes(capture.canonical(index))
    return root, hashlib.sha256((root / "capture_manifest.json").read_bytes()).hexdigest(), hashlib.sha256((root / "capture_index.json").read_bytes()).hexdigest()


def test_extracts_exact_real_gdp_q4_2024_annualized_value_and_anchors():
    row = capture.parse_bea_xlsx(_xlsx(value=2.3), "advance", "2025-01-30", capture.RELEASES[0][2])
    assert row["value"] == 2.3 and type(row["value"]) is float
    assert row["table_label"] == "Real GDP"
    assert row["reference_period"] == "2024-Q4"
    assert row["rate_basis"] == "annualized"
    assert row["worksheet"] == "Table 1" and row["row"] == 5 and row["column"] == 2


@pytest.mark.parametrize("kwargs", [
    {"sheet": "Wrong Table"}, {"row_label": "GDP"}, {"unit": "Annual GDP"},
    {"rate": "Not annualized"}, {"quarter": "Q3 2024"},
])
def test_rejects_wrong_sheet_row_unit_annualization_or_period(kwargs):
    with pytest.raises(ValueError):
        capture.parse_bea_xlsx(_xlsx(**kwargs), "advance", "2025-01-30", capture.RELEASES[0][2])


def test_rejects_duplicate_candidate_values_and_ambiguous_rows():
    with pytest.raises(ValueError, match="ambiguous"):
        capture.parse_bea_xlsx(_xlsx(duplicate=True), "advance", "2025-01-30", capture.RELEASES[0][2])


def test_rejects_invented_precision():
    with pytest.raises(ValueError, match="rounded"):
        capture.parse_bea_xlsx(_xlsx(value=2.34), "advance", "2025-01-30", capture.RELEASES[0][2])


def test_capture_manifest_binds_all_three_release_urls_dates_and_stages(tmp_path):
    root, capture_sha, index_sha = _capture_tree(tmp_path)
    loaded = prepare.load_capture(root, capture_sha, index_sha)
    assert [(r["estimate_stage"], r["release_date"], r["value"]) for r in loaded["records"]] == [
        ("advance", "2025-01-30", 2.3), ("second", "2025-02-27", 2.3), ("third", "2025-03-27", 2.4)
    ]
    assert [r["xlsx_url"] for r in loaded["records"]] == [item[2] for item in capture.RELEASES]
    assert loaded["policy_status"] == "POLICY_PENDING"


def test_capture_blocked_source_status_is_typed_and_never_fabricates(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(capture.urllib.request, "urlopen", fail)
    result = capture.capture(tmp_path / "blocked")
    assert result["status"] == "BLOCKED_SOURCE_CAPTURE"
    manifest = json.loads((tmp_path / "blocked/capture_manifest.json").read_bytes())
    assert manifest["records"] == [] and manifest["scientific_release_allowed"] is False
    assert manifest["blocker"]["kind"] == "BLOCKED_SOURCE_CAPTURE"


def test_build_emits_one_scalar_candidate_with_add_and_two_updates_equal_update(tmp_path):
    cap, cap_sha, index_sha = _capture_tree(tmp_path)
    out = tmp_path / "candidate"
    result = prepare.build(out, cap, cap_sha, index_sha)
    task = json.loads((out / "tasks.jsonl").read_bytes())
    assert [a["operation"] for a in task["actions"]] == ["ADD", "UPDATE", "UPDATE"]
    assert [a["value"] for a in task["actions"]] == [2.3, 2.3, 2.4]
    assert task["queries"][0]["answer_schema"] == "number"
    assert task["target_objects"][0] == {
        "object_type": "macroeconomic_estimate", "namespace": "family_h", "entity": "us-gdp-2024q4",
        "attribute": "real_gdp_growth_annual_rate", "subkey": None,
    }
    assert replay_task_v3(prepare.MemUpdateTaskV3.model_validate(task)).valid
    assert result["status"] == "CANDIDATE_PENDING_HUMAN_REVIEW"


def test_candidate_metadata_policy_pending_release_boundary_and_five_audit_items(tmp_path):
    cap, cap_sha, index_sha = _capture_tree(tmp_path)
    out = tmp_path / "candidate"
    prepare.build(out, cap, cap_sha, index_sha)
    manifest = json.loads((out / "manifest.json").read_bytes())
    audit = json.loads((out / "audit_manifest.json").read_bytes())
    template = json.loads((out / "decisions_template.json").read_bytes())
    assert manifest["source_kind"] == "sequential_release_vintage"
    assert manifest["source_type"] == "other"
    assert manifest["domain"] == "macroeconomics" and manifest["revision_only"] is True
    assert manifest["independent_samples"] is False and manifest["formal_task_release"] is False
    assert manifest["scientific_release_allowed"] is False and manifest["answer_metrics"] is None
    assert manifest["policy_status"] == "POLICY_PENDING"
    assert len(audit["items"]) == 5
    assert [item["kind"] for item in audit["items"]].count("source_snapshot") == 3
    assert audit["item_count"] == 5
    assert template["reviewer"] is None and all(item["decision"] is None for item in template["decisions"])


def test_replay_reference_sanity_and_surface_hash_drift_are_bound(tmp_path):
    cap, cap_sha, index_sha = _capture_tree(tmp_path)
    out = tmp_path / "candidate"
    result = prepare.build(out, cap, cap_sha, index_sha)
    sanity = json.loads((out / "reference_sanity.json").read_bytes())
    assert sanity["status"] == "PASS" and sanity["model_loads"] == sanity["generations"] == 0
    copied = tmp_path / "mutated"
    shutil.copytree(out, copied)
    task = json.loads((copied / "tasks.jsonl").read_bytes())
    task["events"][0]["raw_text"] += " drift"
    task_bytes = prepare.canonical(task)
    (copied / "tasks.jsonl").write_bytes(task_bytes)
    index = json.loads((copied / "index.json").read_bytes())
    for item in index["artifacts"]:
        if item["path"] == "tasks.jsonl":
            item.update(bytes=len(task_bytes), sha256=hashlib.sha256(task_bytes).hexdigest())
    index_bytes = prepare.canonical(index)
    (copied / "index.json").write_bytes(index_bytes)
    with pytest.raises(ValueError, match="surface|binding"):
        prepare.validate_candidate(copied, hashlib.sha256(index_bytes).hexdigest())


def test_capture_and_candidate_are_no_replace_and_hash_tamper_evident(tmp_path):
    cap, cap_sha, index_sha = _capture_tree(tmp_path)
    out = tmp_path / "candidate"
    prepare.build(out, cap, cap_sha, index_sha)
    with pytest.raises((ValueError, FileExistsError), match="exists|replace|nested"):
        prepare.build(out, cap, cap_sha, index_sha)
    changed = cap / "raw/third.xlsx"
    changed.write_bytes(changed.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        prepare.load_capture(cap, cap_sha, index_sha)


def test_no_raw_provider_api_keys_or_absolute_private_paths_in_candidate(tmp_path):
    cap, cap_sha, index_sha = _capture_tree(tmp_path)
    out = tmp_path / "candidate"
    prepare.build(out, cap, cap_sha, index_sha)
    serialized = b"".join(path.read_bytes() for path in out.rglob("*") if path.is_file())
    assert b"api_key" not in serialized.lower() and b"bearer " not in serialized.lower()
    assert b"D:/" not in serialized and b"/NAS/" not in serialized and b"C:\\" not in serialized
