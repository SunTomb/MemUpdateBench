"""Prepare a bounded, review-pending BEA GDP revision candidate."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import AnswerSchema, Difficulty, EvaluationMode, EventRole, Operation, QueryType, SourceType, Split, TaskFamily
from mub.vnext.contracts.v3.adapter import ResetRequestV3
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, object_identity, typed_json_equal
from mub.vnext.contracts.v3.task import CurrentSelector, DerivationStepV3, GeneratorProvenanceV3, GoldActionV3, MemoryEventV3, MemoryQueryV3, MemUpdateTaskV3, QueryGoldEvidenceV3, VersionHistoryEntry, VersionHistoryLedger
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import action_id, core_id, event_id, query_id, stable_id, task_id
from mub.vnext.io import semantic_task_hash_v3
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3
from scripts.vnext_capture_family_h_bea_candidate import CAPTURE_SCHEMA, INDEX_SCHEMA, POLICY_URL, RELEASES, SOURCE_DOCUMENT_ID, SOURCE_GROUP_ID, canonical, parse_bea_xlsx, sha

COMPILER = "family-h-cross-domain-bea-candidate-v1"
KEY = FrozenMemoryObjectKey(object_type="macroeconomic_estimate", namespace="family_h", entity="us-gdp-2024q4", attribute="real_gdp_growth_annual_rate", subkey=None)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _safe_relative(root: Path, value: str) -> Path:
    parsed = PurePosixPath(value)
    require(not parsed.is_absolute() and str(parsed) == value and ".." not in parsed.parts and "\\" not in value and ":" not in value, "capture path must be relative")
    path = root.joinpath(*parsed.parts)
    require(all(not part.is_symlink() for part in (path, *path.parents)), "capture symlink rejected")
    return path


def _artifacts(root: Path, index: dict) -> None:
    expected = {item["path"] for item in index["artifacts"]} | {"capture_index.json"}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == expected, "capture artifact membership changed")
    for item in index["artifacts"]:
        path = _safe_relative(root, item["path"])
        raw = path.read_bytes()
        require(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "capture artifact hash drift")


def _policy_evidence(manifest: dict, root: Path) -> dict:
    status = manifest.get('policy_status')
    policy = manifest.get('policy', {})
    if status == 'POLICY_PENDING':
        return {'status': status, 'source_url': policy.get('url', POLICY_URL), 'source_sha256': policy.get('sha256'),
                'official_citation_requirement': None, 'quotes': {}, 'raw_source_redistribution': None,
                'logos_included': False, 'derivation_notice': None}
    require(status == 'POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY', 'unsupported BEA policy status')
    require(policy.get('url') == 'https://www.bea.gov/about/policies-and-information/linking', 'verified BEA policy URL')
    policy_path = _safe_relative(root, policy.get('path', ''))
    raw = policy_path.read_text(encoding='utf-8', errors='strict')
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    require(policy_hash == policy.get('sha256') and policy_hash == 'e49a712b7e0ec2e8946e256315f4af675af72c29c8d9e019c46626fa56b5a79e', 'verified BEA policy hash')
    parser = HTMLParser()
    paragraphs = []
    current = []
    class _Paragraphs(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == 'p': self.current = []
        def handle_data(self, data):
            if hasattr(self, 'current'): self.current.append(data)
        def handle_endtag(self, tag):
            if tag == 'p' and hasattr(self, 'current'):
                paragraphs.append(' '.join(''.join(self.current).split())); del self.current
    paragraph_parser = _Paragraphs(); paragraph_parser.feed(raw)
    reuse = next((p for p in paragraphs if 'in the public domain' in p and 'may be used or reproduced without specific permission' in p), None)
    endorsement = next((p for p in paragraphs if 'must not contain information that suggests such an endorsement' in p), None)
    external_copyright = next((p for p in paragraphs if 'cannot authorize the use of copyrighted materials' in p), None)
    require(reuse and endorsement and external_copyright, 'BEA policy paragraph evidence missing')
    citation = 'A citation such as "Source: U.S. Bureau of Economic Analysis" would be appreciated.'
    require(citation in reuse, 'BEA citation evidence missing')
    return {'status': status, 'source_url': policy['url'], 'source_sha256': policy_hash,
            'official_citation_requirement': 'appreciated, not stated as mandatory',
            'quotes': {'reuse': reuse, 'citation': reuse, 'endorsement': endorsement,
                       'external_copyright': external_copyright},
            'raw_source_redistribution': False, 'logos_included': False,
            'derivation_notice': 'Derived records are not an official BEA product'}



def load_capture(root: Path, capture_sha256: str, index_sha256: str) -> dict:
    root = Path(root).absolute()
    manifest_raw = (root / "capture_manifest.json").read_bytes()
    index_raw = (root / "capture_index.json").read_bytes()
    require(sha(manifest_raw) == capture_sha256, "capture manifest hash mismatch")
    require(sha(index_raw) == index_sha256, "capture index hash mismatch")
    manifest = json.loads(manifest_raw)
    index = json.loads(index_raw)
    require(manifest["schema"] == CAPTURE_SCHEMA and index["schema"] == INDEX_SCHEMA, "capture schema")
    require(manifest["status"] == "CAPTURED_PENDING_HUMAN_REVIEW", "source capture is not complete")
    require(manifest["source_group_id"] == SOURCE_GROUP_ID, "source group identity")
    require(manifest.get("policy_status") in ("POLICY_PENDING", "POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY"), "BEA policy status unsupported")
    _artifacts(root, index)
    records = []
    for expected_stage, expected_date, expected_xlsx, expected_page, expected_value in RELEASES:
        item = next((item for item in manifest["records"] if item["stage"] == expected_stage), None)
        require(item is not None and item["release_date"] == expected_date and item["xlsx_url"] == expected_xlsx and item["page_url"] == expected_page, "release URL/date/stage identity")
        file_item = item.get("xlsx", item)
        relative = file_item.get("path")
        require(isinstance(relative, str), "xlsx capture path missing")
        raw = _safe_relative(root, relative)
        file_hash = file_item.get("sha256", item.get("sha256"))
        require(sha(raw.read_bytes()) == file_hash == item["normalized"]["source_file_sha256"], "BEA XLSX hash drift")
        parsed = parse_bea_xlsx(raw.read_bytes(), expected_stage, expected_date, expected_xlsx, expected_page)
        supplied = item["normalized"]
        for field in ("release_date", "estimate_stage", "reference_period", "unit", "rate_basis", "table_label", "value", "xlsx_url", "page_url", "source_file_sha256"):
            if field in supplied:
                require(parsed[field] == supplied[field], f"normalized BEA field drift: {field}")
        require(parsed["value"] == expected_value, "pinned release value drift")
        records.append(parsed)
    policy = manifest.get("policy", {})
    policy_path = policy.get("path", "raw/policy.html")
    policy_file = _safe_relative(root, policy_path)
    if policy.get("sha256"):
        require(sha(policy_file.read_bytes()) == policy["sha256"], "policy capture hash drift")
    return {"manifest": manifest, "index": index, "records": records, "policy_status": manifest.get("policy_status"),
            "policy_evidence": _policy_evidence(manifest, root)}


def _guard_output(output_root: Path, capture_root: Path) -> Path:
    output_root = Path(output_root).absolute()
    capture_root = Path(capture_root).absolute()
    require(not output_root.exists(), "output root already exists; no replacement")
    require(capture_root not in output_root.parents and output_root not in capture_root.parents, "output overlaps capture root")
    for parent in output_root.parents:
        if any((parent / name).is_file() for name in ("index.json", "manifest.json", "candidate_manifest.json")):
            raise ValueError("output is nested under an existing candidate")
    return output_root


def _make_task(records: list[dict], capture_sha256: str, code_revision: str, policy_status: str, policy_url: str, policy_evidence: dict):
    require(len(records) == 3, "BEA candidate requires exactly three release vintages")
    require(len({object_identity(KEY)}) == 1, "object identity")
    payload = {"source_group_id": SOURCE_GROUP_ID, "records": [digest(record) for record in records], "identity": list(object_identity(KEY))}
    cid = core_id(TaskFamily.REALISTIC_SOURCE_UPDATE.value, payload)
    tid = task_id(cid, 0)
    trajectory = stable_id("trajectory", {"source_group_id": SOURCE_GROUP_ID, "reference_period": "2024-Q4"})
    events, actions, core_events = [], [], []
    for index, record in enumerate(records):
        operation = Operation.ADD if index == 0 else Operation.UPDATE
        event_identifier, action_identifier = event_id(tid, index), action_id(tid, index, 0)
        role = EventRole.LATEST_GOLD if index == len(records) - 1 else EventRole.HISTORICAL_SUPPORT
        anchor = {"release_date": record["release_date"], "estimate_stage": record["estimate_stage"], "page_url": record["page_url"], "table_url": record["table_url"], "source_file_sha256": record["source_file_sha256"], **record["source_anchor"]}
        visible = {"action": operation.value, "object_key": KEY.model_dump(mode="json"), "value": record["value"], "release_date": record["release_date"], "estimate_stage": record["estimate_stage"]}
        text = canonical(visible).decode("utf-8")
        core_events.append(CoreEvent(operation=operation, object_keys=[MemoryObjectKey(**KEY.model_dump(mode="python"))], value=record["value"], role=role, metadata={"logical_time": record["release_date"], "source_anchor": anchor}))
        events.append(MemoryEventV3(event_id=event_identifier, sequence_index=index, timestamp=record["release_date"], raw_text=text, normalized_text=text, gold_action_ids=(action_identifier,), role=role, source_anchor=anchor, metadata={"source_group_id": SOURCE_GROUP_ID, "revision_only": True}))
        actions.append(GoldActionV3(action_id=action_identifier, event_id=event_identifier, operation=operation, scope="object", target_object_keys=(KEY,), value=record["value"], effective_at=record["release_date"], expected_effect={"canonical_object_id": KEY.canonical_id}))
    core = SemanticCore(core_id=cid, task_family=TaskFamily.REALISTIC_SOURCE_UPDATE, difficulty=Difficulty.MEDIUM, core_index=0, trajectory_id=trajectory, events=core_events, query_targets=[MemoryObjectKey(**KEY.model_dump(mode="python"))], query_type=QueryType.CURRENT_STATE, query_selector=CurrentSelector(), expected_answer=records[-1]["value"], profile={"source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "snapshot_count": 3, "update_depth": 2, "revision_only": True, "independent_samples": False, "source_group_id": SOURCE_GROUP_ID, "candidate_only": True}, stratification={"source_kind": "sequential_release_vintage", "domain": "macroeconomics"})
    ledgers = VersionHistoryLedger(object_key=KEY, entries=tuple(VersionHistoryEntry(version_index=index, status="present", value=action.value, valid_from_event_id=action.event_id, valid_until_event_id=actions[index + 1].event_id if index + 1 < len(actions) else None, logical_time=action.effective_at, source_event_ids=(action.event_id,)) for index, action in enumerate(actions)))
    qid = query_id(tid, 0)
    query = MemoryQueryV3(query_id=qid, query_type="current", text=f"Within the supplied BEA release vintages for 2024-Q4, return the current value for object {KEY.canonical_id}.", selector=CurrentSelector(), target_object_keys=(KEY,), answer_schema=AnswerSchema.NUMBER, evaluation_mode=EvaluationMode.RETRIEVED_PROMPT)
    step = DerivationStepV3(step_id=f"derive_{qid}_read", operation="read_current", supporting_object_keys=(KEY,), supporting_event_ids=(actions[-1].event_id,))
    evidence = QueryGoldEvidenceV3(query_id=qid, answer=records[-1]["value"], supporting_object_keys=(KEY,), supporting_event_ids=(actions[-1].event_id,), derivation_steps=(step,), final_derivation_step_id=step.step_id)
    source = {"source_id": "bea-gdp-2024q4", "source_type": SourceType.OTHER, "source_uri": records[-1]["page_url"], "license_or_privacy": policy_status, "raw_hash": digest([record["source_file_sha256"] for record in records]), "normalized_hash": digest(records), "normalization_version": COMPILER, "provenance": {"source_group_id": SOURCE_GROUP_ID, "source_document_id": SOURCE_DOCUMENT_ID, "source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "language": "en", "revision_only": True, "independent_samples": False, "release_records": records, "policy_url": policy_url, "policy_status": policy_status, "policy_evidence": policy_evidence, "source_audit_status": "NOT_STARTED", "generated_surface_audit_status": "NOT_STARTED"}, "generator": GeneratorProvenanceV3(generator_name="family_h_cross_domain_bea_candidate", seed=0, config_sha256=capture_sha256, code_revision=code_revision, compiler_version=COMPILER)}
    metadata = {"split": Split.EVALUATION_ONLY, "split_key": {"semantic_core_id": cid, "source_group_id": SOURCE_GROUP_ID, "trajectory_id": trajectory, "version_group_id": stable_id("version_group", payload), "split_policy_version": COMPILER}, "profile_name": Difficulty.MEDIUM, "resolved_profile": {"source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "snapshot_count": 3, "update_depth": 2, "revision_only": True, "independent_samples": False, "candidate_only": True}, "generation_config_hash": capture_sha256, "compiler_version": COMPILER, "tags": ("family_h", "cross_domain", "bea", "macroeconomics", "review_pending"), "extra": {"policy_status": policy_status, "policy_evidence": policy_evidence, "formal_task_release": False, "scientific_release_allowed": False, "answer_metrics": None, "natural_language_discovery": "NOT_CLAIMED", "model_backend": 0}}
    return core, MemUpdateTaskV3(task_id=tid, task_family=TaskFamily.REALISTIC_SOURCE_UPDATE.value, difficulty=Difficulty.MEDIUM, source=source, events=tuple(events), target_objects=(KEY,), actions=tuple(actions), queries=(query,), version_history=(ledgers,), gold_evidence=(evidence,), metadata=metadata)


def build(output_root: Path, capture_root: Path, capture_sha256: str, capture_index_sha256: str) -> dict:
    output_root = _guard_output(output_root, capture_root)
    loaded = load_capture(capture_root, capture_sha256, capture_index_sha256)
    code_revision = sha(Path(__file__).read_bytes())
    core, task = _make_task(loaded["records"], capture_sha256, code_revision, loaded["policy_status"], loaded["manifest"].get("policy_url", POLICY_URL), loaded["policy_evidence"])
    replay = replay_task_v3(task)
    evaluated = evaluate_evidence_v3(task.gold_evidence[0], replay, query=task.queries[0], events=task.events)
    require(replay.valid and evaluated.valid and typed_json_equal(evaluated.answer, loaded["records"][-1]["value"]), "reference replay/evidence sanity failed")
    adapter = ReferenceAdapterV3(task)
    require(adapter.reset(ResetRequestV3(namespace="family-h-bea-candidate")).success, "reference adapter reset failed")
    for event in task.events:
        adapter.ingest_event(event)
    prediction = adapter.answer(task.queries[0], "slot_direct").prediction
    adapter.close()
    require(prediction.format_valid and typed_json_equal(prediction.parsed_answer, loaded["records"][-1]["value"]), "reference adapter answer sanity failed")
    snapshot_items = []
    for record in loaded["records"]:
        material = {field: record[field] for field in ("record_id", "release_date", "estimate_stage", "reference_period", "unit", "rate_basis", "table_label", "value", "xlsx_url", "page_url", "table_url", "source_file_sha256", "source_anchor")}
        snapshot_items.append({"audit_id": "snapshot-" + record["estimate_stage"], "kind": "source_snapshot", "binding_sha256": digest(material), "material": material})
    policy_status = loaded["policy_status"]
    policy_evidence = loaded["policy_evidence"]
    policy_url = loaded["manifest"].get("policy_url", POLICY_URL)
    source_material = {"source_group_id": SOURCE_GROUP_ID, "capture_manifest_sha256": capture_sha256, "capture_index_sha256": capture_index_sha256, "policy_url": policy_url, "policy_status": policy_status, "policy_evidence": policy_evidence, "source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "release_count": 3}
    source_item = {"audit_id": "source-bea-gdp-2024q4", "kind": "source_admission", "binding_sha256": digest(source_material), "material": source_material}
    surface_material = {"task_id": task.task_id, "canonical_task_sha256": digest(task.model_dump(mode="json")), "semantic_task_sha256": semantic_task_hash_v3(task), "event_count": 3, "query": task.queries[0].text}
    surface_item = {"audit_id": "surface-bea-gdp-2024q4", "kind": "generated_surface", "binding_sha256": digest(surface_material), "material": surface_material}
    items = [source_item, *snapshot_items, surface_item]
    manifest = {"schema": COMPILER + "-manifest-v1", "status": "CANDIDATE_PENDING_HUMAN_REVIEW", "source_group_id": SOURCE_GROUP_ID, "source_document_id": SOURCE_DOCUMENT_ID, "source_kind": "sequential_release_vintage", "source_type": "other", "domain": "macroeconomics", "language": "en", "revision_only": True, "independent_samples": False, "semantic_core_count": 1, "task_count": 1, "snapshot_count": 3, "event_count": 3, "update_events": 2, "value_changing_updates": 1, "equal_value_update_events": 1, "policy_url": policy_url, "policy_status": policy_status, "policy_evidence": policy_evidence, "source_audit_status": "NOT_STARTED", "generated_surface_audit_status": "NOT_STARTED", "human_approved_sources": 0, "formal_task_release": False, "scientific_release_allowed": False, "answer_metrics": None, "model_loads": 0, "generations": 0, "provider_calls": 0, "capture_manifest_sha256": capture_sha256, "capture_index_sha256": capture_index_sha256, "reference_sanity": {"status": "PASS", "events": 3, "model_loads": 0, "generations": 0, "provider_calls": 0}}
    attribution = ("Source: U.S. Bureau of Economic Analysis.\n"
                   "This candidate contains normalized facts from dated BEA releases; it is not an official BEA product.\n"
                   "Use is subject to the captured BEA linking policy and any third-party material terms.\n")
    files = {"SOURCE_ATTRIBUTION.txt": attribution.encode(), "manifest.json": canonical(manifest), "normalized_records.jsonl": b"".join(canonical(record) for record in loaded["records"]), "semantic_cores.jsonl": canonical(core.model_dump(mode="json")), "tasks.jsonl": canonical(task.model_dump(mode="json")), "audit_manifest.json": canonical({"schema": COMPILER + "-audit-v1", "status": "NOT_STARTED", "items": items, "item_count": 5}), "decisions_template.json": canonical({"schema": COMPILER + "-human-decisions-v1", "status": "NOT_STARTED", "reviewer": None, "candidate_index_sha256": None, "decisions": [{"audit_id": item["audit_id"], "binding_sha256": item["binding_sha256"], "decision": None, "rationale": ""} for item in items]}), "reference_sanity.json": canonical(manifest["reference_sanity"])}
    index = {"schema": COMPILER + "-index-v1", "status": manifest["status"], "scientific_release_allowed": False, "artifacts": [{"path": name, "bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(files.items())]}
    files["index.json"] = canonical(index)
    output_root.mkdir(parents=True)
    for name, raw in files.items():
        (output_root / name).write_bytes(raw)
    return {"status": manifest["status"], "source_group_id": SOURCE_GROUP_ID, "snapshots": 3, "events": 3, "audit_items": 5, "index_sha256": sha(files["index.json"]), "capture_manifest_sha256": capture_sha256, "capture_index_sha256": capture_index_sha256}


def validate_candidate(root: Path, expected_index_sha256: str) -> dict:
    root = Path(root).absolute()
    index_raw = (root / "index.json").read_bytes()
    require(sha(index_raw) == expected_index_sha256, "candidate index hash mismatch")
    index = json.loads(index_raw)
    require(index["status"] == "CANDIDATE_PENDING_HUMAN_REVIEW" and index["scientific_release_allowed"] is False, "candidate boundary")
    for item in index["artifacts"]:
        raw = (root / item["path"]).read_bytes()
        require(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "candidate artifact hash drift")
    task = MemUpdateTaskV3.model_validate(json.loads((root / "tasks.jsonl").read_bytes()))
    require(replay_task_v3(task).valid, "candidate task replay failed")
    audit = json.loads((root / "audit_manifest.json").read_bytes())
    require(audit["item_count"] == 5 and audit["status"] == "NOT_STARTED", "audit boundary")
    surface = next(item for item in audit["items"] if item["kind"] == "generated_surface")
    require(surface["material"]["canonical_task_sha256"] == digest(task.model_dump(mode="json")), "generated surface binding drift")
    require(surface["material"]["semantic_task_sha256"] == semantic_task_hash_v3(task), "semantic surface binding drift")
    require(all(digest(item["material"]) == item["binding_sha256"] for item in audit["items"]), "audit binding drift")
    return {**index, "status": "VALID_PENDING_HUMAN_REVIEW", "snapshots": 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--capture-index-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output_root, args.capture_root, args.capture_sha256, args.capture_index_sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
