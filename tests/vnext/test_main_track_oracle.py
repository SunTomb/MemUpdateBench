from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from mub.vnext.contracts.enums import AnswerDisposition, Operation
from scripts.vnext_run_main_track_oracle import (
    build_oracle_diagnostic,
    execute_reference_oracle,
    load_candidate,
    publish_oracle_diagnostic,
    verify_task_record,
)


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = ROOT / "data" / "vnext" / "main_track_v1_independence_v1"


@pytest.fixture(scope="module")
def candidate():
    return load_candidate(CANDIDATE_ROOT)


@pytest.fixture(scope="module")
def oracle(candidate):
    records = execute_reference_oracle(candidate.test_tasks)
    return candidate, records


def test_selects_exactly_720_test_tasks_and_axis_counts(candidate):
    assert len(candidate.test_tasks) == 720
    assert candidate.family_counts == {
        "entity_attribute_grounding": 240,
        "interleaved_multi_slot_update": 240,
        "noop_write_discipline": 240,
    }
    assert candidate.domain_counts == {
        "civic": 60,
        "community": 40,
        "education": 64,
        "finance": 96,
        "health": 56,
        "household": 56,
        "media": 72,
        "personal": 68,
        "services": 48,
        "software": 64,
        "travel": 32,
        "work": 64,
    }
    assert candidate.attribute_counts == {
        "company": 56,
        "contact_method": 52,
        "hobby": 64,
        "instrument": 64,
        "language": 32,
        "location": 88,
        "priority": 64,
        "project": 72,
        "preference": 52,
        "role": 56,
        "status": 52,
        "timezone": 68,
    }
    assert candidate.language_counts == {"en": 360, "es": 180, "ja": 180}


def test_typed_c_abstentions_and_answers_are_verified(oracle):
    candidate, records = oracle
    by_task = {record.task_id: record for record in records}
    c_tasks = [task for task in candidate.test_tasks if task.task_family == "entity_attribute_grounding"]
    assert any(item.gold_evidence[0].disposition is AnswerDisposition.ABSTAINED for item in c_tasks)
    assert any(item.gold_evidence[0].disposition is AnswerDisposition.ANSWERED for item in c_tasks)
    for task in c_tasks:
        assert verify_task_record(task, by_task[task.task_id]) == ()

    abstained = next(item for item in c_tasks if item.gold_evidence[0].disposition is AnswerDisposition.ABSTAINED)
    prediction = by_task[abstained.task_id].answer_predictions[0]
    assert prediction.disposition is AnswerDisposition.ABSTAINED
    assert prediction.parsed_answer is None

    answered = next(item for item in c_tasks if item.gold_evidence[0].disposition is AnswerDisposition.ANSWERED)
    prediction = by_task[answered.task_id].answer_predictions[0]
    assert prediction.disposition is AnswerDisposition.ANSWERED
    assert prediction.parsed_answer == answered.gold_evidence[0].answer


def test_d_noop_events_replay_without_mutation_errors(oracle):
    candidate, records = oracle
    by_task = {record.task_id: record for record in records}
    task = next(item for item in candidate.test_tasks if item.task_family == "noop_write_discipline")
    record = by_task[task.task_id]
    assert verify_task_record(task, record) == ()
    assert any(action.operation is Operation.NOOP for action in record.parsed_actions)
    assert all(action.operation is not Operation.UPDATE for action in record.parsed_actions[1:])


def test_diagnostic_has_exact_counts_and_dataset_boundary(oracle):
    candidate, records = oracle
    diagnostic = build_oracle_diagnostic(candidate, records)
    assert diagnostic["test_task_count"] == 720
    assert diagnostic["pass_count"] == 720
    assert diagnostic["fail_count"] == 0
    assert diagnostic["evidence_class"] == "dataset_oracle_diagnostic"
    assert diagnostic["review_status"] == "NOT_STARTED"
    assert diagnostic["candidate_artifact_hashes"]
    assert "model" not in diagnostic["evidence_class"]
    assert "external" not in diagnostic["evidence_class"]


def test_publication_is_no_replace_and_rejects_candidate_tamper(oracle, tmp_path):
    candidate, records = oracle
    diagnostic = build_oracle_diagnostic(candidate, records)
    output = tmp_path / "main_track_v1_oracle_test"
    publish_oracle_diagnostic(candidate, diagnostic, output)
    with pytest.raises(FileExistsError):
        publish_oracle_diagnostic(candidate, diagnostic, output)

    tampered_root = tmp_path / "candidate_tampered"
    shutil.copytree(candidate.root, tampered_root)
    tampered_candidate = replace(candidate, root=tampered_root)
    tasks = tampered_root / "tasks.jsonl"
    tasks.write_bytes(tasks.read_bytes() + b"x")
    tampered_output = tmp_path / "main_track_v1_oracle_tampered"

    with pytest.raises(ValueError, match="candidate|changed|hash"):
        publish_oracle_diagnostic(tampered_candidate, diagnostic, tampered_output)
    assert not tampered_output.exists()
