from __future__ import annotations

import copy
import json

import pytest

from scripts.vnext_native_canary_cases import audit_cases, run_cases
from scripts.vnext_prepare_native_multi_object_canary import build_payloads
from tests.vnext.test_qdrant_native_multi_object import FakeQdrantClient, _matches


def prepared():
    payloads = build_payloads()
    requests = [json.loads(line) for line in payloads["requests.jsonl"].decode().splitlines()]
    expectations = [json.loads(line) for line in payloads["expectations.jsonl"].decode().splitlines()]
    return requests, expectations


def make_factory(tmp_path, failures=()):
    made = []

    def factory(case):
        if case["case_id"] in failures:
            raise RuntimeError("factory failure")
        from mub.vnext.external.providers.qdrant_native_multi_object import (
            QdrantNativeMultiObjectProvider,
            build_qdrant_native_multi_object_configuration,
        )

        client = FakeQdrantClient()
        original_query_points = client.query_points
        def query_points_with_filter(**kwargs):
            points = [
                point for point in client.points.values()
                if _matches(point.payload or {}, kwargs.get("query_filter"))
            ]
            client.queries.append((kwargs["collection_name"], kwargs["query"], kwargs["limit"], kwargs.get("query_filter"), True, False))
            return type("QueryResult", (), {
                "points": [type("Point", (), {"id": point.id, "score": 1.0 - i * 0.1, "payload": point.payload})()
                           for i, point in enumerate(points[:kwargs["limit"]])]
            })()
        client.query_points = query_points_with_filter
        provider = QdrantNativeMultiObjectProvider(
            configuration=build_qdrant_native_multi_object_configuration(
                run_id=f"canary-{case['case_id']}-{case['repetition']}",
                path=tmp_path / f"qdrant-{case['case_id']}-{case['repetition']}",
            ),
            client=client,
            runtime_namespace=case["runtime_namespace"],
        )
        made.append((provider, client))

        def cleanup():
            client.delete_collection(collection_name=provider.collection_name)
            provider.close()
            return {"client_closed": client.closed, "collection_deleted": True}

        return provider, cleanup

    return factory, made


def test_run_cases_observes_all_public_cases_and_counts_only_planned_mutations(tmp_path):
    requests, _ = prepared()
    factory, made = make_factory(tmp_path)
    result = run_cases(requests, factory)

    assert result["status"] == "OBSERVATIONS_COMPLETE"
    assert result["counts"]["planned_mutation_operations"] == 50
    assert result["counts"]["mutation_attempts"] == 50
    assert result["counts"]["mutation_completions"] == 50
    assert len(result["cases"]) == 8
    assert all(row["status"] == "OBSERVED" for row in result["cases"])
    assert all(row["cleanup"] == {"client_closed": True, "collection_deleted": True} for row in result["cases"])
    assert all(client.closed for _, client in made)


def test_audit_cases_accepts_run_observations_and_declares_non_scientific_scope(tmp_path):
    requests, expectations = prepared()
    factory, _ = make_factory(tmp_path)
    observations = run_cases(requests, factory)

    audited = audit_cases(requests, expectations, observations)

    assert audited == {
        "scope": "synthetic_api_canary_only",
        "native_admission": "NOT_GRANTED",
        "answer_metrics": None,
        "scientific_release_allowed": False,
        "status": "AUDITED",
        "case_count": 8,
    }


def test_audit_cases_rejects_corrupted_snapshot(tmp_path):
    requests, expectations = prepared()
    factory, _ = make_factory(tmp_path)
    observations = run_cases(requests, factory)
    corrupted = copy.deepcopy(observations)
    corrupted["cases"][0]["mutation_events"][0]["after_snapshot"][0]["payload"]["value"] = "corrupt"

    with pytest.raises(ValueError, match="snapshot"):
        audit_cases(requests, expectations, corrupted)


def test_run_cases_marks_later_cases_not_run_after_factory_failure(tmp_path):
    requests, _ = prepared()
    factory, _ = make_factory(tmp_path, failures={requests[2]["case_id"]})

    result = run_cases(requests, factory)

    assert result["status"] == "BLOCKED"
    assert result["cases"][2]["status"] == "BLOCKED"
    assert all(row["status"] == "NOT_RUN" for row in result["cases"][3:])


def test_run_cases_counts_all_subject_resets_and_keeps_raw_query_witness_separate(tmp_path):
    requests, _ = prepared()
    factory, _ = make_factory(tmp_path)
    result = run_cases(requests, factory)
    assert result['counts']['reset_attempts'] == 48
    first = result['cases'][0]
    assert first['provider_query_points']
    first['provider_query_points'][0]['score'] = 999.0
    with pytest.raises(ValueError, match='retrieval score|retrieval.*mismatch'):
        from scripts.vnext_native_canary_cases import audit_cases
        audit_cases(requests, prepared()[1], result)


def test_audit_requires_independent_raw_query_points(tmp_path):
    requests, expectations = prepared()
    factory, _ = make_factory(tmp_path)
    result = run_cases(requests, factory)
    result['cases'][0]['provider_query_points'] = []
    with pytest.raises(ValueError, match='raw query response|trace'):
        from scripts.vnext_native_canary_cases import audit_cases
        audit_cases(requests, expectations, result)
