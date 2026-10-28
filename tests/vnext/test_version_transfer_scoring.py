import json

import pytest

from mub.vnext.runtime.version_transfer_readout import build_readout, parse_readout_response
from mub.vnext.runtime.version_transfer_scoring import build_readout_run, build_scoring_artifacts
from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3
from scripts.vnext_prepare_version_transfer_decision import build_tasks
from tests.vnext.test_version_transfer_readout import native_result


def rows(task, order='native'):
    requests, predictions = [], []
    for query, gold in zip(task.queries, task.gold_evidence):
        request = build_readout(query, native_result(), namespace='owned', order=order)
        response = {'model':'fixture-model', 'choices':[{'message':{'content':json.dumps({
            'disposition':'answered','answer':gold.answer})},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':12,'total_tokens':112}}
        requests.append(request)
        predictions.append(parse_readout_response(request, response, model_id='fixture-model'))
    return requests, predictions


def test_canonical_readout_scoring_uses_observed_answers_without_fabricated_state():
    task = build_tasks(code_revision='fixture-revision')[0]
    requests, predictions = rows(task)
    run = build_readout_run(task, requests, predictions, run_id='fixture-readout-native', order='native')
    bundle = build_scoring_artifacts((task,), (run,), code_revision='fixture-revision',
                                     qualification_sha256='a'*64, historical_supported=True)
    score = bundle['scores'][0]
    assert score.answer_scores.exact_match == 1.0
    assert score.historical_scores.current_state_accuracy == 1.0
    assert score.historical_scores.point_in_time_accuracy == 1.0
    assert score.state_scores.final_state_accuracy is None
    assert not run.memory_snapshots and not run.parsed_actions
    assert bundle['run_manifest'].adapter_capabilities.exports_object_keys is False
    assert bundle['run_manifest'].adapter_capabilities.supports_native_answer is False
    assert bundle['run_manifest'].native_vs_extracted_field_summary['state_alignment'] == 'unavailable'


def test_query_membership_and_order_are_required_not_inferred_from_gold():
    task = build_tasks(code_revision='fixture-revision')[0]
    requests, predictions = rows(task)
    with pytest.raises(ValueError, match='query'):
        build_readout_run(task, requests[:1], predictions[:1], run_id='incomplete', order='native')
    with pytest.raises(ValueError, match='order'):
        build_readout_run(task, requests, predictions, run_id='wrong-order', order='reverse')
    with pytest.raises(ValueError, match='query'):
        build_readout_run(task, requests, predictions[::-1], run_id='misbound', order='native')


def test_runtime_failure_is_null_and_not_answer_accuracy_zero():
    task = build_tasks(code_revision='fixture-revision')[0]
    requests, predictions = rows(task)
    good = build_readout_run(task, requests, predictions, run_id='failed-run', order='native')
    failed = good.model_copy(update={'completion_status':'failed', 'exceptions':(
        {'phase':'readout','type':'TimeoutError','message':'test-only technical failure'},)})
    bundle = build_scoring_artifacts((task,), (failed,), code_revision='fixture-revision',
                                     qualification_sha256='a'*64, historical_supported=True)
    assert bundle['scores'][0].answer_scores.exact_match is None


def test_scoring_rejects_modified_record_membership():
    task = build_tasks(code_revision='fixture-revision')[0]
    requests, predictions = rows(task)
    run = build_readout_run(task, requests, predictions, run_id='fixture-native', order='native')
    bundle = build_scoring_artifacts((task,), (run,), code_revision='fixture-revision',
                                     qualification_sha256='a'*64, historical_supported=True)
    with pytest.raises(ValueError, match='membership'):
        VerifiedScoringContextV3.from_authenticated_manifests(
            task=task, run=run.model_copy(update={'answer_predictions':()}),
            task_manifest=bundle['task_manifest'], run_manifest=bundle['run_manifest'],
            task_artifact=bundle['task_artifact'], run_artifact=bundle['run_artifact'],
            authenticated_task_manifest_sha256=bundle['task_manifest_sha256'],
            authenticated_run_manifest_sha256=bundle['run_manifest_sha256'])
