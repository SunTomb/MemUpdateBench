"""Canonical v3 scoring artifacts for native evidence plus a separate local reader.

This module verifies internal record bindings, not producer authenticity. The
execution caller must authenticate native receipts, inputs and responses first;
a caller-supplied qualification hash alone is not external scientific evidence.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.manifest import TaskManifestV3, RunManifestV3
from mub.vnext.contracts.v3.runtime import ParserExtractorProvenanceV3, TaskRunRecordV3
from mub.vnext.contracts.v3.score import ScorerConfigV3
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.runtime.answer_model_v3 import ANSWER_MODEL_PARSER_VERSION_V3
from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3


ADAPTER_ID = 'graphiti_native_evidence_plus_local_readout'
METRICS = ('answer_scores.exact_match', 'answer_scores.normalized_match',
           'historical_scores.current_state_accuracy', 'historical_scores.point_in_time_accuracy',
           'system_scores.token_usage')
PROVENANCE = ParserExtractorProvenanceV3(
    action_parser_version='unobserved-native-internal-actions',
    answer_parser_version=ANSWER_MODEL_PARSER_VERSION_V3,
    memory_entry_extractor_version='native-edge-fields-no-gold-alignment-v1',
    redaction_policy_version='private-native-runtime-v1',
)


def build_readout_run(task, requests, predictions, *, run_id, order):
    expected = [query.query_id for query in task.queries]
    if ([request.query.query_id for request in requests] != expected
            or [prediction.query_id for prediction in predictions] != expected):
        raise ValueError('readout query membership/order mismatch')
    if any(request.query != query for query, request in zip(task.queries, requests)):
        raise ValueError('readout query is not bound to the original task')
    if order not in ('native', 'reverse') or any(r.retrieval_trace.context_order != order for r in requests):
        raise ValueError('readout presentation order mismatch')
    return TaskRunRecordV3(
        task_id=task.task_id, adapter_id=ADAPTER_ID, run_id=run_id,
        retrieval_traces=tuple(r.retrieval_trace for r in requests),
        answer_predictions=tuple(predictions), parser_extractor_provenance=PROVENANCE,
        system_events=({'event':'evidence_class','value':'native_evidence_local_answer',
                        'state_alignment':'unavailable','display_order':order},),
        completion_status='completed',
    )


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _ref(path, content, records=None):
    return ArtifactRef(path=path, sha256=hashlib.sha256(content).hexdigest(),
                       media_type='application/x-ndjson' if records is not None else 'application/json',
                       record_count=records)


def build_scoring_artifacts(tasks, runs, *, code_revision, qualification_sha256, historical_supported):
    tasks, runs = tuple(tasks), tuple(runs)
    if not tasks or len(tasks) != len(runs) or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError('task/run membership must be nonempty, unique and complete')
    if [task.task_id for task in tasks] != [run.task_id for run in runs]:
        raise ValueError('task/run membership order mismatch')
    if len({run.run_id for run in runs}) != 1 or any(run.adapter_id != ADAPTER_ID for run in runs):
        raise ValueError('run identity must be one native evidence/readout batch')
    if type(historical_supported) is not bool:
        raise ValueError('history support must be explicitly qualified')
    timestamp = datetime.now(timezone.utc).isoformat()
    task_bytes = b''.join(canonical_json_bytes(t) + b'\n' for t in tasks)
    run_bytes = b''.join(canonical_json_bytes(r) + b'\n' for r in runs)
    task_ref, run_ref = _ref('tasks.jsonl', task_bytes, len(tasks)), _ref('runs.jsonl', run_bytes, len(runs))
    capability_ref = ArtifactRef(path='native-qualification.json',sha256=qualification_sha256,
                                  media_type='application/json')
    config_bytes = _json_bytes({'evidence_class':'native_evidence_local_answer',
                               'state_alignment':'unavailable','native_qualification_sha256':qualification_sha256,
                               'historical_supported':historical_supported})
    info = AdapterInfoV3(adapter_id=ADAPTER_ID, adapter_version='v1',
                         system_name='graphiti_native_plus_local_readout', system_version='v1',
                         configuration_hash=hashlib.sha256(config_bytes).hexdigest())
    # The combined observation has external answers, not a native Graphiti answer API.
    caps = AdapterCapabilitiesV3(exports_entries=True, exports_retrieval_ids=True,
        exports_retrieval_scores=True, exports_timestamps_or_order=True, reports_token_usage=True,
        supports_historical_query=historical_supported)
    config = ScorerConfigV3(requested_metric_fields=METRICS)
    task_manifest = TaskManifestV3(
        data_release_id='version-transfer-diagnostic-slice', split_policy_version='vtd-v1',
        compiler_versions={'version-transfer':'vtd-v1'}, source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(), split_counts=dict(Counter(t.metadata.split.value for t in tasks)),
        family_difficulty_counts=dict(Counter(f'{t.task_family}.{t.difficulty.value}' for t in tasks)),
        semantic_core_counts=dict(Counter(t.metadata.split_key.semantic_core_id for t in tasks)),
        task_file_paths_and_hashes=(task_ref,), task_record_hashes={t.task_id:sha256_model(t) for t in tasks},
        leakage_check_summary={'scope':'declared-diagnostic-slice-no-release-promotion'},
        human_audit_artifacts=(), created_at=timestamp, code_revision=code_revision,
    )
    task_manifest_hash = sha256_model(task_manifest)
    task_manifest_ref = ArtifactRef(path='task-manifest.json',sha256=task_manifest_hash,media_type='application/json')
    run_manifest = RunManifestV3(
        run_id=runs[0].run_id,timestamp=timestamp,code_revision=code_revision,dirty_state=False,
        task_manifest=task_manifest_ref,scorer_config=config,adapter_info=info,adapter_capabilities=caps,
        capability_verification_artifact=capability_ref,
        model_name='Qwen2.5-7B-Instruct',provider='offline_local',
        model_revision='a09a35458c702b33eeacc393d103063234e8bc28',
        prompt_config={'renderer':'version-transfer-native-readout-v1'},
        decoding_config={'temperature':0,'response_parser':ANSWER_MODEL_PARSER_VERSION_V3},
        seed_information={},action_parser_version=PROVENANCE.action_parser_version,
        answer_parser_version=PROVENANCE.answer_parser_version,
        memory_entry_extractor_version=PROVENANCE.memory_entry_extractor_version,
        object_value_extractor_config_hash=hashlib.sha256(config_bytes).hexdigest(),
        redaction_policy_version=PROVENANCE.redaction_policy_version,
        environment_summary={},package_summary={},expected_task_count=len(tasks),
        completed_task_count=sum(r.completion_status.value=='completed' for r in runs),
        failed_task_count=sum(r.completion_status.value in ('failed','partial') for r in runs),
        not_supported_task_count=sum(r.completion_status.value=='not_supported' for r in runs),
        raw_provider_response_artifacts=(),raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(run_ref,),run_record_hashes={r.task_id:sha256_model(r) for r in runs},
        score_artifacts=(),native_vs_extracted_field_summary={
            'state_alignment':'unavailable','source_event_alignment':'unavailable',
            'answers':'separate_local_readout','native_temporal_fields':'preserved'},
    )
    run_manifest_hash = sha256_model(run_manifest)
    scores = []
    for task, run in zip(tasks, runs):
        context = VerifiedScoringContextV3.from_authenticated_manifests(
            task=task,run=run,task_manifest=task_manifest,run_manifest=run_manifest,
            task_artifact=task_ref,run_artifact=run_ref,
            authenticated_task_manifest_sha256=task_manifest_hash,
            authenticated_run_manifest_sha256=run_manifest_hash)
        scores.append(score_task_v3(task, run, context))
    return {'scores':tuple(scores),'task_manifest':task_manifest,'run_manifest':run_manifest,
            'task_artifact':task_ref,'run_artifact':run_ref,
            'task_manifest_sha256':task_manifest_hash,'run_manifest_sha256':run_manifest_hash,
            'payloads':{'tasks.jsonl':task_bytes,'runs.jsonl':run_bytes,
                        'task-manifest.json':canonical_json_bytes(task_manifest),
                        'run-manifest.json':canonical_json_bytes(run_manifest),
                        'scores.jsonl':b''.join(canonical_json_bytes(s)+b'\n' for s in scores)}}
