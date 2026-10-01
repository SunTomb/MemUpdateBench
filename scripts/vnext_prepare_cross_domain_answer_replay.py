"""Prepare four answer-only readouts of frozen v13 retrievals; never execute a model."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema
from mub.vnext.contracts.v3.common import typed_json_equal
from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.runtime.answer_model_v3 import (
    ANSWER_MODEL_PARSER_VERSION_V3, parse_answer_prediction_v3, render_visible_prompt_v3,
)
from mub.vnext.validation.cross_domain_qdrant_state import (
    EXPECTED_ARTIFACT_INDEX_SHA256, EXPECTED_COLLECTION_AUDIT_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256, validate_evidence,
)
from scripts.vnext_promote_family_h_cross_domain_bea import read_release as read_bea
from scripts.vnext_promote_family_h_cross_domain_noaa import read_release as read_noaa
from scripts.vnext_run_cross_domain_state import (
    BEA_INDEX_SHA256, NOAA_INDEX_SHA256, _normalized_entries, _stable_row_signature,
)

POLICY_SHA256 = '30df8bdc52e5f0dd48f7c53cecbb07b46f032c6eae81b2478c0969f9371d681a'
SCHEMA = 'memupdatebench.cross-domain.answer-replay-preparation.v1'
CONFIG = ROOT / 'configs/vnext/post_core/cross_domain_answer_replay_v1.json'
DEFAULT_EVIDENCE = ROOT / 'tests/vnext/fixtures/cross_domain_v13'
DEFAULT_BEA = ROOT / 'data/vnext/family_h_cross_domain_bea_gdp/v1'
DEFAULT_NOAA = ROOT / 'data/vnext/family_h_cross_domain_noaa/v1'
RESULT_NAMES = {'artifact_index.json', 'rows.jsonl', 'summary.json', 'runtime.json',
                'input_binding.json', 'http_accounting.json'}
PACKAGE_NAMES = {'manifest.json', 'plans.jsonl', 'validation.json', 'index.json'}
KEY_FIELDS = ('namespace', 'entity', 'attribute', 'subkey')
SOURCE_FILES = (
    'scripts/vnext_prepare_cross_domain_answer_replay.py',
    'scripts/vnext_run_cross_domain_state.py',
    'scripts/vnext_promote_family_h_cross_domain_bea.py',
    'scripts/vnext_promote_family_h_cross_domain_noaa.py',
    'mub/vnext/runtime/answer_model_v3.py',
    'mub/vnext/contracts/common.py', 'mub/vnext/contracts/enums.py',
    'mub/vnext/contracts/v3/common.py', 'mub/vnext/contracts/v3/runtime.py',
    'mub/vnext/contracts/v3/task.py', 'mub/vnext/validation/replay_v3.py',
    'mub/vnext/validation/cross_domain_qdrant_state.py', 'mub/vnext/io/atomic.py',
    'mub/vnext/external/security.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(canonical(value))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def strict_json(raw):
    def reject(value):
        raise ValueError('nonfinite JSON constant')
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=reject)
    canonical(value)
    return value


def safe_path(path):
    path = Path(path).absolute()
    require('..' not in path.parts, 'path traversal forbidden')
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'symlink or reparse component forbidden')
    return path


def read_file(path):
    path = safe_path(path)
    require(path.is_file() and path.stat().st_size <= 8 * 1024**2, 'missing or oversized input file')
    return path.read_bytes()


def snapshot(root):
    root = safe_path(root)
    require(root.is_dir(), 'missing input directory')
    result = {}
    for path in sorted(root.rglob('*')):
        safe_path(path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(read_file(path))
    return result


def _key(value):
    return tuple(value[field] for field in KEY_FIELDS)


def _equal(left, right, message):
    require(typed_json_equal(left, right), message)


def recompute_evidence(tasks, rows, summary):
    require(len(rows) == 4 and set(tasks) == {'bea', 'noaa'}, 'v13 coverage mismatch')
    expected_pairs = [(name, rep) for name in ('bea', 'noaa') for rep in (0, 1)]
    require([(r['release_name'], r['repetition_index']) for r in rows] == expected_pairs,
            'v13 repetition order mismatch')
    per_source = {name: {'state_matches': 0, 'source_links': 0, 'stale_same_slot': 0,
                         'final_matches': 0, 'retrieval_matches': 0} for name in tasks}
    for row in rows:
        task = tasks[row['release_name']]
        require(type(row['repetition_index']) is int and row['status'] == 'COMPLETE'
                and row['task_id'] == task.task_id and row['task_sha256'] == digest(task.model_dump(mode='json')),
                'row task/byte binding mismatch')
        require(all(row['cleanup'].get(k) is True for k in ('client_closed', 'collection_deleted', 'reset_empty')),
                'row cleanup failed')
        require(len(row['steps']) == len(task.actions), 'incomplete state observations')
        counts = per_source[row['release_name']]
        final_match = False
        for step, action, event in zip(row['steps'], task.actions, task.events, strict=True):
            require(step['event_id'] == action.event_id == event.event_id
                    and type(step['sequence_index']) is int and step['sequence_index'] == event.sequence_index
                    and step['operation'].upper() == action.operation.value, 'state event binding mismatch')
            entries = _normalized_entries(step['entries'])
            require(digest(entries) == step['entries_sha256'], 'state entry hash mismatch')
            key = _key(action.target_object_keys[0].model_dump(mode='json'))
            same_slot = [e for e in entries if _key(e['object_key']) == key]
            current = [e for e in same_slot if action.event_id in e['source_event_ids']]
            state = len(entries) == len(same_slot) == len(current) == 1 and typed_json_equal(current[0]['value'], action.value)
            source = len(current) == 1
            stale = sum(action.event_id not in e['source_event_ids'] for e in same_slot)
            _equal(step['state_match'], state, 'stored state match disagrees with entries')
            _equal(step['current_source_event_link'], source, 'stored state source linkage disagrees')
            _equal(step['stale_same_slot_count'], stale, 'stored stale count disagrees')
            _equal(step['memory_size'], len(entries), 'stored memory size disagrees')
            counts['state_matches'] += int(state)
            counts['source_links'] += int(source)
            counts['stale_same_slot'] += stale
            final_match = state
        _equal(row['final_state_match'], final_match, 'final state observation mismatch')
        counts['final_matches'] += int(final_match)
        trace = row['retrieval']
        entries = _normalized_entries(trace['entries'])
        require(trace['context_order'] == 'provider_order' and digest(entries) == trace['entries_sha256'],
                'retrieval order/hash mismatch')
        gold = task.gold_evidence[0]
        gold_key = _key(gold.supporting_object_keys[0].model_dump(mode='json'))
        retrieved = any(_key(e['object_key']) == gold_key and typed_json_equal(e['value'], gold.answer)
                        and set(gold.supporting_event_ids) <= set(e['source_event_ids']) for e in entries)
        _equal(row['retrieval_typed_object_match'], retrieved, 'stored retrieval match disagrees with entries')
        _equal(trace['typed_object_match'], retrieved, 'stored retrieval annotation disagrees')
        counts['retrieval_matches'] += int(retrieved)
    for index in (0, 2):
        _equal(_stable_row_signature(rows[index]), _stable_row_signature(rows[index + 1]),
               'repetition semantic mismatch')
    total = {name: sum(c[name] for c in per_source.values()) for name in next(iter(per_source.values()))}
    _equal(total, {'state_matches': 20, 'source_links': 20, 'stale_same_slot': 0,
                   'final_matches': 4, 'retrieval_matches': 4}, 'v13 independently recomputed counts mismatch')
    for field, total_field in (('state_step_matches', 'state_matches'), ('final_state_matches', 'final_matches'),
                               ('retrieval_typed_object_matches', 'retrieval_matches')):
        _equal(summary[field], total[total_field], 'summary/recomputed metrics mismatch')
    require(summary['repetitions_equal'] is True, 'summary repetition mismatch')
    return {'status': 'PASS', 'per_source': per_source, 'total': total,
            'base_tasks': 2, 'base_events': 10, 'repetitions_per_task': 2,
            'observed_mutations': 20, 'observed_retrievals': 4,
            'scope': 'historical_v13_manager_evidence_recomputation_not_answer_accuracy'}


@dataclass(frozen=True)
class InputBundle:
    evidence_root: Path
    bea_root: Path
    noaa_root: Path
    config_path: Path
    policy: dict[str, Any]
    tasks: dict[str, MemUpdateTaskV3]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    reference: dict[str, Any]
    hashes: dict[str, Any]
    admitted_payload_sha256: str

    def assert_unchanged(self):
        require(_memory_digest(self.policy, self.tasks, self.rows, self.summary, self.reference, self.hashes)
                == self.admitted_payload_sha256, 'admitted in-memory inputs changed')
        current = _input_hashes(self.evidence_root, self.bea_root, self.noaa_root, self.config_path)
        require(current == self.hashes, 'preparation inputs or source changed')


def _memory_digest(policy, tasks, rows, summary, reference, hashes):
    return digest({'policy': policy, 'tasks': {name: task.model_dump(mode='json') for name, task in tasks.items()},
                   'rows': rows, 'summary': summary, 'reference': reference, 'hashes': hashes})


def _input_hashes(evidence, bea, noaa, config):
    return {'evidence': snapshot(evidence), 'bea': snapshot(bea), 'noaa': snapshot(noaa),
            'configuration': sha(read_file(config)),
            'source': {name: sha(read_file(ROOT / name)) for name in SOURCE_FILES}}


def load_inputs(evidence_root, bea_root, noaa_root, config_path=CONFIG):
    evidence, bea, noaa, config = map(safe_path, (evidence_root, bea_root, noaa_root, config_path))
    before = _input_hashes(evidence, bea, noaa, config)
    require({p.name for p in evidence.iterdir()} == {'results', 'collection_audit.json', 'launch.json', 'supervisor.exitcode'},
            'collected artifact membership mismatch')
    require({p.name for p in (evidence / 'results').iterdir()} == RESULT_NAMES, 'result artifact membership mismatch')
    policy = strict_json(read_file(config))
    require(digest(policy) == POLICY_SHA256, 'frozen answer replay policy hash mismatch')
    require(sha(read_file(ROOT / 'mub/vnext/runtime/answer_model_v3.py')) == policy['renderer_and_parser']['source_sha256'],
            'renderer/parser source drift')
    require(ANSWER_MODEL_PARSER_VERSION_V3 == policy['renderer_and_parser']['parser_version'], 'answer parser version drift')
    validated = validate_evidence(evidence, bea, noaa)
    require(validated['status'] == 'VALID', 'v13 evidence admission failed')
    require(read_file(evidence / 'supervisor.exitcode') == b'0\n', 'supervisor did not complete')
    index = strict_json(read_file(evidence / 'results/artifact_index.json'))
    for row in index['artifacts']:
        raw = read_file(evidence / 'results' / row['path'])
        require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'artifact bytes/hash mismatch')
    tasks = {'bea': read_bea(bea, BEA_INDEX_SHA256)['tasks'][0],
             'noaa': read_noaa(noaa, NOAA_INDEX_SHA256)['tasks'][0]}
    require(all(t.queries[0].answer_schema is AnswerSchema.NUMBER for t in tasks.values()), 'number answer schema required')
    rows = [strict_json(line) for line in read_file(evidence / 'results/rows.jsonl').splitlines()]
    summary = strict_json(read_file(evidence / 'results/summary.json'))
    require(summary['bindings_unchanged'] is True and summary['answer_metrics'] is None
            and summary['error'] is None and summary['scientific_release_allowed'] is False,
            'historical evidence boundary mismatch')
    execution = summary['execution_boundary']
    for field in ('model_loads', 'generations', 'provider_model_calls', 'gpu_calls'):
        _equal(execution[field], 0, 'historical no-model boundary mismatch')
    runtime = strict_json(read_file(evidence / 'results/runtime.json'))
    require(digest(runtime['runtime']) == runtime['runtime_sha256'] == summary['runtime_binding']['runtime_sha256'],
            'runtime fingerprint binding mismatch')
    binding = strict_json(read_file(evidence / 'results/input_binding.json'))
    require(binding['source_manifest_sha256'] == EXPECTED_SOURCE_MANIFEST_SHA256
            and binding['unchanged'] is True and binding['release_pins'] == policy['release_pins'],
            'historical input binding mismatch')
    launch = strict_json(read_file(evidence / 'launch.json'))
    require(launch['source_manifest_sha256'] == EXPECTED_SOURCE_MANIFEST_SHA256
            and launch['archive_sha256'] == '30e2b561575b7ea65616c0ca2f5c0b13adc5f72fc4076bb9c445e8503caff745',
            'launch/source binding mismatch')
    http = strict_json(read_file(evidence / 'results/http_accounting.json'))
    require(len(http['worker_http_calls']) == execution['worker_http_calls'] == 100
            and len(http['setup']) == execution['setup_http_calls'] == 2
            and all(c.get('authenticated') is True and c.get('status') in (200, 201, 202, 204)
                    for c in http['worker_http_calls']), 'HTTP accounting mismatch')
    reference = recompute_evidence(tasks, rows, summary)
    bundle = InputBundle(evidence, bea, noaa, config, policy, tasks, rows, summary, reference, before,
                         _memory_digest(policy, tasks, rows, summary, reference, before))
    bundle.assert_unchanged()
    return bundle


def project_trace(query, retrieval):
    require(query.answer_schema is AnswerSchema.NUMBER, 'number query required')
    require(set(retrieval) == {'entries', 'entries_sha256', 'context_order', 'typed_object_match', 'version_metadata'},
            'unsupported retrieval fields')
    require(retrieval['context_order'] == 'provider_order', 'provider order required')
    _equal(retrieval['version_metadata'], {'provider': 'qdrant', 'provider_entry_ids_preserved': True,
        'provider_order_preserved': True, 'provider_payload_preserved': True, 'provider_scores_preserved': True},
        'unsupported retrieval metadata')
    entries = _normalized_entries(retrieval['entries'])
    require(digest(entries) == retrieval['entries_sha256'], 'retrieval entries hash mismatch')
    records = []
    for entry in entries:
        metadata = entry['version_metadata']
        require(set(metadata) == {'point_id_derivation', 'version_index'}
                and metadata['point_id_derivation'] == 'uuid5-adapter-v1'
                and type(metadata['version_index']) is int and metadata['version_index'] >= 0,
                'unsupported provider entry metadata')
        require(not scan_for_secrets(entry), 'provider entry secret scan failed')
        records.append(MemoryEntryRecordV3(entry_id=entry['entry_id'], content=entry['content'],
            object_key_candidate=entry['object_key'], value_candidate=entry['value'],
            source_event_ids=tuple(entry['source_event_ids']), raw_metadata=metadata))
    return RetrievalTraceV3(query_id=query.query_id, retrieved_entries=tuple(records),
        scores=tuple(e['score'] for e in entries), ranks=tuple(e['rank'] for e in entries),
        context_order='provider_order', retrieval_policy='frozen_v13_provider_order')


def project_prompt(query, retrieval):
    trace = project_trace(query, retrieval)
    # The canonical renderer needs only these public query fields.
    visible_query = SimpleNamespace(query_id=query.query_id, text=query.text, answer_schema=query.answer_schema)
    return render_visible_prompt_v3(query=visible_query, retrieval_trace=trace)


def build_plans(bundle):
    bundle.assert_unchanged()
    plans = []
    for row in bundle.rows:
        task = bundle.tasks[row['release_name']]
        query = task.queries[0]
        trace = project_trace(query, row['retrieval'])
        prompt = project_prompt(query, row['retrieval'])
        plans.append({'request_id': f"{row['release_name']}-repeat-{row['repetition_index']}",
            'release_name': row['release_name'], 'task_id': task.task_id,
            'task_sha256': digest(task.model_dump(mode='json')), 'query_id': query.query_id,
            'repetition_index': row['repetition_index'], 'answer_schema': 'number',
            'semantic_core_id': task.metadata.split_key.semantic_core_id,
            'source_group_id': task.metadata.split_key.source_group_id,
            'retrieval_sha256': digest(row['retrieval']), 'provider_row_sha256': digest(row),
            'trace_sha256': digest(trace.model_dump(mode='json')), 'provider_entry_count': len(trace.retrieved_entries),
            'visible_prompt_sha256': sha(prompt.encode('utf-8')), 'visible_prompt_bytes': len(prompt.encode('utf-8')),
            'context_order': 'provider_order', 'chat_template_sha256': None, 'input_token_count': None,
            'status': 'NOT_RUN'})
    require(len(plans) == 4, 'four replay requests required')
    return plans


def render_plan(bundle, plan):
    plans = build_plans(bundle)
    selected = [p for p in plans if p['request_id'] == plan.get('request_id')]
    require(len(selected) == 1 and selected[0] == plan, 'plan differs from authenticated inputs')
    row = next(r for r in bundle.rows if r['release_name'] == plan['release_name'] and r['repetition_index'] == plan['repetition_index'])
    return project_prompt(bundle.tasks[plan['release_name']].queries[0], row['retrieval'])


def score_readout(task, *, status, raw_output):
    require(status in ('NOT_RUN', 'TECHNICAL_FAILURE', 'UNSUPPORTED', 'COMPLETED'), 'unknown readout status')
    query = task.queries[0]
    require(query.answer_schema is AnswerSchema.NUMBER and len(task.queries) == len(task.gold_evidence) == 1,
            'one numeric query required for descriptive answer scoring')
    if status != 'COMPLETED':
        require(raw_output is None, 'nonexecuted/failed status cannot carry answer output')
        return {'status': status, 'format_valid': None, 'typed_exact_match': None,
                'included_in_denominator': False, 'canonical_score_record': False, 'failure_layer': 'technical_or_not_run'}
    require(type(raw_output) is str, 'completed readout must supply raw output text')
    prediction = parse_answer_prediction_v3(query_id=query.query_id, answer_schema=AnswerSchema.NUMBER, raw_output=raw_output)
    correct = (prediction.format_valid and prediction.disposition is AnswerDisposition.ANSWERED
               and typed_json_equal(prediction.parsed_answer, task.gold_evidence[0].answer))
    return {'status': status, 'format_valid': prediction.format_valid, 'typed_exact_match': bool(correct),
            'included_in_denominator': True, 'canonical_score_record': False,
            'output_sha256': sha(raw_output.encode('utf-8')), 'disposition': prediction.disposition.value,
            'failure_layer': 'answer_format' if not prediction.format_valid else 'correct' if correct else 'answer_content'}


def _payload(bundle):
    plans = build_plans(bundle)
    manifest = {'schema': SCHEMA, 'status': 'PREPARED_NOT_EXECUTED',
        'evidence_class': 'frozen_external_retrieval_answer_preparation_only',
        'scientific_release_allowed': False, 'benchmark_accuracy_claimed': False,
        'execution_authorized': False, 'answer_metrics': None, 'canonical_score_record': False,
        'native_answer_status': 'UNSUPPORTED', 'native_answer_reason': 'qdrant_has_no_native_answer_implementation',
        'model_runtime_status': 'NOT_QUALIFIED_FOR_THIS_REPLAY',
        'prospective_model_intent': bundle.policy['prospective_model_intent'],
        'prospective_requests': 4, 'base_tasks': 2, 'repetitions_per_task': 2,
        'cluster_unit': 'source_trajectory', 'cluster_n': 2, 'independent_samples_claimed': False,
        'confidence_intervals': None, 'repetition_policy': bundle.policy['repetitions'],
        'parser_and_prompt_policy': bundle.policy['renderer_and_parser'], 'scoring_policy': bundle.policy['scoring'],
        'policy_sha256': POLICY_SHA256, 'policy_file_sha256': bundle.hashes['configuration'],
        'historical_evidence_index_sha256': EXPECTED_ARTIFACT_INDEX_SHA256,
        'historical_collection_audit_sha256': EXPECTED_COLLECTION_AUDIT_SHA256,
        'historical_source_manifest_sha256': EXPECTED_SOURCE_MANIFEST_SHA256,
        'release_pins': bundle.policy['release_pins'],
        'input_file_sha256': {key: value for key, value in bundle.hashes.items() if key in ('evidence', 'bea', 'noaa')},
        'preparation_source_sha256': bundle.hashes['source'],
        'source_scope': 'explicit_preparation_files_not_full_environment_or_tokenizer_qualification',
        'execution_boundary': {'model_loads': 0, 'tokenizer_loads': 0, 'generations': 0, 'provider_calls': 0,
                               'database_calls': 0, 'gpu_calls': 0, 'network_calls': 0, 'child_process_launches': 0},
        'claim_boundary': 'Four prospective readouts of two frozen single-entry contexts; not stale-conflict, multi-object, model accuracy or broad benchmark evidence.',
        'remaining_gates': ['current_device_authorization', 'source_and_model_runtime_qualification',
                            'tokenizer_chat_template_and_context_budget', 'bounded_generation_and_unload',
                            'post_run_readout_validation']}
    validation = {'schema': SCHEMA + '.validation', 'status': 'PASS_PREPARATION_ONLY',
                  'manager_reference_recomputation': bundle.reference, 'prompt_binding_count': 4,
                  'answer_execution_status': 'NOT_RUN', 'answer_metrics': None,
                  'model_capability_claimed': False, 'scientific_release_allowed': False}
    payload = {'manifest.json': canonical(manifest), 'plans.jsonl': b''.join(canonical(p) for p in plans),
               'validation.json': canonical(validation)}
    for raw in payload.values():
        require(b'Use only the retrieved memory entries' not in raw and b'"raw_output"' not in raw,
                'raw prompt/output forbidden in preparation package')
    require(not scan_for_secrets(manifest), 'public preparation metadata secret scan failed')
    index = {'schema': SCHEMA + '.index', 'status': manifest['status'], 'scientific_release_allowed': False,
             'artifacts': [{'path': name, 'bytes': len(raw), 'sha256': sha(raw)} for name, raw in sorted(payload.items())]}
    payload['index.json'] = canonical(index)
    return payload


def _output_root(output, bundle):
    path = safe_path(output)
    for source in (bundle.evidence_root, bundle.bea_root, bundle.noaa_root, bundle.config_path):
        require(path != source and path not in source.parents and source not in path.parents, 'output overlaps input')
    for protected in ('data', 'mub', 'scripts', 'tests', 'configs', '.git'):
        root = ROOT / protected
        require(path != root and root not in path.parents and path not in root.parents, 'output overlaps protected root')
    for parent in path.parents:
        require(not any((parent / name).is_file() for name in ('index.json', 'artifact_index.json', 'release_index.json', 'package_manifest.json')),
                'output is nested under indexed artifacts')
    if path.exists():
        raise FileExistsError(path)
    return path


def publish_preparation(bundle, output_root):
    output = _output_root(output_root, bundle)
    bundle.assert_unchanged()
    fresh = load_inputs(bundle.evidence_root, bundle.bea_root, bundle.noaa_root, bundle.config_path)
    require(fresh.hashes == bundle.hashes, 'source changed before preparation')
    payload = _payload(fresh)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    identity = (output.stat().st_dev, output.stat().st_ino)
    def before_publish():
        fresh.assert_unchanged()
        require(safe_path(output).is_dir() and (output.stat().st_dev, output.stat().st_ino) == identity,
                'owned output directory changed')
    source_paths = tuple(root / name for root, label in ((fresh.evidence_root, 'evidence'), (fresh.bea_root, 'bea'),
                        (fresh.noaa_root, 'noaa')) for name in fresh.hashes[label])
    publish_files_atomically({output / name: raw for name, raw in payload.items()}, overwrite=False,
        source_paths=(*source_paths, fresh.config_path, *(ROOT / name for name in SOURCE_FILES)), pre_publish=before_publish)
    return read_preparation(output, sha(payload['index.json']), fresh)


def read_preparation(output_root, expected_index_sha256, bundle):
    root = safe_path(output_root)
    require(root.is_dir() and {p.name for p in root.iterdir()} == PACKAGE_NAMES, 'package membership mismatch')
    raw = read_file(root / 'index.json')
    require(sha(raw) == expected_index_sha256, 'preparation index hash mismatch')
    expected = _payload(load_inputs(bundle.evidence_root, bundle.bea_root, bundle.noaa_root, bundle.config_path))
    for name, content in expected.items():
        require(read_file(root / name) == content, 'preparation package bytes/hash differ from authenticated inputs')
    return {'status': 'PREPARED_NOT_EXECUTED', 'index_sha256': expected_index_sha256,
            'prospective_requests': 4, 'model_loads': 0, 'generations': 0,
            'execution_authorized': False, 'scientific_release_allowed': False}


def _verification_hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise argparse.ArgumentTypeError('expected a 64-character lowercase SHA-256')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--evidence-root', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--bea-root', type=Path, default=DEFAULT_BEA)
    parser.add_argument('--noaa-root', type=Path, default=DEFAULT_NOAA)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--verify-index-sha256', type=_verification_hash)
    args = parser.parse_args(argv)
    try:
        bundle = load_inputs(args.evidence_root, args.bea_root, args.noaa_root, args.config)
        result = (read_preparation(args.output_root, args.verify_index_sha256, bundle) if args.verify_index_sha256 is not None
                  else publish_preparation(bundle, args.output_root))
    except (ValueError, OSError, TypeError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error_type': type(exc).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
