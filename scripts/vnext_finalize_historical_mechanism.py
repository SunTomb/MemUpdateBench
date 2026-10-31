"""Offline Task 4: direct-state historical-serving classification and audited publication.

No services, models, providers or experiments are started. The independent Task 5
receipt must be pinned out of band; PASS means evidence was reopened faithfully,
not that every case succeeded. See AUDIT_SCHEMA / AUDIT_CHECKS / audit_projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import vnext_run_historical_mechanism as runner
from scripts.analyze_version_transfer_mechanism import _time, _time_status, _mentions
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.providers.graphiti_state_witness import GraphitiStateWitness
from mub.vnext.io.atomic import publish_files_atomically

ROLES = ('target_only_control', 'cross_entity_same_relation', 'historical_serving')
LABELS = frozenset(('BLOCKED', 'NO_DISTINGUISHING_SIGNAL', 'CROSS_ENTITY_INVALIDATION_LEAD',
                   'HISTORICAL_SERVING_LEAD', 'CROSS_ENTITY_AND_HISTORICAL_LEAD'))
AUDIT_SCHEMA = 'memupdatebench.historical-mechanism-audit.v1'
AUDIT_CHECKS = ('all_case_rows_reopened', 'artifact_hashes_verified', 'source_verified',
                'packet_verified', 'projection_bindings_verified',
                'extraction_accounting_reconciled', 'cleanup_reopened')
BINDING_KEYS = ('terminal_sha256', 'packet_index_sha256', 'runtime_source_manifest_sha256',
                'private_index_sha256', 'observations_sha256', 'transport_sha256')
COUNTS = runner._COUNT_KEYS


class EvidenceError(ValueError):
    """Only fixed codes created by this module may reach public receipts."""


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def verified_json(path, pin):
    assert_no_reparse_components(path)
    require(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'INVALID_HASH_PIN')
    raw = Path(path).read_bytes()
    require(sha_bytes(raw) == pin, 'ARTIFACT_HASH_MISMATCH')
    return json.loads(raw)


def count_projection(row):
    result = {key: row.get(key) for key in COUNTS}
    require(all(type(v) is int and v >= 0 for v in result.values()), 'INVALID_COUNTS')
    return result


def audit_projection(report):
    """Task 5's exact coverage rows; preserve failed/NOT_RUN statuses and counts."""
    return [{'role': row['role'], 'status': row['status'], **count_projection(row),
             'namespace_empty_after_cleanup': row.get('namespace_empty_after_cleanup')}
            for row in report['cases']]


def verify_audit(path, pin, bindings, report):
    receipt = verified_json(path, pin)
    require(set(receipt) == {'schema', 'status', 'bindings', 'checks', 'cases'}, 'AUDIT_SHAPE')
    require(receipt['schema'] == AUDIT_SCHEMA and receipt['status'] == 'PASS', 'AUDIT_NOT_PASS')
    require(receipt['bindings'] == bindings and set(bindings) == set(BINDING_KEYS), 'AUDIT_BINDING_MISMATCH')
    require(type(receipt['checks']) is dict and set(receipt['checks']) == set(AUDIT_CHECKS)
            and all(v is True for v in receipt['checks'].values()), 'AUDIT_CHECKS_INCOMPLETE')
    require(encoded(receipt['cases']) == encoded(audit_projection(report)), 'AUDIT_COVERAGE_MISMATCH')
    return receipt


def public_schedule(case):
    """Read ONLY the explicit public synthetic sentence, never gold replay."""
    schedule = []
    for event in case.events:
        p = event.public_payload
        match = re.fullmatch(r"At ([^,]+), (Target-00|Other-00)'s relation is ([a-z-]+)\. "
                             r'This statement replaces its earlier relation when present\.', p['raw_text'])
        require(match is not None and match[1] == p['logical_time'], 'PUBLIC_EVENT_UNINTERPRETABLE')
        schedule.append((match[2], match[3], _time(match[1])))
    return schedule


def direct_edges(snapshot, case):
    """Recompute actual Task 2 compact snapshot from its private state payload."""
    group, rows = snapshot['namespace'], snapshot['rows']
    require(group == case.runtime_namespace, 'STATE_NAMESPACE_MISMATCH')
    compact = [{k: v for k, v in row.items() if k != 'properties'} for row in rows]
    compact.sort(key=lambda row: (row['kind'], str(row.get('uuid')), GraphitiStateWitness._encoded(row)))
    require(runner.digest({'namespace': group, 'rows': compact, 'anomalies': []}) == snapshot['sha256'],
            'DIRECT_STATE_HASH_MISMATCH')
    nodes = {}
    seen = set()
    for row in rows:
        require(row.get('group_id') == group and type(row.get('uuid')) is str, 'STATE_IDENTITY_UNKNOWN')
        for key in ('uuid', 'name', 'group_id', 'valid_at', 'invalid_at', 'created_at', 'expired_at'):
            if key in row or key in row['properties']:
                require(row.get(key) == row['properties'].get(key), 'STATE_PROPERTY_ALIGNMENT_UNKNOWN')
            if key.endswith('_at') and row.get(key) is not None:
                _time(row[key])
        require((row['kind'], row['uuid']) not in seen, 'STATE_DUPLICATE_ID')
        seen.add((row['kind'], row['uuid']))
        require(GraphitiStateWitness._value_hash(row['properties']) == row.get('properties_sha256'),
                'STATE_PROPERTIES_MISMATCH')
        if row['kind'] == 'Entity':
            nodes[row['uuid']] = row.get('name')
    aligned = {}
    for row in rows:
        if row['kind'] != 'RELATES_TO':
            continue
        require(row.get('source_group_id') == group and row.get('target_group_id') == group,
                'STATE_ENDPOINT_GROUP_UNKNOWN')
        require(all(k in row for k in ('valid_at', 'invalid_at', 'created_at', 'expired_at')) and row['valid_at'] is not None,
                'STATE_INTERVAL_UNKNOWN')
        require(_time_status(row, _time(row['valid_at'])) is not None, 'STATE_INTERVAL_UNKNOWN')
        fact = row['properties'].get('fact')
        require(type(fact) is str and sha_bytes(fact.encode()) == row.get('fact_sha256'), 'STATE_FACT_HASH_MISMATCH')
        names = (row.get('source_name'), row.get('target_name'))
        target_candidate = 'Target-00' in names or _mentions(fact, 'Target-00')
        if not target_candidate:
            continue
        for side in ('source', 'target'):
            require(nodes.get(row.get(side+'_uuid')) == row.get(side+'_name')
                    and type(row.get(side+'_name')) is str, 'TARGET_ALIGNMENT_UNKNOWN')
        values = [v for v in ('target-alpha', 'target-beta', 'target-gamma') if v in names]
        require('Target-00' in names and len(values) == 1 and _mentions(fact, 'Target-00')
                and _mentions(fact, values[0]) and re.search(r'\brelation\b', fact, re.I),
                'TARGET_ALIGNMENT_UNKNOWN')
        aligned[row['uuid']] = {**row, 'value': values[0]}
    return aligned


def case_state(case, row, artifacts):
    snapshots = row['snapshots']
    require(len(snapshots) == 3 and [s['position'] for s in snapshots] == [0, 1, 2], 'STATE_COVERAGE_INCOMPLETE')
    result = []
    for receipt in snapshots:
        matches = [v for kind, v in artifacts if kind == 'state' and v.get('sha256') == receipt['sha256']
                   and v.get('namespace') == case.runtime_namespace]
        require(bool(matches), 'DIRECT_STATE_UNAVAILABLE')
        require(all(v == matches[0] for v in matches), 'DIRECT_STATE_AMBIGUOUS')
        require(len(matches[0]['rows']) == receipt['rows'], 'DIRECT_STATE_COUNT_MISMATCH')
        result.append(direct_edges(matches[0], case))
    return result


def control_interpretable(states, schedule):
    for position, edges in enumerate(states):
        for i in range(position+1):
            candidates = [e for e in edges.values() if e['value'] == schedule[i][1]]
            require(len(candidates) == 1, 'CONTROL_VERSION_UNKNOWN')
            edge = candidates[0]
            expected_end = schedule[i+1][2] if i < position else None
            require(_time(edge['valid_at']) == schedule[i][2]
                    and (None if edge['invalid_at'] is None else _time(edge['invalid_at'])) == expected_end,
                    'CONTROL_UNEXPLAINED_INTERVAL')
        require(len(edges) == position+1, 'CONTROL_VERSION_UNKNOWN')


def cross_transitions(states, schedule):
    require([s[0] for s in schedule] == ['Target-00', 'Other-00', 'Other-00'], 'CROSS_SCHEDULE_MISMATCH')
    require(len(states[0]) == 1, 'CROSS_INITIAL_TARGET_UNKNOWN')
    initial = next(iter(states[0].values()))
    require(initial['value'] == schedule[0][1] and initial['invalid_at'] is None
            and _time(initial['valid_at']) == schedule[0][2], 'CROSS_INITIAL_TARGET_UNKNOWN')
    previous, transitions, seen = states[0], [], set()
    stable = ('source_uuid', 'target_uuid', 'source_name', 'target_name', 'source_group_id',
              'target_group_id', 'group_id', 'valid_at', 'fact_sha256', 'value')
    for position, current in enumerate(states[1:], 1):
        require(set(current) == set(previous), 'CROSS_TARGET_IDENTITY_UNKNOWN')
        for uuid, edge in current.items():
            before = previous[uuid]
            require(all(edge[k] == before[k] for k in stable), 'CROSS_TARGET_ALIGNMENT_CHANGED')
            if before['invalid_at'] is None and edge['invalid_at'] is not None:
                require(_time_status(before, schedule[position-1][2]) is True
                        and _time(edge['invalid_at']) == schedule[position][2], 'CROSS_TIMING_UNKNOWN')
                if uuid not in seen:
                    transitions.append({'edge_uuid_sha256': sha_bytes(uuid.encode()), 'event_position': position,
                                        'previous_event_position': position-1, 'valid_at': edge['valid_at'],
                                        'invalid_at': edge['invalid_at'], 'fact_sha256': edge['fact_sha256']})
                    seen.add(uuid)
            elif before['invalid_at'] != edge['invalid_at']:
                raise EvidenceError('CROSS_INTERVAL_CHANGED_AMBIGUOUS')
        previous = current
    return transitions


def query_evidence(case, row, final_state, artifacts, schedule):
    projected = runner.project_readout_task(case.task)
    require(row['readout_projection'] == runner.projection_binding(case, projected), 'PROJECTION_BINDING_MISMATCH')
    require(len(row['queries']) == len(case.queries), 'QUERY_COVERAGE_INCOMPLETE')
    outputs = []
    for position, public_query in enumerate(case.queries):
        receipt = row['queries'][position]
        require(receipt['position'] == position and receipt['final_snapshot_sha256'] == row['snapshots'][-1]['sha256'],
                'QUERY_STATE_BINDING_MISMATCH')
        candidates = [v for k, v in artifacts if k == 'retrieval' and runner.digest(v) == receipt['retrieval_sha256']]
        require(bool(candidates) and all(v == candidates[0] for v in candidates), 'RETRIEVAL_ARTIFACT_UNKNOWN')
        native = candidates[0]
        ids = [e['uuid'] for e in native['edges']]
        require(len(ids) == len(set(ids)), 'RETRIEVAL_DUPLICATE_ID')
        query = projected.queries[position]
        match = re.fullmatch(r"As of ([^,]+), what is Target-00's relation\? Return the value only\.", public_query.public_payload['query_text'])
        require(match is not None, 'PUBLIC_QUERY_TIME_UNKNOWN')
        when = _time(match[1])
        visible_versions = [v for v in schedule if v[0] == 'Target-00' and v[2] <= when]
        require(bool(visible_versions), 'PUBLIC_QUERY_VERSION_UNKNOWN')
        expected = visible_versions[-1][1]
        valid = [e for e in final_state.values() if _time_status(e, when) is True]
        require(len(valid) <= 1 and all(e['value'] == expected for e in valid), 'HISTORY_ALIGNMENT_UNKNOWN')
        correct = {}
        visible = False
        for order in ('native', 'reverse', 'oracle'):
            entries = [v for k, v in artifacts if k == 'readout' and v['order'] == order
                       and v['request']['query']['query_id'] == query.query_id]
            require(len(entries) == 1, 'READOUT_COVERAGE_INCOMPLETE')
            item = entries[0]
            request = (runner.oracle_request(projected, query) if order == 'oracle' else
                       runner.build_readout(query, native, namespace=case.runtime_namespace, order=order))
            require(item['request'] == request.model_dump(mode='json'), 'READOUT_REQUEST_MISMATCH')
            prediction = runner.parse_readout_response(request, item['response'], model_id=runner.MODEL_ID)
            require(prediction.format_valid and prediction.model_dump(mode='json') == item['prediction'], 'READOUT_PARSE_MISMATCH')
            parsed = prediction.model_dump(mode='json')
            correct[order] = parsed.get('disposition') == 'answered' and parsed.get('parsed_answer') == expected
            if order == 'native' and valid and valid[0]['uuid'] in ids:
                edge = native['edges'][ids.index(valid[0]['uuid'])]
                stored = valid[0]
                visible = (edge.get('group_id') == case.runtime_namespace
                    and edge.get('source_node_uuid') == stored['source_uuid']
                    and edge.get('target_node_uuid') == stored['target_uuid']
                    and edge.get('fact') == stored['properties']['fact']
                    and edge.get('valid_at') == stored['valid_at'] and edge.get('invalid_at') == stored['invalid_at'])
                require(visible, 'RETURNED_EVIDENCE_ALIGNMENT_UNKNOWN')
        subtype = None
        status = 'ABSENT_TARGET_ALIGNED_WITNESS' if not valid else 'STORED_TIME_VALID'
        if case.role == 'historical_serving' and position == 1 and valid:
            if valid[0]['uuid'] not in ids:
                subtype = 'retrieval'
            elif visible and not correct['native']:
                subtype = 'readout'
        outputs.append({'query_id': query.query_id, 'as_of': match[1], 'status': status,
                        'retrieved_edges': len(ids), 'stored_time_valid_target_edges': len(valid),
                        'native_correct': correct['native'], 'reverse_correct_diagnostic': correct['reverse'],
                        'oracle_correct_diagnostic': correct['oracle'], 'historical_subtype': subtype})
    return outputs


def classify(report, cases, artifacts, observations):
    """Pure injected evaluator. Unverified inputs are NOT authenticated live evidence."""
    result = {'schema': 'memupdatebench.historical-mechanism-decision.v1', 'label': 'BLOCKED',
              'scientific_release_allowed': False, 'independent_signer': False, 'validation_tasks': 0,
              'semantic_input_patterns': 2, 'independent_semantic_cores': None,
              'scope': 'STRUCTURAL_CASE_BOUND_NON_CAUSAL', 'cross_entity_transitions': None,
              'historical_subtype': None, 'transitions': [], 'cases': [], 'counts': {},
              'reason': None}
    states = []
    try:
        require(tuple(c.role for c in cases) == ROLES and len(report['cases']) == 3, 'CASE_MEMBERSHIP_MISMATCH')
        for case, row in zip(cases, report['cases']):
            require(row['role'] == case.role, 'CASE_MEMBERSHIP_MISMATCH')
            status = row['status'] if row['status'] in ('OBSERVED', 'BLOCKED', 'NOT_RUN') else 'BLOCKED'
            result['cases'].append({'role': case.role, 'task_sha256': case.task_hash, 'status': status,
                **count_projection(row), 'planned_events': 3, 'planned_queries': len(case.queries),
                'planned_readouts': len(case.queries)*3, 'state_accuracy': None, 'reason': None,
                'namespace_empty_after_cleanup': row.get('namespace_empty_after_cleanup') if type(row.get('namespace_empty_after_cleanup')) is bool else None,
                'readout_projection': runner.projection_binding(case, runner.project_readout_task(case.task)),
                'internal_accounting': dict.fromkeys(('llm_calls', 'embedding_calls', 'retries', 'input_tokens', 'output_tokens'), None),
                'layers': dict.fromkeys(('state', 'lifecycle', 'retrieval', 'readout'), 'UNAVAILABLE'), 'queries': []})
        result['counts'] = {k: sum(r[k] for r in result['cases']) for k in COUNTS}
        require(result['counts'] == report['counts'], 'COUNT_TOTAL_MISMATCH')
        for case, row, public in zip(cases, report['cases'], result['cases']):
            try:
                require(row['task_hash'] == case.task_hash, 'CASE_MEMBERSHIP_MISMATCH')
                require(row['status'] == 'OBSERVED', 'CASE_NOT_OBSERVED')
                require(row.get('namespace_empty_after_cleanup') is True, 'CASE_CLEANUP_FAILED')
                require(count_projection(row) == dict(zip(COUNTS, (3, 3, len(case.queries), len(case.queries),
                        3*len(case.queries), 3*len(case.queries)))), 'CASE_COVERAGE_INCOMPLETE')
                schedule = public_schedule(case)
                state = case_state(case, row, artifacts)
                if case.role == 'target_only_control':
                    control_interpretable(state, schedule)
                public['layers']['state'] = 'INTERPRETABLE'
                queries = query_evidence(case, row, state[-1], artifacts, schedule)
                public['queries'] = queries
                public['layers'].update(lifecycle='INTERPRETABLE', retrieval='OBSERVED', readout='OBSERVED')
                states.append(state)
            except EvidenceError as exc:
                public['reason'] = str(exc)
                states.append(None)
            except (KeyError, TypeError, ValueError, IndexError, AttributeError):
                public['reason'] = 'MALFORMED_CASE_EVIDENCE'
                states.append(None)
        require(all(s is not None for s in states), 'CASE_EVIDENCE_BLOCKED')
        require(report['status'] == 'OBSERVATIONS_COMPLETE', 'EXECUTION_BLOCKED')
        require(report.get('client_cleanup_errors', []) == []
                and report.get('owned_process_cleanup') and all(v is True for v in report['owned_process_cleanup'])
                and report.get('ports_closed') == {'17687': True, '18043': True}
                and report.get('owned_gpu_process_absent') is True, 'TERMINAL_CLEANUP_FAILED')
        require(bool(observations) and all(r['status'] == 'OBSERVED' for r in observations)
                and runner.digest(observations) == report['extraction_observations']['sha256']
                and len(observations) == report['extraction_observations']['count']
                and report['extraction_observations']['blocked'] == 0, 'EXTRACTION_INCOMPLETE')
        accounting = report.get('accounting')
        require(type(accounting) is dict and accounting['llm']['all_settled'] is True, 'ACCOUNTING_INCOMPLETE')
        require(accounting['llm']['charged'] == {'attempts': len(observations),
            'input_tokens': sum(o['usage']['prompt_tokens'] for o in observations),
            'output_tokens': sum(o['usage']['completion_tokens'] for o in observations)}, 'ACCOUNTING_USAGE_MISMATCH')
        require(type(report.get('elapsed_seconds')) in (float, int) and 0 <= report['elapsed_seconds'] <= runner.LIMITS['seconds']
                and type(report.get('output_bytes')) is int and 0 <= report['output_bytes'] <= runner.STORAGE_BYTES,
                'RESOURCE_LIMIT_UNKNOWN')
        transitions = cross_transitions(states[1], public_schedule(cases[1]))
        subtype = result['cases'][2]['queries'][1]['historical_subtype']
        result.update(cross_entity_transitions=len(transitions), transitions=transitions, historical_subtype=subtype)
        result['label'] = ('CROSS_ENTITY_AND_HISTORICAL_LEAD' if transitions and subtype else
            'CROSS_ENTITY_INVALIDATION_LEAD' if transitions else 'HISTORICAL_SERVING_LEAD' if subtype else
            'NO_DISTINGUISHING_SIGNAL')
    except EvidenceError as exc:
        result['reason'] = str(exc)
    except (KeyError, ValueError, TypeError, IndexError, AttributeError):
        result['reason'] = 'MALFORMED_EXECUTION_EVIDENCE'
    return result


def reopen_run(*, run_root, terminal_sha256, packet_root, source_manifest, source_manifest_sha256,
               runtime_revision, audit_receipt, audit_sha256, source_root=ROOT):
    """Verify raw byte pins BEFORE decoding artifacts or evaluating any state."""
    run_root = Path(run_root)
    source = runner.verify_runtime_source(source_manifest, source_manifest_sha256, runtime_revision, root=source_root)
    _, cases = runner.load_cases(packet_root, runner.PACKET_PIN)
    terminal = verified_json(run_root/'terminal.json', terminal_sha256)
    require(terminal.get('runtime_revision') == runtime_revision
            and terminal.get('runtime_source_manifest_sha256') == source_manifest_sha256
            and terminal.get('packet_index_sha256') == runner.PACKET_PIN
            and terminal.get('packet_source_revision') == runner.PACKET_SOURCE, 'TERMINAL_SOURCE_MISMATCH')
    require(terminal.get('runtime_source_map_sha256') == runner.digest(source['files']), 'TERMINAL_SOURCE_MAP_MISMATCH')
    require(len(terminal['cases']) == 3, 'CASE_MEMBERSHIP_MISMATCH')
    for case, row in zip(cases, terminal['cases']):
        require(row.get('role') == case.role and row.get('task_hash') == case.task_hash, 'CASE_MEMBERSHIP_MISMATCH')
        if 'readout_projection' in row:
            require(row['readout_projection'] == runner.projection_binding(case, runner.project_readout_task(case.task)),
                    'PROJECTION_BINDING_MISMATCH')
    bindings = dict(zip(BINDING_KEYS, (terminal_sha256, runner.PACKET_PIN, source_manifest_sha256,
        terminal['private_artifacts']['index_sha256'], terminal.get('observations_sha256'), terminal.get('transport_sha256'))))
    verify_audit(audit_receipt, audit_sha256, bindings, terminal)
    index = verified_json(run_root/'private/index.json', bindings['private_index_sha256'])
    require(type(index) is dict and len(index) == terminal['private_artifacts']['count'], 'PRIVATE_INDEX_COUNT_MISMATCH')
    require({p.name for p in (run_root/'private').iterdir()} == set(index) | {'index.json'},
            'PRIVATE_INDEX_MEMBERSHIP_MISMATCH')
    artifacts = []
    for name, pin in sorted(index.items()):
        match = re.fullmatch(r'[0-9]{5}_([A-Za-z0-9_.-]+)\.json', name)
        require(match is not None, 'PRIVATE_ARTIFACT_NAME_INVALID')
        artifacts.append((match[1], verified_json(run_root/'private'/name, pin)))
    observations = (verified_json(run_root/'compact/observations.json', bindings['observations_sha256'])
                    if bindings['observations_sha256'] is not None else [])
    if bindings['transport_sha256'] is not None:
        verified_json(run_root/'compact/transport.json', bindings['transport_sha256'])
    result = classify(terminal, cases, artifacts, observations)
    result['bindings'] = {**bindings, 'audit_sha256': audit_sha256,
                          'packet_source_revision': runner.PACKET_SOURCE, 'runtime_revision': runtime_revision,
                          'analysis_source_sha256': sha_bytes(Path(__file__).read_bytes())}
    result['resources'] = resource_projection(terminal, observations if bindings['observations_sha256'] is not None else None)
    return result


def resource_projection(report, observations=None):
    """Explicit numeric leaf allowlist; never copy arbitrary provider metadata."""
    def number(value):
        return value if type(value) in (int, float) and 0 <= value < float('inf') else None
    a = report.get('accounting') or {}
    return {'elapsed_seconds': number(report.get('elapsed_seconds')), 'output_bytes': number(report.get('output_bytes')),
        'llm': {k: number(a.get('llm', {}).get('charged', {}).get(k)) for k in ('attempts', 'input_tokens', 'output_tokens')},
        'embeddings': {k: number(a.get('embeddings', {}).get(k)) for k in ('attempts', 'items', 'input_tokens')},
        'actual_http_attempts': number(a.get('actual_http_attempts')),
        'roles': {k: number(a.get('llm', {}).get('roles', {}).get(k)) for k in ('graphiti', 'readout', 'oracle_readout')},
        'finish_reasons': None if observations is None else {
            'stop': sum(o.get('finish_reason') == 'stop' for o in observations),
            'length': sum(o.get('finish_reason') == 'length' for o in observations),
            'other': sum(o.get('finish_reason') not in (None, 'stop', 'length') for o in observations),
            'unknown': sum(o.get('finish_reason') is None for o in observations)},
        'openai_transport_retries': number(a.get('openai_transport_retries')),
        'native_retry_attempts': None}


def finalize(*, output_root, results_parent=ROOT/'results/vnext', **kwargs):
    parent = Path(results_parent).resolve(strict=True)
    output = Path(output_root).absolute()
    require(output.parent.resolve(strict=True) == parent and re.fullmatch(
        r'version_transfer_historical_mechanism_20261030_v[1-9][0-9]*', output.name), 'OUTPUT_OUTSIDE_PROTOCOL')
    assert_no_reparse_components(output)
    require(not output.exists(), 'OUTPUT_ALREADY_EXISTS')
    decision = reopen_run(**kwargs)
    payload = encoded(decision)
    index = {'schema': 'memupdatebench.historical-mechanism-index.v1', 'artifacts': {'decision.json': sha_bytes(payload)},
             'bindings': decision['bindings'], 'scientific_release_allowed': False, 'independent_signer': False,
             'validation_tasks': 0}
    output.mkdir(exist_ok=False)
    publish_files_atomically({output/'decision.json': payload, output/'index.json': encoded(index)}, overwrite=False)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('run-root', 'packet-root', 'source-manifest', 'audit-receipt', 'output-root'):
        parser.add_argument('--'+name, required=True, type=Path)
    for name in ('terminal-sha256', 'source-manifest-sha256', 'runtime-revision', 'audit-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(**vars(args))
    except (ValueError, OSError, KeyError, TypeError):
        print(json.dumps({'status': 'BLOCKED', 'error_code': 'FINALIZATION_REFUSED', 'scientific_release_allowed': False}))
        return 1
    print(json.dumps({'label': result['label'], 'scientific_release_allowed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
