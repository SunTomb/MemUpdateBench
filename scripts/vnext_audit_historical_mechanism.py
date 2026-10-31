"""Task 5 offline same-assistant integrity audit of an immutable Task 3 run.

PASS authenticates faithful recorded evidence, including blocked/partial outcomes;
it is not a successful protocol, semantic accuracy, human review, signature, or
current live cleanup assertion. No native SDK, database, model or network opens.
The exact compact receipt is consumed by Task 4 using its raw-byte SHA-256.
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

from scripts import vnext_finalize_historical_mechanism as finalizer
from scripts import vnext_run_historical_mechanism as runner
from scripts.analyze_version_transfer_mechanism import _time, _time_status
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.providers.graphiti_observation_recorder import ObservationRecorder, ObservationQualificationError
from mub.vnext.external.providers.graphiti_state_witness import GraphitiStateWitness as Witness
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.contracts.v3.manifest import TaskManifestV3, RunManifestV3
from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3


class AuditError(ValueError):
    """Fixed, redacted code only; never contains source/provider exception text."""


def require(condition, code):
    if not condition:
        raise AuditError(code)


def same(left, right, code):
    require(finalizer.encoded(left) == finalizer.encoded(right), code)


def read_json(path, pin=None):
    assert_no_reparse_components(path)
    raw = Path(path).read_bytes()
    if pin is not None:
        require(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'INVALID_HASH_PIN')
        require(hashlib.sha256(raw).hexdigest() == pin, 'ARTIFACT_HASH_MISMATCH')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(AuditError('NONFINITE_JSON')))


def ledger_rows(path):
    assert_no_reparse_components(path)
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


class Evidence:
    def __init__(self, root, report):
        index = read_json(root/'private/index.json', report['private_artifacts']['index_sha256'])
        require(type(index) is dict and len(index) == report['private_artifacts']['count'], 'PRIVATE_INDEX_COUNT')
        require({p.name for p in (root/'private').iterdir()} == set(index) | {'index.json'}, 'PRIVATE_MEMBERSHIP')
        self.items = []
        for position, (name, pin) in enumerate(sorted(index.items()), 1):
            match = re.fullmatch(r'([0-9]{5})_([A-Za-z0-9_.-]+)\.json', name)
            require(match is not None and int(match[1]) == position, 'PRIVATE_ARTIFACT_NAME')
            kind = match[2]
            require(kind in ('state', 'mutation', 'retrieval', 'readout', 'observation', 'transport', 'canonical_run')
                    or re.fullmatch(r'scoring_(native|reverse)_(tasks.jsonl|runs.jsonl|task-manifest.json|run-manifest.json|scores.jsonl|projection)', kind),
                    'UNEXPECTED_PRIVATE_PAYLOAD')
            self.items.append((kind, read_json(root/'private'/name, pin)))
        self.used = set()

    def of(self, kind):
        return [(i, value) for i, (key, value) in enumerate(self.items) if key == kind]

    def take(self, kind, predicate=lambda _: True):
        matches = [(i, v) for i, v in self.of(kind) if i not in self.used and predicate(v)]
        require(len(matches) == 1, 'ARTIFACT_BINDING_MISSING_OR_AMBIGUOUS')
        i, value = matches[0]
        self.used.add(i)
        return i, value

    def snapshot(self, group, receipt):
        matches = [(i, v) for i, v in self.of('state') if v['namespace'] == group and v['sha256'] == receipt['sha256']]
        require(bool(matches), 'DIRECT_STATE_MISSING')
        for i, value in matches:
            require(len(value['rows']) == receipt['rows'], 'DIRECT_STATE_ROW_COUNT')
            same(value, matches[0][1], 'DIRECT_STATE_AMBIGUOUS')
            self.used.add(i)
        return matches[0]


def verify_snapshot(snapshot, namespaces):
    require(set(snapshot) == {'namespace', 'sha256', 'rows'} and snapshot['namespace'] in namespaces, 'STATE_SHAPE')
    group, compact, seen, nodes = snapshot['namespace'], [], set(), {}
    for row in snapshot['rows']:
        kind, properties = row['kind'], row['properties']
        require(kind in ('Entity', 'Episodic', 'RELATES_TO') and type(properties) is dict, 'STATE_ROW_SHAPE')
        require(type(row.get('uuid')) is str and row['uuid'] and row.get('group_id') == group, 'STATE_IDENTITY')
        require((kind, row['uuid']) not in seen, 'STATE_DUPLICATE_UUID')
        seen.add((kind, row['uuid']))
        require(Witness._value_hash(properties) == row['properties_sha256'], 'STATE_PROPERTIES_HASH')
        expected = {'kind': kind, **{k: v for k, v in properties.items() if k in (
            'uuid', 'name', 'group_id', 'valid_at', 'invalid_at', 'created_at', 'expired_at',
            'episodes', 'episode_ids', 'entity_edges')}}
        if 'fact' in properties:
            expected['fact_sha256'] = Witness._value_hash(properties['fact'])
        for key, value in properties.items():
            require(not key.endswith('embedding'), 'UNEXPECTED_RAW_VECTOR')
            if key.endswith('embedding_sha256'):
                require(value is None or type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'STATE_VECTOR_HASH')
                expected[key] = value
        if kind == 'RELATES_TO':
            for side in ('source', 'target'):
                for suffix in ('uuid', 'name', 'group_id'):
                    key = side+'_'+suffix
                    require(key in row, 'STATE_ENDPOINT_MISSING')
                    expected[key] = row[key]
                require(row[side+'_group_id'] == group, 'STATE_ENDPOINT_NAMESPACE')
        else:
            nodes[row['uuid']] = row.get('name')
            for key in ('linked_episode_uuids', 'linked_entity_uuids'):
                if key in row:
                    require(row[key] is None or row[key] == sorted(row[key]), 'STATE_LINK_ORDER')
                    expected[key] = row[key]
        expected['properties_sha256'] = Witness._value_hash(properties)
        same({k:v for k,v in row.items() if k != 'properties'}, expected, 'STATE_PROPERTY_BINDING')
        for key in ('valid_at', 'invalid_at', 'created_at', 'expired_at'):
            if row.get(key) is not None:
                _time(row[key])
        compact.append(expected)
    for row in snapshot['rows']:
        if row['kind'] == 'RELATES_TO':
            for side in ('source', 'target'):
                require(row[side+'_uuid'] in nodes and nodes[row[side+'_uuid']] == row[side+'_name'], 'STATE_ENDPOINT_BINDING')
    compact.sort(key=lambda r: (r['kind'], str(r.get('uuid')), Witness._encoded(r)))
    same(runner.digest({'namespace': group, 'rows': compact, 'anomalies': []}), snapshot['sha256'], 'DIRECT_STATE_HASH')


def verify_mutation(value, case, position):
    event = case.events[position].public_payload
    episode = value['episode']
    require(episode['group_id'] == case.runtime_namespace and episode['name'] == event['event_id']
            and _time(episode['valid_at']) == _time(event['logical_time'])
            and episode['content'] == event['raw_text'] and type(episode['uuid']) is str, 'NATIVE_EPISODE_BINDING')
    for item in value['nodes'] + value['edges']:
        require(item['group_id'] == case.runtime_namespace, 'NATIVE_MUTATION_NAMESPACE')
        for key in ('valid_at', 'invalid_at', 'created_at', 'expired_at'):
            if item.get(key) is not None:
                _time(item[key])


def verify_snapshot_episodes(snapshot, case, prior_mutations):
    """Bind every ingested episode, including old episodes in later/query states."""
    for event, mutation in zip(case.events, prior_mutations):
        public, episode = event.public_payload, mutation['episode']
        matches = [r for r in snapshot['rows'] if r['kind'] == 'Episodic'
                   and r.get('name') == public['event_id']]
        require(len(matches) == 1, 'SNAPSHOT_EPISODE_BINDING')
        row = matches[0]
        properties = row['properties']
        require(row.get('uuid') == properties.get('uuid') == episode['uuid']
                and row.get('name') == properties.get('name') == episode['name'] == public['event_id']
                and row.get('group_id') == properties.get('group_id') == episode['group_id']
                == public['runtime_namespace'] == case.runtime_namespace
                and properties.get('content') == episode['content'] == public['raw_text'],
                'SNAPSHOT_EPISODE_BINDING')
        try:
            require(_time(row.get('valid_at')) == _time(properties.get('valid_at'))
                    == _time(episode['valid_at']) == _time(public['logical_time']), 'SNAPSHOT_EPISODE_BINDING')
        except ValueError:
            raise AuditError('SNAPSHOT_EPISODE_BINDING') from None


def verify_cases(report, cases, evidence):
    require(report['status'] in ('BLOCKED', 'OBSERVATIONS_COMPLETE') and len(report['cases']) == 3, 'TERMINAL_STATUS')
    require(tuple(c.role for c in cases) == finalizer.ROLES, 'CASE_MEMBERSHIP')
    require(report.get('scientific_release_allowed') is False and report.get('validation_tasks') == 0, 'EVIDENCE_CLASS')
    all_runs = {'native': [], 'reverse': []}
    readouts, mutations, retrievals = evidence.of('readout'), evidence.of('mutation'), evidence.of('retrieval')
    mutation_cursor = retrieval_cursor = readout_cursor = 0
    stopped = False
    for case_index, (case, row) in enumerate(zip(cases, report['cases'])):
        require(row['role'] == case.role and row['task_hash'] == case.task_hash and row['state_accuracy'] is None, 'CASE_BINDING')
        projected = runner.project_readout_task(case.task)
        if 'readout_projection' in row:
            same(row['readout_projection'], runner.projection_binding(case, projected), 'PROJECTION_BINDING')
        else:
            require(report['status'] == 'BLOCKED' and not any(report['counts'].values())
                    and all('readout_projection' not in r for r in report['cases']), 'PROJECTION_BINDING_MISSING')
        require(row['status'] in ('OBSERVED', 'BLOCKED', 'NOT_RUN'), 'CASE_STATUS')
        counts = finalizer.count_projection(row)
        for prefix, limit in (('event', 3), ('retrieval', len(case.queries)), ('readout', 3*len(case.queries))):
            attempted, completed = (counts[prefix+'_'+suffix] for suffix in ('attempts', 'completions'))
            require(0 <= completed <= attempted <= limit and attempted-completed <= 1, 'CASE_COUNTS')
        require(not stopped or row['status'] == 'NOT_RUN' and not any(counts.values()), 'CASE_EXECUTION_ORDER')
        if row['status'] != 'OBSERVED':
            stopped = True
        if row['status'] == 'NOT_RUN':
            require(not any(counts.values()) and not row['snapshots'] and not row['queries'], 'NOT_RUN_HAS_WORK')
        if row['status'] == 'OBSERVED':
            same(counts, dict(zip(runner._COUNT_KEYS, (3, 3, len(case.queries), len(case.queries),
                                                     3*len(case.queries), 3*len(case.queries)))), 'OBSERVED_COVERAGE')
        require(len(row['snapshots']) <= counts['event_completions'] and counts['event_completions']-len(row['snapshots']) <= 1,
                'SNAPSHOT_COVERAGE')
        case_mutations = mutations[mutation_cursor:mutation_cursor+counts['event_completions']]
        require(len(case_mutations) == counts['event_completions'], 'MUTATION_COVERAGE')
        for position, (index, mutation) in enumerate(case_mutations):
            evidence.used.add(index)
            verify_mutation(mutation, case, position)
        for snapshot_index, snapshot in evidence.of('state'):
            if snapshot['namespace'] == case.runtime_namespace:
                prior_mutations = [mutation for index, mutation in case_mutations if index < snapshot_index]
                verify_snapshot_episodes(snapshot, case, prior_mutations)
        mutation_cursor += len(case_mutations)
        states = []
        for position, receipt in enumerate(row['snapshots']):
            require(receipt['position'] == position, 'SNAPSHOT_POSITION')
            mi, mutation = case_mutations[position]
            require(receipt['mutation_sha256'] == runner.digest(mutation), 'MUTATION_HASH_BINDING')
            si, state = evidence.snapshot(case.runtime_namespace, receipt)
            require(si > mi, 'SNAPSHOT_BEFORE_MUTATION')
            if position+1 < len(case_mutations):
                require(si < case_mutations[position+1][0], 'SNAPSHOT_AFTER_LATER_MUTATION')
            states.append(state)
        if 'interrupted_event_snapshot' in row:
            require(counts['event_attempts'] > len(states), 'UNEXPECTED_INTERRUPTED_STATE')
            evidence.snapshot(case.runtime_namespace, row['interrupted_event_snapshot'])
        require(counts['retrieval_attempts'] == 0 or len(states) == 3, 'RETRIEVAL_BEFORE_WRITES')
        native_rows = retrievals[retrieval_cursor:retrieval_cursor+counts['retrieval_completions']]
        require(len(native_rows) == counts['retrieval_completions'], 'RETRIEVAL_COVERAGE')
        retrieval_cursor += len(native_rows)
        require(len(row['queries']) <= len(native_rows) and len(native_rows)-len(row['queries']) <= 1, 'QUERY_COVERAGE')
        requests, predictions = {o: [] for o in all_runs}, {o: [] for o in all_runs}
        available_readouts = []
        remaining = counts['readout_attempts']
        successful = 0
        for position, (ri, native) in enumerate(native_rows):
            evidence.used.add(ri)
            query = projected.queries[position]
            # Rebuilding both traces validates UUID order, score membership and all retained fields.
            for edge in native['edges']:
                for key in ('valid_at', 'invalid_at', 'created_at', 'expired_at'):
                    if edge.get(key) is not None:
                        _time(edge[key])
            runner.build_readout(query, native, namespace=case.runtime_namespace)
            if position < len(row['queries']):
                receipt = row['queries'][position]
                require(receipt['position'] == position and receipt['retrieval_sha256'] == runner.digest(native)
                        and receipt['final_snapshot_sha256'] == states[-1]['sha256'], 'QUERY_STATE_BINDING')
                matching_states = [(i, v) for i, v in evidence.of('state') if v['namespace'] == case.runtime_namespace
                                   and v['sha256'] == states[-1]['sha256']]
                require(any(i < ri for i, _ in matching_states) and any(i > ri for i, _ in matching_states), 'QUERY_STATE_REOPENING')
                for i, _ in matching_states:
                    evidence.used.add(i)
            for order in ('native', 'reverse', 'oracle'):
                if not remaining:
                    break
                request = (runner.oracle_request(projected, query) if order == 'oracle' else
                           runner.build_readout(query, native, namespace=case.runtime_namespace, order=order))
                expected = {'case': case_index, 'kind': 'readout', 'position': position, 'order': order}
                # A call can fail before a readout artifact exists; only the final attempted call may be absent.
                if readout_cursor >= len(readouts) or readouts[readout_cursor][1]['request']['query']['query_id'] != query.query_id:
                    require(row['status'] == 'BLOCKED' and remaining == 1, 'READOUT_MISSING')
                    remaining -= 1
                    break
                index, item = readouts[readout_cursor]
                readout_cursor += 1
                remaining -= 1
                evidence.used.add(index)
                require(index > ri and item['order'] == order, 'READOUT_ORDER')
                same(item['request'], request.model_dump(mode='json'), 'READOUT_PROMPT_BINDING')
                prediction = runner.parse_readout_response(request, item['response'], model_id=runner.MODEL_ID)
                same(item['prediction'], prediction.model_dump(mode='json'), 'READOUT_PREDICTION_BINDING')
                available_readouts.append((item, expected, request))
                if prediction.format_valid and item['response']['choices'][0]['finish_reason'] == 'stop':
                    successful += 1
                    if order != 'oracle':
                        requests[order].append(request)
                        predictions[order].append(prediction)
        require(remaining == 0 and successful == counts['readout_completions'], 'READOUT_COMPLETION_ACCOUNTING')
        # Recompute public-time witnesses/UUID transitions as diagnostics only. Unknown
        # affirmative surfaces are authentic data, never an integrity/accuracy PASS.
        schedule = finalizer.public_schedule(case)
        for state in states:
            for edge in state['rows']:
                if edge['kind'] == 'RELATES_TO' and edge.get('valid_at') is not None:
                    for public in case.queries:
                        match = re.match(r'As of ([^,]+), ', public.public_payload['query_text'])
                        require(match is not None, 'PUBLIC_QUERY_TIME')
                        _time_status(edge, _time(match[1]))
        try:
            aligned = [finalizer.direct_edges(s, case) for s in states]
            if len(aligned) == 3 and case.role == 'target_only_control':
                finalizer.control_interpretable(aligned, schedule)
            elif len(aligned) == 3 and case.role == 'cross_entity_same_relation':
                finalizer.cross_transitions(aligned, schedule)
        except finalizer.EvidenceError:
            pass  # Classifier, not integrity auditor, owns semantic eligibility.
        for order in all_runs:
            entries = [(i, v) for i, v in evidence.of('canonical_run') if v['task_id'] == projected.task_id
                       and v['run_id'] == f'historical-mechanism-{order}']
            if entries:
                require(len(entries) == 1, 'CANONICAL_RUN_MEMBERSHIP')
                expected = runner.build_readout_run(projected, requests[order], predictions[order],
                    run_id=f'historical-mechanism-{order}', order=order)
                same(entries[0][1], expected.model_dump(mode='json'), 'CANONICAL_RUN_BINDING')
                evidence.used.add(entries[0][0])
                all_runs[order].append(expected)
            else:
                require(row['status'] != 'OBSERVED', 'CANONICAL_RUN_MISSING')
        yield case_index, available_readouts
    require(mutation_cursor == len(mutations) and retrieval_cursor == len(retrievals) and readout_cursor == len(readouts), 'UNCLAIMED_CASE_ARTIFACT')
    same(report['counts'], {key: sum(r[key] for r in report['cases']) for key in runner._COUNT_KEYS}, 'TOTAL_COUNTS')
    if report['status'] == 'OBSERVATIONS_COMPLETE':
        require(all(r['status'] == 'OBSERVED' for r in report['cases']), 'TERMINAL_CASE_STATUS')
    verify_scores(report, cases, all_runs, evidence)


def verify_scores(report, cases, runs, evidence):
    tasks = [runner.project_readout_task(c.task) for c in cases]
    for order in ('native', 'reverse'):
        prefix = 'scoring_'+order+'_'
        entries = [(i, k, v) for i, (k, v) in enumerate(evidence.items) if k.startswith(prefix)]
        if not entries:
            require(order not in report['canonical_scores'] and report['status'] == 'BLOCKED', 'SCORE_BUNDLE_MISSING')
            continue
        require(len(runs[order]) == 3, 'INCOMPLETE_SCORE_RUNS')
        bundle = runner.build_scoring_artifacts(tasks, runs[order], code_revision=report['runtime_revision'],
                    qualification_sha256=runner.QUALIFICATION_PIN, historical_supported=True)
        values = {}
        for name in ('tasks.jsonl', 'runs.jsonl', 'task-manifest.json', 'run-manifest.json', 'scores.jsonl', 'projection'):
            _, values[name] = evidence.take(prefix+name)
        require(len(entries) == len(values), 'SCORE_BUNDLE_MEMBERSHIP')
        same(values['tasks.jsonl'], [t.model_dump(mode='json') for t in tasks], 'SCORE_TASKS_BINDING')
        same(values['runs.jsonl'], [r.model_dump(mode='json') for r in runs[order]], 'SCORE_RUNS_BINDING')
        bindings = [runner.projection_binding(c, t) for c, t in zip(cases, tasks)]
        same(values['projection'], bindings, 'SCORE_PROJECTION_BINDING')
        tm = TaskManifestV3.model_validate(values['task-manifest.json'])
        rm = RunManifestV3.model_validate(values['run-manifest.json'])
        expected_task = bundle['task_manifest'].model_dump(mode='json')
        expected_task['created_at'] = values['task-manifest.json']['created_at']
        same(values['task-manifest.json'], expected_task, 'TASK_MANIFEST_BINDING')
        expected_run = bundle['run_manifest'].model_dump(mode='json')
        expected_run['timestamp'] = values['run-manifest.json']['timestamp']
        expected_run['task_manifest']['sha256'] = sha256_model(tm)
        same(values['run-manifest.json'], expected_run, 'RUN_MANIFEST_BINDING')
        recomputed = []
        for task, record in zip(tasks, runs[order]):
            context = VerifiedScoringContextV3.from_authenticated_manifests(task=task, run=record,
                task_manifest=tm, run_manifest=rm, task_artifact=bundle['task_artifact'], run_artifact=bundle['run_artifact'],
                authenticated_task_manifest_sha256=sha256_model(tm), authenticated_run_manifest_sha256=sha256_model(rm))
            score = score_task_v3(task, record, context)
            require(all(value is None for value in score.model_dump(mode='json')['state_scores'].values()), 'STATE_ACCURACY_NOT_NULL')
            recomputed.append(score)
        same(values['scores.jsonl'], [s.model_dump(mode='json') for s in recomputed], 'SCORE_RECOMPUTATION')
        score_bytes = b''.join(canonical_json_bytes(s)+b'\n' for s in recomputed)
        same(report['canonical_scores'][order], {'projection_bindings_sha256': runner.digest(bindings),
            'projection_policy': runner.PROJECTION_POLICY, 'score_sha256': hashlib.sha256(score_bytes).hexdigest(),
            'task_manifest_sha256': sha256_model(tm), 'run_manifest_sha256': sha256_model(rm), 'tasks': 3,
            'state_alignment': 'UNAVAILABLE_NOT_INFERRED_FROM_GOLD'}, 'SCORE_RECEIPT_BINDING')


def sdk_wire_binding(decoded, wire):
    """SDK serialization may add null defaults, never alter non-null wire data."""
    if type(wire) is dict:
        require(type(decoded) is dict and set(wire).issubset(decoded), 'SDK_WIRE_SHAPE')
        require(all(decoded[key] is None for key in set(decoded)-set(wire)), 'SDK_WIRE_EXTRA_VALUE')
        for key in wire:
            sdk_wire_binding(decoded[key], wire[key])
    elif type(wire) is list:
        require(type(decoded) is list and len(decoded) == len(wire), 'SDK_WIRE_SEQUENCE')
        for left, right in zip(decoded, wire):
            sdk_wire_binding(left, right)
    else:
        same(decoded, wire, 'SDK_WIRE_VALUE')


def verify_accounting(root, report, observations, transport, evidence, readouts):
    private_obs = evidence.of('observation')
    require(len(private_obs) == len(observations), 'OBSERVATION_MEMBERSHIP')
    replay = ObservationRecorder()
    for (index, raw), row in zip(private_obs, observations):
        require(raw['observation_id'] == row['observation_id'], 'OBSERVATION_ID')
        try:
            parsed = replay.record_response(raw['response'], role=row['role'], schema_name=row['schema_name'],
                                            timestamp=row.get('timestamp'))
        except ObservationQualificationError as exc:
            parsed = exc.observation
        same(parsed, row, 'OBSERVATION_RECOMPUTATION')
        choices = raw['response'].get('choices', []) if type(raw['response']) is dict else []
        output = choices[0].get('message', {}).get('content') if len(choices) == 1 else None
        require(raw['raw_output'] == output, 'OBSERVATION_RAW_OUTPUT')
        evidence.used.add(index)
    if 'extraction_observations' in report:
        same(report['extraction_observations'], {'count': len(observations), 'sha256': runner.digest(observations),
            'blocked': sum(r['status'] == 'BLOCKED' for r in observations)}, 'EXTRACTION_RECEIPT_BINDING')
    else:
        require(not observations and report['status'] == 'BLOCKED', 'EXTRACTION_RECEIPT_MISSING')
    transport_private = evidence.of('transport')
    require(len(transport_private) == sum('response_sha256' in r for r in transport), 'TRANSPORT_MEMBERSHIP')
    matched_observations, pending, chat, embedding_rows = set(), [], [], []
    for position, row in enumerate(transport, 1):
        require(row['transport_attempt'] == position and row['kind'] in ('completions', 'embeddings')
                and row['status'] in ('OBSERVED', 'BLOCKED', 'PENDING'), 'TRANSPORT_SEQUENCE')
        op = row['operation']
        if op.get('kind') != 'startup':
            require(type(op.get('case')) is int and 0 <= op['case'] < 3 and op['kind'] in ('event', 'query', 'readout'), 'TRANSPORT_OPERATION')
            case = report['cases'][op['case']]
            prefix = {'event':'event', 'query':'retrieval', 'readout':'retrieval'}[op['kind']]
            require(type(op.get('position')) is int and 0 <= op['position'] < case[prefix+'_attempts'], 'TRANSPORT_CASE_BINDING')
        if row['kind'] == 'completions':
            chat.append(row)
        else:
            embedding_rows.append(row)
        if 'response_sha256' not in row:
            require(row['status'] == 'PENDING' and report['status'] == 'BLOCKED', 'MISSING_TRANSPORT_RESPONSE')
            pending.append(row)
            continue
        index, raw = evidence.take('transport', lambda v: v['operation'] == op
            and runner.digest(v['request']) == row['request_sha256'] and runner.digest(v['response']) == row['response_sha256'])
        request, response = raw['request'], raw['response']
        require(request['model'] == (runner.MODEL_ID if row['kind'] == 'completions' else runner.EMBEDDING_ID), 'TRANSPORT_MODEL')
        if row['kind'] == 'embeddings':
            require(row['role'] == 'graphiti', 'EMBEDDING_ROLE')
            continue
        schema = request.get('response_format', {}).get('json_schema', {})
        require(row['schema_name'] == schema.get('name') and row['schema_sha256'] == (runner.digest(schema['schema']) if 'schema' in schema else None), 'TRANSPORT_SCHEMA')
        oid = row.get('observation_id')
        require(type(oid) is int and 1 <= oid <= len(observations) and oid not in matched_observations, 'TRANSPORT_OBSERVATION_ID')
        matched_observations.add(oid)
        observation = observations[oid-1]
        raw_obs = private_obs[oid-1][1]
        same(raw_obs['response'], response, 'TRANSPORT_RESPONSE_BINDING')
        same(raw_obs['private_material'], {'operation': op, 'request': request}, 'TRANSPORT_REQUEST_BINDING')
        require(observation['role'] == row['role'] and observation['schema_name'] == row['schema_name'], 'TRANSPORT_OBSERVATION_ROLE')
        if observation['status'] == 'OBSERVED':
            require(row.get('call_id') == observation['call_id'] and row.get('usage') == observation['usage'], 'TRANSPORT_CALL_BINDING')
        expected_status = ('OBSERVED' if observation['status'] == 'OBSERVED' and row['http_status'] == 200
            and row['schema_sha256'] is not None and re.fullmatch(r'study-attempt-[1-9][0-9]*', observation['call_id'] or '') else 'BLOCKED')
        require(row['status'] == expected_status, 'TRANSPORT_STATUS')
    require(matched_observations == set(range(1, len(observations)+1)), 'UNBOUND_OBSERVATION')
    for item, op, request in readouts:
        matches = [r for r in transport if r.get('call_id') == item['response'].get('id')]
        require(len(matches) == 1 and matches[0]['operation'] == op, 'READOUT_CALL_ID_BINDING')
        row = matches[0]
        require(row['role'] == ('oracle_readout' if op['order'] == 'oracle' else 'readout'), 'READOUT_TRANSPORT_BINDING')
        raw_matches = [v for _, v in transport_private if runner.digest(v['request']) == row['request_sha256']
                       and runner.digest(v['response']) == row['response_sha256'] and v['operation'] == op]
        require(len(raw_matches) == 1, 'READOUT_RAW_TRANSPORT')
        sdk_wire_binding(item['response'], raw_matches[0]['response'])
        body = raw_matches[0]['request']
        same(body['messages'], [{'role':'user', 'content':request.rendered_prompt}], 'READOUT_TRANSPORT_PROMPT')
        same(body['response_format'], {'type':'json_schema', 'json_schema':{'name':'VersionTransferAnswer', 'schema':runner.ANSWER_SCHEMA}}, 'READOUT_TRANSPORT_SCHEMA')
    llm_path = root/'llm-calls.jsonl'
    if llm_path.exists():
        audit = runner.audit_llm_ledger(llm_path)
        ledger = ledger_rows(llm_path)
        reserved = [r for r in ledger if r['event'] == 'reserved']
        settled = {r['attempt']: r for r in ledger if r['event'] == 'settled'}
        same(report.get('llm_ledger_audit'), audit, 'LEDGER_AUDIT_BINDING')
    else:
        require(not chat and not observations and report['status'] == 'BLOCKED'
                and report.get('llm_ledger_audit') is None, 'LLM_LEDGER_MISSING')
        reserved, settled, audit = [], {}, None
    seen = set()
    unknown = [r for r in chat if not re.fullmatch(r'study-attempt-[1-9][0-9]*', r.get('call_id') or '')]
    for reservation in reserved:
        ticket = reservation['attempt']
        require(reservation['reserved_output_tokens'] <= 2048 and reservation['elapsed_seconds'] >= 0, 'LEDGER_RESERVATION')
        matches = [o for o in observations if o.get('call_id') == f'study-attempt-{ticket}']
        if matches:
            require(len(matches) == 1 and ticket in settled, 'RESPONSE_WITHOUT_SETTLEMENT')
            observation = matches[0]
            require(observation['role'] == reservation['role'] and observation['usage'] is not None
                    and observation['usage']['prompt_tokens'] == reservation['input_tokens']
                    and observation['usage']['completion_tokens'] == settled[ticket]['completion_tokens']
                    and observation['finish_reason'] == settled[ticket]['finish_reason'], 'LEDGER_USAGE_BINDING')
            seen.add(observation['observation_id'])
        else:
            require(report['status'] == 'BLOCKED' and unknown, 'RESERVATION_WITHOUT_TRANSPORT')
            candidate = unknown.pop(0)
            require(candidate['role'] == reservation['role'] and (ticket not in settled or settled[ticket]['finish_reason'] == 'error'), 'UNSETTLED_TRANSPORT_BINDING')
    for row in observations:
        require(row['observation_id'] in seen or row['status'] == 'BLOCKED'
                and not re.fullmatch(r'study-attempt-[1-9][0-9]*', row.get('call_id') or ''), 'OBSERVATION_WITHOUT_RESERVATION')
    embedding_path = root/'embedding-calls.jsonl'
    embeddings = ledger_rows(embedding_path) if embedding_path.exists() else []
    require(embedding_path.exists() or not embedding_rows and report['status'] == 'BLOCKED', 'EMBEDDING_LEDGER_MISSING')
    a, b = [], {}
    for row in embeddings:
        if row['event'] == 'embedding_reserved':
            require(row['attempt'] == len(a)+1 and row['role'] == 'graphiti'
                    and all(type(row[k]) is int and row[k] > 0 for k in ('items', 'input_tokens')), 'EMBEDDING_RESERVATION')
            a.append(row)
        else:
            require(row['event'] == 'embedding_settled' and row['attempt'] not in b
                    and 1 <= row['attempt'] <= len(a) and row['elapsed_seconds'] >= 0, 'EMBEDDING_SETTLEMENT')
            b[row['attempt']] = row
    require(len(a) <= len(embedding_rows), 'EMBEDDING_ATTEMPT_MEMBERSHIP')
    for reservation, row in zip(a, embedding_rows):
        matches = [v for _, v in transport_private if v['operation'] == row['operation']
                   and runner.digest(v['request']) == row['request_sha256'] and 'input' in v['request']]
        if not matches:
            require(row['status'] == 'PENDING' and report['status'] == 'BLOCKED', 'EMBEDDING_PRIVATE_MISSING')
            continue
        require(len(matches) == 1, 'EMBEDDING_PRIVATE_AMBIGUOUS')
        raw = matches[0]
        texts = raw['request']['input']
        texts = [texts] if type(texts) is str else texts
        require(len(texts) == reservation['items'] and hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
                == reservation['input_sha256'], 'EMBEDDING_INPUT_BINDING')
        if row['status'] == 'OBSERVED':
            response = raw['response']
            require(reservation['attempt'] in b and response['model'] == runner.EMBEDDING_ID
                    and response['usage'] == {'prompt_tokens': reservation['input_tokens'], 'total_tokens': reservation['input_tokens']}
                    and [v['index'] for v in response['data']] == list(range(reservation['items'])), 'EMBEDDING_USAGE_BINDING')
    if report.get('accounting') is not None:
        require(audit is not None and audit['all_settled'] and len(a) == len(b) == len(embedding_rows)
                and not pending and all(r['status'] == 'OBSERVED' for r in transport), 'ACCOUNTING_FALSE_SUCCESS')
        expected = {'llm': audit, 'aggregate_charged_llm': {k: runner.PRIOR_TOTALS[k]+audit['charged'][k] for k in runner.PRIOR_TOTALS},
            'embeddings': {'attempts': len(a), 'items': sum(r['items'] for r in a), 'input_tokens': sum(r['input_tokens'] for r in a)},
            'native_sdk_retry_policy': 'qualified_native_tenacity_up_to_4_preserved',
            'openai_transport_retries': 0, 'actual_http_attempts': len(transport)}
        same(report['accounting'], expected, 'ACCOUNTING_TOTAL_BINDING')
    else:
        require(report['status'] == 'BLOCKED', 'ACCOUNTING_MISSING')


def verify_state_stream(report, cases, evidence):
    """Match actual snapshot multiplicity and position, not just reusable hashes."""
    if not report['empty_state_preflight']:
        require(not any(r['event_attempts'] for r in report['cases']), 'STATE_PREFLIGHT_MISSING')
        states = evidence.of('state')
        require(len(states) <= 3 and [s['namespace'] for _, s in states]
                == [c.runtime_namespace for c in cases[:len(states)]], 'STARTUP_STATE_MEMBERSHIP')
        return
    retrievals = iter(evidence.of('retrieval'))
    for case, row in zip(cases, report['cases']):
        states = [(i, s) for i, s in evidence.of('state') if s['namespace'] == case.runtime_namespace]
        require(states and not states[0][1]['rows'], 'PREFLIGHT_STATE_MISSING')
        cursor = 1
        for receipt in row['snapshots']:
            require(cursor < len(states) and states[cursor][1]['sha256'] == receipt['sha256'], 'WRITE_STATE_STREAM')
            cursor += 1
        for position in range(row['retrieval_completions']):
            ri, _ = next(retrievals)
            if position < len(row['queries']):
                require(cursor+1 < len(states), 'QUERY_STATE_PAIR_MISSING')
                before, after = states[cursor:cursor+2]
                pin = row['queries'][position]['final_snapshot_sha256']
                require(before[0] < ri < after[0] and before[1]['sha256'] == after[1]['sha256'] == pin, 'QUERY_STATE_PAIR_BINDING')
                readers = [i for i, v in evidence.of('readout') if v['request']['query']['query_id']
                           == case.task.queries[position].query_id]
                require(not readers or after[0] < min(readers), 'QUERY_STATE_AFTER_READOUT')
                cursor += 2
        if 'interrupted_event_snapshot' in row:
            require(cursor < len(states) and states[cursor][1]['sha256'] == row['interrupted_event_snapshot']['sha256'], 'INTERRUPTED_STATE_STREAM')
            cursor += 1
        remaining = len(states)-cursor
        if remaining:
            # The only unreceipted snapshots are a failed write's snapshot or the
            # before/after witnesses of the active failed retrieval. No fillers.
            active_failure = row['status'] == 'BLOCKED' and ('error_type' in row or 'interrupted_state_error_type' in row)
            unreceipted_query = row['retrieval_attempts'] > len(row['queries'])
            unreceipted_write = row['event_completions'] > len(row['snapshots'])
            changed_before_query = (row['event_completions'] == 3 and row['retrieval_attempts'] < len(case.queries)
                and states[-1][1]['sha256'] != row['snapshots'][-1]['sha256'])
            require(active_failure and (unreceipted_query and remaining <= 2 or unreceipted_write and remaining == 1
                    or changed_before_query and remaining == 1), 'UNCLAIMED_STATE_SNAPSHOT')


def verify_cleanup(report, cases, evidence):
    runtime_recorded = any('readout_projection' in r for r in report['cases']) or bool(report['empty_state_preflight'])
    if runtime_recorded:
        require(type(report.get('client_cleanup_errors')) is list
                and all(type(v) is str and re.fullmatch('[A-Za-z][A-Za-z0-9_]*', v) for v in report['client_cleanup_errors']),
                'CLIENT_CLEANUP_MISSING')
    for case, row in zip(cases, report['cases']):
        require('namespace_empty_after_cleanup' in row and (row['namespace_empty_after_cleanup'] is None
                or type(row['namespace_empty_after_cleanup']) is bool), 'CLEANUP_STATUS_MISSING')
        if row['status'] == 'OBSERVED':
            require(row['namespace_empty_after_cleanup'] is True, 'OBSERVED_CLEANUP_INCONSISTENT')
    require('owned_process_cleanup' in report and type(report['owned_process_cleanup']) is list
            and all(type(v) is bool for v in report['owned_process_cleanup']), 'PROCESS_CLEANUP_MISSING')
    require(set(report['ports_closed']) == {'17687', '18043'} and all(type(v) is bool for v in report['ports_closed'].values()), 'PORT_CLEANUP_MISSING')
    require('owned_gpu_process_absent' in report and (report['owned_gpu_process_absent'] is None
            or type(report['owned_gpu_process_absent']) is bool), 'GPU_CLEANUP_MISSING')
    if report['status'] == 'OBSERVATIONS_COMPLETE':
        require(report.get('client_cleanup_errors') == [] and report['owned_process_cleanup']
                and all(report['owned_process_cleanup']) and all(report['ports_closed'].values())
                and report['owned_gpu_process_absent'] is True, 'COMPLETE_CLEANUP_INCONSISTENT')
    preflight = report['empty_state_preflight']
    require(len(preflight) <= 3 and [p['role'] for p in preflight] == [c.role for c in cases[:len(preflight)]], 'PREFLIGHT_MEMBERSHIP')
    for receipt, case in zip(preflight, cases):
        require(receipt['empty'] is True and receipt['rows'] == 0
                and receipt['evidence_class'] == 'empty_state_runtime_preflight_not_native_state', 'PREFLIGHT_BINDING')
        evidence.snapshot(case.runtime_namespace, receipt)
    if any(r['event_attempts'] for r in report['cases']):
        require(len(preflight) == 3, 'PREFLIGHT_MISSING')


def _audit(*, run_root, terminal_sha256, packet_root, source_manifest, source_manifest_sha256,
           runtime_revision, output_path, source_root):
    root, output = Path(run_root).resolve(strict=True), Path(output_path).absolute()
    assert_no_reparse_components(root)
    assert_no_reparse_components(output)
    require(output.parent.resolve(strict=True) == root.parent and not output.is_relative_to(root)
            and not output.exists() and not output.is_symlink(), 'OUTPUT_MUST_BE_NEW_SIBLING')
    source = runner.verify_runtime_source(source_manifest, source_manifest_sha256, runtime_revision, root=source_root)
    _, cases = runner.load_cases(packet_root, runner.PACKET_PIN)
    report = read_json(root/'terminal.json', terminal_sha256)
    require(report['runtime_revision'] == runtime_revision and report['runtime_source_manifest_sha256'] == source_manifest_sha256
            and report['runtime_source_map_sha256'] == runner.digest(source['files'])
            and report['packet_index_sha256'] == runner.PACKET_PIN and report['packet_source_revision'] == runner.PACKET_SOURCE, 'TERMINAL_SOURCE_BINDING')
    evidence = Evidence(root, report)
    for _, state in evidence.of('state'):
        verify_snapshot(state, {c.runtime_namespace for c in cases})
    expected_compact = {kind+'.json' for kind in ('observations', 'transport') if report.get(kind+'_sha256') is not None}
    require((root/'compact').is_dir() and {p.name for p in (root/'compact').iterdir()} == expected_compact, 'COMPACT_MEMBERSHIP')
    compact = {}
    for kind in ('observations', 'transport'):
        pin, path = report.get(kind+'_sha256'), root/f'compact/{kind}.json'
        if pin is None:
            require(not path.exists() and report['status'] == 'BLOCKED', 'MISSING_COMPACT_PIN')
            compact[kind] = []
        else:
            compact[kind] = read_json(path, pin)
            require(type(compact[kind]) is list, 'COMPACT_SHAPE')
    readouts = []
    for _, available in verify_cases(report, cases, evidence):
        readouts.extend(available)
    verify_accounting(root, report, compact['observations'], compact['transport'], evidence, readouts)
    verify_cleanup(report, cases, evidence)
    verify_state_stream(report, cases, evidence)
    # Reopened state occurrences include guarded startup/interruption witnesses;
    # the stream check above preserves them without promoting failed writes.
    for i, _ in evidence.of('state'):
        evidence.used.add(i)
    require(evidence.used == set(range(len(evidence.items))), 'UNCLAIMED_PRIVATE_PAYLOAD')
    bindings = dict(zip(finalizer.BINDING_KEYS, (terminal_sha256, runner.PACKET_PIN, source_manifest_sha256,
        report['private_artifacts']['index_sha256'], report.get('observations_sha256'), report.get('transport_sha256'))))
    receipt = {'schema': finalizer.AUDIT_SCHEMA, 'status': 'PASS', 'bindings': bindings,
               'checks': dict.fromkeys(finalizer.AUDIT_CHECKS, True), 'cases': finalizer.audit_projection(report)}
    payload = finalizer.encoded(receipt)
    with output.open('xb') as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def audit(*, run_root, terminal_sha256, packet_root, source_manifest, source_manifest_sha256,
          runtime_revision, output_path, source_root=ROOT):
    """Write a new sibling receipt; return its raw-byte SHA. Never repair inputs."""
    try:
        return _audit(run_root=run_root, terminal_sha256=terminal_sha256, packet_root=packet_root,
            source_manifest=source_manifest, source_manifest_sha256=source_manifest_sha256,
            runtime_revision=runtime_revision, output_path=output_path, source_root=source_root)
    except AuditError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise AuditError('EVIDENCE_VERIFICATION_REFUSED') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('run-root', 'packet-root', 'source-manifest', 'output-path'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    for name in ('terminal-sha256', 'source-manifest-sha256', 'runtime-revision'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    try:
        pin = audit(**vars(args))
    except AuditError as exc:
        print(json.dumps({'status':'FAIL', 'error_code':str(exc)}))
        return 1
    print(json.dumps({'status':'PASS', 'audit_sha256':pin}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
