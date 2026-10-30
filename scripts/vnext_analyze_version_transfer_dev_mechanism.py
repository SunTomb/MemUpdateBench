"""Hash-bound offline diagnosis of saved native history-query observations.

Never opens a graph/model, reconstructs a final store, or promotes the prior
INCONCLUSIVE decision. Output contains hashes/IDs/counts, not native fact text.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_local = runpy.run_path(str(Path(__file__).with_name('analyze_version_transfer_mechanism.py')))
analyze_target_lifecycle = _local['analyze_target_lifecycle']


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _no_symlinks(path):
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('artifact path contains traversal or symlink')
    return path


def _read_pinned(path, expected):
    if type(expected) is not str or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('explicit lowercase SHA-256 pin required')
    path = _no_symlinks(path)
    raw = path.read_bytes()
    if _hash(raw) != expected:
        raise ValueError(f'artifact hash mismatch: {path.name}')
    return json.loads(raw)


def _indexed_member(root, name, expected):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('unsafe indexed artifact member')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('indexed artifact escapes root')
    return _read_pinned(path, expected)


def build_report(rows):
    rows = list(rows)
    aggregate = collections.Counter()
    witnesses = collections.Counter()
    same_fact = collections.Counter()
    for row in rows:
        depth = row['depth']
        history = row['history']
        transitions = row['target_invalidations']
        aggregate[(depth, 'queries')] += 1
        aggregate[(depth, 'observed_non_target_invalidation_transition')] += int(bool(row['non_target_event_target_invalidations']))
        aggregate[(depth, 'non_target_transition_excluding_query_candidate')] += int(any(
            t['at_non_target_event'] and t['interval_covered_query_before_change'] is True
            and t['interval_covers_query_after_change'] is False and t['mentions_gold_value'] is True
            for t in transitions))
        aggregate[(depth, 'empty_retrieval')] += int(history['total_edges_returned'] == 0)
        aggregate[(depth, 'no_target_mention_returned')] += int(history['target_edges_returned'] == 0)
        aggregate[(depth, 'time_valid_target_mention_returned')] += int(history['target_time_valid_edges_returned'] > 0)
        candidates = row['candidate_observations']
        aggregate[(depth, 'last_seen_compatible_gold_candidate_not_returned')] += int(bool(candidates['candidate_ids_absent_from_returned_set']))
        aggregate[(depth, 'time_valid_gold_candidate_returned')] += int(bool(candidates['time_compatible_gold_candidate_ids_returned']))
        if 'answer' in row:
            aggregate[(depth, 'historical_answer_correct')] += int(row['answer']['correct'])
            aggregate[(depth, 'historical_abstained')] += int(row['answer']['disposition'] == 'abstained')
        for transition in row['target_invalidations']:
            if transition['at_non_target_event']:
                witnesses[transition['target_endpoint_witness']] += 1
                same_fact[str(transition['same_fact_text'])] += 1
    return {
        'schema': 'memupdatebench.version-transfer-mechanism-diagnostic.v1',
        'status': 'OFFLINE_ARTIFACT_DIAGNOSTIC_ONLY', 'model_calls': 0, 'database_calls': 0,
        'scientific_decision': 'PRIOR_INCONCLUSIVE_UNCHANGED',
        'final_state_coverage': 'UNAVAILABLE_MUTATION_OUTPUTS_ONLY',
        'identity_scope': 'lexical_synthetic_identifiers_not_general_semantic_grounding',
        'interpretation_limits': [
            'An invalidation transition is observed across returned mutations, not a complete final-state dump.',
            'A candidate absent from retrieval is not proof that it persisted or was deleted.',
            'Text and endpoint witnesses do not uniquely identify whether extraction, resolution or storage caused an error.',
            'Historical findings are exploratory and cannot replace the pre-specified current-value primary endpoint.',
            'No new inference, validation trajectory or causal intervention was executed.',
        ],
        'rows': rows,
        'aggregate': {'counts': {f'u{depth}/{kind}': value for (depth, kind), value in sorted(aggregate.items())},
                     'non_target_transition_endpoint_witnesses': dict(sorted(witnesses.items())),
                     'non_target_transition_same_fact': dict(sorted(same_fact.items()))},    }


def _identity(key):
    return key['namespace'], key['entity'], key['attribute'], key.get('subkey')


def analyze(run_root, packet_root, *, terminal_sha256, packet_index_sha256, audit_path, audit_sha256):
    run_root, packet_root = _no_symlinks(run_root), _no_symlinks(packet_root)
    terminal = _read_pinned(run_root/'terminal.json', terminal_sha256)
    packet = _read_pinned(packet_root/'index.json', packet_index_sha256)
    audit = _read_pinned(audit_path, audit_sha256)
    if terminal.get('status') != 'OBSERVATIONS_COMPLETE' or terminal.get('validation_tasks') != 0:
        raise ValueError('only completed development evidence is admissible')
    if (audit.get('status') != 'AUDITED_DEVELOPMENT_OBSERVATIONS'
            or audit.get('terminal_sha256') != terminal_sha256
            or audit.get('packet_index_sha256') != packet_index_sha256):
        raise ValueError('independent audit binding mismatch')
    # Validate every saved input before deriving any new observation.
    for name, pin in terminal['index'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('unsafe indexed artifact member')
        path = _no_symlinks(run_root/relative)
        if not path.resolve().is_relative_to(run_root.resolve()) or _hash(path.read_bytes()) != pin:
            raise ValueError('terminal artifact hash mismatch')
    if {p.name for p in packet_root.iterdir()} != {*packet['artifacts'], 'index.json'}:
        raise ValueError('packet artifact membership mismatch')
    for name, ref in packet['artifacts'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('unsafe packet artifact member')
        raw = _no_symlinks(packet_root/relative).read_bytes()
        if _hash(raw) != ref['sha256'] or len(raw) != ref['bytes']:
            raise ValueError('packet artifact hash mismatch')
    all_tasks = [json.loads(line) for line in (packet_root/'tasks.jsonl').read_bytes().splitlines()]
    tasks = {row['task_id']:row for row in all_tasks if row['metadata']['split'] == 'dev'}
    if len({row['task_id'] for row in all_tasks}) != len(all_tasks):
        raise ValueError('duplicate task IDs')
    cases = _indexed_member(run_root, 'cases.json', terminal['index']['cases.json'])
    if cases != terminal['cases'] or len(cases) != len(tasks) or {c['task_id'] for c in cases} != set(tasks):
        raise ValueError('development case coverage mismatch')
    audited = [row for row in audit['query_rows'] if row['order'] == 'native' and row['query_kind'] == 'point_in_time']
    answers = {row['task_id']:row for row in audited}
    if len(answers) != len(audited) or set(answers) != set(tasks):
        raise ValueError('audited history answer coverage mismatch')
    rows = []
    for index, case in enumerate(cases):
        task = tasks[case['task_id']]
        queries = [query for query in task['queries'] if query['query_type'] == 'point_in_time']
        if len(queries) != 1 or queries[0]['selector']['kind'] != 'logical_time_anchor':
            raise ValueError('one public logical-time history query required')
        query = queries[0]
        query_position = task['queries'].index(query)
        if len(query['target_object_keys']) != 1:
            raise ValueError('one historical target required')
        target = query['target_object_keys'][0]
        event_positions = {event['event_id']:i for i,event in enumerate(task['events'])}
        if len(event_positions) != len(task['events']):
            raise ValueError('duplicate event IDs')
        positions = {event_positions[action['event_id']] for action in task['actions']
                     if any(_identity(key) == _identity(target) for key in action['target_object_keys'])}
        def read_run(name):
            return _indexed_member(run_root, name, terminal['index'][name])
        events = [read_run(f'task_{index:02d}/event_{i:02d}.json') for i in range(len(task['events']))]
        native = read_run(f'task_{index:02d}/retrieval_{query_position}.json')
        request = read_run(f'task_{index:02d}/native_request_{query_position}.json')
        prediction = read_run(f'task_{index:02d}/native_prediction_{query_position}.json')
        if request['query'] != query or prediction['query_id'] != query['query_id']:
            raise ValueError('history request/query mismatch')
        when = query['selector']['logical_time']
        if not query['text'].startswith(f'As of {when}, '):
            raise ValueError('time anchor is not public in query')
        expected_group = f'vtd_dev_{run_root.name}_{index}'
        for event, observed in zip(task['events'], events):
            episode = observed['episode']
            if episode['name'] != event['event_id'] or episode['content'] != event['raw_text'] or episode['group_id'] != expected_group:
                raise ValueError('event input binding mismatch')
            if any(e['group_id'] != expected_group for e in observed['edges']):
                raise ValueError('event edge namespace mismatch')
        if any(e['group_id'] != expected_group for e in native['edges']):
            raise ValueError('retrieval namespace mismatch')
        golds = [gold for gold in task['gold_evidence'] if gold['query_id'] == query['query_id']]
        if len(golds) != 1 or answers[task['task_id']]['gold'] != golds[0]['answer']:
            raise ValueError('gold/audit binding mismatch')
        result = analyze_target_lifecycle(target_entity=target['entity'], target_update_positions=positions,
            event_results=events, history_result=native, history_time=when, gold_value=golds[0]['answer'])
        answer = answers[task['task_id']]
        if prediction['parsed_answer'] != answer['answer'] or prediction['disposition'] != answer['disposition']:
            raise ValueError('saved answer/audit mismatch')
        rows.append({'task_id':case['task_id'],'core_id':case['core_id'],'depth':case['update_count'],
                     'answer':{k:answer[k] for k in ('correct','disposition','format_valid')}, **result})
    report = build_report(rows)
    report['input_bindings'] = {'terminal_sha256':terminal_sha256,'packet_index_sha256':packet_index_sha256,
                               'independent_audit_sha256':audit_sha256,'indexed_artifacts':len(terminal['index'])}
    report['analyzer_source_sha256'] = {p.name:_hash(p.read_bytes()) for p in (
        Path(__file__), Path(__file__).with_name('analyze_version_transfer_mechanism.py'))}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--packet-root', type=Path, required=True)
    parser.add_argument('--terminal-sha256', required=True)
    parser.add_argument('--packet-index-sha256', required=True)
    parser.add_argument('--audit-path', type=Path, required=True)
    parser.add_argument('--audit-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = _no_symlinks(args.output)
    for source in (args.run_root, args.packet_root, args.audit_path.parent):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('output must not be inside any immutable input root')
    if output.exists():
        raise FileExistsError(output)
    result = analyze(args.run_root,args.packet_root,terminal_sha256=args.terminal_sha256,
                     packet_index_sha256=args.packet_index_sha256,audit_path=args.audit_path,
                     audit_sha256=args.audit_sha256)
    raw = (json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'status':result['status'],'rows':len(result['rows']),
                      'aggregate':result['aggregate'],'report_sha256':_hash(raw)},sort_keys=True))


if __name__ == '__main__':
    main()
