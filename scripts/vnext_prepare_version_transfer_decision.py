"""Prepare a bounded version-transfer study using canonical v3 tasks; no inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.io import canonical_json_bytes
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3

CONFIG_PATH = ROOT / 'configs/vnext/version_transfer_decision_v1.json'
SOURCE_FILES = (
    'scripts/vnext_prepare_version_transfer_decision.py',
    'mub/vnext/contracts/v3/task.py',
    'mub/vnext/contracts/v3/common.py',
    'mub/vnext/validation/replay_v3.py',
    'mub/vnext/external/visibility.py',
)
# Distinct changing predicates, not renamings counted as independent samples.
CORES = (
    ('profile', 'residence_city', 'residence city', 'city'),
    ('profile', 'employer', 'employer', 'company'),
    ('profile', 'preferred_editor', 'preferred code editor', 'editor'),
    ('profile', 'commute_mode', 'commute mode', 'transport'),
    ('profile', 'preferred_drink', 'preferred drink', 'drink'),
    ('profile', 'training_location', 'training location', 'venue'),
    ('profile', 'delivery_address', 'delivery address', 'address'),
    ('profile', 'contact_channel', 'preferred contact channel', 'channel'),
    ('project', 'deployment_region', 'deployment region', 'region'),
    ('project', 'database_engine', 'database engine', 'database'),
    ('project', 'release_channel', 'release channel', 'release'),
    ('project', 'build_runner', 'build runner', 'runner'),
    ('project', 'artifact_bucket', 'artifact bucket', 'bucket'),
    ('project', 'log_sink', 'log destination', 'sink'),
    ('project', 'service_owner', 'service owner team', 'team'),
    ('project', 'alert_route', 'alert routing destination', 'route'),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _time(position):
    return (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=position)).isoformat().replace('+00:00', 'Z')


def _task(core_index, depth, code_revision, *, positions=None, repeated=False, smoke_case=None):
    config = json.loads(CONFIG_PATH.read_bytes())
    required = {'core_count': 16, 'dev_cores': 8, 'validation_cores': 8,
                'depths': [1, 4, 16], 'events_per_task': 17, 'history_event_position': 8}
    if any(config.get(name) != value for name, value in required.items()):
        raise ValueError('config disagrees with the frozen study design')
    config_hash = digest(canonical(config))
    domain, attribute, label, value_kind = CORES[core_index]
    entity = f'{"Person" if domain == "profile" else "Project"}-{core_index:02d}'
    # Depth conditions use the same value at a given public time and the same final value.
    size = 17 if positions is None else max(positions) + 1
    updates = set(range(17 - depth, 17)) if positions is None else set(positions[1:])
    target_positions = [0, *sorted(updates)]
    task_id = f'vtd-core-{core_index:02d}-u{depth}' + (f'-{smoke_case}' if smoke_case else '')
    target = {'namespace': 'version-transfer', 'entity': entity,
              'attribute': attribute, 'subkey': None, 'object_type': 'slot'}
    events, actions, ledgers = [], [], {}
    keys = {}
    for position in range(size):
        is_target = position in target_positions
        key = target if is_target else {
            **target, 'entity': f'{"Person" if domain == "profile" else "Project"}-{core_index + 16:02d}'
        }
        name = key['entity']
        value = f'{value_kind}-value-{0 if repeated and is_target else position:02d}'
        event_id, action_id = f'{task_id}-e{position:02d}', f'{task_id}-a{position:02d}'
        text = (f'Effective {_time(position)}, the {label} for {name} is {value}. '
                'This single-valued setting replaces its earlier setting from that time onward.')
        identity = key['entity']
        previous = ledgers.setdefault(identity, [])
        operation = 'UPDATE' if previous else 'ADD'
        if previous:
            previous[-1]['valid_until_event_id'] = event_id
        previous.append({'version_index': len(previous), 'status': 'present', 'value': value,
                         'valid_from_event_id': event_id, 'logical_time': _time(position),
                         'source_event_ids': [event_id]})
        keys[identity] = key
        events.append({'event_id': event_id, 'sequence_index': position,
                       'timestamp': _time(position), 'raw_text': text, 'normalized_text': text,
                       'speaker': 'user', 'role': 'neutral', 'gold_action_ids': [action_id]})
        actions.append({'action_id': action_id, 'event_id': event_id, 'operation': operation,
                        'scope': 'object', 'target_object_keys': [key], 'value': value,
                        'effective_at': _time(position)})
    query_positions = (size - 1, min(config['history_event_position'], size - 2))
    queries, evidence = [], []
    for kind, position in zip(('current', 'history'), query_positions):
        selected = max(p for p in target_positions if p <= position)
        event_id = f'{task_id}-e{selected:02d}'
        query_id = f'{task_id}-{kind}'
        selector = {'kind': 'current'} if kind == 'current' else {'kind': 'logical_time_anchor', 'logical_time': _time(position)}
        queries.append({'query_id': query_id, 'query_type': 'current' if kind == 'current' else 'point_in_time',
                        'text': f'As of {_time(position)}, what is the {label} for {entity}? Return the setting value only.',
                        'selector': selector, 'target_object_keys': [target], 'answer_schema': 'string',
                        'evaluation_mode': 'retrieved_prompt'})
        evidence.append({'query_id': query_id,
                         'answer': f'{value_kind}-value-{0 if repeated else selected:02d}',
                         'supporting_object_keys': [target], 'supporting_event_ids': [event_id],
                         'derivation_steps': [{'step_id': 'read', 'operation': 'read',
                                               'supporting_object_keys': [target],
                                               'supporting_event_ids': [event_id]}],
                         'final_derivation_step_id': 'read'})
    raw_hash = digest(canonical(events))
    split = 'dev' if core_index % 8 < 4 else 'test'
    core_id = f'vtd-core-{core_index:02d}'
    extra = {'update_count': depth, 'domain': domain, 'event_budget': size}
    if smoke_case:
        extra['smoke_case'] = smoke_case
    task = MemUpdateTaskV3.model_validate({
        'task_id': task_id, 'task_family': 'F', 'difficulty': 'hard',
        'source': {'source_id': core_id, 'source_type': 'synthetic', 'source_uri': None,
                   'license_or_privacy': 'synthetic controlled diagnostic, no real personal data',
                   'raw_hash': raw_hash, 'normalized_hash': raw_hash, 'normalization_version': 'vtd-v1',
                   'generator': {'generator_name': 'version-transfer-decision', 'seed': core_index,
                                 'config_sha256': config_hash, 'code_revision': code_revision,
                                 'compiler_version': 'vtd-v1'}},
        'events': events, 'target_objects': list(keys.values()), 'actions': actions,
        'queries': queries, 'version_history': [{'object_key': keys[k], 'entries': v} for k, v in ledgers.items()],
        'gold_evidence': evidence,
        'metadata': {'split': split, 'split_key': {'semantic_core_id': core_id, 'source_group_id': core_id,
                    'trajectory_id': core_id, 'split_policy_version': 'vtd-v1'},
                     'profile_name': 'hard', 'generation_config_hash': config_hash,
                     'compiler_version': 'vtd-v1', 'extra': extra},
    })
    replay = replay_task_v3(task)
    if replay.issues:
        raise ValueError(f'replay failed: {replay.issues}')
    for query, gold in zip(task.queries, task.gold_evidence):
        result = resolve_query_v3(query, replay, task.events)
        if result.issues or result.answer != gold.answer:
            raise ValueError(f'gold query disagrees with replay: {query.query_id}')
    return task


def build_tasks(*, code_revision):
    return tuple(_task(index, depth, code_revision)
                 for index in range(len(CORES)) for depth in (1, 4, 16))


def build_smoke(*, code_revision):
    return tuple(_task(i, 2, code_revision, positions=[0, 2, 4],
                       repeated=case == 'repeated_value', smoke_case=case)
                 for i, case in enumerate(('replacement', 'repeated_value', 'history', 'similar_entity')))


def publish(output_root, *, code_revision):
    output = Path(output_root).absolute()
    assert_no_reparse_components(output)
    protected = ROOT / 'data'
    results = ROOT / 'results'
    if output == protected or protected in output.parents:
        raise ValueError('protected data roots cannot receive study preparation')
    if output == results or results in output.parents:
        relative = output.relative_to(results)
        if len(relative.parts) != 2 or relative.parts[0] != 'vnext' or not relative.parts[1].startswith('version_transfer_decision_'):
            raise ValueError('protected result roots cannot receive study preparation')
    if output.exists():
        raise FileExistsError(output)
    tasks, smoke = build_tasks(code_revision=code_revision), build_smoke(code_revision=code_revision)
    artifacts = {
        'tasks.jsonl': b''.join(canonical_json_bytes(t) + b'\n' for t in tasks),
        'smoke.jsonl': b''.join(canonical_json_bytes(t) + b'\n' for t in smoke),
        'config.json': CONFIG_PATH.read_bytes(),
    }
    manifest = {'schema': 'memupdatebench.version-transfer-preparation.v1',
                'status': 'PREPARED_NOT_EXECUTED', 'scientific_release_allowed': False,
                'code_revision': code_revision, 'task_count': len(tasks), 'core_count': 16,
                'independent_sample_claim': '16_declared_synthetic_trajectories_not_population_independence',
                'tokenizer_length_validation': 'PENDING_MODEL_QUALIFICATION',
                'source_hash_policy': 'python_lf_normalized',
                'source_hashes': {name: digest((ROOT / name).read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')) for name in SOURCE_FILES},
                'smoke_count': len(smoke), 'model_calls': 0,
                'task_hashes': {t.task_id: digest(canonical_json_bytes(t)) for t in tasks},
                'semantic_hashes': {t.task_id: t.semantic_hash for t in tasks}}
    artifacts['manifest.json'] = canonical(manifest)
    index = {'schema': 'memupdatebench.version-transfer-index.v1',
             'artifacts': {name: {'sha256': digest(raw), 'bytes': len(raw)} for name, raw in artifacts.items()}}
    index_raw = canonical(index)
    output.mkdir(parents=True, exist_ok=False)
    publish_files_atomically({output / name: raw for name, raw in {**artifacts, 'index.json': index_raw}.items()}, overwrite=False)
    return {'index_sha256': digest(index_raw), 'status': 'PREPARED_NOT_EXECUTED', 'task_count': 48}


def read_packet(root, index_sha256):
    root = Path(root).absolute()
    assert_no_reparse_components(root)
    raw = (root / 'index.json').read_bytes()
    if digest(raw) != index_sha256:
        raise ValueError('index hash mismatch')
    index = json.loads(raw)
    if set(index['artifacts']) != {'tasks.jsonl', 'smoke.jsonl', 'config.json', 'manifest.json'}:
        raise ValueError('artifact membership mismatch')
    if {p.name for p in root.iterdir()} != {*index['artifacts'], 'index.json'}:
        raise ValueError('artifact membership mismatch')
    for name, ref in index['artifacts'].items():
        assert_no_reparse_components(root / name)
        content = (root / name).read_bytes()
        if len(content) != ref['bytes'] or digest(content) != ref['sha256']:
            raise ValueError(f'artifact hash mismatch: {name}')
    tasks = tuple(MemUpdateTaskV3.model_validate_json(line) for line in (root / 'tasks.jsonl').read_bytes().splitlines())
    manifest = json.loads((root / 'manifest.json').read_bytes())
    if {t.task_id: digest(canonical_json_bytes(t)) for t in tasks} != manifest['task_hashes']:
        raise ValueError('task record binding mismatch')
    for task in tasks:
        replay = replay_task_v3(task)
        if replay.issues:
            raise ValueError('task replay failed')
        for query, gold in zip(task.queries, task.gold_evidence):
            resolved = resolve_query_v3(query, replay, task.events)
            if resolved.issues or resolved.answer != gold.answer:
                raise ValueError('task query binding failed')
    smoke = tuple(MemUpdateTaskV3.model_validate_json(line) for line in (root / 'smoke.jsonl').read_bytes().splitlines())
    if len(tasks) != manifest['task_count'] or len(smoke) != manifest['smoke_count']:
        raise ValueError('artifact record count mismatch')
    if any(replay_task_v3(task).issues for task in smoke):
        raise ValueError('smoke replay failed')
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(publish(args.output_root, code_revision=args.code_revision), sort_keys=True))


if __name__ == '__main__':
    main()
