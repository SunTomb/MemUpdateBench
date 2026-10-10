"""Prepare synthetic native-API canary inputs; never construct or qualify a provider."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, object_identity, thaw_json
from mub.vnext.contracts.v3.native_multi_object import NativeMultiObjectMutationRequestV3, NativeRetrievalRequestV3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.native_multi_object import check_forbidden_metadata
from mub.vnext.io.atomic import publish_files_atomically

SOURCE_FILES = (
    'scripts/vnext_prepare_native_multi_object_canary.py',
    'mub/vnext/contracts/common.py',
    'mub/vnext/contracts/enums.py',
    'mub/vnext/contracts/v3/common.py',
    'mub/vnext/contracts/v3/native_multi_object.py',
    'mub/vnext/external/native_multi_object.py',
    'mub/vnext/external/artifacts.py',
    'mub/vnext/io/atomic.py',
)
DEFAULT_OUTPUT = 'results/vnext/native_multi_object_canary_preparation_20261010_v1'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_binding():
    result = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        assert_no_reparse_components(path)
        raw = path.read_bytes().replace(bytes((13, 10)), bytes((10,))).replace(bytes((13,)), bytes((10,)))
        result[name] = digest(raw)
    return result


_LOADED_SOURCE_BINDING = source_binding()


def _key(namespace='research', attribute='city', subkey=None, object_type='profile'):
    return FrozenMemoryObjectKey(namespace=namespace, entity='Mira', attribute=attribute,
                                 subkey=subkey, object_type=object_type)


def _request(case_id, sequence, operation, objects=()):
    event_id = f'{case_id}-e{sequence}'
    return NativeMultiObjectMutationRequestV3(
        request_id=f'{case_id}-m{sequence}', event_id=event_id, sequence_index=sequence,
        operation=Operation(operation), objects=objects, source_event_ids=(event_id,),
        provenance={'kind':'declared', 'provider_owned':False},
    )


def build_cases():
    city, timezone = _key(), _key(attribute='timezone')
    other_namespace = _key(namespace='research-us')
    subkey = _key(subkey='primary')
    classified = _key(object_type='fact')
    missing = _key(attribute='missing')
    scenarios = (
        ('interleaved_updates', (
            ('ADD', ({'object_key':city, 'value':'Paris'}, {'object_key':timezone, 'value':'UTC+1'})),
            ('UPDATE', ({'object_key':city, 'value':'Lyon'},)),
            ('NOOP', ()),
            ('UPDATE', ({'object_key':city, 'value':'Lyon'},)),
            ('UPDATE', ({'object_key':timezone, 'value':'UTC+2'},)),
            ('UPDATE', ({'object_key':city, 'value':'Bordeaux'},)),
            ('DELETE', ({'object_key':timezone},)),
        )),
        ('four_part_identity', (
            ('ADD', ({'object_key':city, 'value':1}, {'object_key':subkey, 'value':'primary'},
                     {'object_key':other_namespace, 'value':'other'})),
            ('UPDATE', ({'object_key':classified, 'value':1.0},)),
            ('UPDATE', ({'object_key':subkey, 'value':None},)),
            ('DELETE', ({'object_key':other_namespace},)),
            ('NOOP', ()),
        )),
        ('no_effect_batches', (
            ('UPDATE', ({'object_key':city, 'value':'not-written'},)),
            ('ADD', ({'object_key':city, 'value':{'active':True}},)),
            ('ADD', ({'object_key':city, 'value':'not-written'}, {'object_key':timezone, 'value':None})),
            ('UPDATE', ({'object_key':missing, 'value':'not-written'}, {'object_key':timezone, 'value':False})),
            ('DELETE', ({'object_key':missing}, {'object_key':city})),
            ('NOOP', ()),
        )),
        ('delete_readd', (
            ('ADD', ({'object_key':city, 'value':1}, {'object_key':timezone, 'value':'1'})),
            ('DELETE', ({'object_key':city},)),
            ('UPDATE', ({'object_key':city, 'value':'not-written'},)),
            ('ADD', ({'object_key':city, 'value':None},)),
            ('UPDATE', ({'object_key':city, 'value':2}, {'object_key':timezone, 'value':'latest'})),
            ('DELETE', ({'object_key':timezone},)),
            ('NOOP', ()),
        )),
    )
    return tuple((case_id, tuple(_request(case_id, i, op, objects)
                                 for i, (op, objects) in enumerate(actions)))
                 for case_id, actions in scenarios)


def reference_expectations(mutations):
    # This dictionary is an offline expectation oracle, never provider-side state.
    state, steps = {}, []
    for original in mutations:
        request = NativeMultiObjectMutationRequestV3.model_validate_json(
            original.model_dump_json(exclude_unset=True))
        outcomes = []
        for item in request.objects:
            identity = object_identity(item.object_key)
            present = identity in state
            reason = None
            if request.operation is Operation.ADD and present:
                reason = 'already_present'
            elif request.operation is Operation.UPDATE and not present:
                reason = 'absent_update'
            elif request.operation is Operation.DELETE and not present:
                reason = 'absent_delete'
            if reason is None:
                if request.operation is Operation.DELETE:
                    del state[identity]
                else:
                    state[identity] = {
                        'object_key':item.object_key.model_dump(mode='json'),
                        'value':thaw_json(item.value),
                        'source_event_ids':list(request.source_event_ids),
                        'sequence_index':request.sequence_index,
                    }
            outcomes.append({'object_id':item.object_key.canonical_id,
                             'status':'no_effect' if reason else 'executed', 'reason':reason})
        statuses = {item['status'] for item in outcomes}
        atomicity = ('not_applicable' if request.operation is Operation.NOOP else
                     'partial' if len(statuses) > 1 else 'unknown')
        steps.append({'request_id':request.request_id, 'event_id':request.event_id,
                      'atomicity':atomicity, 'outcomes':outcomes,
                      'after_state':deepcopy(sorted(state.values(), key=lambda e:canonical(e['object_key'])))})
    return {'steps':steps, 'final_state':deepcopy(sorted(state.values(), key=lambda e:canonical(e['object_key'])))}


def build_payloads():
    binding = source_binding()
    if binding != _LOADED_SOURCE_BINDING:
        raise ValueError('loaded source changed on disk; restart preparation from unchanged source')
    requests, expectations = [], []
    cases = build_cases()
    for case_id, mutations in cases:
        reference = reference_expectations(mutations)
        for repetition in range(2):
            namespace = f'native_canary_{case_id}_r{repetition}'
            retrieval = NativeRetrievalRequestV3(
                query_id=f'{case_id}-q{repetition}', query_text='Return all current memory objects.',
                k=8, namespace=namespace, filters={}, options={},
            )
            check_forbidden_metadata(retrieval.model_dump(mode='json'))
            requests.append({'case_id':case_id, 'repetition':repetition, 'runtime_namespace':namespace,
                             'mutations':[r.model_dump(mode='json', exclude_unset=True) for r in mutations],
                             'retrieval':retrieval.model_dump(mode='json')})
            expectations.append({'case_id':case_id, 'repetition':repetition, **deepcopy(reference)})
    manifest = {
        'schema':'memupdatebench.native-api-canary-preparation.v1',
        'status':'PREPARED_NOT_EXECUTED', 'execution_authorized':False,
        'scientific_release_allowed':False, 'native_admission':'NOT_REQUESTED', 'answer_metrics':None,
        'evidence_class':'synthetic_api_contract_preparation',
        'scenario_count':len(cases), 'planned_repetitions':2, 'real_source_trajectories':0,
        'planned_mutation_requests':sum(len(m) for _, m in cases) * 2,
        'planned_retrieval_requests':len(requests),
        'source_sha256':binding, 'source_hash_policy':'python_lf_normalized',
        'execution':{'provider_calls':0, 'model_loads':0, 'generations':0,
                     'network_calls':0, 'gpu_calls':0, 'additional_executable_calls':0},
        'gates':{name:'NOT_REQUESTED' for name in ('source_runtime_identity', 'direct_api',
                                                'isolation_reset', 'trace_validation', 'deterministic_repeat')},
        'boundary':'Reference expectations only; no provider observations, retrieval ordering, capability, accuracy or native qualification.',
    }
    if binding != source_binding():
        raise ValueError('source changed during preparation')
    return {'manifest.json':canonical(manifest),
            'requests.jsonl':b''.join(canonical(r) for r in requests),
            'expectations.jsonl':b''.join(canonical(r) for r in expectations)}


def _output_path(path):
    path = Path(path).absolute()
    if '..' in path.parts:
        raise ValueError('output traversal into protected input is forbidden')
    assert_no_reparse_components(path)
    for relative in ('data', 'tests/vnext/fixtures', '.git', 'mub', 'scripts', 'configs'):
        protected = (ROOT / relative).absolute()
        if path == protected or protected in path.parents:
            raise ValueError('output overlaps a protected input root')
    return path


def publish_preparation(output_root):
    output = _output_path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    identity = (output.stat().st_dev, output.stat().st_ino)
    def check_owned_directory():
        assert_no_reparse_components(output)
        if (output.stat().st_dev, output.stat().st_ino) != identity:
            raise ValueError('output directory changed during publication')
    try:
        payloads = build_payloads()
        index = {'schema':'memupdatebench.native-api-canary-preparation-index.v1',
                 'status':'PREPARED_NOT_EXECUTED', 'scientific_release_allowed':False,
                 'artifacts':[{'path':name, 'bytes':len(raw), 'sha256':digest(raw)}
                              for name, raw in sorted(payloads.items())]}
        all_payloads = {**payloads, 'index.json':canonical(index)}
        def revalidate():
            check_owned_directory()
            if payloads != build_payloads():
                raise ValueError('source changed before publication')
        publish_files_atomically({output / name:raw for name, raw in all_payloads.items()},
                                 overwrite=False, source_paths=tuple(ROOT / name for name in SOURCE_FILES),
                                 pre_publish=revalidate)
        check_owned_directory()
        pin = digest(all_payloads['index.json'])
        read_preparation(output, pin)
        return {'status':'PREPARED_NOT_EXECUTED', 'index_sha256':pin,
                'scientific_release_allowed':False}
    except BaseException:
        try:
            check_owned_directory()
            output.rmdir()
        except (OSError, ValueError):
            pass
        raise


def read_preparation(root, expected_index_sha256):
    root = Path(root).absolute()
    assert_no_reparse_components(root)
    if type(expected_index_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_index_sha256) is None:
        raise ValueError('expected index SHA-256 must be supplied')
    assert_no_reparse_components(root / 'index.json')
    raw = (root / 'index.json').read_bytes()
    if digest(raw) != expected_index_sha256:
        raise ValueError('index hash mismatch')
    expected = build_payloads()
    index = {'schema':'memupdatebench.native-api-canary-preparation-index.v1',
             'status':'PREPARED_NOT_EXECUTED', 'scientific_release_allowed':False,
             'artifacts':[{'path':name, 'bytes':len(value), 'sha256':digest(value)}
                          for name, value in sorted(expected.items())]}
    if raw != canonical(index):
        raise ValueError('index disagrees with preparation reconstruction')
    if {p.name for p in root.iterdir()} != {'index.json', *expected}:
        raise ValueError('preparation membership mismatch')
    for name, value in expected.items():
        path = root / name
        assert_no_reparse_components(path)
        if path.read_bytes() != value:
            raise ValueError('artifact disagrees with preparation reconstruction')
    return json.loads(expected['manifest.json'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', type=Path, default=ROOT / DEFAULT_OUTPUT)
    parser.add_argument('--verify-index-sha256')
    args = parser.parse_args(argv)
    if args.verify_index_sha256 is not None:
        result = read_preparation(args.output_root, args.verify_index_sha256)
        result = {'status':result['status'], 'index_sha256':args.verify_index_sha256,
                  'execution_authorized':False, 'scientific_release_allowed':False}
    else:
        result = publish_preparation(args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
