from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import vnext_prepare_native_multi_object_canary as canary
from mub.vnext.contracts.v3.common import typed_json_equal
from mub.vnext.contracts.v3.native_multi_object import NativeMultiObjectMutationRequestV3, NativeRetrievalRequestV3


def rows(raw):
    return [json.loads(line) for line in raw.splitlines()]


def test_preparation_is_deterministic_and_cannot_qualify_provider():
    first = canary.build_payloads()
    assert first == canary.build_payloads()
    manifest = json.loads(first['manifest.json'])
    assert manifest['status'] == 'PREPARED_NOT_EXECUTED'
    assert manifest['scientific_release_allowed'] is False
    assert manifest['execution_authorized'] is False
    assert manifest['native_admission'] == 'NOT_REQUESTED'
    assert manifest['answer_metrics'] is None
    assert all(value == 0 for value in manifest['execution'].values())
    assert manifest['scenario_count'] == 4 and manifest['planned_repetitions'] == 2
    assert manifest['real_source_trajectories'] == 0


def test_provider_requests_use_existing_contracts_without_oracle_fields():
    payloads = canary.build_payloads()
    requests = rows(payloads['requests.jsonl'])
    assert len(requests) == 8
    assert len({row['runtime_namespace'] for row in requests}) == 8
    for row in requests:
        assert set(row) == {'case_id', 'repetition', 'runtime_namespace', 'mutations', 'retrieval'}
        for mutation in row['mutations']:
            parsed = NativeMultiObjectMutationRequestV3.model_validate_json(json.dumps(mutation))
            assert parsed.provenance.kind.value == 'declared' and not parsed.provenance.provider_owned
        retrieval = NativeRetrievalRequestV3.model_validate_json(json.dumps(row['retrieval']))
        assert retrieval.namespace == row['runtime_namespace']
        assert dict(retrieval.filters) == dict(retrieval.options) == {}
        assert 'expected' not in json.dumps(row) and 'gold' not in json.dumps(row)
    assert 'expectations.jsonl' in payloads


def test_reference_equal_update_keeps_new_source_link_without_collateral():
    expectations = rows(canary.build_payloads()['expectations.jsonl'])
    row = next(r for r in expectations if r['case_id'] == 'interleaved_updates' and r['repetition'] == 0)
    assert row['steps'][3]['outcomes'][0]['status'] == 'executed'
    assert row['steps'][3]['after_state'][0]['source_event_ids'] == ['interleaved_updates-e3']
    assert [e['value'] for e in row['final_state']] == ['Bordeaux']
    assert row['steps'][2]['atomicity'] == 'not_applicable'


def test_reference_four_part_identity_and_json_types_are_exact():
    row = next(r for r in rows(canary.build_payloads()['expectations.jsonl'])
               if r['case_id'] == 'four_part_identity' and r['repetition'] == 0)
    assert len(row['steps'][0]['after_state']) == 3
    assert len(row['final_state']) == 2
    values = [e['value'] for e in row['final_state']]
    assert any(value is None for value in values)
    assert any(type(value) is float and value == 1.0 for value in values)
    assert not typed_json_equal(1, 1.0)


def test_no_effect_mixed_batch_is_not_atomic():
    row = next(r for r in rows(canary.build_payloads()['expectations.jsonl'])
               if r['case_id'] == 'no_effect_batches' and r['repetition'] == 0)
    assert row['steps'][0]['outcomes'][0]['reason'] == 'absent_update'
    assert [o['status'] for o in row['steps'][2]['outcomes']] == ['no_effect', 'executed']
    assert row['steps'][2]['atomicity'] == 'partial'
    assert row['final_state'][0]['value'] is False


def test_publish_and_reopen_are_no_replace_and_index_pinned(tmp_path):
    output = tmp_path / 'packet'
    result = canary.publish_preparation(output)
    assert canary.read_preparation(output, result['index_sha256'])['status'] == 'PREPARED_NOT_EXECUTED'
    before = {p.name:p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        canary.publish_preparation(output)
    assert before == {p.name:p.read_bytes() for p in output.iterdir() if p.is_file()}
    for invalid in ('0' * 64, ''):
        with pytest.raises(ValueError, match='index'):
            canary.read_preparation(output, invalid)


def test_rehashed_oracle_tamper_is_rejected(tmp_path):
    output = tmp_path / 'packet'
    canary.publish_preparation(output)
    values = rows((output / 'expectations.jsonl').read_bytes())
    values[0]['final_state'][0]['value'] = 'tampered'
    raw = b''.join(canary.canonical(row) for row in values)
    (output / 'expectations.jsonl').write_bytes(raw)
    index = json.loads((output / 'index.json').read_bytes())
    record = next(r for r in index['artifacts'] if r['path'] == 'expectations.jsonl')
    record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    index_raw = canary.canonical(index)
    (output / 'index.json').write_bytes(index_raw)
    with pytest.raises(ValueError, match='reconstruction'):
        canary.read_preparation(output, hashlib.sha256(index_raw).hexdigest())


def test_preparation_does_not_import_or_call_runtime_providers():
    code = '''import sys, importlib.abc
class NoProvider(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('qdrant_client','torch','transformers','sentence_transformers'):
   raise AssertionError('provider import forbidden: '+fullname)
sys.meta_path.insert(0, NoProvider())
from scripts.vnext_prepare_native_multi_object_canary import build_payloads
assert build_payloads()
print('NO_PROVIDER_IMPORTS')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'NO_PROVIDER_IMPORTS' in result.stdout


def test_protected_output_is_rejected_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(canary, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='protected'):
        canary.publish_preparation(tmp_path / 'data' / 'vnext' / 'core' / 'v3' / 'canary')
    assert not (tmp_path / 'data').exists()


def test_python_source_binding_is_checkout_line_ending_invariant(monkeypatch, tmp_path):
    monkeypatch.setattr(canary, 'ROOT', tmp_path)
    monkeypatch.setattr(canary, 'SOURCE_FILES', ('source.py',))
    source = tmp_path / 'source.py'
    source.write_bytes(b'print(1)\nprint(2)\n')
    first = canary.source_binding()
    for newline in (b'\r\n', b'\r'):
        source.write_bytes(b'print(1)' + newline + b'print(2)' + newline)
        assert canary.source_binding() == first
    source.write_bytes(b'print(3)\n')
    assert canary.source_binding() != first


def test_parent_traversal_cannot_publish_into_protected_root(monkeypatch, tmp_path):
    import shutil
    for name in canary.SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canary.ROOT / name, target)
    monkeypatch.setattr(canary, 'ROOT', tmp_path)
    (tmp_path / 'results').mkdir()
    with pytest.raises(ValueError, match='protected|traversal'):
        canary.publish_preparation(tmp_path / 'results' / '..' / 'data' / 'vnext' / 'core' / 'v3' / 'packet')
    assert not (tmp_path / 'data').exists()


def test_output_directory_created_during_build_is_not_adopted(monkeypatch, tmp_path):
    output = tmp_path / 'packet'
    original = canary.build_payloads
    first = [True]
    def competing_writer():
        if first[0]:
            first[0] = False
            output.mkdir()
            (output / 'sentinel.txt').write_bytes(b'other writer')
        return original()
    monkeypatch.setattr(canary, 'build_payloads', competing_writer)
    with pytest.raises(FileExistsError):
        canary.publish_preparation(output)
    assert not (output / 'index.json').exists()


def test_generator_modified_after_import_cannot_bind_new_disk_code(tmp_path):
    import importlib.util
    import shutil
    for name in canary.SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canary.ROOT / name, target)
    source = tmp_path / 'scripts/vnext_prepare_native_multi_object_canary.py'
    spec = importlib.util.spec_from_file_location('temporary_canary', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = source.read_bytes()
    changed = before.replace(b"'Paris'", b"'Strasbourg'")
    assert changed != before
    source.write_bytes(changed)
    with pytest.raises(ValueError, match='loaded source'):
        module.publish_preparation(tmp_path / 'packet')
    assert not (tmp_path / 'packet' / 'index.json').exists()
