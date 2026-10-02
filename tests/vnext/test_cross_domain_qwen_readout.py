from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import vnext_run_cross_domain_qwen_readout as run
from scripts.vnext_run_cross_domain_answer_replay import PreparedInput, GenerationOutput, PublicRequest


def test_device_admission_is_uuid_bound_and_never_uses_busy_or_low_memory():
    row = {'index': run.GPU_INDEX, 'uuid': run.GPU_UUID, 'name': 'NVIDIA A40', 'free_mib': 45408}
    assert run.check_device_rows('Tang2', [row], [])['uuid'] == run.GPU_UUID
    for changed, processes in [({**row, 'uuid': 'wrong'}, []), ({**row, 'free_mib': 30000}, []),
                                (row, [{'uuid': run.GPU_UUID, 'pid': 123}])]:
        with pytest.raises(ValueError): run.check_device_rows('Tang2', [changed], processes)
    with pytest.raises(ValueError): run.check_device_rows('other-node', [row], [])
    after_load={**row,'free_mib':24000}
    assert run.check_device_rows('Tang2',[after_load],[{'uuid':run.GPU_UUID,'pid':123}],
                                 allowed_pid=123,minimum_free_mib=1024)['uuid']==run.GPU_UUID
    with pytest.raises(ValueError):
        run.check_device_rows('Tang2',[after_load],[],minimum_free_mib=1024)


def test_source_manifest_uses_platform_independent_member_order(tmp_path):
    for name in ('z.py','Upper.py','a.py'):(tmp_path/name).write_bytes(name.encode())
    manifest=run.make_source_manifest(tmp_path)
    assert [r['path'] for r in manifest['files']]==['Upper.py','a.py','z.py']


def test_source_manifest_rejects_changed_bytes(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'one.py').write_bytes(b'pass\n')
    manifest = run.make_source_manifest(root)
    raw = run.canonical(manifest)
    (root / 'source_manifest.json').write_bytes(raw)
    assert run.verify_source(root, run.sha(raw)) == manifest
    (root / 'one.py').write_bytes(b'changed\n')
    with pytest.raises(ValueError): run.verify_source(root, run.sha(raw))


def test_snapshot_uses_streamed_file_hashes_and_exact_membership(tmp_path, monkeypatch):
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    entries = []
    for i in range(16):
        p = snapshot / f'f{i}'
        raw = f'weight{i}'.encode()
        p.write_bytes(raw)
        entries.append({'path': p.name, 'bytes': len(raw), 'sha256': run.sha(raw)})
    binding = {'repo': run.MODEL_ID, 'revision': run.REVISION, 'tree_sha256': run.MODEL_TREE,
               'file_count': 16, 'total_bytes': sum(e['bytes'] for e in entries), 'entries': entries}
    receipt = tmp_path / 'binding.json'
    receipt.write_bytes(run.canonical(binding))
    monkeypatch.setattr(run, 'SNAPSHOT_BINDING_SHA', run.sha(receipt.read_bytes()))
    before = run.verify_snapshot(snapshot, receipt)
    assert before['file_count'] == 16
    (snapshot / 'f0').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='snapshot'):
        run.verify_snapshot(snapshot, receipt)


class Fake:
    def __init__(self, fail=None):
        self.fail=fail; self.closed=False; self.loaded=False; self.generations=0
        self.observations={'load_attempts':0,'tokenizer_loads':1,'generation_attempts':0}
    def prepare(self, request):
        text='chat:'+request.visible_prompt
        if self.loaded and self.fail=='drift': text+='changed'
        return PreparedInput(request.request_id,text,'1'*64,12)
    def load(self):
        self.observations['load_attempts']+=1
        if self.fail=='load': raise MemoryError('private-detail')
        self.loaded=True
    def generate(self, prepared, **settings):
        self.generations+=1; self.observations['generation_attempts']+=1
        if self.fail=='generate': raise TimeoutError('private-detail')
        return GenerationOutput('{"disposition":"answered","answer":2.4}',12,10)
    def close(self):
        self.closed=True
        if self.fail=='close': raise RuntimeError('private-detail')
        return {'allocated_bytes_after':0,'reserved_bytes_after':0}


def requests_and_binding(backend):
    requests=[PublicRequest('r'+str(i),'public '+str(i)) for i in range(4)]
    bindings=[run.template_binding(r,backend.prepare(r)) for r in requests]
    return requests,bindings


def test_real_worker_loop_is_bounded_and_cleanup_precedes_result():
    backend=Fake(); requests,bindings=requests_and_binding(backend); events=[]
    def emit(event):
        assert 'raw_output' not in event
        events.append(event)
        if event['phase']=='before_generation': assert backend.generations < 4
    result=run.perform_readouts(requests,bindings,backend,emit,lambda:None)
    assert result['status']=='WORKER_COMPLETE' and backend.closed and backend.generations==4
    assert result['lifecycle']['load_completed'] is True
    assert result['lifecycle']['close']['allocated_bytes_after']==0
    assert len(result['rows'])==4 and all(r['status']=='COMPLETED' for r in result['rows'])
    assert 'public 0' not in json.dumps(result) and 'private-detail' not in json.dumps(result)


@pytest.mark.parametrize('failure',['load','generate','close','drift'])
def test_worker_failure_never_retries_and_always_closes(failure):
    backend=Fake(failure); requests,bindings=requests_and_binding(backend)
    result=run.perform_readouts(requests,bindings,backend,lambda e:None,lambda:None)
    assert result['status']=='BLOCKED' and backend.closed
    assert backend.observations['load_attempts']==1
    assert backend.generations<=1 if failure!='close' else backend.generations==4
    assert len(result['rows'])==4


def test_lifecycle_not_published_as_complete_without_process_and_gpu_cleanup():
    worker={'status':'WORKER_COMPLETE','rows':[]}
    good={'worker_exited':True,'owned_group_empty':True,'gpu_pid_absent':True,'temporary_root_removed':True}
    assert run.final_status(worker,good,True)=='COMPLETE'
    for key in good:
        assert run.final_status(worker,{**good,key:False},True)=='BLOCKED'
    assert run.final_status(worker,good,False)=='BLOCKED'


def test_phase_deadlines_are_enforced_by_parent_clock():
    state={'sequence':3,'phase':'before_generation'}
    current=run.watch_phase(None,state,10.0)
    assert run.watch_phase(current,state,20.0)==current
    with pytest.raises(TimeoutError): run.watch_phase(current,state,200.0)
    with pytest.raises(ValueError): run.watch_phase(current,{'sequence':2,'phase':'before_load'},30.0)


def test_runtime_proxy_admission_is_exact_and_retains_implementation_binding():
    proxy_type = type('_Ops', (), {'__module__': 'torch._ops'})
    proxy = proxy_type()
    proxy.__file__ = '_ops.py'
    proxy.__spec__ = None
    modules = {'torch._ops': SimpleNamespace(_Ops=proxy_type, __file__='measured/torch/_ops.py')}
    assert run.framework_proxy_implementation('torch.ops', proxy, modules) == 'torch._ops'
    assert run.framework_proxy_implementation('torch.other', proxy, modules) is None
    for change in ({'__file__':'outside.py'}, {'__spec__':SimpleNamespace(origin='outside.py')}):
        bad = proxy_type()
        bad.__file__ = '_ops.py'; bad.__spec__ = None
        bad.__dict__.update(change)
        with pytest.raises(ValueError):
            run.framework_proxy_implementation('torch.ops', bad, modules)
    with pytest.raises(ValueError):
        run.framework_proxy_implementation('torch.ops', SimpleNamespace(__file__='_ops.py', __spec__=None), modules)
    with pytest.raises(ValueError):
        run.framework_proxy_implementation('torch.ops', proxy, {})


def test_runtime_classes_proxy_has_distinct_exact_binding():
    proxy_type = type('_Classes', (), {'__module__':'torch._classes'})
    proxy = proxy_type(); proxy.__file__ = '_classes.py'; proxy.__spec__ = None
    modules = {'torch._classes':SimpleNamespace(_Classes=proxy_type, __file__='measured/torch/_classes.py')}
    assert run.framework_proxy_implementation('torch.classes', proxy, modules) == 'torch._classes'
    with pytest.raises(ValueError):
        run.framework_proxy_implementation('torch.ops', proxy, modules)


def test_authorization_profile_cannot_widen_the_trial():
    p=run.profile('1'*64)
    run.validate_profile(p)
    p['maximum_generations']=5
    with pytest.raises(ValueError): run.validate_profile(p)
