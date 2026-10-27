from types import SimpleNamespace
import json

from scripts import vnext_prepare_version_transfer_decision as prep
from scripts import vnext_run_version_transfer_mem0_smoke as runner
from tests.vnext.test_mem0_native_episode import Memory


def test_mem0_smoke_enumerates_current_and_explicit_unsupported_history(tmp_path):
    shapes=[]
    class ObservedMemory(Memory):
        def add(self,*args,**kwargs):
            shapes.append({'is_extraction':True,'valid_memory_list':True})
            return super().add(*args,**kwargs)
    memory=ObservedMemory()
    cases=runner.run_cases(memory,prep.build_smoke(code_revision='test-only'),tmp_path,shapes)
    assert len(cases)==4
    assert all(c['event_attempts']==5 and c['event_completions']==5 for c in cases)
    assert all(c['current_retrieval_completions']==1 and c['history_queries_unsupported']==1 for c in cases)
    assert all(c['namespace_empty_after_cleanup'] for c in cases)
    assert sum(c[0]=='add' for c in memory.calls)==20
    assert sum(c[0]=='search' for c in memory.calls)==4
    for i in range(4):
        row=json.loads((tmp_path/f'case_{i}_query_1.json').read_bytes())
        assert row['status']=='UNSUPPORTED' and row['answer_metrics'] is None


def test_native_cleanup_attempts_every_owned_client_even_if_one_close_fails():
    called=[]
    def fail():
        called.append('http')
        raise RuntimeError('test-only close failure')
    api=SimpleNamespace(close=fail)
    memory=SimpleNamespace(vector_store=SimpleNamespace(client=SimpleNamespace(close=lambda:called.append('vector'))),
                           db=SimpleNamespace(close=lambda:called.append('history')))
    result=runner.close_native_memory(memory,api)
    assert called==['http','vector','history']
    assert result=={'http':False,'vector':True,'history':True}
