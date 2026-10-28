from pathlib import Path

import pytest

from scripts import vnext_run_version_transfer_dev as dev
from scripts.vnext_prepare_version_transfer_decision import build_tasks


def test_development_selection_excludes_validation_and_keeps_paired_blocks():
    tasks = build_tasks(code_revision='test-only')
    selected = dev.select_dev_tasks(tasks)
    assert len(selected) == 24
    assert all(t.metadata.split.value == 'dev' for t in selected)
    for offset in range(0, len(selected), 3):
        block = selected[offset:offset+3]
        assert len({t.metadata.split_key.semantic_core_id for t in block}) == 1
        assert {t.metadata.extra['update_count'] for t in block} == {1,4,16}
    assert len({t.metadata.extra['update_count'] for t in selected[::3]}) == 3
    with pytest.raises(ValueError, match='complete'):
        dev.select_dev_tasks(tasks[1:])


def test_oracle_readout_is_labeled_and_does_not_touch_native_inputs():
    task = build_tasks(code_revision='test-only')[0]
    request = dev.oracle_request(task, task.queries[1])
    assert request.retrieval_trace.retrieval_policy == 'oracle_evidence_diagnostic'
    assert request.retrieval_trace.version_metadata['evidence_origin'] == 'privileged_reference'
    assert task.gold_evidence[1].answer in request.rendered_prompt
    assert 'latest' not in request.rendered_prompt
    assert 'gold_answer' not in request.rendered_prompt


def test_output_admission_rejects_paths_outside_nas_before_loading_dependencies():
    with pytest.raises(ValueError, match='owned'):
        dev.admit_output(Path('somewhere-outside-nas'))


def test_estimator_counts_native_internal_calls_and_all_readout_conditions():
    estimate = dev.estimate_dev_cost(
        {'llm_attempts':72,'events':20,'elapsed_seconds':938.066},
        task_count=24, events_per_task=17)
    assert estimate['episode_requests'] == 408
    assert estimate['readout_generations'] == 144
    assert estimate['estimated_llm_attempts'] > estimate['episode_requests']
    assert estimate['uses_measured_cached_runtime'] is False


def test_usage_audit_charges_unsettled_attempts_conservatively(tmp_path):
    import json
    path = tmp_path / 'llm.jsonl'
    events = [
        {'event':'reserved','attempt':1,'role':'graphiti','input_tokens':100,'reserved_output_tokens':2048},
        {'event':'settled','attempt':1,'completion_tokens':20,'finish_reason':'stop'},
        {'event':'reserved','attempt':2,'role':'readout','input_tokens':200,'reserved_output_tokens':128},
    ]
    path.write_text(''.join(json.dumps(row)+'\n' for row in events))
    result = dev.audit_llm_ledger(path)
    assert result['charged'] == {'attempts':2,'input_tokens':300,'output_tokens':148}
    assert result['unsettled_attempts'] == [2]
    assert result['all_settled'] is False
    with path.open('a') as f:
        f.write(json.dumps(events[-1])+'\n')
    with pytest.raises(ValueError, match='duplicate'):
        dev.audit_llm_ledger(path)


@pytest.mark.parametrize('fail_second_event', [False, True])
def test_injected_dev_pipeline_writes_real_contracts_and_cleans_owned_scope(
    tmp_path, monkeypatch, fail_second_event
):
    import asyncio
    import hashlib
    import json
    import sys
    from types import SimpleNamespace
    from pydantic import BaseModel

    class Config(BaseModel):
        limit: int = 8

    class Filter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Driver:
        _init_task = None
        cleanup_calls = 0

        async def execute_query(self, query, **kwargs):
            if 'DELETE' in query:
                self.cleanup_calls += 1
            return [{'n':0}], None, None

    class Graph:
        events = 0
        closed = False

        async def build_indices_and_constraints(self):
            pass

        async def add_episode(self, **kwargs):
            self.events += 1
            if fail_second_event and self.events == 2:
                raise RuntimeError('injected native failure')
            episode = SimpleNamespace(group_id=kwargs['group_id'], valid_at=kwargs['reference_time'])
            return SimpleNamespace(episode=episode, nodes=[], edges=[],
                model_dump=lambda **unused:{'episode':{'group_id':kwargs['group_id']},'nodes':[],'edges':[]})

        async def search_(self, **kwargs):
            return SimpleNamespace(edges=[], model_dump=lambda **unused:{'edges':[],'edge_reranker_scores':[]})

        async def close(self):
            self.closed = True

    class Api:
        calls = 0

        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            Api.calls += 1
            value = {'model':dev.MODEL_ID,'choices':[{'message':{'content':
                '{"disposition":"answered","answer":"deliberately-wrong-fixture"}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':12,'total_tokens':112}}
            return SimpleNamespace(model_dump=lambda **unused:value)

        async def close(self):
            pass

    graph, driver, api = Graph(), Driver(), Api()
    monkeypatch.setattr(dev, 'native_graph_clients', lambda **kwargs:(graph, driver, api))
    modules = {
        'openai':SimpleNamespace(AsyncOpenAI=Api),
        'graphiti_core.nodes':SimpleNamespace(EpisodeType=SimpleNamespace(text='text')),
        'graphiti_core.search.search_config_recipes':SimpleNamespace(EDGE_HYBRID_SEARCH_RRF=Config()),
        'graphiti_core.search.search_filters':SimpleNamespace(SearchFilters=Filter,DateFilter=Filter,
            ComparisonOperator=SimpleNamespace(less_than_equal='<=',greater_than='>',is_null='IS NULL')),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(dev, 'BASE', tmp_path)
    qualification = tmp_path / dev.QUALIFICATION_REL
    qualification.parent.mkdir(parents=True)
    qualification.write_bytes(b'{"evidence":"injected-unit-test-only"}')
    monkeypatch.setattr(dev, 'QUALIFICATION_PIN', hashlib.sha256(qualification.read_bytes()).hexdigest())
    out = tmp_path/'output'
    out.mkdir()
    task = build_tasks(code_revision='fixture-revision')[0]
    if fail_second_event:
        with pytest.raises(RuntimeError, match='injected'):
            asyncio.run(dev.execute_dev((task,),out,source_revision='fixture-revision'))
        rows = json.loads((out/'cases.json').read_bytes())
        assert rows[0]['status']=='BLOCKED' and rows[0]['events_completed']==1
        assert not (out/'native_canonical').exists() and Api.calls==0
    else:
        rows = asyncio.run(dev.execute_dev((task,),out,source_revision='fixture-revision'))
        assert rows[0]['status']=='OBSERVED' and rows[0]['events_completed']==17
        assert rows[0]['readouts_completed']==6 and Api.calls==6
        score = json.loads((out/'native_canonical/scores.jsonl').read_bytes())
        assert score['answer_scores']['exact_match']==0.0
        assert score['state_scores']['final_state_accuracy'] is None
    assert graph.closed and driver.cleanup_calls==1
    assert rows[0]['namespace_empty_after_cleanup'] is True
