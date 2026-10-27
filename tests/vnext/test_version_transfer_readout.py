from copy import deepcopy

import pytest

from mub.vnext.runtime.version_transfer_readout import build_readout, parse_readout_response
from scripts.vnext_prepare_version_transfer_decision import build_tasks


def native_result():
    return {'edges':[
        {'uuid':'old','fact':'Person-00 lives in city-value-00','name':'RESIDES_IN',
         'group_id':'owned','valid_at':'2025-01-01T00:00:00Z','invalid_at':'2025-01-17T00:00:00Z',
         'created_at':'2026-10-27T00:00:00Z','expired_at':'2026-10-27T00:00:01Z',
         'source_node_uuid':'person','target_node_uuid':'city0','episodes':['native-episode-0']},
        {'uuid':'new','fact':'Person-00 lives in city-value-16','name':'RESIDES_IN',
         'group_id':'owned','valid_at':'2025-01-17T00:00:00Z','invalid_at':None,
         'created_at':'2026-10-27T00:00:01Z','expired_at':None,
         'source_node_uuid':'person','target_node_uuid':'city16','episodes':['native-episode-16']}
    ],'edge_reranker_scores':[0.7,0.4]}


def test_paired_prompts_preserve_evidence_bytes_and_only_reverse_order():
    task=build_tasks(code_revision='test-only')[0]
    result=native_result(); before=deepcopy(result)
    forward=build_readout(task.queries[0],result,order='native',namespace='owned')
    reverse=build_readout(task.queries[0],result,order='reverse',namespace='owned')
    a=forward.retrieval_trace; b=reverse.retrieval_trace
    assert [e.entry_id for e in a.retrieved_entries]==['old','new']
    assert [e.content for e in a.retrieved_entries]==list(reversed([e.content for e in b.retrieved_entries]))
    assert a.scores==(0.7,0.4) and b.scores==(0.4,0.7)
    assert a.ranks==(1,2) and b.ranks==(2,1)
    assert all(e.object_key_candidate is None and e.value_candidate is None and not e.source_event_ids
               for e in a.retrieved_entries)
    assert a.gold_in_context is None and a.stale_in_context is None
    assert forward.prompt_hash!=reverse.prompt_hash
    assert result==before
    assert 'invalid_at' in forward.rendered_prompt and 'expired_at' in forward.rendered_prompt


def test_gold_metadata_does_not_change_provider_visible_prompt():
    query=build_tasks(code_revision='test-only')[0].queries[0]
    first=build_readout(query,native_result(),namespace='owned')
    # Valid evaluator-only identity changes do not reach the readout text.
    other=build_tasks(code_revision='test-only')[3].queries[0]
    modified=query.model_copy(update={'target_object_keys':other.target_object_keys})
    second=build_readout(modified,native_result(),namespace='owned')
    assert first.rendered_prompt==second.rendered_prompt
    assert 'target_object_keys' not in first.rendered_prompt and 'selector' not in first.rendered_prompt


def test_native_identity_scores_and_namespace_are_required():
    query=build_tasks(code_revision='test-only')[0].queries[0]
    for changed in (
        {**native_result(),'edge_reranker_scores':[0.1]},
        {**native_result(),'edges':[native_result()['edges'][0]]*2},
    ):
        with pytest.raises(ValueError):
            build_readout(query,changed,namespace='owned')
    changed=native_result(); changed['edges'][0]['group_id']='foreign'
    with pytest.raises(ValueError,match='namespace'):
        build_readout(query,changed,namespace='owned')


def test_canonical_parser_rejects_schema_echo_and_preserves_real_usage():
    query=build_tasks(code_revision='test-only')[0].queries[0]
    request=build_readout(query,native_result(),namespace='owned')
    response={'model':'exact-model','choices':[{'message':{'content':'{"disposition":"answered","answer":"city-value-16"}'},'finish_reason':'stop'}],
              'usage':{'prompt_tokens':100,'completion_tokens':12,'total_tokens':112}}
    parsed=parse_readout_response(request,response,model_id='exact-model')
    assert parsed.format_valid and parsed.parsed_answer=='city-value-16'
    assert parsed.usage['prompt_tokens']==100
    response['choices'][0]['message']['content']='{"properties":{},"title":"Answer"}'
    assert parse_readout_response(request,response,model_id='exact-model').format_valid is False
    response['usage']['prompt_tokens']=0
    with pytest.raises(ValueError,match='usage'):
        parse_readout_response(request,response,model_id='exact-model')


def test_unknown_native_metadata_is_preserved_not_replaced_by_a_field_allowlist():
    import json
    query=build_tasks(code_revision='test-only')[0].queries[0]
    result=native_result()
    result['edges'][0]['attributes']={'native_effective_end':'2025-01-17T00:00:00Z'}
    result['edges'][0]['fact_embedding']=[0.1,0.2]
    request=build_readout(query,result,namespace='owned')
    projected=json.loads(request.retrieval_trace.retrieved_entries[0].content)
    assert projected['attributes']==result['edges'][0]['attributes']
    assert 'fact_embedding' not in projected
