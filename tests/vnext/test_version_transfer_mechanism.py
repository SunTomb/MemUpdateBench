from copy import deepcopy

import pytest

from scripts import analyze_version_transfer_mechanism as mechanism
from scripts import vnext_analyze_version_transfer_dev_mechanism as report


def edge(uuid, fact, valid_at, invalid_at=None):
    return {'uuid': uuid, 'fact': fact, 'valid_at': valid_at, 'invalid_at': invalid_at,
            'group_id': 'owned'}


def classify(events=(), returned=(), **kwargs):
    return mechanism.analyze_target_lifecycle(
        target_entity='Person-00', target_update_positions={0, 3},
        event_results=list(events), history_result={'edges': list(returned)},
        history_time='2025-01-03T00:00:00Z', **kwargs)


def test_counts_only_observed_invalidation_transitions_not_repeated_returns():
    original = edge('old', 'Person-00 lives in city-value-11', '2025-01-01T00:00:00Z')
    invalidated = {**original, 'invalid_at': '2025-01-02T00:00:00Z'}
    events = [{'edges':[original]}, {'edges':[invalidated]}, {'edges':[invalidated]}]
    before = deepcopy(events)
    result = classify(events)
    assert result['target_invalidation_count'] == 1
    assert result['non_target_event_target_invalidations'] == [1]
    assert result['target_update_event_invalidations'] == []
    assert result['target_invalidations'][0]['previous_event_position'] == 0
    assert 'fact' not in result['target_invalidations'][0]
    assert events == before


def test_first_seen_already_invalidated_is_not_attributed_to_that_event():
    result = classify([{'edges':[edge('old','Person-00 lives in A',
        '2025-01-01T00:00:00Z','2025-01-02T00:00:00Z')]}])
    assert result['target_invalidation_count'] == 0
    assert result['first_seen_invalidated_target_edges'] == 1


def test_entity_identifier_boundaries_prevent_substring_aliasing():
    result = classify(returned=[
        edge('a','Person-000 uses A','2025-01-01T00:00:00Z'),
        edge('b','the setting for_Person-00_is A','2025-01-01T00:00:00Z'),
        edge('c','Person-00的城市为 A','2025-01-01T00:00:00Z'),
    ])
    assert result['history']['target_edges_returned'] == 2
    assert result['history']['without_target_mention_edges_returned'] == 1


def test_time_filter_uses_exclusive_end_and_marks_missing_valid_time_unknown():
    result = classify(returned=[
        edge('ok','Person-00 A','2025-01-01T00:00:00Z','2025-01-05T00:00:00Z'),
        edge('future','Person-00 B','2025-01-05T00:00:00Z'),
        edge('ended','Person-00 C','2025-01-01T00:00:00Z','2025-01-03T00:00:00Z'),
        edge('unknown','Person-00 D',None),
    ])
    assert result['history']['target_time_valid_edges_returned'] == 1
    assert result['history']['target_time_invalid_edges_returned'] == 2
    assert result['history']['target_time_unknown_edges_returned'] == 1


def test_endpoint_names_strengthen_observation_but_do_not_invent_state():
    original = edge('old','Person-00 A','2025-01-01T00:00:00Z')
    original.update(source_node_uuid='person',target_node_uuid='value')
    events = [
        {'nodes':[{'uuid':'person','name':'Person-00'},{'uuid':'value','name':'A'}], 'edges':[original]},
        {'nodes':[{'uuid':'other','name':'Person-16'}],
         'edges':[{**original,'invalid_at':'2025-01-02T00:00:00Z'}]},
    ]
    result = classify(events)
    assert result['target_invalidations'][0]['target_endpoint_witness'] == 'exact_observed_name'
    assert result['final_persisted_state_observed'] is False


def test_saved_candidate_not_returned_is_labeled_observation_not_storage_loss():
    original = edge('gold','Person-00 lives in city-value-11','2025-01-01T00:00:00Z')
    result = classify([{'edges':[original]}], gold_value='city-value-11')
    assert result['candidate_observations']['last_seen_time_compatible_gold_candidate_ids'] == ['gold']
    assert result['candidate_observations']['candidate_ids_absent_from_returned_set'] == ['gold']
    assert result['candidate_observations']['scope'] == 'lexical_last_observation_not_final_store'


@pytest.mark.parametrize('timestamp', ['not-time', '2025-01-03', '2025-01-03T00:00:00'])
def test_invalid_or_timezone_naive_timestamps_fail_closed(timestamp):
    with pytest.raises(ValueError, match='timestamp'):
        mechanism.analyze_target_lifecycle(target_entity='Person-00', target_update_positions={0},
            event_results=[], history_result={'edges':[]}, history_time=timestamp)


def test_duplicate_ids_and_invalid_positions_fail_closed():
    with pytest.raises(ValueError, match='duplicate'):
        classify(returned=[edge('same','Person-00 A','2025-01-01T00:00:00Z')]*2)
    with pytest.raises(ValueError, match='positions'):
        mechanism.analyze_target_lifecycle(target_entity='Person-00',target_update_positions={True},
            event_results=[],history_result={'edges':[]},history_time='2025-01-03T00:00:00Z')


def test_aggregate_does_not_label_missing_target_as_an_empty_retrieval():
    result = classify(returned=[edge('other','Person-16 A','2025-01-01T00:00:00Z')])
    value = report.build_report([{'depth':1, **result}])
    counts = value['aggregate']['counts']
    assert counts['u1/empty_retrieval'] == 0
    assert counts['u1/no_target_mention_returned'] == 1
    assert value['model_calls'] == 0
    assert value['final_state_coverage'] == 'UNAVAILABLE_MUTATION_OUTPUTS_ONLY'
