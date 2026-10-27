from copy import deepcopy

import pytest

from scripts import vnext_analyze_version_transfer_decision as analysis


def record(core, depth, answer, *, domain=None, system='graphiti_native', status='COMPLETED'):
    return {'core_id': core, 'domain': domain or ('profile' if core < 4 else 'project'),
            'split': 'test', 'system': system, 'update_count': depth,
            'query_kind': 'current', 'readout': 'native_order_full_time',
            'status': status, 'answer_correct': answer,
            'sufficient_time_information': True, 'attribution': 'version_resolution'}


def sample():
    return [record(core, depth, not (depth == 16 and core in (0, 1, 4)))
            for core in range(8) for depth in (1, 16)]


def test_go_requires_paired_depth_degradation_and_fresh_replication():
    rows = sample()
    report = analysis.decide(rows, qualified=True, replication_core_ids=(0, 1, 4))
    assert report['decision'] == 'GO_TO_CONFIRMATION'
    assert report['systems']['graphiti_native']['deteriorating_core_ids'] == [0, 1, 4]
    assert report['scientific_release_allowed'] is False
    assert analysis.decide(rows, qualified=True)['decision'] == 'INCONCLUSIVE'


def test_low_depth_failures_do_not_trigger_go():
    rows = sample()
    for row in rows:
        row['answer_correct'] = False
    assert analysis.decide(rows, qualified=True, replication_core_ids=(0, 1, 4))['decision'] == 'INCONCLUSIVE'


def test_unsupported_is_missing_not_zero_and_blocks_insufficient_pairs():
    rows = sample()
    for row in rows:
        if row['core_id'] < 3:
            row.update(status='UNSUPPORTED', answer_correct=None)
    report = analysis.decide(rows, qualified=True, replication_core_ids=(0, 1, 4))
    assert report['decision'] == 'INCONCLUSIVE'
    assert report['systems']['graphiti_native']['complete_pairs'] == 5
    assert report['systems']['graphiti_native']['missing_core_ids'] == [0, 1, 2]


def test_duplicate_rows_and_inconsistent_domain_rejected():
    rows = sample()
    with pytest.raises(ValueError, match='duplicate'):
        analysis.decide([*rows, rows[0]], qualified=True)
    rows[0]['domain'] = 'other'
    with pytest.raises(ValueError, match='domain'):
        analysis.decide(rows, qualified=True)


def test_blocked_and_pivot_do_not_masquerade_as_scientific_failure():
    assert analysis.decide([], qualified=False)['decision'] == 'BLOCKED'
    rows = sample()
    for row in rows:
        row['sufficient_time_information'] = False
    assert analysis.decide(rows, qualified=True, adapter_only_effect_confirmed=True)['decision'] == 'PIVOT'
    assert analysis.decide(rows, qualified=True)['decision'] == 'INCONCLUSIVE'


def test_validation_only_correct_system_and_native_full_time():
    rows = sample()
    for row in rows:
        row['split'] = 'dev'
    assert analysis.decide(rows, qualified=True, replication_core_ids=(0, 1, 4))['decision'] == 'INCONCLUSIVE'
    rows = sample()
    for row in rows:
        row['readout'] = 'stripped_time'
    assert analysis.decide(rows, qualified=True, replication_core_ids=(0, 1, 4))['decision'] == 'INCONCLUSIVE'
