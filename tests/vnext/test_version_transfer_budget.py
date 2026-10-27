import json

import pytest

from mub.vnext.runtime.version_transfer_budget import BudgetBlocked, StudyBudget


def test_reserve_and_settle_record_exact_usage(tmp_path):
    clock = [0.0]
    budget = StudyBudget(tmp_path / 'calls.jsonl', attempts=2, input_tokens=20,
                         output_tokens=8, seconds=10, clock=lambda: clock[0])
    ticket = budget.reserve(role='graphiti', prompt_tokens=7, max_output_tokens=5)
    budget.settle(ticket, completion_tokens=3, finish_reason='stop')
    assert budget.totals == {'attempts': 1, 'input_tokens': 7, 'output_tokens': 3}
    rows = [json.loads(x) for x in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert [r['event'] for r in rows] == ['reserved', 'settled']
    assert all('prompt' not in row and 'content' not in row for row in rows)
    budget.close()


def test_blocked_reservation_does_not_start_generation(tmp_path):
    budget = StudyBudget(tmp_path / 'calls.jsonl', attempts=1, input_tokens=10,
                         output_tokens=4, seconds=10, clock=lambda: 0.0)
    ticket = budget.reserve(role='mem0', prompt_tokens=5, max_output_tokens=4)
    with pytest.raises(BudgetBlocked):
        budget.reserve(role='mem0', prompt_tokens=1, max_output_tokens=1)
    with pytest.raises(ValueError):
        budget.settle(ticket, completion_tokens=5, finish_reason='length')
    budget.settle(ticket, completion_tokens=2, finish_reason='stop')
    with pytest.raises(ValueError):
        budget.settle(ticket, completion_tokens=2, finish_reason='stop')
    budget.close()


def test_budget_timeout_and_unknown_role_fail_closed(tmp_path):
    clock = [0.0]
    budget = StudyBudget(tmp_path / 'calls.jsonl', attempts=4, input_tokens=50,
                         output_tokens=20, seconds=10, clock=lambda: clock[0])
    with pytest.raises(ValueError):
        budget.reserve(role='unclassified', prompt_tokens=1, max_output_tokens=1)
    clock[0] = 10.1
    with pytest.raises(BudgetBlocked):
        budget.reserve(role='readout', prompt_tokens=1, max_output_tokens=1)
    budget.close()
    with pytest.raises(FileExistsError):
        StudyBudget(tmp_path / 'calls.jsonl', attempts=4, input_tokens=50,
                    output_tokens=20, seconds=10)
