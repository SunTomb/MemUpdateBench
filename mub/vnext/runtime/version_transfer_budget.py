"""Fail-closed attempt/token ledger for the bounded version-transfer study."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


class BudgetBlocked(RuntimeError):
    pass


class StudyBudget:
    def __init__(self, ledger, *, attempts, input_tokens, output_tokens, seconds,
                 clock=time.monotonic):
        for value in (attempts, input_tokens, output_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError('budget limits must be positive integers')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('elapsed limit must be positive and finite')
        self.limits = {'attempts': attempts, 'input_tokens': input_tokens, 'output_tokens': output_tokens}
        self._totals = {'attempts': 0, 'input_tokens': 0, 'output_tokens': 0}
        self._clock, self._seconds, self._started = clock, seconds, clock()
        self._pending = {}
        self._closed = False
        self._stream = Path(ledger).open('x', encoding='utf-8')

    @property
    def totals(self):
        return dict(self._totals)

    def _record(self, event):
        self._stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        self._stream.flush()

    def reserve(self, *, role, prompt_tokens, max_output_tokens):
        if self._closed:
            raise BudgetBlocked('ledger is closed')
        if role not in {'graphiti', 'mem0', 'readout', 'oracle_readout'}:
            raise ValueError('unknown model-call role')
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError('prompt token count must be positive')
        if type(max_output_tokens) is not int or not 0 < max_output_tokens <= 2048:
            raise ValueError('output reservation exceeds per-call ceiling')
        elapsed = self._clock() - self._started
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed >= self._seconds:
            raise BudgetBlocked('model uptime limit exhausted or clock invalid')
        proposed = {'attempts': self._totals['attempts'] + 1,
                    'input_tokens': self._totals['input_tokens'] + prompt_tokens,
                    'output_tokens': self._totals['output_tokens'] + max_output_tokens}
        if any(proposed[k] > limit for k, limit in self.limits.items()):
            raise BudgetBlocked('study attempt/token budget exhausted')
        ticket = proposed['attempts']
        self._record({'event': 'reserved', 'attempt': ticket, 'role': role,
                      'input_tokens': prompt_tokens, 'reserved_output_tokens': max_output_tokens,
                      'elapsed_seconds': elapsed})
        self._totals = proposed
        self._pending[ticket] = max_output_tokens
        return ticket

    def settle(self, ticket, *, completion_tokens, finish_reason):
        if self._closed or type(ticket) is not int or ticket not in self._pending:
            raise ValueError('unknown, settled or closed attempt')
        ceiling = self._pending[ticket]
        if type(completion_tokens) is not int or not 0 <= completion_tokens <= ceiling:
            raise ValueError('completion tokens exceed reservation')
        if finish_reason not in {'stop', 'length', 'error'}:
            raise ValueError('unknown finish reason')
        self._record({'event': 'settled', 'attempt': ticket,
                      'completion_tokens': completion_tokens, 'finish_reason': finish_reason})
        self._totals['output_tokens'] -= ceiling - completion_tokens
        del self._pending[ticket]

    def close(self):
        # Unsettled attempts retain their maximum reservation after a technical failure.
        if not self._closed:
            self._stream.close()
            self._closed = True
