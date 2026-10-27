"""Strict local completion protocol; no aliases, hidden truncation or fake usage."""
from __future__ import annotations

import math
import threading


class LocalCompletionBoundary:
    def __init__(self, backend, budget, *, model_id):
        if type(model_id) is not str or not model_id.strip():
            raise ValueError('exact model identity is required')
        self.backend, self.budget, self.model_id = backend, budget, model_id
        self._lock = threading.Lock()

    def complete(self, request, *, role):
        if type(request) is not dict or request.get('model') != self.model_id:
            raise ValueError('request model must equal pinned model identity')
        if request.get('stream', False) is not False or request.get('n', 1) != 1:
            raise ValueError('only a single nonstreaming response is supported')
        if request.get('tools') or request.get('tool_choice') not in (None, 'none'):
            raise ValueError('tool execution is unsupported')
        temperature = request.get('temperature', 0)
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature != 0:
            raise ValueError('deterministic temperature zero is required')
        response_format = request.get('response_format', {'type': 'text'})
        if type(response_format) is not dict or response_format.get('type') not in ('text', 'json_object'):
            raise ValueError('JSON schema constrained decoding is not implemented')
        messages = request.get('messages')
        if type(messages) is not list or not messages:
            raise ValueError('nonempty text messages required')
        if any(type(m) is not dict or m.get('role') not in ('system', 'user', 'assistant')
               or type(m.get('content')) is not str for m in messages):
            raise ValueError('only text chat messages are supported')
        if 'max_tokens' in request and 'max_completion_tokens' in request:
            raise ValueError('ambiguous output limits')
        maximum = request.get('max_tokens', request.get('max_completion_tokens', 512))
        if type(maximum) is not int or not 0 < maximum <= 2048:
            raise ValueError('output token limit exceeds study ceiling')
        with self._lock:
            prepared = self.backend.prepare(messages)
            prompt_tokens = prepared['input_tokens']
            if type(prompt_tokens) is not int or prompt_tokens <= 0:
                raise ValueError('backend must report exact input token count')
            if prompt_tokens + maximum > self.backend.context_tokens:
                raise ValueError('context limit exceeded; input is never truncated')
            ticket = self.budget.reserve(role=role, prompt_tokens=prompt_tokens,
                                         max_output_tokens=maximum)
            # Failures retain the conservative output reservation in the ledger.
            generated = self.backend.generate(prepared, maximum)
            tokens, finish = generated['completion_tokens'], generated['finish_reason']
            if type(generated['text']) is not str:
                raise ValueError('backend text response is malformed')
            self.budget.settle(ticket, completion_tokens=tokens, finish_reason=finish)
            return {'id': f'study-attempt-{ticket}', 'object': 'chat.completion',
                    'model': self.model_id,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': generated['text']},
                                 'finish_reason': finish}],
                    'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': tokens,
                              'total_tokens': prompt_tokens + tokens}}
