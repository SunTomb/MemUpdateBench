import pytest

from mub.vnext.runtime.version_transfer_local import LocalCompletionBoundary
from mub.vnext.runtime.version_transfer_budget import StudyBudget


class Backend:
    context_tokens = 32

    def __init__(self):
        self.generated = 0
        self.input_size = 10

    def prepare(self, messages):
        return {'input_tokens': self.input_size, 'messages': messages}

    def generate(self, prepared, max_tokens):
        self.generated += 1
        return {'text': '{"ok":true}', 'completion_tokens': 3, 'finish_reason': 'stop'}


def setup_boundary(tmp_path):
    backend = Backend()
    budget = StudyBudget(tmp_path / 'ledger.jsonl', attempts=10, input_tokens=100,
                         output_tokens=100, seconds=100)
    boundary = LocalCompletionBoundary(backend, budget, model_id='pinned-model')
    request = {'model': 'pinned-model', 'messages': [{'role': 'user', 'content': 'test'}],
               'max_tokens': 8, 'temperature': 0, 'response_format': {'type': 'json_object'}}
    return boundary, backend, budget, request


def test_real_usage_and_exact_identity_returned(tmp_path):
    boundary, backend, budget, request = setup_boundary(tmp_path)
    result = boundary.complete(request, role='graphiti')
    assert result['model'] == 'pinned-model'
    assert result['usage'] == {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}
    assert result['choices'][0]['message']['content'] == '{"ok":true}'
    assert backend.generated == 1
    budget.close()


def test_context_overflow_rejected_without_truncation_or_generation(tmp_path):
    boundary, backend, budget, request = setup_boundary(tmp_path)
    backend.input_size = 30
    with pytest.raises(ValueError, match='context'):
        boundary.complete(request, role='mem0')
    assert backend.generated == 0 and budget.totals['attempts'] == 0
    budget.close()


@pytest.mark.parametrize('changes', [
    {'model': 'silent-alias'}, {'stream': True}, {'temperature': 0.7},
    {'response_format': {'type': 'json_schema'}}, {'max_tokens': 2049},
    {'tools': [{'type': 'function'}]}, {'logprobs': True}, {'logit_bias': {'123': 1}},
])
def test_unsupported_request_fails_before_generation(tmp_path, changes):
    boundary, backend, budget, request = setup_boundary(tmp_path)
    request.update(changes)
    with pytest.raises(ValueError):
        boundary.complete(request, role='graphiti')
    assert backend.generated == 0
    budget.close()


def test_loopback_http_smoke_reports_usage_without_a_model(tmp_path):
    import json
    import queue
    import threading
    from urllib.request import Request, urlopen
    from scripts.vnext_serve_version_transfer_local import serve

    boundary, backend, budget, request = setup_boundary(tmp_path)
    ready = queue.Queue()
    thread = threading.Thread(target=serve, kwargs={
        'boundary': boundary, 'port': 0, 'seconds': 0.2, 'on_ready': ready.put,
    }, daemon=True)
    thread.start()
    host, port = ready.get(timeout=5)
    try:
        req = Request(f'http://{host}:{port}/v1/chat/completions',
                      data=json.dumps(request).encode(),
                      headers={'Content-Type': 'application/json', 'X-MUB-Role': 'graphiti'})
        with urlopen(req, timeout=5) as response:
            result = json.load(response)
        assert result['usage']['total_tokens'] == 13
        assert backend.generated == 1
    finally:
        thread.join(timeout=5)
        budget.close()
    assert not thread.is_alive()


def test_declared_schema_backend_receives_exact_schema_without_output_repair(tmp_path):
    from copy import deepcopy

    class ConstrainedBackend(Backend):
        supports_json_schema = True

        def prepare(self, messages, response_schema=None):
            self.schema = deepcopy(response_schema)
            return super().prepare(messages)

    _, _, budget, request = setup_boundary(tmp_path)
    backend = ConstrainedBackend()
    boundary = LocalCompletionBoundary(backend, budget, model_id='pinned-model')
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}
    request['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'check', 'schema': schema}}
    result = boundary.complete(request, role='graphiti')
    assert backend.schema == schema
    assert schema['required'] == ['ok']
    assert result['choices'][0]['message']['content'] == '{"ok":true}'
    assert backend.generated == 1
    budget.close()
