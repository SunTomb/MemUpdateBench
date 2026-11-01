import base64
import json
import struct

import pytest

from mub.vnext.runtime.version_transfer_embedding import LocalEmbeddingBoundary, _mean_pool_normalize


class Embedder:
    dimension = 3
    max_tokens = 8

    def __init__(self):
        self.calls = 0

    def token_count(self, text):
        return len(text) + 2

    def encode(self, texts):
        self.calls += 1
        return [[0.25, 0.5, 0.75] for _ in texts]


def test_native_embedding_protocol_and_base64_usage(tmp_path):
    backend = Embedder()
    boundary = LocalEmbeddingBoundary(backend, tmp_path / 'embedding.jsonl', model_id='fixed-embedding', max_requests=2)
    response = boundary.create({'model': 'fixed-embedding', 'input': ['a', 'bb'], 'encoding_format': 'base64'}, role='graphiti')
    assert response['usage'] == {'prompt_tokens': 7, 'total_tokens': 7}
    assert struct.unpack('<3f', base64.b64decode(response['data'][0]['embedding'])) == (0.25, 0.5, 0.75)
    assert backend.calls == 1
    assert boundary.totals == {'attempts': 1, 'items': 2, 'input_tokens': 7}
    boundary.close()
    events = [json.loads(x) for x in (tmp_path / 'embedding.jsonl').read_text().splitlines()]
    assert [r['event'] for r in events] == ['embedding_reserved', 'embedding_settled']
    assert all('input' not in r for r in events)


def test_embedding_never_silently_truncates_and_cannot_exceed_attempt_budget(tmp_path):
    backend = Embedder()
    boundary = LocalEmbeddingBoundary(backend, tmp_path / 'embedding.jsonl', model_id='fixed-embedding', max_requests=1)
    with pytest.raises(ValueError, match='context'):
        boundary.create({'model': 'fixed-embedding', 'input': 'too-long-input'}, role='graphiti')
    assert backend.calls == 0
    boundary.create({'model': 'fixed-embedding', 'input': 'ok'}, role='graphiti')
    with pytest.raises(RuntimeError, match='budget'):
        boundary.create({'model': 'fixed-embedding', 'input': 'ok'}, role='graphiti')
    assert backend.calls == 1
    boundary.close()


def test_embedding_http_route_is_explicit_and_measured(tmp_path):
    import queue
    import threading
    from types import SimpleNamespace
    from urllib.request import Request, urlopen
    from scripts.vnext_serve_version_transfer_local import serve

    embedding = LocalEmbeddingBoundary(Embedder(), tmp_path / 'embedding.jsonl', model_id='fixed-embedding')
    ready = queue.Queue()
    thread = threading.Thread(target=serve, kwargs={
        'boundary': SimpleNamespace(model_id='fixed-llm'), 'embedding': embedding,
        'port': 0, 'seconds': 0.2, 'on_ready': ready.put,
    }, daemon=True)
    thread.start()
    host, port = ready.get(timeout=5)
    try:
        req = Request(f'http://{host}:{port}/v1/embeddings',
                      data=json.dumps({'model': 'fixed-embedding', 'input': 'ok'}).encode(),
                      headers={'Content-Type': 'application/json', 'X-MUB-Role': 'graphiti'})
        with urlopen(req, timeout=5) as response:
            value = json.load(response)
        assert value['usage']['prompt_tokens'] == 4
    finally:
        thread.join(timeout=5)
        embedding.close()
    assert not thread.is_alive()


def test_native_mem0_batch_of_100_keeps_item_and_token_ceilings(tmp_path):
    backend = Embedder()
    boundary = LocalEmbeddingBoundary(backend, tmp_path / 'mem0-batch.jsonl', model_id='fixed-embedding', max_items=100)
    response = boundary.create({'model':'fixed-embedding','input':['a'] * 100}, role='mem0')
    assert len(response['data']) == 100
    assert boundary.totals == {'attempts':1,'items':100,'input_tokens':300}
    with pytest.raises(RuntimeError, match='budget'):
        boundary.create({'model':'fixed-embedding','input':'a'}, role='mem0')
    assert backend.calls == 1
    boundary.close()


def test_minilm_pooling_is_dependency_free_and_normalized():
    vectors = _mean_pool_normalize([[[3.0, 4.0], [100.0, 100.0]]], [[1, 0]])
    assert vectors == [[0.6, 0.8]]


def test_minilm_pooling_uses_only_unmasked_tokens():
    vectors = _mean_pool_normalize([[[1.0, 0.0], [0.0, 2.0]]], [[1, 1]])
    assert vectors == [[0.4472135954999579, 0.8944271909999159]]


def test_minilm_backend_uses_transformers_hidden_states_without_sentence_transformers():
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, value):
            self.value = value
        def detach(self):
            return self
        def cpu(self):
            return self
        def tolist(self):
            return self.value

    class Torch:
        class _Inference:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        @staticmethod
        def inference_mode(): return Torch._Inference()

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {'padding': True, 'truncation': False, 'return_tensors': 'pt'}
            return {'attention_mask': Tensor([[1, 1], [1, 0]])}

    class Model:
        def __call__(self, **encoded):
            assert set(encoded) == {'attention_mask'}
            return SimpleNamespace(last_hidden_state=Tensor([
                [[3.0, 4.0], [0.0, 0.0]],
                [[8.0, 6.0], [100.0, 100.0]],
            ]))

    from mub.vnext.runtime.version_transfer_embedding import MiniLMBackend
    backend = object.__new__(MiniLMBackend)
    backend.dimension = 2
    backend._torch, backend.tokenizer, backend.model = Torch, Tokenizer(), Model()
    vectors = backend.encode(['first', 'second'])
    assert vectors == [[0.6, 0.8], [0.8, 0.6]]
