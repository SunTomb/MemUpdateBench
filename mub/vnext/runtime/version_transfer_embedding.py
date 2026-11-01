"""Accounted local embedding protocol; truncation and provider fallback forbidden."""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from mub.vnext.runtime.version_transfer_budget import BudgetBlocked


class LocalEmbeddingBoundary:
    def __init__(self, backend, ledger, *, model_id, max_requests=2048,
                 max_items=16384, max_tokens=1048576):
        if any(type(v) is not int or v <= 0 for v in (max_requests, max_items, max_tokens)):
            raise ValueError('embedding ceilings must be positive integers')
        self.backend, self.model_id = backend, model_id
        self.limits = {'attempts': max_requests, 'items': max_items, 'input_tokens': max_tokens}
        self._totals = dict.fromkeys(self.limits, 0)
        self._stream = Path(ledger).open('x', encoding='utf-8')

    @property
    def totals(self):
        return dict(self._totals)

    def _record(self, row):
        self._stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        self._stream.flush()

    def create(self, request, *, role):
        if self._stream.closed:
            raise BudgetBlocked('embedding budget ledger closed')
        if role not in ('graphiti', 'mem0') or type(request) is not dict or request.get('model') != self.model_id:
            raise ValueError('embedding requires declared role and exact model')
        encoding = request.get('encoding_format', 'float')
        if encoding not in ('float', 'base64'):
            raise ValueError('unsupported embedding encoding')
        if request.get('dimensions', self.backend.dimension) != self.backend.dimension:
            raise ValueError('embedding dimension changes are unsupported')
        texts = request.get('input')
        texts = [texts] if type(texts) is str else texts
        if type(texts) is not list or not texts or len(texts) > 100 or any(type(t) is not str for t in texts):
            raise ValueError('embedding requires at most 100 text inputs')
        counts = [self.backend.token_count(t) for t in texts]
        if any(type(c) is not int or c <= 0 or c > self.backend.max_tokens for c in counts):
            raise ValueError('embedding context overflow; truncation is forbidden')
        proposed = {'attempts': self._totals['attempts'] + 1,
                    'items': self._totals['items'] + len(texts),
                    'input_tokens': self._totals['input_tokens'] + sum(counts)}
        if any(proposed[k] > limit for k, limit in self.limits.items()):
            raise BudgetBlocked('embedding budget exhausted')
        attempt = proposed['attempts']
        self._record({'event': 'embedding_reserved', 'attempt': attempt, 'role': role,
                      'items': len(texts), 'input_tokens': sum(counts),
                      'input_sha256': hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()})
        self._totals = proposed
        started = time.monotonic()
        vectors = self.backend.encode(texts)
        if len(vectors) != len(texts) or any(len(v) != self.backend.dimension for v in vectors):
            raise ValueError('backend embedding shape mismatch')
        if any(type(value) not in (float, int) or not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError('nonfinite embedding data')
        data = []
        for index, vector in enumerate(vectors):
            value = ([float(x) for x in vector] if encoding == 'float' else
                     base64.b64encode(struct.pack('<' + 'f' * len(vector), *vector)).decode('ascii'))
            data.append({'object': 'embedding', 'index': index, 'embedding': value})
        self._record({'event': 'embedding_settled', 'attempt': attempt,
                      'elapsed_seconds': time.monotonic() - started})
        return {'object': 'list', 'model': self.model_id, 'data': data,
                'usage': {'prompt_tokens': sum(counts), 'total_tokens': sum(counts)}}

    def close(self):
        self._stream.close()


def _mean_pool_normalize(token_embeddings, attention_mask):
    """Mean-pool unmasked token vectors and L2-normalize each item."""
    result = []
    for tokens, mask in zip(token_embeddings, attention_mask):
        if not tokens or len(tokens) != len(mask):
            raise ValueError('embedding token/mask shape mismatch')
        active = sum(int(value) for value in mask)
        if active <= 0:
            raise ValueError('embedding input has no active tokens')
        dimensions = len(tokens[0])
        pooled = [sum(float(row[index]) * int(weight) for row, weight in zip(tokens, mask)) / active
                  for index in range(dimensions)]
        norm = math.sqrt(sum(value * value for value in pooled))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('embedding pooled vector is not normalizable')
        result.append([value / norm for value in pooled])
    return result


class MiniLMBackend:
    dimension = 384

    def __init__(self, snapshot):
        import os
        import torch
        from transformers import AutoModel, AutoTokenizer
        snapshot = Path(snapshot).resolve(strict=True)
        if Path('/NAS/yesh') not in snapshot.parents:
            raise ValueError('embedding snapshot outside authorized path')
        if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
            raise ValueError('offline embedding flags required')
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False)
        self.model.eval()
        configured = int(getattr(self.model.config, 'max_position_embeddings', 512))
        tokenizer_limit = int(getattr(self.tokenizer, 'model_max_length', configured))
        if tokenizer_limit <= 0 or tokenizer_limit > 1_000_000:
            tokenizer_limit = configured
        self.max_tokens = min(configured, tokenizer_limit)

    def token_count(self, text):
        return len(self.tokenizer.encode(text, truncation=False, add_special_tokens=True))

    def encode(self, texts):
        encoded = self.tokenizer(list(texts), padding=True, truncation=False, return_tensors='pt')
        with self._torch.inference_mode():
            output = self.model(**encoded)
        hidden = output.last_hidden_state.detach().cpu().tolist()
        mask = encoded['attention_mask'].detach().cpu().tolist()
        vectors = _mean_pool_normalize(hidden, mask)
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError('MiniLM embedding dimension mismatch')
        return vectors

    def close(self):
        self.model = self.tokenizer = None
        self._torch = None
