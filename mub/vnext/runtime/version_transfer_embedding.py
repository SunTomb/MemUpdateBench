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
        if type(texts) is not list or not texts or len(texts) > 32 or any(type(t) is not str for t in texts):
            raise ValueError('embedding requires at most 32 text inputs')
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


class MiniLMBackend:
    dimension = 384

    def __init__(self, snapshot):
        import os
        snapshot = Path(snapshot).resolve(strict=True)
        if Path('/NAS/yesh') not in snapshot.parents:
            raise ValueError('embedding snapshot outside authorized path')
        if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
            raise ValueError('offline embedding flags required')
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(snapshot), device='cpu', local_files_only=True, trust_remote_code=False)
        self.max_tokens = self.model.max_seq_length

    def token_count(self, text):
        return len(self.model.tokenizer.encode(text, truncation=False, add_special_tokens=True))

    def encode(self, texts):
        return self.model.encode(texts, batch_size=1, show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True).tolist()

    def close(self):
        self.model = None
