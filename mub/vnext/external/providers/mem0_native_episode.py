"""Thin visible-episode observer for native Mem0 OSS infer=True calls.

No custom CRUD policy, historical reconstruction, target mapping or reranking.
The caller owns SDK identity admission, loopback transports and resource cleanup.
"""
from __future__ import annotations

import hashlib
import json

from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1, validate_visible_payload


class NativeHistoryUnsupported(RuntimeError):
    pass


class Mem0NativeEpisode:
    def __init__(self, memory, *, namespace, response_shapes=None):
        if type(namespace) is not str or not namespace.strip():
            raise ValueError('owned namespace required')
        self.memory, self.namespace = memory, namespace
        self.response_shapes = response_shapes if response_shapes is not None else []
        self.require_response_monitor = response_shapes is not None
        self.counts = {'episode_attempts':0,'retrieval_attempts':0}

    def _admit(self, request, model):
        if type(request) is not model:
            raise TypeError('exact public input contract required')
        request = model.model_validate_json(request.model_dump_json())
        validate_visible_payload(request.model_dump(mode='python'))
        if request.runtime_namespace != self.namespace:
            raise ValueError('request namespace is not owned')
        return request

    def ingest(self, request):
        event = self._admit(request, ProviderEventInputV1)
        metadata = {'mub_source_event_id':event.event_id,'mub_sequence_index':event.sequence_index}
        if event.logical_time is not None:
            metadata['mub_logical_time'] = event.logical_time
        prior = len(self.response_shapes)
        self.counts['episode_attempts'] += 1
        result = self.memory.add([{'role':'user','content':event.raw_text}],
                                 user_id=self.namespace, metadata=metadata, infer=True)
        # OSS catches some extraction parse errors and returns an empty list.
        # Observe its response callback; do not replace, retry or repair that output.
        observed = self.response_shapes[prior:]
        if self.require_response_monitor and not any(e.get('is_extraction') for e in observed):
            raise ValueError('native extraction response monitor did not observe a required call')
        if any(e.get('is_extraction') and not e.get('valid_memory_list') for e in observed):
            raise ValueError('native extraction format invalid; empty result cannot be assumed valid')
        return result

    def retrieve(self, request, *, query_kind):
        query = self._admit(request, ProviderQueryInputV1)
        if query_kind != 'current':
            raise NativeHistoryUnsupported('OSS reference_date/as-of API is unsupported')
        self.counts['retrieval_attempts'] += 1
        return self.memory.search(query.query_text, top_k=query.k,
                                  filters={'user_id':self.namespace},threshold=0.0,rerank=False)

    def export(self):
        return self.memory.get_all(filters={'user_id':self.namespace},top_k=10000)

    def reset(self):
        return self.memory.delete_all(user_id=self.namespace)

    @staticmethod
    def response_monitor(events, *, extraction_system_prompt, native_json_parser=None):
        def observe(client, response, params):
            content = response.choices[0].message.content or ''
            messages = params.get('messages') or []
            extraction = bool(messages and messages[0].get('content') == extraction_system_prompt)
            try:
                parsed = native_json_parser(content) if native_json_parser is not None else json.loads(content)
                valid = (type(parsed) is dict and type(parsed.get('memory')) is list
                         and all(type(item) is dict and type(item.get('text')) is str and bool(item['text'].strip())
                                 for item in parsed['memory']))
            except (ValueError, TypeError):
                valid = False
            events.append({'is_extraction':extraction,'valid_memory_list':valid,
                           'output_sha256':hashlib.sha256(content.encode()).hexdigest()})
        return observe
