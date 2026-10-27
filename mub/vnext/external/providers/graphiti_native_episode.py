"""Public-episode bridge to native Graphiti; not an admitted v3 adapter.

The client owns extraction, resolution, graph state and search order. This bridge
never parses gold CRUD, maps target objects or mutates provider search results.
Qualification and internal model-call accounting belong to the runtime gate.
"""
from __future__ import annotations

from datetime import datetime
import re

from mub.vnext.external.visibility import (
    ProviderEventInputV1, ProviderQueryInputV1, validate_visible_payload,
)


_AS_OF = re.compile(r'^As of (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z), ')
_UTC = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


def _public_time(value):
    if type(value) is not str or not _UTC.fullmatch(value):
        raise ValueError('public event time must be ISO UTC')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


class GraphitiNativeEpisode:
    def __init__(self, client, *, namespace, source_type, filter_factory):
        if type(namespace) is not str or not namespace.strip():
            raise ValueError('owned namespace must be nonblank')
        if not callable(filter_factory):
            raise ValueError('native time filter factory is required')
        self.client = client
        self.namespace = namespace
        self.source_type = source_type
        self.filter_factory = filter_factory
        self.counts = {'episode_attempts': 0, 'retrieval_attempts': 0}
        self.closed = False

    def _admit(self, request, expected_type):
        if type(request) is not expected_type:
            raise TypeError('requires exact public input contract')
        # Revalidation catches model_copy/construct bypasses at this boundary.
        rebuilt = expected_type.model_validate_json(request.model_dump_json())
        validate_visible_payload(rebuilt.model_dump(mode='python'))
        if rebuilt.runtime_namespace != self.namespace:
            raise ValueError('request namespace is not owned by this bridge')
        if self.closed:
            raise ValueError('bridge is closed')
        return rebuilt

    async def ingest(self, request):
        event = self._admit(request, ProviderEventInputV1)
        reference = _public_time(event.logical_time)
        self.counts['episode_attempts'] += 1
        return await self.client.add_episode(
            name=event.event_id, episode_body=event.raw_text,
            source_description='Controlled synthetic version-transfer study',
            reference_time=reference, source=self.source_type, group_id=self.namespace,
        )

    async def retrieve(self, request):
        query = self._admit(request, ProviderQueryInputV1)
        matched = _AS_OF.match(query.query_text)
        if matched is None:
            raise ValueError('query must state public as-of ISO UTC time')
        native_filter = self.filter_factory(_public_time(matched.group(1)))
        self.counts['retrieval_attempts'] += 1
        return await self.client.search(query=query.query_text, group_ids=[self.namespace],
                                        num_results=query.k, search_filter=native_filter)

    async def close(self):
        if not self.closed:
            await self.client.close()
            self.closed = True

    @staticmethod
    def validity_filter_factory(search_filters, date_filter, operators):
        def build(when):
            return search_filters(
                valid_at=[[date_filter(date=when, comparison_operator=operators.less_than_equal)]],
                invalid_at=[
                    [date_filter(comparison_operator=operators.is_null)],
                    [date_filter(date=when, comparison_operator=operators.greater_than)],
                ],
            )
        return build
