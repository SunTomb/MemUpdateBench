"""Direct Graphiti state evidence; no retrieval, ranking, or replacement state."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FIELDS = ('uuid', 'name', 'group_id', 'valid_at', 'invalid_at', 'created_at',
           'expired_at', 'episodes', 'episode_ids', 'entity_edges')
_ENDPOINTS = ('source_uuid', 'target_uuid', 'source_name', 'target_name',
              'source_group_id', 'target_group_id')


@dataclass(frozen=True)
class StateSnapshot:
    namespace: str
    rows: tuple[dict[str, Any], ...]
    sha256: str
    anomalies: tuple[dict[str, Any], ...] = ()


class GraphitiStateWitness:
    """Inject the actual async Neo4jDriver and explicitly admit owned namespaces.

    ``private_sink(payload)`` is an optional synchronous callback for raw facts
    and node properties. Even that payload replaces vectors with hashes. Compact
    rows are allowlisted metadata only; missing properties are not filled in.
    Snapshot ordering is canonical evidence serialization, not search ordering.
    """

    def __init__(self, driver: Any, *, owned_namespaces: str | Iterable[str],
                 private_sink: Callable[[dict[str, Any]], None] | None = None):
        if not callable(getattr(driver, 'execute_query', None)):
            raise TypeError('driver must expose async execute_query(query, **kwargs)')
        admitted = [owned_namespaces] if isinstance(owned_namespaces, str) else owned_namespaces
        self._owned = frozenset(self._namespace(item) for item in admitted)
        if not self._owned:
            raise ValueError('at least one explicitly owned namespace is required')
        if private_sink is not None and not callable(private_sink):
            raise TypeError('private_sink must be callable')
        self.driver, self._private_sink = driver, private_sink

    @staticmethod
    def _namespace(namespace: str) -> str:
        if type(namespace) is not str:
            raise TypeError('namespace must be a string')
        if not _NAMESPACE.fullmatch(namespace):
            raise ValueError('namespace must be a nonblank bounded identifier')
        return namespace

    def _admit(self, namespace: str) -> str:
        group = self._namespace(namespace)
        if group not in self._owned:
            raise ValueError('namespace was not explicitly admitted as owned')
        return group

    @staticmethod
    def _stable_value(value: Any) -> Any:
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, (list, tuple)):
            return [GraphitiStateWitness._stable_value(item) for item in value]
        if isinstance(value, Mapping):
            return {key: GraphitiStateWitness._stable_value(item)
                    for key, item in sorted(value.items())}
        # Neo4j temporal objects use iso_format; Python datetime uses isoformat.
        for name in ('iso_format', 'isoformat'):
            formatter = getattr(value, name, None)
            if callable(formatter):
                return formatter()
        raise TypeError(f'unsupported provider value type: {type(value).__name__}')

    @classmethod
    def _encoded(cls, value: Any) -> bytes:
        return json.dumps(cls._stable_value(value), ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')

    @classmethod
    def _value_hash(cls, value: Any) -> str | None:
        if value is None:
            return None
        payload = value.encode('utf-8') if type(value) is str else cls._encoded(value)
        return hashlib.sha256(payload).hexdigest()

    async def _query(self, query: str, namespace: str) -> list[Mapping[str, Any]]:
        records, _summary, _keys = await self.driver.execute_query(query, group=namespace)
        rows = []
        for record in records:
            if isinstance(record, Mapping):
                rows.append(record)
            elif callable(getattr(record, 'data', None)):
                rows.append(record.data())
            else:
                raise TypeError('execute_query must return (mapping records, summary, keys)')
        return rows

    @classmethod
    def _private_properties(cls, properties: Mapping[str, Any]) -> dict[str, Any]:
        return {key + '_sha256' if key.endswith('embedding') else key:
                cls._value_hash(value) if key.endswith('embedding') else cls._stable_value(value)
                for key, value in properties.items()}

    async def snapshot(self, namespace: str) -> StateSnapshot:
        group = self._admit(namespace)
        query = """
        MATCH (n) WHERE n.group_id = $group AND (n:Entity OR n:Episodic)
        OPTIONAL MATCH (episode:Episodic)-[:MENTIONS]->(n)
        WITH n, collect(episode.uuid) AS linked_episode_uuids
        OPTIONAL MATCH (n)-[:MENTIONS]->(entity:Entity)
        RETURN CASE WHEN n:Entity THEN 'Entity' ELSE 'Episodic' END AS kind,
               properties(n) AS properties, linked_episode_uuids,
               collect(entity.uuid) AS linked_entity_uuids,
               null AS source_uuid, null AS target_uuid,
               null AS source_name, null AS target_name,
               null AS source_group_id, null AS target_group_id
        UNION ALL
        MATCH (s)-[r:RELATES_TO]->(t)
        WHERE r.group_id = $group OR s.group_id = $group OR t.group_id = $group
        RETURN 'RELATES_TO' AS kind, properties(r) AS properties,
               null AS linked_episode_uuids, null AS linked_entity_uuids,
               s.uuid AS source_uuid, t.uuid AS target_uuid,
               s.name AS source_name, t.name AS target_name,
               s.group_id AS source_group_id, t.group_id AS target_group_id
        """
        records = await self._query(query, group)
        rows, private_rows, anomalies = [], [], []
        for record in records:
            kind, properties = record['kind'], record['properties']
            if kind not in ('Entity', 'Episodic', 'RELATES_TO') or not isinstance(properties, Mapping):
                raise ValueError('invalid direct-state record')
            row = {'kind': kind, **{key: self._stable_value(properties[key])
                                   for key in _FIELDS if key in properties}}
            for key in ('fact', 'name_embedding', 'fact_embedding'):
                if key in properties:
                    row[key + '_sha256'] = self._value_hash(properties[key])
            private_properties = self._private_properties(properties)
            row['properties_sha256'] = self._value_hash(private_properties)
            if kind == 'RELATES_TO':
                row.update({key: self._stable_value(record[key]) for key in _ENDPOINTS if key in record})
                if any(value != group for value in (properties.get('group_id'),
                       record.get('source_group_id'), record.get('target_group_id'))):
                    anomalies.append({'kind': kind, 'uuid': properties.get('uuid'),
                                      'error': 'cross_namespace_relationship'})
            else:
                if properties.get('group_id') != group:
                    raise ValueError('node query returned a foreign namespace')
                for key in ('linked_episode_uuids', 'linked_entity_uuids'):
                    if key in record:
                        value = record[key]
                        row[key] = None if value is None else sorted(value)
            rows.append(row)
            private_rows.append({**self._stable_value(row),
                                 'properties': private_properties})
        rows.sort(key=lambda row: (row['kind'], str(row.get('uuid')), self._encoded(row)))
        anomalies.sort(key=self._encoded)
        payload = {'namespace': group, 'rows': rows, 'anomalies': anomalies}
        digest = hashlib.sha256(self._encoded(payload)).hexdigest()
        if self._private_sink is not None:
            private_rows.sort(key=self._encoded)
            self._private_sink({'namespace': group, 'sha256': digest, 'rows': private_rows})
        return StateSnapshot(group, tuple(rows), digest, tuple(anomalies))

    async def delete_namespace(self, namespace: str) -> None:
        group = self._admit(namespace)
        # The guard and conditional mutation share one query. No foreign endpoint
        # or foreign-group relationship can be removed by an unguarded DETACH.
        records = await self._query("""
        MATCH (n) WHERE n.group_id = $group AND (n:Entity OR n:Episodic)
        WITH collect(n) AS nodes
        OPTIONAL MATCH (s)-[r]->(t)
        WHERE s IN nodes OR t IN nodes OR r.group_id = $group
        WITH nodes, collect(DISTINCT r) AS relationships,
             sum(CASE WHEN r IS NOT NULL AND
                 (NOT (s IN nodes) OR NOT (t IN nodes) OR
                  (r.group_id IS NOT NULL AND r.group_id <> $group))
                 THEN 1 ELSE 0 END) AS anomaly_count
        FOREACH (r IN CASE WHEN anomaly_count = 0 THEN relationships ELSE [] END | DELETE r)
        FOREACH (n IN CASE WHEN anomaly_count = 0 THEN nodes ELSE [] END | DELETE n)
        RETURN anomaly_count
        """, group)
        if len(records) != 1 or type(records[0].get('anomaly_count')) is not int:
            raise ValueError('cleanup query returned an invalid result')
        if records[0]['anomaly_count'] != 0:
            raise ValueError('cross-namespace relationship prevents cleanup')
        if not await self.verify_empty(group):
            raise ValueError('owned namespace is not empty after cleanup')

    async def verify_empty(self, namespace: str) -> bool:
        group = self._admit(namespace)
        records = await self._query("""
        OPTIONAL MATCH (n) WHERE n.group_id = $group
        WITH count(n) AS node_count
        OPTIONAL MATCH ()-[r]->() WHERE r.group_id = $group
        RETURN node_count, count(r) AS edge_count
        """, group)
        if len(records) != 1 or any(type(records[0].get(key)) is not int or records[0][key] < 0
                                    for key in ('node_count', 'edge_count')):
            raise ValueError('empty-state query returned an invalid result')
        return records[0]['node_count'] == 0 and records[0]['edge_count'] == 0
