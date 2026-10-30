"""Classify saved Graphiti mutation observations, never final graph state.

Entity/value matching is an explicit lexical diagnostic, not gold object
alignment or a semantic-entailment judge. Repeated edge returns are not counted
as new invalidations; missing observations are not proof of storage loss.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import re


def _time(value):
    if type(value) is not str:
        raise ValueError('timestamp must be a string')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('timestamp is invalid') from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('timestamp must carry a timezone')
    return parsed.astimezone(timezone.utc)


def _mentions(text, identifier):
    # The synthetic identifiers include hyphens; do not match Person-00 in
    # Person-000. Underscore/Chinese surrounds occur in saved native facts.
    return bool(re.search(r'(?<![A-Za-z0-9-])' + re.escape(identifier) + r'(?![A-Za-z0-9-])', text))


def _edge_rows(result):
    if type(result) is not dict or type(result.get('edges')) is not list:
        raise ValueError('edge result must contain a list')
    seen = set()
    for edge in result['edges']:
        if type(edge) is not dict or type(edge.get('uuid')) is not str or not edge['uuid'].strip():
            raise ValueError('edge UUID is required')
        if edge['uuid'] in seen:
            raise ValueError('duplicate edge UUID')
        seen.add(edge['uuid'])
        if type(edge.get('fact')) is not str:
            raise ValueError('edge fact is required')
        for field in ('valid_at', 'invalid_at'):
            if edge.get(field) is not None:
                _time(edge[field])
        yield edge


def _time_status(edge, when):
    if edge.get('valid_at') is None or 'invalid_at' not in edge:
        return None
    start = _time(edge['valid_at'])
    end = _time(edge['invalid_at']) if edge['invalid_at'] is not None else None
    if end is not None and end < start:
        return None
    return start <= when and (end is None or when < end)


def _witness(edge, node_names, target_entity):
    names = [node_names.get(edge.get(field), set()) for field in ('source_node_uuid', 'target_node_uuid')]
    if any(values == {target_entity} for values in names):
        return 'exact_observed_name'
    if any(target_entity in values for values in names):
        return 'ambiguous_observed_name'
    return 'no_exact_endpoint_name_witness'


def analyze_target_lifecycle(*, target_entity, target_update_positions,
                             event_results, history_result, history_time, gold_value=None):
    if type(target_entity) is not str or not target_entity.strip():
        raise ValueError('target entity is required')
    if any(type(position) is not int or position < 0 for position in target_update_positions):
        raise ValueError('target update positions must be nonnegative integers')
    if gold_value is not None and (type(gold_value) is not str or not gold_value):
        raise ValueError('lexical gold value must be a nonempty string')
    when = _time(history_time)
    last, names = {}, defaultdict(set)
    transitions, first_invalidated = [], []
    for position, result in enumerate(event_results):
        for node in result.get('nodes', []):
            if type(node.get('uuid')) is str and type(node.get('name')) is str:
                names[node['uuid']].add(node['name'])
        for edge in _edge_rows(result):
            previous = last.get(edge['uuid'])
            target = _mentions(edge['fact'], target_entity)
            if target and edge.get('invalid_at') is not None:
                if previous is None:
                    first_invalidated.append(edge['uuid'])
                elif ('invalid_at' in previous[1] and previous[1]['invalid_at'] is None
                      and _mentions(previous[1]['fact'], target_entity)):
                    transitions.append({
                        'event_position': position, 'previous_event_position': previous[0],
                        'edge_uuid': edge['uuid'], 'invalid_at': edge['invalid_at'],
                        'valid_at': edge.get('valid_at'),
                        'same_fact_text': previous[1]['fact'] == edge['fact'],
                        'fact_sha256': hashlib.sha256(edge['fact'].encode()).hexdigest(),
                        'target_endpoint_witness': _witness(edge, names, target_entity),
                        'at_non_target_event': position not in target_update_positions,
                        'interval_covers_query_after_change': _time_status(edge, when),
                        'interval_covered_query_before_change': _time_status(previous[1], when),
                        'mentions_gold_value': _mentions(edge['fact'], gold_value) if gold_value is not None else None,
                    })
            last[edge['uuid']] = (position, edge)
    returned = list(_edge_rows(history_result))
    target_returned = [edge for edge in returned if _mentions(edge['fact'], target_entity)]
    time_statuses = [_time_status(edge, when) for edge in target_returned]
    candidate = lambda edge: (gold_value is not None and _mentions(edge['fact'], target_entity)
                              and _mentions(edge['fact'], gold_value))
    last_candidates = [edge for _, edge in last.values() if candidate(edge)]
    compatible = [edge['uuid'] for edge in last_candidates if _time_status(edge, when) is True]
    returned_ids = {edge['uuid'] for edge in returned}
    returned_candidates = [edge['uuid'] for edge in target_returned if candidate(edge) and _time_status(edge, when) is True]
    return {
        'identity_method': 'exact_synthetic_identifier_lexical_mention_with_optional_native_endpoint_witness',
        'final_persisted_state_observed': False,
        'target_invalidation_count': len(transitions),
        'first_seen_invalidated_target_edges': len(first_invalidated),
        'non_target_event_target_invalidations': [r['event_position'] for r in transitions if r['at_non_target_event']],
        'target_update_event_invalidations': [r['event_position'] for r in transitions if not r['at_non_target_event']],
        'target_invalidations': transitions,
        'history': {
            'total_edges_returned': len(returned),
            'target_edges_returned': len(target_returned),
            'target_time_valid_edges_returned': sum(value is True for value in time_statuses),
            'target_time_invalid_edges_returned': sum(value is False for value in time_statuses),
            'target_time_unknown_edges_returned': sum(value is None for value in time_statuses),
            'without_target_mention_edges_returned': len(returned) - len(target_returned),
        },
        'candidate_observations': {
            'scope': 'lexical_last_observation_not_final_store',
            'gold_matching_observed_edge_ids': sorted(edge['uuid'] for edge in last_candidates),
            'last_seen_time_compatible_gold_candidate_ids': sorted(compatible),
            'candidate_ids_absent_from_returned_set': sorted(set(compatible) - returned_ids),
            'time_compatible_gold_candidate_ids_returned': sorted(returned_candidates),
        },
    }
