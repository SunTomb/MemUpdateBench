"""Pure pilot screening rules; input rows require separate execution authentication."""
from __future__ import annotations

from collections import defaultdict


SYSTEMS = ('graphiti_native', 'mem0_native')
DOMAINS = {'profile', 'project'}


def decide(rows, *, qualified, replication_core_ids=(), adapter_only_effect_confirmed=False):
    """Evaluate declared paired observations, not their provenance or qualification receipts.

    A runner must authenticate rows and fresh-ingestion replication before treating
    this screening result as a research investment decision. Unit fixtures are not
    observed model results, even when this function returns GO_TO_CONFIRMATION.
    """
    if type(qualified) is not bool or type(adapter_only_effect_confirmed) is not bool:
        raise ValueError('qualification and adapter effect flags must be booleans')
    report = {'decision': 'BLOCKED' if not qualified else 'INCONCLUSIVE',
              'evidence_class': 'declared_observation_screen_only',
              'scientific_release_allowed': False, 'systems': {}}
    if not qualified:
        return report
    grouped = defaultdict(dict)
    domains = {}
    seen = set()
    for row in rows:
        key = (row['system'], row['core_id'], row['split'], row['query_kind'],
               row['readout'], row['update_count'])
        if key in seen:
            raise ValueError('duplicate observation')
        seen.add(key)
        if row['domain'] not in DOMAINS:
            raise ValueError('unknown domain')
        if row['core_id'] in domains and domains[row['core_id']] != row['domain']:
            raise ValueError('core domain is inconsistent')
        domains[row['core_id']] = row['domain']
        if row['system'] not in SYSTEMS:
            raise ValueError('system must be a declared native system')
        if row['update_count'] not in (1, 4, 16):
            raise ValueError('unknown update depth')
        if row['status'] == 'COMPLETED':
            if type(row['answer_correct']) is not bool:
                raise ValueError('completed answers require typed correctness')
        elif row['status'] not in ('UNSUPPORTED', 'BLOCKED', 'FAILED', 'NOT_RUN'):
            raise ValueError('unknown status')
        elif row['answer_correct'] is not None:
            raise ValueError('unavailable observations must remain null')
        if row['split'] == 'test' and row['query_kind'] == 'current' and row['readout'] == 'native_order_full_time':
            grouped[row['system']].setdefault(row['core_id'], {})[row['update_count']] = row
    any_signal = False
    for system, cores in sorted(grouped.items()):
        complete, missing, worse, better = [], [], [], []
        for core, depths in sorted(cores.items()):
            low, high = depths.get(1), depths.get(16)
            if not all(r and r['status'] == 'COMPLETED' and r['sufficient_time_information'] is True for r in (low, high)):
                missing.append(core)
                continue
            complete.append(core)
            if low['answer_correct'] and not high['answer_correct']:
                worse.append(core)
            if not low['answer_correct'] and high['answer_correct']:
                better.append(core)
        replicated = sorted(set(worse) & set(replication_core_ids))
        traceable = all(cores[c][16].get('attribution') in {
            'event_interpretation', 'version_resolution', 'retrieval', 'answer'
        } for c in worse)
        signal = (len(complete) >= 6 and len(worse) >= 3 and len(worse) - len(better) >= 2
                  and {domains[c] for c in worse} == DOMAINS and traceable
                  and len(replicated) >= 3 and {domains[c] for c in replicated} == DOMAINS)
        report['systems'][system] = {
            'complete_pairs': len(complete), 'complete_core_ids': complete,
            'missing_core_ids': missing, 'deteriorating_core_ids': worse,
            'improving_core_ids': better, 'replicated_deteriorating_core_ids': replicated,
            'net_deteriorating_cores': len(worse) - len(better), 'candidate_signal': signal,
        }
        any_signal |= signal
    if adapter_only_effect_confirmed and any_signal:
        raise ValueError('adapter-only attribution contradicts native signal')
    if any_signal:
        report['decision'] = 'GO_TO_CONFIRMATION'
    elif adapter_only_effect_confirmed:
        report['decision'] = 'PIVOT'
    return report
