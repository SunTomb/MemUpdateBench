"""Offline NOAA capture, scalar task and review-binding regressions."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import vnext_prepare_family_h_cross_domain_candidate as prepare

PROJECT = Path(__file__).resolve().parents[2]
CAPTURE = PROJECT / 'external/family_h_cross_domain_noaa_20260928_v5'
CAPTURE_SHA = '773c28cdeae721fce84b66ba7bc26a31390cbbfb1020c93bd05614befa2d0aee'
INDEX_SHA = 'd37a2a7a5ad5c0842fd1dae9846b3e2bb09264d2b238b31506b75766ff0be832'
pytestmark = pytest.mark.skipif(
    not CAPTURE.is_dir(),
    reason='requires the separately retained original NOAA capture; raw source is not bundled with source',
)


def raw(advisory='001'):
    return (CAPTURE / 'raw' / (advisory + '.shtml')).read_bytes()


def test_number_slot_uses_scalar_and_preserves_corrected_archive_status():
    row = prepare.parse_advisory(raw(), '001')
    assert row['value'] == 35 and type(row['value']) is int
    assert row['object_key']['attribute'] == 'max_sustained_wind_mph'
    assert row['object_key']['entity'] == 'hurricane-beryl-2024'
    assert row['issued_at_utc'] == '2024-06-28T21:00:00Z'
    assert row['corrected_archive_text'] is True
    assert row['source_anchor']['raw_sha256'] == hashlib.sha256(raw()).hexdigest()
    assert row['source_kind'] == 'sequential_public_advisory'


def test_ast_conversion_crosses_utc_day_and_distinguishes_intermediate():
    row = prepare.parse_advisory(raw('002'), '002')
    assert row['issued_at_utc'] == '2024-06-29T03:00:00Z'
    row = prepare.parse_advisory(raw('003a'), '003a')
    assert row['issued_at_utc'] == '2024-06-29T12:00:00Z'
    assert row['intermediate'] and row['corrected_archive_text']


@pytest.mark.parametrize('change', ['wrong_id', 'wrong_storm', 'duplicate_wind', 'no_pre', 'wrong_utc', 'wrong_weekday'])
def test_parser_rejects_ambiguous_or_mismatched_official_fields(change):
    text = raw().decode('utf-8')
    if change == 'wrong_id': text = text.replace('Advisory Number   1', 'Advisory Number   2')
    if change == 'wrong_storm': text = text.replace('AL022024', 'AL032024')
    if change == 'duplicate_wind': text = text.replace('MAXIMUM SUSTAINED WINDS...35 MPH...55 KM/H', 'MAXIMUM SUSTAINED WINDS...35 MPH...55 KM/H\nMAXIMUM SUSTAINED WINDS...40 MPH...65 KM/H')
    if change == 'no_pre': text = text.replace('<pre>', '').replace('</pre>', '')
    if change == 'wrong_utc': text = text.replace('...2100 UTC...INFORMATION', '...0900 UTC...INFORMATION')
    if change == 'wrong_weekday': text = text.replace('Fri Jun 28 2024', 'Sat Jun 28 2024')
    with pytest.raises(ValueError):
        prepare.parse_advisory(text.encode(), '001')


def test_parser_ignores_false_fields_outside_bulletin():
    noise = b'<div>100 PM AST Fri Jun 28 2024 MAXIMUM SUSTAINED WINDS...999 MPH...999 KM/H</div>'
    assert prepare.parse_advisory(noise + raw(), '001')['value'] == 35
    with pytest.raises(ValueError):
        prepare.parse_advisory(raw() + raw(), '001')


def test_all_captured_bytes_revalidated_not_only_manifest(tmp_path):
    cloned = tmp_path / 'capture'
    shutil.copytree(CAPTURE, cloned)
    proof = prepare.load_capture(cloned, CAPTURE_SHA, INDEX_SHA)
    assert len(proof['rows']) == 7 and proof['rows'][4]['value'] == proof['rows'][5]['value'] == 65
    path = cloned / 'raw/004a.shtml'
    path.write_bytes(path.read_bytes().replace(b'65 MPH', b'60 MPH'))
    with pytest.raises(ValueError, match='hash'):
        prepare.load_capture(cloned, CAPTURE_SHA, INDEX_SHA)


def test_policy_capture_tamper_is_rejected(tmp_path):
    cloned = tmp_path / 'capture'
    shutil.copytree(CAPTURE, cloned)
    (cloned / 'raw/nws_disclaimer.html').write_bytes(b'changed policy')
    with pytest.raises(ValueError, match='hash'):
        prepare.load_capture(cloned, CAPTURE_SHA, INDEX_SHA)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp('noaa') / 'candidate'
    result = prepare.build(out, CAPTURE, CAPTURE_SHA, INDEX_SHA)
    return out, result


def test_candidate_numeric_type_chronology_and_full_audit(built):
    root, result = built
    manifest = json.loads((root/'manifest.json').read_bytes())
    task = json.loads((root/'tasks.jsonl').read_bytes())
    audit = json.loads((root/'audit_manifest.json').read_bytes())
    template = json.loads((root/'decisions_template.json').read_bytes())
    core = json.loads((root/'semantic_cores.jsonl').read_bytes())
    assert task['source']['source_type'] == 'other'
    assert task['source']['provenance']['source_kind'] == 'sequential_public_advisory'
    assert task['queries'][0]['answer_schema'] == 'number'
    assert task['gold_evidence'][0]['answer'] == 75
    values = [json.loads(e['raw_text'])['value'] for e in task['events']]
    assert values == [35, 40, 50, 60, 65, 65, 75]
    assert all(type(value) is int for value in values)
    assert [a['operation'] for a in task['actions']] == ['ADD'] + ['UPDATE'] * 6
    assert manifest['update_events'] == 6 and manifest['value_changing_updates'] == 5
    assert manifest['equal_value_update_events'] == 1
    assert manifest['human_approved_sources'] == 0 and not manifest['formal_task_release']
    assert task['metadata']['split'] == 'evaluation_only'
    assert core['expected_answer'] == 75
    assert audit['item_count'] == 9
    assert [i['kind'] for i in audit['items']].count('source_snapshot') == 7
    assert all(prepare.digest(i['material']) == i['binding_sha256'] for i in audit['items'])
    assert len(template['decisions']) == 9 and all(d['decision'] is None for d in template['decisions'])
    assert prepare.validate_candidate(root, result['index_sha256'])['status'] == 'VALID_PENDING_HUMAN_REVIEW'
    assert 'D:/' not in (root/'manifest.json').read_text() and '/NAS/' not in (root/'manifest.json').read_text()


def test_build_is_reproducible_and_no_replace(built, tmp_path):
    original, result = built
    second = tmp_path / 'second'
    again = prepare.build(second, CAPTURE, CAPTURE_SHA, INDEX_SHA)
    assert result == again
    for p in original.iterdir():
        assert p.read_bytes() == (second / p.name).read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        prepare.build(second, CAPTURE, CAPTURE_SHA, INDEX_SHA)


def test_full_surface_drift_cannot_reuse_review_binding(built, tmp_path):
    root, _ = built
    copied = tmp_path / 'mutated'; shutil.copytree(root, copied)
    task = json.loads((copied/'tasks.jsonl').read_bytes())
    task['events'][0]['raw_text'] += ' changed text'
    payload = prepare.canonical(task)
    (copied/'tasks.jsonl').write_bytes(payload)
    index = json.loads((copied/'index.json').read_bytes())
    for member in index['artifacts']:
        if member['path'] == 'tasks.jsonl':
            member.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    index_raw = prepare.canonical(index); (copied/'index.json').write_bytes(index_raw)
    with pytest.raises(ValueError, match='surface|event'):
        prepare.validate_candidate(copied, hashlib.sha256(index_raw).hexdigest())


def test_review_completion_remains_manual(built):
    root, result = built
    audit = json.loads((root/'audit_manifest.json').read_bytes())
    review = json.loads((root/'decisions_template.json').read_bytes())
    with pytest.raises(ValueError, match='reviewer'):
        prepare.validate_review(audit, review, result['index_sha256'])
    review['reviewer'] = 'unit-test fixture, not a human audit'
    review['candidate_index_sha256'] = result['index_sha256']
    with pytest.raises(ValueError, match='decision'):
        prepare.validate_review(audit, review, result['index_sha256'])
    for d in review['decisions']:
        d['decision'] = 'NEEDS_FIX'; d['rationale'] = 'Synthetic negative unit-test decision.'
    review['candidate_index_sha256'] = result['index_sha256']
    assert prepare.validate_review(audit, review, result['index_sha256'])['all_release_ready'] is False
    review['decisions'][0]['binding_sha256'] = '0'*64
    with pytest.raises(ValueError, match='binding'):
        prepare.validate_review(audit, review, result['index_sha256'])


def test_output_cannot_nest_in_capture_or_older_candidate(built):
    root, _ = built
    with pytest.raises(ValueError):
        prepare.build(root/'new', CAPTURE, CAPTURE_SHA, INDEX_SHA)
    with pytest.raises(ValueError):
        prepare.build(CAPTURE/'new', CAPTURE, CAPTURE_SHA, INDEX_SHA)


def test_old_diagnostic_is_preserved():
    old = PROJECT / 'results/vnext/family_h_cross_domain_noaa_candidate_20260927_v1'
    assert hashlib.sha256((old/'index.json').read_bytes()).hexdigest() == '0679e7ac7c82473bf80b34bfe2e6c5463cf63b4741db44ecf4482cfcce0fa365'
