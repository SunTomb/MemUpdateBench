"""BEA verified-policy claims must be evidence-bound, not a mutable status flag."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil

import pytest

from scripts import vnext_prepare_family_h_bea_candidate as prepare

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'external/family_h_cross_domain_bea_gdp_20260930_v6_policy_verified'
MANIFEST_SHA = '9395b50167b4b47f64e19b9ebcbbdafe6c3ba70e1084ebb9a273bc3ba179eb1b'
INDEX_SHA = '17aa9bd3a576effa7b30b79f87e6090e07edf58839594518f808049ac11e6313'
POLICY_SHA = 'e49a712b7e0ec2e8946e256315f4af675af72c29c8d9e019c46626fa56b5a79e'
URL = 'https://www.bea.gov/about/policies-and-information/linking'
pytestmark = pytest.mark.skipif(
    not SOURCE.is_dir(),
    reason='requires the separately retained original BEA capture and policy bytes; raw source is not bundled',
)


def save_reindexed(root, manifest):
    (root/'capture_manifest.json').write_bytes(prepare.canonical(manifest))
    index = json.loads((root/'capture_index.json').read_bytes())
    for row in index['artifacts']:
        raw = (root/row['path']).read_bytes()
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (root/'capture_index.json').write_bytes(prepare.canonical(index))
    return prepare.sha((root/'capture_manifest.json').read_bytes()), prepare.sha((root/'capture_index.json').read_bytes())


@pytest.mark.parametrize('change', ['fake_verified_text', 'wrong_url', 'missing_sha'])
def test_verified_status_requires_the_pinned_official_policy(tmp_path, change):
    root = tmp_path/'capture'; shutil.copytree(SOURCE, root)
    manifest = json.loads((root/'capture_manifest.json').read_bytes())
    if change == 'fake_verified_text':
        raw = b'<p>policy pending, no reproduction permission</p>'
        (root/'raw/linking.html').write_bytes(raw)
        manifest['policy'].update(bytes=len(raw), sha256=prepare.sha(raw))
    elif change == 'wrong_url':
        manifest['policy']['url'] = 'https://example.invalid/linking'
    else:
        del manifest['policy']['sha256']
    a, b = save_reindexed(root, manifest)
    with pytest.raises(ValueError, match='policy'):
        prepare.load_capture(root, a, b)


def test_policy_is_propagated_consistently_and_contains_exact_limits(tmp_path):
    result = prepare.build(tmp_path/'candidate', SOURCE, MANIFEST_SHA, INDEX_SHA)
    root = tmp_path/'candidate'
    manifest = json.loads((root/'manifest.json').read_bytes())
    task = json.loads((root/'tasks.jsonl').read_bytes())
    audit = json.loads((root/'audit_manifest.json').read_bytes())
    assert task['source']['license_or_privacy'] != 'policy_pending'
    for payload in (manifest, task['source']['provenance'], task['metadata']['extra'], audit['items'][0]['material']):
        assert payload['policy_status'] == 'POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY'
        assert payload['policy_evidence']['source_sha256'] == POLICY_SHA
        assert payload['policy_evidence']['source_url'] == URL
    policy = manifest['policy_evidence']
    assert policy['official_citation_requirement'] == 'appreciated, not stated as mandatory'
    assert 'unless stated otherwise' in policy['quotes']['reuse']
    assert 'would be appreciated' in policy['quotes']['reuse']
    assert policy['raw_source_redistribution'] is False and policy['logos_included'] is False
    assert 'not an official BEA product' in policy['derivation_notice']
    class Paragraphs(HTMLParser):
        def __init__(self):
            super().__init__(); self.active = False; self.text = ''; self.rows = []
        def handle_starttag(self, tag, attrs):
            if tag == 'p': self.active = True; self.text = ''
        def handle_data(self, data):
            if self.active: self.text += data
        def handle_endtag(self, tag):
            if tag == 'p' and self.active:
                self.rows.append(' '.join(self.text.split())); self.active = False
    parser = Paragraphs(); parser.feed((SOURCE/'raw/linking.html').read_text(encoding='utf-8'))
    assert all(text in parser.rows for text in policy['quotes'].values())
    assert 'logo' in policy['quotes']['reuse']
    assert 'must not contain information that suggests such an endorsement' in policy['quotes']['endorsement']
    assert 'cannot authorize the use of copyrighted materials' in policy['quotes']['external_copyright']
    assert (root/'SOURCE_ATTRIBUTION.txt').is_file()
    assert prepare.validate_candidate(root, result['index_sha256'])['status'] == 'VALID_PENDING_HUMAN_REVIEW'
    assert manifest['source_audit_status'] == 'NOT_STARTED' and manifest['human_approved_sources'] == 0


def test_pending_capture_remains_unchanged_and_readable():
    root = PROJECT/'external/family_h_cross_domain_bea_gdp_20260930_v5'
    result = prepare.load_capture(root, '92f52342687578802b7d53213ac157068b00785a98f3a6031155a4fa9520fd24',
                                  'f1c68542f5a5ee96acb6840c668ea0a20e137348dbb5644879cb6f769a3911b9')
    assert result['policy_status'] == 'POLICY_PENDING'
