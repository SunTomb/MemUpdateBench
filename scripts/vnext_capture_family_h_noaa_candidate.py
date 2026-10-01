"""Capture one bounded NHC public-advisory trajectory; no release or runtime claim."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

ADVISORIES = (
    ('001', 'https://www.nhc.noaa.gov/archive/2024/al02/al022024.public.001.shtml'),
    ('002', 'https://www.nhc.noaa.gov/archive/2024/al02/al022024.public.002.shtml'),
    ('003', 'https://www.nhc.noaa.gov/archive/2024/al02/al022024.public.003.shtml'),
    ('003a', 'https://www.nhc.noaa.gov/archive/2024/al02/al022024.public_a.003.shtml'),
    ('004', 'https://www.nhc.noaa.gov/archive/2024/al02/al022024.public.004.shtml'),
    ('004a', 'https://www.nhc.noaa.gov/archive/2024/al02/al022024.public_a.004.shtml'),
    ('005', 'https://www.nhc.noaa.gov/archive/2024/al02/al022024.public.005.shtml'),
)
DISCLAIMER_URL = 'https://www.weather.gov/disclaimer'
USER_AGENT = 'MemUpdateBench-Family-H-cross-domain-capture/1.0'
SOURCE_ID = 'noaa-nhc-beryl-al02-2024'
SOURCE_GROUP = 'family-h-cross-domain-noaa-beryl-2024'
SOURCE_DOCUMENT = 'nhc-al022024-public-advisories'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def fetch(url: str, output: Path) -> dict:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError('source exceeds bounded capture size')
        status = response.status
        content_type = response.headers.get('Content-Type')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return {'url': url, 'http_status': status, 'content_type': content_type,
            'bytes': len(raw), 'sha256': sha(raw), 'path': output.relative_to(output.parents[1]).as_posix()}


def pre_text(raw: bytes) -> str:
    text = raw.decode('utf-8', errors='strict')
    if len(re.findall(r'<pre\b', text, flags=re.I)) != 1 or len(re.findall(r'</pre>', text, flags=re.I)) != 1:
        raise ValueError('NHC advisory must contain exactly one preformatted body')
    match = re.search(r'<pre[^>]*>(?P<body>.*?)</pre>', text, flags=re.I | re.S)
    if not match:
        raise ValueError('NHC advisory preformatted body missing')
    return html.unescape(match.group('body')).replace('\r\n', '\n').replace('\r', '\n')


def parse_advisory(raw: bytes, advisory_id: str, url: str = '') -> dict:
    text = pre_text(raw)
    expected_number = re.fullmatch(r'(\d+)(a)?', advisory_id)
    if not expected_number:
        raise ValueError('invalid advisory identifier')
    number, letter = str(int(expected_number.group(1))), expected_number.group(2)
    header = re.search(
        r'BULLETIN\n(?P<title>[^\n]*Advisory Number\s+[^\n]*)\n'
        r'NWS National Hurricane Center[^\n]*\s+AL022024\s*\n'
        r'(?P<issued>\d{3,4}\s+[AP]M\s+AST\s+[A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d{4})',
        text,
    )
    if not header:
        raise ValueError(f'official NHC header missing: {advisory_id}')
    title = header.group('title')
    title_match = re.search(r'Advisory Number\s+(\d+)([A-Z])?', title)
    if not title_match or title_match.group(1) != number or (title_match.group(2) or '').lower() != (letter or ''):
        raise ValueError(f'advisory number mismatch: {advisory_id}')
    issued_text = header.group('issued')
    issued = datetime.strptime(issued_text, '%I%M %p AST %a %b %d %Y')
    if issued.strftime('%a') != issued_text.split()[3]:
        raise ValueError(f'advisory weekday mismatch: {advisory_id}')
    issued_utc = (issued + timedelta(hours=4)).replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
    section_start = text.find('SUMMARY OF')
    section_end = text.find('\nWATCHES', section_start)
    section = text[section_start:section_end if section_end >= 0 else len(text)]
    if len(re.findall(r'MAXIMUM SUSTAINED WINDS\.\.\.\d+ MPH\.\.\.\d+ KM/H', section)) != 1:
        raise ValueError(f'advisory summary field ambiguity: {advisory_id}')
    summary_match = re.search(
        r'SUMMARY OF\s+(?P<local>\d{3,4}\s+[AP]M AST)\.\.\.(?P<utc>\d{4}) UTC\.\.\.INFORMATION\n'
        r'-+\nLOCATION\.\.\.(?P<location>[0-9.]+N\s+[0-9.]+W)(?P<body>.*?)'
        r'(?P<winds>MAXIMUM SUSTAINED WINDS\.\.\.(?P<mph>\d+) MPH\.\.\.(?P<kmh>\d+) KM/H)',
        text, flags=re.S,
    )
    if not summary_match:
        raise ValueError(f'advisory summary field ambiguity: {advisory_id}')
    summary = summary_match
    local_time = ' '.join(issued_text.split()[:3])
    if not summary or summary.group('local') != local_time:
        raise ValueError(f'advisory summary mismatch: {advisory_id}')
    if issued_utc[11:13] + issued_utc[14:16] != summary.group('utc'):
        raise ValueError(f'advisory UTC mismatch: {advisory_id}')
    wind_mph, wind_kmh = int(summary.group('mph')), int(summary.group('kmh'))
    return {
        'record_id': f'beryl-public-{advisory_id}',
        'source_id': SOURCE_ID,
        'source_kind': 'sequential_public_advisory',
        'source_type': 'public_advisory',
        'source_domain': 'weather',
        'upstream_name': 'National Hurricane Center',
        'logical_locator': url,
        'source_group_id': SOURCE_GROUP,
        'source_document_id': SOURCE_DOCUMENT,
        'advisory_id': advisory_id,
        'advisory_number': int(number),
        'intermediate': bool(letter),
        'corrected_archive_text': 'Corrected' in title,
        'issue_time_text': issued_text,
        'issued_at_utc': issued_utc,
        'location_text': summary.group('location'),
        'value': wind_mph,
        'value_unit': 'mph',
        'max_sustained_wind_mph': wind_mph,
        'max_sustained_wind_kmh': wind_kmh,
        'object_key': {'namespace': 'family_h', 'entity': 'hurricane-beryl-2024',
                       'attribute': 'max_sustained_wind_mph', 'subkey': None, 'object_type': 'weather_observation'},
        'source_anchor': {'advisory_url': url, 'advisory_id': advisory_id,
                          'raw_sha256': sha(raw), 'summary_marker': f'SUMMARY OF {summary.group("utc")} UTC'},
    }


def capture(output_root: Path) -> dict:
    output_root = output_root.absolute()
    if output_root.exists():
        raise FileExistsError(output_root)
    raw_root = output_root / 'raw'
    records = []
    for advisory_id, url in ADVISORIES:
        target = raw_root / f'{advisory_id}.shtml'
        record = fetch(url, target)
        raw = target.read_bytes()
        record.update({'advisory_id': advisory_id, 'normalized': parse_advisory(raw, advisory_id, url),
                       'raw_public_domain_policy': 'NWS web-page information is public domain unless specifically noted; third-party material may have separate terms.'})
        records.append(record)
    disclaimer = fetch(DISCLAIMER_URL, raw_root / 'nws_disclaimer.html')
    manifest = {
        'schema': 'memupdatebench.family-h.cross-domain-source-capture.v2',
        'status': 'CAPTURED_PENDING_HUMAN_REVIEW', 'source_id': SOURCE_ID,
        'source_kind': 'sequential_public_advisory', 'source_group_id': SOURCE_GROUP,
        'source_document_id': SOURCE_DOCUMENT, 'upstream_name': 'National Hurricane Center',
        'source_domain': 'weather', 'source_type': 'public_advisory',
        'public_domain_policy_url': DISCLAIMER_URL, 'source_window': {'start': '2024-06-28', 'end': '2024-06-29'},
        'advisory_count': len(records), 'records': records, 'disclaimer_capture': disclaimer,
        'privacy_status': 'public', 'export_status': 'raw_hash_bound_only', 'natural_language_discovery': 'NOT_CLAIMED',
        'formal_task_release': False, 'scientific_release_allowed': False, 'model_loads': 0, 'generations': 0,
        'external_system_calls': 0,
    }
    (output_root / 'capture_manifest.json').write_bytes(canonical(manifest))
    (output_root / 'normalized_records.jsonl').write_bytes(b''.join(canonical(r['normalized']) for r in records))
    artifacts = [{'path': p.relative_to(output_root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                 for p in sorted(output_root.rglob('*')) if p.is_file()]
    (output_root / 'capture_index.json').write_bytes(canonical({
        'schema': 'memupdatebench.family-h.cross-domain-source-capture-index.v2',
        'status': manifest['status'], 'artifacts': artifacts, 'scientific_release_allowed': False,
    }))
    return {'status': manifest['status'], 'advisories': len(records),
            'capture_manifest_sha256': sha((output_root / 'capture_manifest.json').read_bytes()),
            'index_sha256': sha((output_root / 'capture_index.json').read_bytes()), 'source_group_id': SOURCE_GROUP}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(capture(args.output_root), sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
