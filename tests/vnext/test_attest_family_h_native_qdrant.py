from __future__ import annotations

from pathlib import Path
import shutil
import pytest

from scripts import vnext_attest_family_h_native_qdrant as attestor

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'results/vnext/family_h_native_qdrant_full_20261011_v2_collected'
RELEASE=ROOT/'data/vnext/family_h_independent/v1_published'
RELEASE_SHA='c092c94312d6dc8793862ea84549ffc6f0616fc33da37e1c8ca528685db25261'
SOURCE_SHA='e640e984668cb7866837e58f88453eda61054c83945c25914d71903fe3f9afe2'

def test_attest_family_h_v8_state_retrieval_without_answer_release():
    report=attestor.attest(EVIDENCE,RELEASE,RELEASE_SHA,SOURCE_SHA)
    assert report['status']=='FAMILY_H_NATIVE_STATE_RETRIEVAL_ATTESTED'
    assert report['scientific_release_allowed'] is False
    assert report['answer_metrics'] is None
    assert report['counts']=={'tasks':3,'events':27,'state_steps':27,'source_event_links':27,'final_matches':3,'retrieval_matches':3,'stale_same_slot':0}
    assert report['cleanup']=={'all_case_clients_closed':True,'all_case_collections_deleted':True,'all_final_empty':True}

def test_attest_rejects_release_hash_drift():
    with pytest.raises(ValueError,match='release index'):
        attestor.attest(EVIDENCE,RELEASE,'0'*64,SOURCE_SHA)

def test_attest_rejects_tampered_observation(tmp_path):
    evidence=tmp_path/'evidence'; shutil.copytree(EVIDENCE,evidence)
    obs=attestor.read_json(evidence/'run/observations.json'); obs['rows'][0]['mutation_rows'][0]['export'][0]['payload']['value']['version']='tampered'
    (evidence/'run/observations.json').write_bytes(attestor.canonical(obs))
    with pytest.raises(ValueError,match='state|source|snapshot'):
        attestor.attest(evidence,RELEASE,RELEASE_SHA,SOURCE_SHA)

def test_publish_attestation_no_replace(tmp_path):
    report=attestor.attest(EVIDENCE,RELEASE,RELEASE_SHA,SOURCE_SHA)
    out=tmp_path/'out'; attestor.publish(report,out)
    with pytest.raises(FileExistsError): attestor.publish(report,out)
