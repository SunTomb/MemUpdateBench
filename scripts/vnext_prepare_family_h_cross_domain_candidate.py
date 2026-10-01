"""Compile a review-pending cross-domain NOAA advisory candidate."""
from __future__ import annotations
import argparse, hashlib, json, re
from datetime import datetime
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import AnswerSchema, Difficulty, EvaluationMode, EventRole, Operation, QueryType, SourceType, Split, TaskFamily
from mub.vnext.contracts.v3.adapter import ResetRequestV3
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, typed_json_equal, object_identity
from mub.vnext.contracts.v3.task import CurrentSelector, DerivationStepV3, GeneratorProvenanceV3, GoldActionV3, MemoryEventV3, MemoryQueryV3, MemUpdateTaskV3, QueryGoldEvidenceV3, VersionHistoryEntry, VersionHistoryLedger
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import action_id, core_id, event_id, query_id, stable_id, task_id
from mub.vnext.io import semantic_task_hash_v3
from mub.vnext.validation.replay_v3 import replay_task_v3, evaluate_evidence_v3
from scripts.vnext_capture_family_h_noaa_candidate import parse_advisory
COMPILER = 'family-h-cross-domain-noaa-candidate-v2'
SOURCE_GROUP = 'family-h-cross-domain-noaa-beryl-2024'


def canonical(value): return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)

def load_json(path): return json.loads(Path(path).read_bytes())

def _artifact_records(root):
    index = load_json(root/'capture_index.json')
    required = {row['path'] for row in index['artifacts']} | {'capture_index.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == required, 'capture artifact membership changed')
    for row in index['artifacts']:
        path = root/row['path']; raw = path.read_bytes()
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'capture artifact hash drift')
    return index

def load_capture(root, capture_sha256, index_sha256):
    root = Path(root).absolute(); cap_raw = (root/'capture_manifest.json').read_bytes(); idx_raw = (root/'capture_index.json').read_bytes()
    require(hashlib.sha256(cap_raw).hexdigest() == capture_sha256, 'capture manifest hash mismatch')
    require(hashlib.sha256(idx_raw).hexdigest() == index_sha256, 'capture index hash mismatch')
    index = _artifact_records(root); cap = json.loads(cap_raw)
    require(cap['status'] == 'CAPTURED_PENDING_HUMAN_REVIEW' and cap['schema'].endswith('v2'), 'capture status/schema')
    require(cap['source_group_id'] == SOURCE_GROUP and cap['source_document_id'] == 'nhc-al022024-public-advisories', 'source identity')
    from scripts.vnext_capture_family_h_noaa_candidate import parse_advisory
    records = []
    for item in cap['records']:
        raw = (root/'raw'/f"{item['advisory_id']}.shtml").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == item['sha256'] == item['normalized']['source_anchor']['raw_sha256'], 'advisory bytes drift')
        require(parse_advisory(raw, item['advisory_id'], item['url']) == item['normalized'], 'normalized advisory drift')
        records.append(item['normalized'])
    dates = [datetime.fromisoformat(row['issued_at_utc'].replace('Z', '+00:00')) for row in records]
    require(dates == sorted(dates) and len({r['advisory_id'] for r in records}) == 7, 'advisory chronology/cardinality')
    require(cap['disclaimer_capture']['sha256'] == hashlib.sha256((root/'raw/nws_disclaimer.html').read_bytes()).hexdigest(), 'policy capture drift')
    return {'capture': cap, 'records': records, 'rows': records, 'index': index}

def _guard_output(output_root, capture_root):
    output_root = Path(output_root).absolute(); capture_root = Path(capture_root).absolute()
    require(not output_root.exists(), 'output root already exists')
    require(capture_root not in output_root.parents and output_root not in capture_root.parents, 'output overlaps capture root')
    for parent in output_root.parents:
        if (parent/'tasks.jsonl').is_file() or (parent/'index.json').is_file() or (parent/'candidate_manifest.json').is_file():
            raise ValueError('output is nested under an existing candidate')
    return output_root



    key = FrozenMemoryObjectKey(object_type='weather_observation', namespace='family_h', entity='hurricane-beryl-2024', attribute='max_sustained_wind_mph', subkey=None)
    payload = {'source_group_id': SOURCE_GROUP, 'records': [digest(r) for r in records], 'identity': list(object_identity(key))}
    cid = core_id(TaskFamily.REALISTIC_SOURCE_UPDATE.value, payload); tid = task_id(cid, 0); trajectory = stable_id('trajectory', payload)
    events, actions, core_events = [], [], []
    for i, row in enumerate(records):
        op = Operation.ADD if i == 0 else Operation.UPDATE; eid = event_id(tid, i); aid = action_id(tid, i, 0)
        role = EventRole.LATEST_GOLD if i == len(records)-1 else EventRole.HISTORICAL_SUPPORT
        visible = {'action': op.value, 'object_key': key.model_dump(mode='json'), 'value': row['value'], 'source_heading_anchor': row['record_id'], 'normalized_record_sha256': digest(row)}
        text = canonical(visible).decode()
        core_events.append(CoreEvent(operation=op, object_keys=[MemoryObjectKey(**key.model_dump(mode='python'))], value=row['value'], role=role, metadata={'logical_time': f'{i:08d}', 'source_anchor': row['source_anchor']}))
        events.append(MemoryEventV3(event_id=eid, sequence_index=i, timestamp=f'{i:08d}', raw_text=text, normalized_text=text, gold_action_ids=(aid,), role=role, source_anchor=row['source_anchor'], metadata={'source_group_id': SOURCE_GROUP}))
        actions.append(GoldActionV3(action_id=aid, event_id=eid, operation=op, scope='object', target_object_keys=(key,), value=row['value'], effective_at=f'{i:08d}', expected_effect={'canonical_object_id': key.canonical_id}))
    core = SemanticCore(core_id=cid, task_family=TaskFamily.REALISTIC_SOURCE_UPDATE, difficulty=Difficulty.HARD, core_index=0, trajectory_id=trajectory, events=core_events, query_targets=[MemoryObjectKey(**key.model_dump(mode='python'))], query_type=QueryType.CURRENT_STATE, query_selector=CurrentSelector(), expected_answer=records[-1]['value'], profile={'source_type': 'public_advisory', 'source_domain': 'weather', 'snapshot_count': 7, 'update_depth': 6, 'candidate_only': True, 'source_group_id': SOURCE_GROUP}, stratification={'source_type': 'public_advisory', 'source_domain': 'weather'})
    ledger = VersionHistoryLedger(object_key=key, entries=tuple(VersionHistoryEntry(version_index=i, status='present', value=a.value, valid_from_event_id=a.event_id, valid_until_event_id=actions[i+1].event_id if i+1 < len(actions) else None, logical_time=a.effective_at, source_event_ids=(a.event_id,)) for i, a in enumerate(actions)))
    qid = query_id(tid, 0); query = MemoryQueryV3(query_id=qid, query_type='current', text=f'Within the supplied public advisories, return the current wind value for object {key.canonical_id}.', selector=CurrentSelector(), target_object_keys=(key,), answer_schema=AnswerSchema.NUMBER, evaluation_mode=EvaluationMode.RETRIEVED_PROMPT)
    step = DerivationStepV3(step_id=f'derive_{qid}_read', operation='read_current', supporting_object_keys=(key,), supporting_event_ids=(actions[-1].event_id,))
    evidence = QueryGoldEvidenceV3(query_id=qid, answer=records[-1]['value'], supporting_object_keys=(key,), supporting_event_ids=(actions[-1].event_id,), derivation_steps=(step,), final_derivation_step_id=step.step_id)
    source = {'source_id': 'noaa-nhc-beryl-al02-2024', 'source_type': SourceType.OTHER, 'source_uri': 'https://www.nhc.noaa.gov/archive/2024/al02/', 'license_or_privacy': 'public_domain;public', 'raw_hash': capture_sha256, 'normalized_hash': digest(records), 'normalization_version': COMPILER, 'provenance': {'source_group_id': SOURCE_GROUP, 'source_document_id': 'nhc-al022024-public-advisories', 'source_domain': 'weather', 'source_kind': 'sequential_public_advisory', 'public_domain_policy_url': 'https://www.weather.gov/disclaimer', 'source_audit_status': 'NOT_STARTED', 'generated_surface_audit_status': 'NOT_STARTED'}, 'generator': GeneratorProvenanceV3(generator_name='family_h_cross_domain_noaa_candidate', seed=0, config_sha256=capture_sha256, code_revision=compiler_hash, compiler_version=COMPILER)}
    task = MemUpdateTaskV3(task_id=tid, task_family=TaskFamily.REALISTIC_SOURCE_UPDATE.value, difficulty=Difficulty.HARD, source=source, events=tuple(events), target_objects=(key,), actions=tuple(actions), queries=(query,), version_history=(ledger,), gold_evidence=(evidence,), metadata={'split': Split.EVALUATION_ONLY, 'split_key': {'semantic_core_id': core.core_id, 'source_group_id': SOURCE_GROUP, 'trajectory_id': trajectory, 'source_document_id': 'nhc-al022024-public-advisories', 'version_group_id': stable_id('version_group', payload), 'split_policy_version': COMPILER}, 'profile_name': Difficulty.HARD, 'resolved_profile': {'task_family': TaskFamily.REALISTIC_SOURCE_UPDATE.value, 'source_type': 'public_advisory', 'source_domain': 'weather', 'snapshot_count': 7, 'update_depth': 6, 'candidate_only': True}, 'generation_config_hash': capture_sha256, 'compiler_version': COMPILER, 'tags': ('family_h', 'cross_domain', 'weather', 'review_pending'), 'extra': {'source_audit_status': 'NOT_STARTED', 'generated_surface_audit_status': 'NOT_STARTED', 'scientific_release_allowed': False, 'natural_language_discovery': 'NOT_CLAIMED', 'public_surface': 'generated structured JSON from public advisory facts'}})
    return core, task

def validate_candidate(root, expected_index_sha256):
    root = Path(root).absolute(); raw = (root/'index.json').read_bytes(); require(hashlib.sha256(raw).hexdigest() == expected_index_sha256, 'candidate index hash')
    idx = json.loads(raw); require(idx['status'] == 'CANDIDATE_PENDING_HUMAN_REVIEW' and idx['scientific_release_allowed'] is False, 'candidate status')
    for row in idx['artifacts']:
        data = (root/row['path']).read_bytes(); require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'candidate artifact hash')
    tasks = [MemUpdateTaskV3.model_validate(json.loads(line)) for line in (root/'tasks.jsonl').read_bytes().splitlines()]
    require(len(tasks) == 1 and replay_task_v3(tasks[0]).valid, 'candidate task replay')
    audit = json.loads((root/'audit_manifest.json').read_bytes()); require(audit['status'] == 'NOT_STARTED' and audit['item_count'] == 9, 'audit boundary')
    surface = next(i for i in audit['items'] if i['kind'] == 'generated_surface'); require(surface['material']['canonical_task_sha256'] == digest(tasks[0].model_dump(mode='json')), 'surface task binding')
    require(all(digest(item['material']) == item['binding_sha256'] for item in audit['items']), 'audit item binding')
    return {**idx, 'status': 'VALID_PENDING_HUMAN_REVIEW'}

def validate_review(audit, review, candidate_index_sha256):
    require(review.get('reviewer'), 'reviewer required'); require(review.get('candidate_index_sha256') == candidate_index_sha256, 'candidate review binding')
    expected = {i['audit_id']: i['binding_sha256'] for i in audit['items']}; rows = review.get('decisions'); require(len(rows) == len(expected), 'decision cardinality')
    seen=set();counts={}
    for row in rows:
        require(set(row) == {'audit_id','binding_sha256','decision','rationale'}, 'decision schema'); identifier=row['audit_id']; require(identifier in expected and identifier not in seen, 'decision ID');seen.add(identifier);require(row['binding_sha256']==expected[identifier],'decision binding');require(row['decision'] in ('RELEASE_READY','NEEDS_FIX','UNSUPPORTED'),'decision');require(row['rationale'].strip(),'rationale');counts[row['decision']]=counts.get(row['decision'],0)+1
    return {'status':'DECISIONS_VALIDATED','decision_count':len(rows),'counts':counts,'all_release_ready':counts.get('RELEASE_READY')==len(expected),'scientific_release_allowed':False}

    return {'capture': cap, 'records': records, 'rows': records, 'index': index}

def _make_task(records, capture_sha256, compiler_hash):
    key = FrozenMemoryObjectKey(object_type='weather_observation', namespace='family_h', entity='hurricane-beryl-2024', attribute='max_sustained_wind_mph', subkey=None)
    payload = {'source_group_id': SOURCE_GROUP, 'records': [digest(r) for r in records], 'identity': list(object_identity(key))}
    cid = core_id(TaskFamily.REALISTIC_SOURCE_UPDATE.value, payload); tid = task_id(cid, 0); trajectory = stable_id('trajectory', payload)
    events=[]; actions=[]; core_events=[]
    for i,row in enumerate(records):
        op=Operation.ADD if i==0 else Operation.UPDATE; eid=event_id(tid,i); aid=action_id(tid,i,0); role=EventRole.LATEST_GOLD if i==len(records)-1 else EventRole.HISTORICAL_SUPPORT
        visible={'action':op.value,'object_key':key.model_dump(mode='json'),'value':row['value'],'source_heading_anchor':row['record_id'],'normalized_record_sha256':digest(row)}
        text=canonical(visible).decode()
        core_events.append(CoreEvent(operation=op,object_keys=[MemoryObjectKey(**key.model_dump(mode='python'))],value=row['value'],role=role,metadata={'logical_time':f'{i:08d}','source_anchor':row['source_anchor']}))
        events.append(MemoryEventV3(event_id=eid,sequence_index=i,timestamp=f'{i:08d}',raw_text=text,normalized_text=text,gold_action_ids=(aid,),role=role,source_anchor=row['source_anchor'],metadata={'source_group_id':SOURCE_GROUP}))
        actions.append(GoldActionV3(action_id=aid,event_id=eid,operation=op,scope='object',target_object_keys=(key,),value=row['value'],effective_at=f'{i:08d}',expected_effect={'canonical_object_id':key.canonical_id}))
    core=SemanticCore(core_id=cid,task_family=TaskFamily.REALISTIC_SOURCE_UPDATE,difficulty=Difficulty.HARD,core_index=0,trajectory_id=trajectory,events=core_events,query_targets=[MemoryObjectKey(**key.model_dump(mode='python'))],query_type=QueryType.CURRENT_STATE,query_selector=CurrentSelector(),expected_answer=records[-1]['value'],profile={'source_type':'public_advisory','source_domain':'weather','snapshot_count':7,'update_depth':6,'candidate_only':True,'source_group_id':SOURCE_GROUP},stratification={'source_type':'public_advisory','source_domain':'weather'})
    ledger=VersionHistoryLedger(object_key=key,entries=tuple(VersionHistoryEntry(version_index=i,status='present',value=a.value,valid_from_event_id=a.event_id,valid_until_event_id=actions[i+1].event_id if i+1<len(actions) else None,logical_time=a.effective_at,source_event_ids=(a.event_id,)) for i,a in enumerate(actions)))
    qid=query_id(tid,0);query=MemoryQueryV3(query_id=qid,query_type='current',text=f'Within the supplied public advisories, return the current wind value for object {key.canonical_id}.',selector=CurrentSelector(),target_object_keys=(key,),answer_schema=AnswerSchema.NUMBER,evaluation_mode=EvaluationMode.RETRIEVED_PROMPT)
    step=DerivationStepV3(step_id=f'derive_{qid}_read',operation='read_current',supporting_object_keys=(key,),supporting_event_ids=(actions[-1].event_id,));evidence=QueryGoldEvidenceV3(query_id=qid,answer=records[-1]['value'],supporting_object_keys=(key,),supporting_event_ids=(actions[-1].event_id,),derivation_steps=(step,),final_derivation_step_id=step.step_id)
    source={'source_id':'noaa-nhc-beryl-al02-2024','source_type':SourceType.OTHER,'source_uri':'https://www.nhc.noaa.gov/archive/2024/al02/','license_or_privacy':'public_domain;public','raw_hash':capture_sha256,'normalized_hash':digest(records),'normalization_version':COMPILER,'provenance':{'source_group_id':SOURCE_GROUP,'source_document_id':'nhc-al022024-public-advisories','source_domain':'weather','source_kind':'sequential_public_advisory','public_domain_policy_url':'https://www.weather.gov/disclaimer','source_audit_status':'NOT_STARTED','generated_surface_audit_status':'NOT_STARTED'},'generator':GeneratorProvenanceV3(generator_name='family_h_cross_domain_noaa_candidate',seed=0,config_sha256=capture_sha256,code_revision=compiler_hash,compiler_version=COMPILER)}
    return core,MemUpdateTaskV3(task_id=tid,task_family=TaskFamily.REALISTIC_SOURCE_UPDATE.value,difficulty=Difficulty.HARD,source=source,events=tuple(events),target_objects=(key,),actions=tuple(actions),queries=(query,),version_history=(ledger,),gold_evidence=(evidence,),metadata={'split':Split.EVALUATION_ONLY,'split_key':{'semantic_core_id':cid,'source_group_id':SOURCE_GROUP,'trajectory_id':trajectory,'source_document_id':'nhc-al022024-public-advisories','version_group_id':stable_id('version_group',payload),'split_policy_version':COMPILER},'profile_name':Difficulty.HARD,'resolved_profile':{'task_family':TaskFamily.REALISTIC_SOURCE_UPDATE.value,'source_type':'public_advisory','source_domain':'weather','snapshot_count':7,'update_depth':6,'candidate_only':True},'generation_config_hash':capture_sha256,'compiler_version':COMPILER,'tags':('family_h','cross_domain','weather','review_pending'),'extra':{'source_audit_status':'NOT_STARTED','generated_surface_audit_status':'NOT_STARTED','scientific_release_allowed':False,'natural_language_discovery':'NOT_CLAIMED','public_surface':'generated structured JSON from public advisory facts'}})

def build(output_root: Path, capture_root: Path, capture_sha256: str, capture_index_sha256: str):
    output_root=_guard_output(output_root,capture_root);loaded=load_capture(capture_root,capture_sha256,capture_index_sha256);compiler_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();core,task=_make_task(loaded['records'],capture_sha256,compiler_hash);replay=replay_task_v3(task);evidence=evaluate_evidence_v3(task.gold_evidence[0],replay,query=task.queries[0],events=task.events);require(replay.valid and evidence.valid and typed_json_equal(evidence.answer,loaded['records'][-1]['value']),'reference sanity');adapter=ReferenceAdapterV3(task);require(adapter.reset(ResetRequestV3(namespace='family-h-noaa-candidate')).success,'reset');[adapter.ingest_event(e) for e in task.events];pred=adapter.answer(task.queries[0],'slot_direct').prediction;adapter.close();require(pred.format_valid and typed_json_equal(pred.parsed_answer,loaded['records'][-1]['value']),'reference answer')
    snapshot_items=[]
    for row in loaded['records']:
        material={'record_id':row['record_id'],'advisory_id':row['advisory_id'],'issued_at_utc':row['issued_at_utc'],'value':row['value'],'value_unit':row['value_unit'],'source_anchor':row['source_anchor'],'raw_sha256':row['source_anchor']['raw_sha256']};snapshot_items.append({'audit_id':'snapshot-'+row['advisory_id'],'kind':'source_snapshot','binding_sha256':digest(material),'material':material})
    surface={'task_id':task.task_id,'canonical_task_sha256':digest(task.model_dump(mode='json')),'semantic_task_sha256':semantic_task_hash_v3(task),'event_count':7,'query':task.queries[0].text};source={'audit_id':'source-noaa-beryl','kind':'source_admission','binding_sha256':digest(loaded['capture']),'material':{'source_group_id':SOURCE_GROUP,'capture_manifest_sha256':capture_sha256,'capture_index_sha256':capture_index_sha256,'advisory_count':7,'source_domain':'weather','public_domain_policy_url':'https://www.weather.gov/disclaimer'}};items=[source,*snapshot_items,{'audit_id':'surface-noaa-beryl','kind':'generated_surface','binding_sha256':digest(surface),'material':surface}]
    for item in items:
        item['binding_sha256'] = digest(item['material'])
    manifest={'schema':COMPILER+'-manifest-v3','status':'CANDIDATE_PENDING_HUMAN_REVIEW','source_domain':'weather','source_group_count':1,'semantic_core_count':1,'task_count':1,'snapshot_count':7,'event_count':7,'source_count_semantics':'one upstream storm trajectory, not statistically independent samples','source_audit_status':'NOT_STARTED','generated_surface_audit_status':'NOT_STARTED','human_approved_sources':0,'capture_manifest_sha256':capture_sha256,'capture_index_sha256':capture_index_sha256,'public_domain_policy_url':'https://www.weather.gov/disclaimer','formal_task_release':False,'scientific_release_allowed':False,'answer_metrics':None,'update_events':6,'value_changing_updates':5,'equal_value_update_events':1,'reference_sanity':{'status':'PASS','events':7,'model_loads':0,'generations':0}}
    core_json=core.model_dump(mode='json');files={'manifest.json':canonical(manifest),'normalized_records.jsonl':b''.join(canonical(r) for r in loaded['records']),'semantic_cores.jsonl':canonical(core_json),'tasks.jsonl':canonical(task.model_dump(mode='json')),'audit_manifest.json':canonical({'schema':COMPILER+'-audit-v3','status':'NOT_STARTED','items':items,'item_count':len(items)}),'decisions_template.json':canonical({'schema':COMPILER+'-human-decisions-v1','status':'NOT_STARTED','reviewer':None,'candidate_index_sha256':None,'decisions':[{'audit_id':i['audit_id'],'binding_sha256':i['binding_sha256'],'decision':None,'rationale':''} for i in items]}),'reference_sanity.json':canonical(manifest['reference_sanity'])};index={'schema':COMPILER+'-index-v3','status':manifest['status'],'scientific_release_allowed':False,'artifacts':[{'path':n,'bytes':len(r),'sha256':hashlib.sha256(r).hexdigest()} for n,r in sorted(files.items())]};files['index.json']=canonical(index);output_root.mkdir(parents=True);[ (output_root/n).write_bytes(r) for n,r in files.items()];return {'status':manifest['status'],'advisories':7,'events':7,'index_sha256':hashlib.sha256(files['index.json']).hexdigest(),'capture_manifest_sha256':capture_sha256,'capture_index_sha256':capture_index_sha256}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--capture-root',type=Path,required=True);p.add_argument('--capture-sha256',required=True);p.add_argument('--capture-index-sha256',required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(build(a.output_root,a.capture_root,a.capture_sha256,a.capture_index_sha256),sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
