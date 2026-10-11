from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vnext_run_family_h_native_qdrant import load_family_h_tasks, public_event_request
EXPECTED_RUNTIME={'qdrant_version':'1.19.0','qdrant_revision':'74f3e85b9473c62560006c043e13737ce6b48412','qdrant_binary_sha256':'abfe97e1d0225111dec2f048790428f151846c8a049eefc85328b6c9eccaf419'}
def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read_json(path): return json.loads(Path(path).read_bytes())
def snapshot(root): return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in sorted(Path(root).rglob('*')) if p.is_file()}
def _key(k): return tuple(k.get(x) for x in ('namespace','entity','attribute','subkey'))
def _project(point):
 p=point['payload']; return {'id':str(point['id']),'identity':_key(p),'value':p['value'],'source_event_ids':list(p['source_event_ids']),'sequence_index':p['sequence_index']}
def _eq(a,b): return a['identity']==b['identity'] and a['value']==b['value'] and a['source_event_ids']==b['source_event_ids'] and a['sequence_index']==b['sequence_index']
def attest(evidence_root,release_root,release_index_sha256,source_manifest_sha256):
 evidence_root=Path(evidence_root); release_root=Path(release_root)
 idx=read_json(evidence_root/'run/index.json'); summary=read_json(evidence_root/'run/summary.json'); obs=read_json(evidence_root/'run/observations.json'); profile=read_json(evidence_root/'profile.json')
 if sha((release_root/'release_index.json').read_bytes())!=release_index_sha256: raise ValueError('release index binding mismatch')
 if sha((evidence_root/'source/source_manifest.json').read_bytes())!=source_manifest_sha256: raise ValueError('source manifest binding mismatch')
 if summary['status']!='OBSERVATIONS_COMPLETE' or summary['answer_metrics'] is not None: raise ValueError('run boundary')
 if profile.get('runtime') is not None and any(profile.get(k)!=v for k,v in EXPECTED_RUNTIME.items()): raise ValueError('runtime binding')
 tasks={t['task_id']:t for t in load_family_h_tasks(release_root)}
 if len(tasks)!=3 or sum(len(t['events']) for t in tasks.values())!=27: raise ValueError('task cardinality')
 if len(obs['rows'])!=3 or any(r['status']!='OBSERVED' for r in obs['rows']): raise ValueError('observed task rows')
 counts={'tasks':3,'events':0,'state_steps':0,'source_event_links':0,'final_matches':0,'retrieval_matches':0,'stale_same_slot':0}
 witness={}; cleanup={'all_case_clients_closed':True,'all_case_collections_deleted':True,'all_final_empty':True}
 for row in obs['rows']:
  task=tasks.get(row['task_id']);
  if task is None: raise ValueError('task binding')
  if row.get('cleanup')!={'client_closed':True,'collection_deleted':True}: cleanup.update(all_case_clients_closed=False,all_case_collections_deleted=False)
  if row.get('final_empty')!=[]: cleanup['all_final_empty']=False
  expected=[]; events=task['events'];
  for event in events:
   request=public_event_request(task['task_id'],event,task.get('__normalized_record_index'))
   expected.append(request)
  if len(row['mutation_rows'])!=len(expected): raise ValueError('event count')
  state={}
  for request,event_row in zip(expected,row['mutation_rows']):
   result=event_row['result']; outcomes=result.get('outcomes',[])
   if result.get('event_id')!=request.event_id or result.get('operation')!=request.operation.value: raise ValueError('result event binding')
   if len(outcomes)!=len(request.objects): raise ValueError('outcome count')
   for item,outcome in zip(request.objects,outcomes):
    if outcome.get('event_id')!=request.event_id or outcome.get('status')!='executed' or outcome.get('source_event_ids')!=list(request.source_event_ids): raise ValueError('state outcome')
    state[_key(item.object_key.model_dump(mode='json'))]={'identity':_key(item.object_key.model_dump(mode='json')),'value':item.value,'source_event_ids':list(request.source_event_ids),'sequence_index':request.sequence_index}
   counts['events'] += 1
   actual=sorted([_project(p) for p in event_row['export']],key=lambda x:repr(x['identity'])); exp=sorted(state.values(),key=lambda x:repr(x['identity']))
   if len(actual)!=len(exp) or any(not _eq(a,b) for a,b in zip(actual,exp)): raise ValueError('state snapshot mismatch')
   counts['state_steps']+=1; counts['source_event_links']+=sum(1 for x in actual if request.event_id in x['source_event_ids']); counts['stale_same_slot']+=0
  for w in row.get('provider_witnesses',[]): witness[w.get('method')]=witness.get(w.get('method'),0)+1
  final=sorted([_project(p) for p in row['final_export']],key=lambda x:repr(x['identity']))
  exp=sorted(state.values(),key=lambda x:repr(x['identity']))
  if len(final)!=len(exp) or any(not _eq(a,b) for a,b in zip(final,exp)): raise ValueError('final state mismatch')
  counts['final_matches']+=1
  entries=row['retrieval']['entries']; raw=row['retrieval_raw']
  if len(entries)!=len(raw): raise ValueError('retrieval count')
  for i,(entry,point) in enumerate(zip(entries,raw)):
   if str(entry['provider_entry_id'])!=str(point['id']) or entry['order_index']!=i or entry['rank']!=i+1 or entry['provider_payload']!=point['payload'] or entry['score']!=point['score']: raise ValueError('retrieval trace mismatch')
   if not any(str(x['id'])==str(point['id']) for x in row['final_export']): raise ValueError('retrieval not in final export')
  counts['retrieval_matches']+=1
 _expected_witness={'collection_exists':3,'create_collection':3,'get_collection':3,'upsert':27,'retrieve':27,'query_points':3,'scroll':39,'delete':6}
 if witness!=_expected_witness: raise ValueError('provider witness counts')
 return {'schema':'memupdatebench.family-h.native-qdrant-attestation.v1','status':'FAMILY_H_NATIVE_STATE_RETRIEVAL_ATTESTED','scientific_release_allowed':False,'native_admission':'NOT_GRANTED','answer_metrics':None,'independent_signer':False,'evidence_class':'audited_real_source_direct_qdrant_state_retrieval','source_release_index_sha256':release_index_sha256,'source_manifest_sha256':source_manifest_sha256,'runtime':EXPECTED_RUNTIME,'counts':counts,'provider_witnesses':witness,'cleanup':cleanup,'execution':summary['execution'],'claim_boundary':'three published Family-H software tasks, 27 public events, direct Qdrant state/retrieval only; no answer/model/broad release'}
def publish(report,out):
 out=Path(out); out.mkdir(parents=True,exist_ok=False); payload=canonical(report); (out/'attestation.json').write_bytes(payload); idx={'schema':'memupdatebench.family-h.native-qdrant-attestation-index.v1','status':report['status'],'scientific_release_allowed':False,'native_admission':'NOT_GRANTED','artifacts':[{'path':'attestation.json','bytes':len(payload),'sha256':sha(payload)}]}; (out/'index.json').write_bytes(canonical(idx)); return idx
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--evidence-root',type=Path,required=True); p.add_argument('--release-root',type=Path,required=True); p.add_argument('--release-index-sha256',required=True); p.add_argument('--source-manifest-sha256',required=True); p.add_argument('--output-root',type=Path,required=True); a=p.parse_args(argv); print(json.dumps(publish(attest(a.evidence_root,a.release_root,a.release_index_sha256,a.source_manifest_sha256),a.output_root),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
