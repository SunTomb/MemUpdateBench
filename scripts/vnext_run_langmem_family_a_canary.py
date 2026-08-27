from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from mub.vnext.contracts.v3.adapter import ResetRequestV3,RetrievalRequestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.bridge import WorkerRequestV1
from mub.vnext.external.providers.langmem import build_langmem_adapter_configuration
from mub.vnext.external.providers.langmem_adapter import LangMemExternalAdapterV3
from mub.vnext.external.workers.langmem_worker import OfficialLangMemBackendV1,LangMemWorkerServiceV1
TASK_SHA='ef352d6eb719389bcab39d4746ad97fe7f1b0489f4fa402f15e039e33c5c2ac6'
canon=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
sha=lambda b:hashlib.sha256(b).hexdigest()
class Bridge:
 def __init__(self,s):self.s=s;self.closed=False
 def request(self,r:WorkerRequestV1):return self.s.handle(r)
 def close(self):self.closed=True

def core(t):return t.metadata.extra['semantic_core_id']
def main():
 p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--output-root',required=True);a=p.parse_args();tp=Path(a.tasks);out=Path(a.output_root)
 if out.exists():raise FileExistsError(out)
 raw=tp.read_bytes();assert sha(raw)==TASK_SHA
 tasks=[MemUpdateTaskV3.model_validate(json.loads(x)) for x in raw.splitlines()];tasks.sort(key=lambda t:(core(t).encode(),t.task_id.encode()));cores=sorted({core(t) for t in tasks})[:8];selected=[t for t in tasks if core(t) in cores];assert len(selected)==32 and all(sum(core(t)==c for t in selected)==4 for c in cores)
 out.mkdir(parents=True);rows=[]
 for t in selected:
  start=time.monotonic();gold=t.gold_evidence[0].answer
  if len(t.target_objects)!=1:
   rows.append({'schema_version':'memupdatebench.external.langmem.family-a-canary-row.v1','task_id':t.task_id,'semantic_core_id':core(t),'status':'NOT_SUPPORTED','unsupported_reason':'profile_single_record_only','state_accuracy':None,'final_memory_size':None,'stale_same_slot_count':None,'gold_retrieved_k16':None,'stale_retrieved_k16':None,'stable_entry_id':None,'gold_sha256':sha(canon(gold)),'latency_ms':(time.monotonic()-start)*1000});(out/'rows.jsonl').write_bytes(b''.join(canon(x)+b'\n' for x in rows));continue
  cfg=build_langmem_adapter_configuration(run_id=f'langmem-canary-{t.task_id}');bridge=Bridge(LangMemWorkerServiceV1(OfficialLangMemBackendV1(cfg)));adapter=LangMemExternalAdapterV3(bridge=bridge,configuration=cfg,target_objects=tuple(t.target_objects));status='PASS';err=None
  try:
   adapter.reset(ResetRequestV3(namespace=f'langmem_canary_{t.task_id}'));ids=[]
   for e in t.events:
    r=adapter.ingest_event(e)
    if r.affected_entry_ids:ids.extend(r.affected_entry_ids)
   entries=adapter.export_entries().entries;query=t.queries[0];retr=adapter.retrieve(RetrievalRequestV3(query=query,k=16)).trace;ans=adapter.answer(query,'slot_direct').prediction;gold=t.gold_evidence[0].answer
   state_acc=bool(ans.format_valid and ans.parsed_answer==gold);memory_size=len(entries);stale=sum(e.value_candidate!=gold for e in entries);retr_gold=any(e.value_candidate==gold for e in retr.retrieved_entries);retr_stale=sum(e.value_candidate!=gold for e in retr.retrieved_entries);stable_ids=len(set(ids))<=1
  except Exception as exc:
   status='FAIL';err='VISIBLE_SURFACE_UNSUPPORTED' if t.events and t.events[0].raw_text!=t.events[0].normalized_text else type(exc).__name__;state_acc=False;memory_size=None;stale=None;retr_gold=None;retr_stale=None;stable_ids=False;gold=t.gold_evidence[0].answer
  finally:adapter.close()
  row={'schema_version':'memupdatebench.external.langmem.family-a-canary-row.v1','task_id':t.task_id,'semantic_core_id':core(t),'status':status,'error_class':err,'state_accuracy':state_acc,'final_memory_size':memory_size,'stale_same_slot_count':stale,'gold_retrieved_k16':retr_gold,'stale_retrieved_k16':retr_stale,'stable_entry_id':stable_ids,'gold_sha256':sha(canon(gold)),'latency_ms':(time.monotonic()-start)*1000};rows.append(row);(out/'rows.jsonl').write_bytes(b''.join(canon(x)+b'\n' for x in rows))
 supported=[r for r in rows if r['status']!='NOT_SUPPORTED'];passed=[r for r in supported if r['status']=='PASS'];n=len(passed)
 summary={'schema_version':'memupdatebench.external.langmem.family-a-canary.v1','candidate_id':'langmem_0_0_30_profile','admission_scope':'profile_single_record_only','task_view_sha256':TASK_SHA,'requested_task_count':32,'supported_task_count':len(supported),'unsupported_task_count':32-len(supported),'semantic_core_count':8,'pass_count':n,'fail_count':sum(r['status']=='FAIL' for r in supported),'failure_classes':dict(__import__('collections').Counter(r['error_class'] for r in supported if r.get('error_class'))),'state_accuracy_mean':sum(bool(r['state_accuracy']) for r in passed)/n if n else None,'avg_memory_size':sum(r['final_memory_size'] for r in passed)/n if n else None,'avg_stale_same_slot':sum(r['stale_same_slot_count'] for r in passed)/n if n else None,'gold_retrieval_rate_k16':sum(bool(r['gold_retrieved_k16']) for r in passed)/n if n else None,'avg_stale_retrieved_k16':sum(r['stale_retrieved_k16'] for r in passed)/n if n else None,'stable_id_rate':sum(bool(r['stable_entry_id']) for r in passed)/n if n else None,'prompted_answer_status':'BLOCKED_BY_INGEST','llm_used':False,'api_used':False,'gpu_used':False,'rows_sha256':sha((out/'rows.jsonl').read_bytes())};summary['payload_sha256']=sha(canon(summary));(out/'canary_receipt.json').write_bytes(canon(summary));print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
