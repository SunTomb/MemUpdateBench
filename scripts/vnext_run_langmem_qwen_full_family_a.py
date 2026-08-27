from __future__ import annotations
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.providers.langmem import build_langmem_adapter_configuration
from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1
from mub.vnext.external.workers.langmem_worker import OfficialLangMemBackendV1
TASK_SHA='ef352d6eb719389bcab39d4746ad97fe7f1b0489f4fa402f15e039e33c5c2ac6';MODEL='/NAS/HuggingFaceModels/Qwen3.5-9B';REV='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
canon=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode();sha=lambda b:hashlib.sha256(b).hexdigest()
def core(t):return t.metadata.extra['semantic_core_id']
class Extractor:
 def load(self):
  import torch;from transformers import AutoModelForCausalLM,AutoTokenizer
  self.t=torch;self.tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,local_files_only=True,trust_remote_code=False);self.m=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager').eval()
 def extract(self,text,object_id):
  attribute=object_id.split('|')[2]
  prompt='Convert the visible memory event into one JSON object only. Return exactly keys operation and value. operation is one of add, update, noop, delete. The memory attribute is '+json.dumps(attribute)+'. For add/update, value must be only the scalar value of that attribute, never a profile/object. For noop/delete set value to null. The controller has already bound the only allowed memory object.\nEvent: '+text
  rendered=self.tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False);enc=self.tok(rendered,return_tensors='pt').to('cuda:0');start=time.monotonic()
  with self.t.inference_mode():out=self.m.generate(**enc,do_sample=False,num_beams=1,max_new_tokens=96,use_cache=True,pad_token_id=self.tok.eos_token_id)
  raw=self.tok.decode(out[0,enc.input_ids.shape[-1]:],skip_special_tokens=True).strip();data=json.loads(raw)
  if set(data)!={'operation','value'} or data['operation'] not in {'add','update','noop','delete'}:raise ValueError('invalid extraction')
  data['object_id']=object_id
  return data,raw,int(out.shape[-1]-enc.input_ids.shape[-1]),(time.monotonic()-start)*1000
 def answer(self,query_text,attribute,retrieved):
  payload={'query':query_text,'attribute':attribute,'retrieved_entries':[{'value':e.value,'content':e.content} for e in retrieved]};prompt='Use only the retrieved memory entries to answer the query. Return exactly one JSON object: {"disposition":"answered","answer":...} or {"disposition":"abstained"}.\n'+json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
  rendered=self.tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False);enc=self.tok(rendered,return_tensors='pt').to('cuda:0');start=time.monotonic()
  with self.t.inference_mode():out=self.m.generate(**enc,do_sample=False,num_beams=1,max_new_tokens=64,use_cache=True,pad_token_id=self.tok.eos_token_id)
  raw=self.tok.decode(out[0,enc.input_ids.shape[-1]:],skip_special_tokens=True).strip();data=json.loads(raw)
  if set(data) not in ({'disposition','answer'},{'disposition'}) or data.get('disposition') not in {'answered','abstained'}:raise ValueError('invalid answer envelope')
  return data,raw,int(out.shape[-1]-enc.input_ids.shape[-1]),(time.monotonic()-start)*1000
 def close(self):self.m=None;self.tok=None;gc.collect();self.t.cuda.empty_cache();self.t.cuda.synchronize()
def explicit(data):
 op=data['operation'];oid=data['object_id'];value=data['value']
 if op=='noop':return 'No memory object changes.'
 if op=='delete':return f'Delete {oid} [scope=object; enumerated_targets={oid}; event_logical_time=none; effective_at=now].'
 return f'{op.title()} {oid} with value {json.dumps(value,ensure_ascii=False)}.'
def run(args):
 out=Path(args.output_root)
 if out.exists():raise FileExistsError(out)
 raw=Path(args.tasks).read_bytes();assert sha(raw)==TASK_SHA
 tasks=[MemUpdateTaskV3.model_validate(json.loads(x)) for x in raw.splitlines()];tasks.sort(key=lambda t:(core(t).encode(),t.task_id.encode()));selected=tasks;assert len(selected)==80;out.mkdir(parents=True)
 ex=Extractor();ex.load();rows=[]
 try:
  for t in selected:
   start=time.monotonic();gold=t.gold_evidence[0].answer
   if len(t.target_objects)!=1:
    row={'task_id':t.task_id,'semantic_core_id':core(t),'status':'NOT_SUPPORTED','reason':'profile_single_record_only','state_accuracy':None};rows.append(row);(out/'rows.jsonl').write_bytes(b''.join(canon(x)+b'\n' for x in rows));continue
   oid=t.target_objects[0].canonical_id;backend=OfficialLangMemBackendV1(build_langmem_adapter_configuration(run_id='langmem-qwen-'+t.task_id));ns='langmem_qwen_'+t.task_id;backend.reset_namespace(ns);extracts=[];entries=();status='PASS';err=None
   try:
    for e in t.events:
     data,raw_output,tokens,lat=ex.extract(e.raw_text,oid)
     if data['operation']=='update' and not backend.export_entries(ns): data['operation']='add'
     event=ProviderEventInputV1(event_id=e.event_id,sequence_index=e.sequence_index,logical_time=e.timestamp,raw_text=explicit(data),runtime_namespace=ns);result=backend.ingest_event(event);extracts.append({'event_id':e.event_id,'operation':data['operation'],'output_sha256':sha(raw_output.encode()),'generated_tokens':tokens,'latency_ms':lat,'effective_operation':result.effective_operation})
    entries=backend.export_entries(ns);answer=entries[0].value if len(entries)==1 else None;state=answer==gold;stable=len({e.entry_id for e in entries})<=1;retr=backend.retrieve(ProviderQueryInputV1(runtime_namespace=ns,query_id=t.queries[0].query_id,query_text=t.queries[0].text,k=16));gold_retrieved=any(e.value==gold for e in retr.entries);stale=sum(e.value!=gold for e in retr.entries);envelope,answer_raw,answer_tokens,answer_latency=ex.answer(t.queries[0].text,t.target_objects[0].attribute,retr.entries);prompted_answer=envelope.get('answer') if envelope.get('disposition')=='answered' else None;prompted_em=prompted_answer==gold;answer_meta={'output_sha256':sha(answer_raw.encode()),'generated_tokens':answer_tokens,'latency_ms':answer_latency,'format_valid':True,'disposition':envelope.get('disposition')}
   except Exception as exc:status='FAIL';err=type(exc).__name__;answer=None;state=False;stable=False;gold_retrieved=None;stale=None;prompted_answer=None;prompted_em=False;answer_meta=None
   row={'task_id':t.task_id,'semantic_core_id':core(t),'status':status,'error_class':err,'state_accuracy':state,'parsed_final_value':answer,'prompted_answer':prompted_answer,'prompted_exact_match':prompted_em,'answer_meta':answer_meta,'gold_sha256':sha(canon(gold)),'final_memory_size':len(entries) if status=='PASS' else None,'stable_entry_id':stable,'gold_retrieved_k16':gold_retrieved,'stale_retrieved_k16':stale,'extractions':extracts,'latency_ms':(time.monotonic()-start)*1000};rows.append(row);(out/'rows.jsonl').write_bytes(b''.join(canon(x)+b'\n' for x in rows))
 finally:ex.close()
 supported=[r for r in rows if r['status']!='NOT_SUPPORTED'];passed=[r for r in supported if r['status']=='PASS'];summary={'schema_version':'memupdatebench.external.langmem-qwen-extraction.full-family-a.v1','requested':80,'supported':len(supported),'unsupported':80-len(supported),'pass':len(passed),'fail':len(supported)-len(passed),'state_accuracy':sum(r['state_accuracy'] for r in supported)/len(supported),'prompted_answer_em':sum(bool(r.get('prompted_exact_match')) for r in supported)/len(supported),'gold_retrieval_rate':sum(bool(r['gold_retrieved_k16']) for r in passed)/len(passed) if passed else None,'avg_memory_size':sum(r['final_memory_size'] for r in passed)/len(passed) if passed else None,'rows_sha256':sha((out/'rows.jsonl').read_bytes()),'llm':'Qwen/Qwen3.5-9B','llm_role':'visible_event_crud_extraction','provider_calls':0,'api_calls':0,'retries':0};summary['payload_sha256']=sha(canon(summary));(out/'canary_receipt.json').write_bytes(canon(summary));print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--output-root',required=True);run(p.parse_args())
