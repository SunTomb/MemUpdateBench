"""Four-case native Mem0 infer=True smoke; no main study or historical reconstruction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.providers.mem0_native_episode import Mem0NativeEpisode
from mub.vnext.external.visibility import visible_event_input, visible_query_input
from scripts.vnext_prepare_version_transfer_decision import read_packet
from scripts.vnext_run_version_transfer_decision import available, wait_port, stop_owned, write_new, sha, MODEL_ID, EMBEDDING_ID


def create_native_memory(run, shapes):
    from importlib.metadata import distribution
    from mub.vnext.external.workers.mem0_worker import (
        _installed_mem0_content_digest, MEM0_PACKAGE_VERSION,
        MEM0_INSTALLED_CONTENT_SHA256, MEM0_INSTALLED_CONTENT_FILE_COUNT,
    )
    dist = distribution('mem0ai')
    content_hash, count = _installed_mem0_content_digest(dist)
    if (dist.version,content_hash,count) != (MEM0_PACKAGE_VERSION,MEM0_INSTALLED_CONTENT_SHA256,MEM0_INSTALLED_CONTENT_FILE_COUNT):
        raise ValueError('native Mem0 installed identity differs from pinned package')
    from openai import OpenAI
    from mem0 import Memory
    from mem0.memory import main as native_main
    from mem0.memory.telemetry import MEM0_TELEMETRY
    assert MEM0_TELEMETRY is False

    def native_parser(content):
        content = native_main.remove_code_blocks(content)
        try:
            return json.loads(content, strict=False)
        except json.JSONDecodeError:
            return json.loads(native_main.extract_json(content), strict=False)

    monitor = Mem0NativeEpisode.response_monitor(shapes,
        extraction_system_prompt=native_main.ADDITIVE_EXTRACTION_PROMPT,
        native_json_parser=native_parser)
    config = {
        'llm':{'provider':'openai','config':{'model':MODEL_ID,'api_key':'local-only-placeholder',
            'openai_base_url':'http://127.0.0.1:18043/v1','temperature':0.0,'top_p':1.0,
            'max_tokens':2048,'is_reasoning_model':False,'response_callback':monitor}},
        'embedder':{'provider':'openai','config':{'model':EMBEDDING_ID,'api_key':'local-only-placeholder',
            'openai_base_url':'http://127.0.0.1:18043/v1','embedding_dims':384}},
        'vector_store':{'provider':'qdrant','config':{'collection_name':'owned_version_transfer_mem0_smoke',
            'path':str(run/'qdrant'),'embedding_model_dims':384,'on_disk':True}},
        'history_db_path':str(run/'history.db'),'version':'v1.1',
    }
    memory = Memory.from_config(config)
    # Change transport only: native prompts/inference/CRUD/search remain SDK-owned.
    assert str(memory.llm.client.base_url)=='http://127.0.0.1:18043/v1/'
    assert str(memory.embedding_model.client.base_url)=='http://127.0.0.1:18043/v1/'
    assert memory.llm.config.response_callback is monitor
    memory.llm.client.close()
    memory.embedding_model.client.close()
    api = OpenAI(api_key='local-only-placeholder',base_url='http://127.0.0.1:18043/v1',
                 default_headers={'X-MUB-Role':'mem0'},max_retries=0,timeout=120)
    memory.llm.client = api
    memory.embedding_model.client = api
    return memory,api,{'package':'mem0ai','version':dist.version,'installed_sha256':content_hash,'files':count}


def run_cases(memory, tasks, run, shapes):
    rows = [{'case':t.metadata.extra['smoke_case'],'status':'NOT_RUN','event_attempts':0,
             'event_completions':0,'current_retrieval_completions':0,'history_queries_unsupported':0}
            for t in tasks]
    try:
        for i,task in enumerate(tasks):
            row=rows[i]
            bridge=Mem0NativeEpisode(memory,namespace=f'version_transfer_mem0_owned_smoke_{i}',response_shapes=shapes)
            try:
                for j,event in enumerate(task.events):
                    row['event_attempts']+=1
                    result=bridge.ingest(visible_event_input(event,runtime_namespace=bridge.namespace))
                    write_new(run/f'case_{i}_event_{j}.json',result)
                    write_new(run/f'case_{i}_state_{j}.json',bridge.export())
                    row['event_completions']+=1
                for j,query in enumerate(task.queries):
                    if query.query_type.value != 'current':
                        write_new(run/f'case_{i}_query_{j}.json',{'query_id':query.query_id,
                            'status':'UNSUPPORTED','reason':'native_oss_reference_date_api_unavailable',
                            'answer_metrics':None})
                        row['history_queries_unsupported']+=1
                        continue
                    public=visible_query_input(query,k=8,runtime_namespace=bridge.namespace)
                    result=bridge.retrieve(public,query_kind='current')
                    write_new(run/f'case_{i}_query_{j}.json',result)
                    row['current_retrieval_completions']+=1
                row['status']='OBSERVED'
            except Exception as exc:
                row.update(status='BLOCKED',error_type=type(exc).__name__,error=str(exc))
                raise
            finally:
                bridge.reset()
                empty=bridge.export()
                row['namespace_empty_after_cleanup']=empty.get('results')==[]
                write_new(run/f'case_{i}_receipt.json',row)
        return rows
    finally:
        write_new(run/'native_case_summary.json',rows)
        write_new(run/'response_shape_observations.json',shapes)


def close_native_memory(memory, api):
    if memory is None:
        return {}
    status={}
    for name,close in (('http',api.close),('vector',memory.vector_store.client.close),('history',memory.db.close)):
        try:
            close()
            status[name]=True
        except Exception:
            status[name]=False
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--packet-root',type=Path,required=True)
    parser.add_argument('--packet-index-sha256',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args()
    base=Path('/NAS/yesh/MemUpdateBench')
    run=args.output_root.absolute()
    if '..' in run.parts or not run.is_relative_to(base/'_local') or run.exists():
        raise ValueError('new owned NAS-local output root required')
    assert_no_reparse_components(run)
    if not args.packet_root.resolve().is_relative_to(base):
        raise ValueError('packet outside authorized project')
    read_packet(args.packet_root,args.packet_index_sha256)
    tasks=[MemUpdateTaskV3.model_validate_json(line) for line in (args.packet_root/'smoke.jsonl').read_bytes().splitlines()]
    assert len(tasks)==4 and all(len(t.events)<=5 for t in tasks)
    policy=ROOT/'configs/vnext/version_transfer_decision_v1.json'
    assert json.loads(policy.read_bytes())['budget']['smoke_trajectories']==9
    available(18043)
    free=subprocess.check_output(['nvidia-smi','--id='+args.gpu_uuid,'--query-gpu=memory.free',
                                  '--format=csv,noheader,nounits'],text=True)
    if int(free.strip())<24576:
        raise RuntimeError('GPU admission blocked before inference')
    run.mkdir()
    for name in ('home','tmp','cache'):
        (run/name).mkdir()
    env=dict(os.environ,HOME=str(run/'home'),TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),
        MEM0_DIR=str(run/'home/.mem0'),MEM0_TELEMETRY='false',GRAPHITI_TELEMETRY_ENABLED='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_CACHE='/NAS/yesh/hf_cache/hub',
        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',
        http_proxy='',https_proxy='',all_proxy='',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    for key in ('OPENROUTER_API_KEY','GIT_DIR','GIT_WORK_TREE'):
        env.pop(key,None)
        os.environ.pop(key,None)
    os.environ.update({k:env[k] for k in ('HOME','TMPDIR','XDG_CACHE_HOME','MEM0_DIR','MEM0_TELEMETRY',
        'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','NO_PROXY','no_proxy')})
    report={'status':'BLOCKED','evidence_class':'native_mem0_capability_smoke','scientific_evidence':False,
            'paid_api_calls':0,'study_validation_tasks':0,'model':MODEL_ID,'embedding_model':EMBEDDING_ID,
            'historical_query_support':'UNSUPPORTED','prior_native_smoke_attempts':5,
            'reserved_attempts_this_run':4,'smoke_attempt_cap':9,'budget_policy_sha256':sha(policy),
            'packet_index_sha256':args.packet_index_sha256,'admitted_free_mib':int(free.strip()),
            'runtime_source_hashes':{n:sha(ROOT/n) for n in ('scripts/vnext_run_version_transfer_mem0_smoke.py',
                'scripts/vnext_run_version_transfer_decision.py','mub/vnext/external/visibility.py',
                'mub/vnext/external/workers/mem0_worker.py',
                'mub/vnext/external/providers/mem0_native_episode.py','scripts/vnext_serve_version_transfer_local.py',
                'mub/vnext/runtime/version_transfer_budget.py','mub/vnext/runtime/version_transfer_local.py',
                'mub/vnext/runtime/version_transfer_embedding.py')}}
    model=memory=api=None
    shapes=[]
    started=time.monotonic()
    log=None
    try:
        memory,api,identity=create_native_memory(run,shapes)
        report['mem0_identity']=identity
        free_now=subprocess.check_output(['nvidia-smi','--id='+args.gpu_uuid,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True)
        if int(free_now.strip())<24576:
            raise RuntimeError('GPU headroom changed before model load')
        log=(run/'model-console.log').open('xb')
        snapshot=base.parent/'hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
        mini=base.parent/'hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
        model=subprocess.Popen([str(base/'external/version_transfer_constraints_py310_v1/venv/bin/python'),
            '-B','-I',str(ROOT/'scripts/vnext_serve_version_transfer_local.py'),'--snapshot',str(snapshot),
            '--gpu-uuid',args.gpu_uuid,'--model-id',MODEL_ID,'--ledger',str(run/'llm-calls.jsonl'),
            '--attempts','256','--input-tokens','1000000','--output-tokens','524288','--seconds','3600',
            '--embedding-snapshot',str(mini),'--embedding-model-id',EMBEDDING_ID,
            '--embedding-ledger',str(run/'embedding-calls.jsonl')],cwd=base,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        wait_port(model,18043,600)
        with urlopen('http://127.0.0.1:18043/health',timeout=5) as response:
            assert json.load(response)['model']==MODEL_ID
        report['cases']=run_cases(memory,tasks,run,shapes)
        if all(r['status']=='OBSERVED' and r['event_completions']==5 and r['current_retrieval_completions']==1
               and r['history_queries_unsupported']==1 and r['namespace_empty_after_cleanup'] for r in report['cases']):
            report['status']='READY_CURRENT_ONLY'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        try:
            report['native_cleanup']=close_native_memory(memory,api)
            report['native_clients_and_history_closed']=(bool(report['native_cleanup'])
                and all(report['native_cleanup'].values()))
        finally:
            report['owned_model_process_exited']=stop_owned(model)
            if log is not None:
                log.close()
        with socket.socket() as sock:
            report['owned_port_closed']=sock.connect_ex(('127.0.0.1',18043))!=0
        try:
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).split()
            report['owned_gpu_process_absent']=model is not None and str(model.pid) not in pids
        except Exception:
            report['owned_gpu_process_absent']=None
        summary=run/'native_case_summary.json'
        if summary.exists():
            report['cases']=json.loads(summary.read_bytes())
            report['actual_trajectory_attempts_this_run']=sum(c['event_attempts']>0 for c in report['cases'])
        for kind,name in (('llm','llm-calls.jsonl'),('embedding','embedding-calls.jsonl')):
            path=run/name
            if path.exists():
                events=[json.loads(x) for x in path.read_bytes().splitlines()]
                report[kind+'_ledger_sha256']=sha(path)
                report[kind+'_event_counts']={k:sum(x['event']==k for x in events) for k in sorted({x['event'] for x in events})}
        lc=report.get('llm_event_counts',{}); ec=report.get('embedding_event_counts',{})
        report['call_accounting_complete']=(lc.get('reserved',0)>0 and lc.get('reserved')==lc.get('settled')
            and ec.get('embedding_reserved',0)>0 and ec.get('embedding_reserved')==ec.get('embedding_settled'))
        report['elapsed_seconds']=time.monotonic()-started
        if not (report.get('native_clients_and_history_closed') and report['owned_model_process_exited']
                and report['owned_port_closed'] and report['owned_gpu_process_absent'] and report['call_accounting_complete']):
            report['status']='BLOCKED'
        write_new(run/'qualification.json',report)
        print(json.dumps(report,sort_keys=True),flush=True)
    return 0 if report['status']=='READY_CURRENT_ONLY' else 1


if __name__=='__main__':
    raise SystemExit(main())
