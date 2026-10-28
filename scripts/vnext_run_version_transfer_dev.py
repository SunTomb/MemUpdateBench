"""Bounded Graphiti-only development pilot. Validation is deliberately inaccessible.

Native ingestion/search, public time filters, and a separately metered local
reader are retained. This is not a cross-system experiment or a GO decision.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import zipfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.providers.graphiti_native_episode import GraphitiNativeEpisode
from mub.vnext.external.visibility import visible_event_input, visible_query_input
from mub.vnext.io import canonical_json_bytes
from mub.vnext.runtime.version_transfer_readout import build_readout, parse_readout_response
from mub.vnext.runtime.version_transfer_scoring import build_readout_run, build_scoring_artifacts
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3
from scripts.vnext_prepare_version_transfer_decision import read_packet
from scripts.vnext_run_version_transfer_decision import (
    MODEL_ID, EMBEDDING_ID, GRAPHITI_REVISION, available, database_config,
    native_graph_clients, sha, stop_owned, wait_port, write_new,
)

BASE = Path('/NAS/yesh/MemUpdateBench')
QUALIFICATION_REL = '_local/native_graphiti_smoke_2ff1544_run_v1/qualification.json'
QUALIFICATION_PIN = '60b8ec95cab095228e028461d53985106f3471997a35675d7078292cf20fc73a'
LIMITS = {'attempts': 2000, 'input_tokens': 4000000, 'output_tokens': 256000, 'seconds': 21600}
# Conservative allocation for all prior admitted protocol/native qualification
# lifetimes, including the failed load; not a measurement of exact GPU runtime.
PRIOR_TIME_RESERVATION_SECONDS = 7200
ANSWER_SCHEMA = {'type':'object','properties':{'disposition':{'type':'string','enum':['answered','abstained']},
                  'answer':{'type':'string'}},'required':['disposition'],'additionalProperties':False}


def select_dev_tasks(tasks):
    groups = defaultdict(list)
    for task in tasks:
        if task.metadata.split.value == 'dev':
            groups[task.metadata.split_key.semantic_core_id].append(task)
    if len(groups) != 8:
        raise ValueError('eight complete development cores required')
    ordered = []
    for index, core in enumerate(sorted(groups)):
        group = groups[core]
        by_depth = {task.metadata.extra['update_count']: task for task in group}
        if len(group) != 3 or set(by_depth) != {1,4,16}:
            raise ValueError('complete development depth pairs required')
        depths = (1,4,16)
        rotate = index % 3
        ordered.extend(by_depth[d] for d in depths[rotate:] + depths[:rotate])
    return tuple(ordered)


def oracle_request(task, query):
    replay = replay_task_v3(task)
    resolution = resolve_query_v3(query, replay, task.events)
    if resolution.issues or len(resolution.selected_versions) != 1:
        raise ValueError('oracle query must resolve exactly one version')
    version = resolution.selected_versions[0]
    selected = {e.event_id:e for e in task.events}[resolution.selected_event_ids[0]]
    ledger = next(h for h in task.version_history if h.object_key.canonical_id == query.target_object_keys[0].canonical_id)
    following = next((entry for entry in ledger.entries if entry.version_index == version.version_index+1), None)
    edge = {'uuid':f'oracle-{query.query_id}','fact':selected.raw_text,'group_id':'privileged-reference',
            'valid_at':version.logical_time,'invalid_at':following.logical_time if following else None,
            'created_at':None,'expired_at':None,'evidence_origin':'privileged_reference'}
    request = build_readout(query, {'edges':[edge],'edge_reranker_scores':[1.0]},namespace='privileged-reference')
    trace = request.retrieval_trace.model_copy(update={'retrieval_policy':'oracle_evidence_diagnostic',
        'version_metadata':{'evidence_origin':'privileged_reference','not_native_retrieval':True}})
    return request.model_copy(update={'retrieval_trace':trace})


def estimate_dev_cost(smoke, *, task_count, events_per_task):
    episodes = task_count * events_per_task
    answers = task_count * 2 * 3
    native_attempts = math.ceil(episodes * smoke['llm_attempts'] / smoke['events'])
    # Do not take the CPU map optimization as a demonstrated model speedup.
    seconds = episodes * smoke['elapsed_seconds'] / smoke['events'] + answers * 10 + 600
    return {'episode_requests':episodes,'readout_generations':answers,
            'estimated_llm_attempts':native_attempts+answers,'estimated_seconds':seconds,
            'uses_measured_cached_runtime':False,
            'estimate_is_not_a_guarantee':True}


def admit_output(path):
    path = Path(path).absolute()
    if '..' in path.parts or not path.is_relative_to(BASE/'_local') or path.exists() or path.is_symlink():
        raise ValueError('new owned NAS-local output required')
    assert_no_reparse_components(path)
    return path


def audit_llm_ledger(path):
    reserved, settled = {}, {}
    for row in (json.loads(line) for line in Path(path).read_bytes().splitlines()):
        attempt = row.get('attempt')
        if type(attempt) is not int or attempt <= 0:
            raise ValueError('invalid attempt identifier')
        if row.get('event') == 'reserved':
            if attempt in reserved:
                raise ValueError('duplicate reservation')
            if attempt != len(reserved) + 1 or row.get('role') not in ('graphiti','readout','oracle_readout'):
                raise ValueError('attempt sequence or role mismatch')
            if any(type(row.get(k)) is not int or row[k] <= 0 for k in ('input_tokens','reserved_output_tokens')):
                raise ValueError('invalid reservation token count')
            reserved[attempt] = row
        elif row.get('event') == 'settled':
            if attempt in settled or attempt not in reserved:
                raise ValueError('duplicate or unreserved settlement')
            amount = row.get('completion_tokens')
            if type(amount) is not int or not 0 <= amount <= reserved[attempt]['reserved_output_tokens']:
                raise ValueError('invalid settlement token count')
            if row.get('finish_reason') not in ('stop','length','error'):
                raise ValueError('unknown settlement finish reason')
            settled[attempt] = row
        else:
            raise ValueError('unknown ledger event')
    return {'charged':{'attempts':len(reserved),
                       'input_tokens':sum(r['input_tokens'] for r in reserved.values()),
                       'output_tokens':sum(settled[i]['completion_tokens'] if i in settled
                                           else r['reserved_output_tokens'] for i,r in reserved.items())},
            'unsettled_attempts':sorted(set(reserved)-set(settled)),
            'all_settled':set(reserved)==set(settled),
            'roles':{role:sum(r['role']==role for r in reserved.values())
                     for role in ('graphiti','readout','oracle_readout')}}


def verify_prior_budget():
    known = (
        ('_local/version_transfer_model_protocol_v3','calls.jsonl','eb36bce8a0cd33e15e8f56de955d1f017e5ffdbfd24b516dcc73ef0e3da25892'),
        ('_local/native_graphiti_smoke_95b7193_run_v2','llm-calls.jsonl','e4ebec9764cef15dbf8f611e6ff1cf1fdfa78436d0244a8f41ec3f7817bb5a32'),
        ('_local/native_graphiti_smoke_2ff1544_run_v1','llm-calls.jsonl','6e46827bab31ee046b95e164c59870753cda0009e9b4f339372317149c919d98'),
        ('_local/native_mem0_smoke_5d3d584_run_v1','llm-calls.jsonl','678b39e62dda570c867c7985ed7df3e7d4834396614e0dab645be373f218a2bc'),
    )
    totals = {'attempts':0,'input_tokens':0,'output_tokens':0}
    for directory, name, pin in known:
        path = BASE/directory/name
        if sha(path) != pin:
            raise ValueError('prior generation ledger changed')
        events = [json.loads(line) for line in path.read_bytes().splitlines()]
        a = [row for row in events if row['event']=='reserved']
        b = [row for row in events if row['event']=='settled']
        if [r['attempt'] for r in a] != [r['attempt'] for r in b] or len({r['attempt'] for r in a}) != len(a):
            raise ValueError('prior accounting is incomplete or duplicated')
        totals['attempts'] += len(a)
        totals['input_tokens'] += sum(r['input_tokens'] for r in a)
        totals['output_tokens'] += sum(r['completion_tokens'] for r in b)
    if totals != {'attempts':84,'input_tokens':128445,'output_tokens':6286}:
        raise ValueError('unexpected prior study consumption; re-audit before launch')
    return totals


def verify_sdk_python_sources():
    from importlib.metadata import distribution
    package = distribution('graphiti-core')
    if package.version != '0.31.2':
        raise ValueError('Graphiti package version changed')
    archive = BASE/'external/version_transfer_graphiti_5c27edf_v1/source_download_v1/graphiti.zip'
    if sha(archive) != '2435eff43e4deee0d869c90bb0f96b3e3fb8e5ff076977f68ed6b2c1850a15ef':
        raise ValueError('pinned upstream source archive changed')
    root = Path(package.locate_file('graphiti_core')).resolve(strict=True)
    if not root.is_relative_to(BASE):
        raise ValueError('native SDK outside authorized project environment')
    prefix = f'graphiti-{GRAPHITI_REVISION}/graphiti_core/'
    with zipfile.ZipFile(archive) as zipped:
        expected = {name[len(prefix):]:hashlib.sha256(zipped.read(name)).hexdigest()
                    for name in zipped.namelist() if name.startswith(prefix) and name.endswith('.py')}
    observed = {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*.py') if p.is_file()}
    if not expected or observed != expected:
        raise ValueError('installed native Python sources differ from pinned upstream commit')
    return {'version':package.version,'python_files':len(expected),
            'source_map_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()}


def verify_qualification():
    path = BASE/QUALIFICATION_REL
    if sha(path) != QUALIFICATION_PIN:
        raise ValueError('native qualification pin changed')
    row = json.loads(path.read_bytes())
    if not (row['status']=='READY' and row['structured_output_mode']=='json_schema'
            and row['model']==MODEL_ID and row['embedding_model']==EMBEDDING_ID
            and row['graphiti_revision']==GRAPHITI_REVISION
            and row['retrieval_recipe']=='EDGE_HYBRID_SEARCH_RRF'
            and row['call_accounting_complete'] and len(row['cases'])==4):
        raise ValueError('qualified native configuration mismatch')
    return row


async def execute_dev(tasks, run, *, source_revision):
    from openai import AsyncOpenAI
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
    from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator
    graph, driver, graph_api = native_graph_clients(structured_output_mode='json_schema')
    read_api = AsyncOpenAI(api_key='local-only-placeholder',base_url='http://127.0.0.1:18043/v1',
        default_headers={'X-MUB-Role':'readout'},max_retries=0,timeout=120)
    oracle_api = AsyncOpenAI(api_key='local-only-placeholder',base_url='http://127.0.0.1:18043/v1',
        default_headers={'X-MUB-Role':'oracle_readout'},max_retries=0,timeout=120)
    records = {order:[] for order in ('native','reverse')}
    cases = [{'task_id':task.task_id,'core_id':task.metadata.split_key.semantic_core_id,
              'update_count':task.metadata.extra['update_count'],'status':'NOT_RUN',
              'events_completed':0,'retrievals_completed':0,'readouts_completed':0}
             for task in tasks]
    try:
        if driver._init_task is not None:
            await driver._init_task
        else:
            await graph.build_indices_and_constraints()
        for index, task in enumerate(tasks):
            if sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) > 2*1024**3:
                raise RuntimeError('development storage ceiling exceeded')
            case = cases[index]
            directory = run/f'task_{index:02d}'
            directory.mkdir()
            group = f'vtd_dev_{run.name}_{index}'
            bridge = GraphitiNativeEpisode(graph,namespace=group,source_type=EpisodeType.text,
                filter_factory=GraphitiNativeEpisode.validity_filter_factory(SearchFilters,DateFilter,ComparisonOperator),
                search_config=EDGE_HYBRID_SEARCH_RRF)
            print(json.dumps({'event':'task_start','index':index,'task_id':task.task_id}),flush=True)
            requests = {order:[] for order in records}
            predictions = {order:[] for order in records}
            try:
                for position, event in enumerate(task.events):
                    result = await bridge.ingest(visible_event_input(event,runtime_namespace=group))
                    if (result.episode.group_id != group or result.episode.valid_at.isoformat().replace('+00:00','Z') != event.timestamp
                            or any(n.group_id != group for n in result.nodes) or any(e.group_id != group for e in result.edges)):
                        raise ValueError('native ingestion provenance mismatch')
                    write_new(directory/f'event_{position:02d}.json',result.model_dump(mode='json'))
                    case['events_completed'] += 1
                for position, query in enumerate(task.queries):
                    native = await bridge.retrieve(visible_query_input(query,k=8,runtime_namespace=group))
                    result = native.model_dump(mode='json')
                    write_new(directory/f'retrieval_{position}.json',result)
                    case['retrievals_completed'] += 1
                    orders = ('native','reverse') if index % 2 == 0 else ('reverse','native')
                    for order in (*orders, 'oracle'):
                        request = (oracle_request(task,query) if order=='oracle' else
                                   build_readout(query,result,namespace=group,order=order))
                        api = oracle_api if order=='oracle' else read_api
                        response = await api.chat.completions.create(model=MODEL_ID,
                            messages=[{'role':'user','content':request.rendered_prompt}],
                            temperature=0,max_tokens=128,
                            response_format={'type':'json_schema','json_schema':{'name':'VersionTransferAnswer','schema':ANSWER_SCHEMA}})
                        raw = response.model_dump(mode='json')
                        prediction = parse_readout_response(request,raw,model_id=MODEL_ID)
                        write_new(directory/f'{order}_response_{position}.json',raw)
                        (directory/f'{order}_request_{position}.json').write_bytes(canonical_json_bytes(request))
                        (directory/f'{order}_prediction_{position}.json').write_bytes(canonical_json_bytes(prediction))
                        if raw['choices'][0]['finish_reason'] != 'stop':
                            raise ValueError('readout length/technical terminal is not scientific correctness')
                        if order != 'oracle':
                            requests[order].append(request)
                            predictions[order].append(prediction)
                        case['readouts_completed'] += 1
                for order in records:
                    record = build_readout_run(task,requests[order],predictions[order],
                        run_id=f'{run.name}-{order}',order=order)
                    records[order].append(record)
                    (directory/f'{order}_run.json').write_bytes(canonical_json_bytes(record))
                case['status']='OBSERVED'
            except BaseException as exc:
                case.update(status='BLOCKED',error_type=type(exc).__name__,error=str(exc))
                raise
            finally:
                await driver.execute_query('MATCH (n) WHERE n.group_id = $group DETACH DELETE n',group=group)
                rows, _, _ = await driver.execute_query('MATCH (n) WHERE n.group_id = $group RETURN count(n) AS n',group=group)
                case['namespace_empty_after_cleanup']=rows[0]['n']==0
                write_new(directory/'case.json',case)
                print(json.dumps({'event':'task_terminal',**case}),flush=True)
            if not case['namespace_empty_after_cleanup']:
                raise ValueError('owned namespace cleanup failed')
        for order, runs in records.items():
            target = run/f'{order}_canonical'
            target.mkdir()
            bundle = build_scoring_artifacts(tasks,runs,code_revision=source_revision,
                                            qualification_sha256=QUALIFICATION_PIN,historical_supported=True)
            for name, raw in bundle['payloads'].items():
                with (target/name).open('xb') as stream:
                    stream.write(raw)
            (target/'native-qualification.json').write_bytes((BASE/QUALIFICATION_REL).read_bytes())
            index = {'scope':'graphiti_only_dev_no_scientific_decision',
                     'artifacts':{p.name:sha(p) for p in target.iterdir() if p.is_file()}}
            write_new(target/'index.json',index)
        return cases
    finally:
        try:
            await graph.close()
        finally:
            await graph_api.close()
            await read_api.close()
            await oracle_api.close()
            write_new(run/'cases.json',cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--packet-root',type=Path,required=True)
    parser.add_argument('--packet-index-sha256',required=True)
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    run = admit_output(args.output_root)
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):
        raise ValueError('exact execution source revision required')
    if not args.packet_root.resolve(strict=True).is_relative_to(BASE):
        raise ValueError('packet must remain within authorized project')
    tasks = select_dev_tasks(read_packet(args.packet_root,args.packet_index_sha256))
    qualification = verify_qualification()
    sdk_identity = verify_sdk_python_sources()
    prior = verify_prior_budget()
    policy = json.loads((ROOT/'configs/vnext/version_transfer_decision_v1.json').read_bytes())['budget']
    for key, policy_key in (('attempts','llm_attempts'),('input_tokens','input_tokens'),('output_tokens','output_tokens')):
        if prior[key]+LIMITS[key]>policy[policy_key]:
            raise ValueError('prospective dev reservation exceeds remaining whole-study budget')
    if PRIOR_TIME_RESERVATION_SECONDS+LIMITS['seconds']>policy['gpu_hours']*3600:
        raise ValueError('conservative GPU-time reservation exceeds approved budget')
    estimate = estimate_dev_cost({'llm_attempts':72,'events':20,
        'elapsed_seconds':qualification['elapsed_seconds']},task_count=len(tasks),events_per_task=17)
    if estimate['estimated_llm_attempts']>LIMITS['attempts'] or estimate['estimated_seconds']>LIMITS['seconds']:
        raise ValueError('development pilot does not fit the bounded forecast')
    admission = {'status':'DEV_PLAN_READY_NOT_EXECUTED','scope':'graphiti_only_development',
        'validation_tasks':0,'mem0_status':'BLOCKED_NOT_RETRIED','planned_tasks':24,
        'planned_query_count':48,'planned_readout_count':144,'native_smoke_attempts_consumed_by_plan':0,
        'qualification_sha256':QUALIFICATION_PIN,'packet_index_sha256':args.packet_index_sha256,
        'source_revision':args.source_revision,'sdk_identity':sdk_identity,
        'prior_usage':prior,'run_ceilings':LIMITS,
        'prior_time_reservation_seconds':PRIOR_TIME_RESERVATION_SECONDS,'estimate':estimate,
        'runtime_source_hashes':{name:sha(ROOT/name) for name in (
            'scripts/vnext_run_version_transfer_dev.py','scripts/vnext_run_version_transfer_decision.py',
            'scripts/vnext_serve_version_transfer_local.py','mub/vnext/runtime/version_transfer_readout.py',
            'mub/vnext/runtime/version_transfer_scoring.py','mub/vnext/runtime/version_transfer_budget.py',
            'mub/vnext/runtime/version_transfer_local.py','mub/vnext/runtime/version_transfer_embedding.py',
            'mub/vnext/external/providers/graphiti_native_episode.py')}}
    if not args.execute:
        print(json.dumps(admission,sort_keys=True))
        return 0
    if not args.gpu_uuid or not args.gpu_uuid.startswith('GPU-'):
        raise ValueError('freshly admitted GPU UUID required')
    for port in (17687,18043):
        available(port)
    free=int(subprocess.check_output(['nvidia-smi','--id='+args.gpu_uuid,'--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip())
    if free<24576:
        raise RuntimeError('GPU admission blocked before model/database launch')
    # Single study-wide one-shot dev reservation; failing runs are not silently resumed.
    reservation = BASE/'_local/version_transfer_graphiti_dev_reservation_v1'
    reservation.mkdir(exist_ok=False)
    write_new(reservation/'reservation.json',{**admission,'output_root':str(run)})
    run.mkdir()
    write_new(run/'admission.json',admission)
    neo_base=BASE/'external/version_transfer_neo4j_5260_v1'
    neo=neo_base/'database_runtime/neo4j-community-5.26.0'
    java=neo_base/'java_runtime/zulu21.54.107-ca-jre21.0.13-linux_x64'
    conf=database_config(run,neo)
    env=dict(os.environ,HOME=str(run/'home'),TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_CACHE='/NAS/yesh/hf_cache/hub',
        GRAPHITI_TELEMETRY_ENABLED='false',EMBEDDING_DIM='384',PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env[key]=''
    env.update(NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    for key in ('GIT_DIR','GIT_WORK_TREE'):
        env.pop(key,None)
    os.environ.update({key:env[key] for key in ('HOME','TMPDIR','XDG_CACHE_HOME','GRAPHITI_TELEMETRY_ENABLED',
        'EMBEDDING_DIM','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','NO_PROXY','no_proxy')})
    report={'status':'BLOCKED','scope':'graphiti_only_development','scientific_decision':'NOT_EVALUATED',
            'validation_tasks':0,'mem0_status':'BLOCKED_NOT_RUN','planned_task_ids':[t.task_id for t in tasks]}
    processes=[]; streams=[]; started=time.monotonic()
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    try:
        db_env=dict(env,JAVA_HOME=str(java),NEO4J_HOME=str(neo),NEO4J_CONF=str(conf),
            JAVA_TOOL_OPTIONS=f'-Djava.io.tmpdir={run/"tmp"} -XX:-UsePerfData')
        log=(run/'database-console.log').open('xb'); streams.append(log)
        database=subprocess.Popen(['bash',str(neo/'bin/neo4j'),'console'],cwd=BASE,env=db_env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True); processes.append(database)
        wait_port(database,17687,120)
        free=int(subprocess.check_output(['nvidia-smi','--id='+args.gpu_uuid,'--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'],text=True).strip())
        if free<24576:
            raise RuntimeError('GPU headroom changed before model load')
        model_log=(run/'model-console.log').open('xb'); streams.append(model_log)
        cache=BASE.parent/'hf_cache/hub'
        model=subprocess.Popen([str(BASE/'external/version_transfer_constraints_py310_v1/venv/bin/python'),'-B','-I',
            str(ROOT/'scripts/vnext_serve_version_transfer_local.py'),
            '--snapshot',str(cache/'models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'),
            '--gpu-uuid',args.gpu_uuid,'--model-id',MODEL_ID,'--require-json-schema',
            '--ledger',str(run/'llm-calls.jsonl'),'--attempts',str(LIMITS['attempts']),
            '--input-tokens',str(LIMITS['input_tokens']),'--output-tokens',str(LIMITS['output_tokens']),
            '--seconds',str(LIMITS['seconds']),
            '--embedding-snapshot',str(cache/'models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'),
            '--embedding-model-id',EMBEDDING_ID,'--embedding-ledger',str(run/'embedding-calls.jsonl')],
            cwd=BASE,env=env,stdout=model_log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(model)
        wait_port(model,18043,600)
        with urlopen('http://127.0.0.1:18043/health',timeout=5) as response:
            health=json.load(response)
        if health['model']!=MODEL_ID or not health.get('supports_json_schema'):
            raise ValueError('local model identity/schema readiness mismatch')
        remaining=LIMITS['seconds']-(time.monotonic()-started)
        if remaining<=0:
            raise TimeoutError('development runtime ceiling exhausted during startup')
        cases=asyncio.run(asyncio.wait_for(execute_dev(tasks,run,source_revision=args.source_revision),timeout=remaining))
        report['cases']=cases
        if len(cases)==24 and all(c['status']=='OBSERVED' and c['namespace_empty_after_cleanup'] for c in cases):
            report['status']='OBSERVATIONS_COMPLETE'
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        report['owned_process_cleanup']=[stop_owned(p) for p in reversed(processes)]
        for stream in streams:
            stream.close()
        report['ports_closed']={}
        for port in (17687,18043):
            with socket.socket() as sock:
                report['ports_closed'][str(port)]=sock.connect_ex(('127.0.0.1',port))!=0
        if (run/'cases.json').exists():
            report['cases']=json.loads((run/'cases.json').read_bytes())
        report['elapsed_seconds']=time.monotonic()-started
        try:
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).split()
            report['owned_gpu_process_absent']=all(str(p.pid) not in pids for p in processes)
        except Exception:
            report['owned_gpu_process_absent']=None
        if (run/'llm-calls.jsonl').exists():
            try:
                report['llm_accounting']=audit_llm_ledger(run/'llm-calls.jsonl')
                report['aggregate_charged_llm']={key:prior[key]+report['llm_accounting']['charged'][key]
                                                 for key in prior}
                if not report['llm_accounting']['all_settled'] or any(
                        report['llm_accounting']['charged'][key]>LIMITS[key] for key in prior):
                    report['status']='BLOCKED'
            except Exception as exc:
                report.update(status='BLOCKED',accounting_error=str(exc))
        else:
            report['status']='BLOCKED'
        if (run/'embedding-calls.jsonl').exists():
            try:
                embeddings=[json.loads(line) for line in (run/'embedding-calls.jsonl').read_bytes().splitlines()]
                reservations=[r for r in embeddings if r['event']=='embedding_reserved']
                settlements=[r for r in embeddings if r['event']=='embedding_settled']
                expected=list(range(1,len(reservations)+1))
                if [r['attempt'] for r in reservations] != expected or [r['attempt'] for r in settlements] != expected:
                    raise ValueError('embedding accounting mismatch')
                report['embedding_usage']={'attempts':len(reservations),'items':sum(r['items'] for r in reservations),
                                          'input_tokens':sum(r['input_tokens'] for r in reservations)}
            except Exception as exc:
                report.update(status='BLOCKED',embedding_accounting_error=str(exc))
        else:
            report['status']='BLOCKED'
        report['index']={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*.json')
                         if 'database-' not in p.relative_to(run).parts[0] and p.is_file()}
        for name in ('llm-calls.jsonl','embedding-calls.jsonl'):
            if (run/name).exists():
                report['index'][name]=sha(run/name)
        report['output_bytes']=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
        if (not all(report['owned_process_cleanup']) or not all(report['ports_closed'].values())
                or report['owned_gpu_process_absent'] is not True or report['output_bytes']>2*1024**3):
            report['status']='BLOCKED'
        write_new(run/'terminal.json',report)
        write_new(reservation/'terminal.json',{'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
            'terminal_sha256':sha(run/'terminal.json'),'run_ceilings':LIMITS})
        print(json.dumps({k:v for k,v in report.items() if k not in ('index','planned_task_ids','cases')},sort_keys=True),flush=True)
    return 0 if report['status']=='OBSERVATIONS_COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())
