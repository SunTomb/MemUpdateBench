"""One explicitly authorized four-request Qwen readout, with an owned-worker watchdog."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'Qwen/Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
MODEL_TREE = 'e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db'
SNAPSHOT = Path('/NAS/HuggingFaceModels/Qwen3.5-9B')
BASE = Path('/NAS/yesh/MemUpdateBench/external')
BINDING = BASE / 'post_core_shared_qwen_binding_20260822_v1.json'
SNAPSHOT_BINDING_SHA = '924cb994248cf56d1d41df0da3bca06ce7abe1184cde5f0af489d4d364a1d9c2'
HISTORY = BASE / 'post_core_qwen35_tang1_gpu3_runtime_gate_20260826_v2.json'
HISTORY_SHA = '5d06cb1cbacd43beb0b0a2aaafd1bd7a5b75e8f6d283f5dbbd899b8429ff202f'
PYTHON = BASE / 'qdrant_qwen_environment_20260905_v1/venv/bin/python'
GPU_INDEX = 6
GPU_UUID = 'GPU-9fff6cbd-a32d-2b97-12f6-786d30d15ee9'
TEMPLATE_SHA = 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'
PREPARATION_SHA = '1112a371f07326130634715cfc15fe656aa11a48539f2d16b06f39c400f9e687'
RUN_ID = 'cross_domain_qwen_readout_20261002_v4'
RUN_ROOT = BASE / RUN_ID
TEMP_ROOT = Path('/tmp/mub-cross-domain-qwen-20261002-v4')
PHASE_BUDGETS = {'qualification': 1800, 'template_preflight': 240, 'before_load': 300,
                 'template_recheck': 120, 'before_generation': 90, 'after_generation': 60,
                 'before_close': 90, 'after_close': 120, 'post_validation': 1800,
                 'worker_complete': 60, 'technical_failure': 90}
TOTAL_SECONDS = 3600


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def safe(path):
    import stat
    p = Path(path).absolute()
    require('..' not in p.parts, 'path traversal forbidden')
    for part in (p, *p.parents):
        try: info = part.lstat()
        except FileNotFoundError: continue
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&0x400,
                'symlink/reparse path forbidden')
    return p


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    raw = canonical(value)
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return sha(raw)


def profile(source_sha):
    return {'schema':'memupdatebench.cross-domain.qwen-authorization.v1', 'run_id':RUN_ID,
        'authorization_basis':'explicit_user_authorization_for_one_bounded_trial_20261002',
        'node':'Tang2', 'gpu_index':GPU_INDEX, 'gpu_uuid':GPU_UUID, 'minimum_free_mib':30720,
        'model_id':MODEL_ID, 'revision':REVISION, 'model_tree_sha256':MODEL_TREE,
        'source_manifest_sha256':source_sha, 'preparation_index_sha256':PREPARATION_SHA,
        'maximum_load_attempts':1, 'maximum_generations':4, 'max_new_tokens':64,
        'maximum_generated_tokens':256, 'retries':0, 'context_token_cap':4096,
        'phase_budgets_seconds':PHASE_BUDGETS, 'total_seconds':TOTAL_SECONDS,
        'paid_provider_calls_allowed':False, 'qdrant_calls_allowed':False,
        'scientific_release_allowed':False, 'approval_is_cryptographic_signature':False}


def validate_profile(value):
    pin = value.get('source_manifest_sha256')
    require(type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin) and pin!='0'*64,'invalid source pin')
    require(canonical(value)==canonical(profile(pin)),'authorization scope mismatch')


def check_device_rows(node, rows, processes, allowed_pid=None, minimum_free_mib=30720):
    require(node=='Tang2','wrong execution node')
    selected=[r for r in rows if r['uuid']==GPU_UUID]
    require(len(selected)==1 and selected[0]['index']==GPU_INDEX and selected[0]['name']=='NVIDIA A40', 'device UUID/index mismatch')
    require(minimum_free_mib in (30720,1024) and (minimum_free_mib==30720 or allowed_pid is not None),
            'lower memory floor requires the admitted worker')
    require(selected[0]['free_mib']>=minimum_free_mib,'insufficient current memory headroom')
    require(not any(p['uuid']==GPU_UUID and p['pid']!=allowed_pid for p in processes),'selected GPU busy')
    return {**selected[0],'node':node}


def observe_device(allowed_pid=None, minimum_free_mib=30720):
    rows_raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[]
    for line in rows_raw.splitlines():
        index,uuid,name,free=[v.strip() for v in line.split(',')]
        rows.append({'index':int(index),'uuid':uuid,'name':name,'free_mib':int(free)})
    processes=gpu_processes()
    node=Path('/proc/sys/kernel/hostname').read_text().strip()
    return check_device_rows(node,rows,processes,allowed_pid,minimum_free_mib)


def gpu_processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    return [{'uuid':v[0].strip(),'pid':int(v[1])} for line in raw.splitlines() if line.strip() for v in [line.split(',')]]


def make_source_manifest(root):
    root=safe(root)
    rows=[]
    for path in sorted(root.rglob('*')):
        safe(path)
        if path.is_file() and path.name!='source_manifest.json':
            rows.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':file_sha(path)})
    rows.sort(key=lambda r:r['path'])
    return {'schema':'memupdatebench.cross-domain.qwen-source.v1','files':rows,
            'tree_sha256':sha(canonical(rows))}


def verify_source(root, expected):
    raw=(safe(root)/'source_manifest.json').read_bytes()
    require(sha(raw)==expected,'source manifest pin mismatch')
    manifest=json.loads(raw)
    require(manifest==make_source_manifest(root),'source files changed')
    return manifest


def verify_snapshot(root=SNAPSHOT, receipt=BINDING):
    root=safe(root); receipt=safe(receipt)
    raw=receipt.read_bytes()
    require(sha(raw)==SNAPSHOT_BINDING_SHA,'snapshot binding receipt hash mismatch')
    value=json.loads(raw)
    require(value['repo']==MODEL_ID and value['revision']==REVISION and value['tree_sha256']==MODEL_TREE
            and value['file_count']==16,'snapshot identity mismatch')
    rows=value['entries']; expected={r['path']:r for r in rows}
    require(len(rows)==len(expected)==16,'snapshot duplicate member')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual==set(expected),'snapshot membership mismatch')
    observations=[]
    for name,row in sorted(expected.items()):
        p=safe(root/name)
        require(root in p.parents and p.is_file(),'snapshot unsafe member')
        require(p.stat().st_size==row['bytes'] and file_sha(p)==row['sha256'],'snapshot file changed')
        observations.append({'path':name,'bytes':row['bytes'],'sha256':row['sha256']})
    require(sum(r['bytes'] for r in observations)==value['total_bytes'],'snapshot size mismatch')
    return {'model_id':MODEL_ID,'revision':REVISION,'tree_sha256':MODEL_TREE,'binding_receipt_sha256':sha(raw),
            'file_count':16,'total_bytes':value['total_bytes'],'files':observations}


def template_binding(request, prepared):
    require(prepared.request_id==request.request_id and type(prepared.input_tokens) is int
            and 0<prepared.input_tokens<=4032 and type(prepared.rendered_text) is str,'template/context contract mismatch')
    require(type(prepared.template_sha256) is str and re.fullmatch('[0-9a-f]{64}',prepared.template_sha256), 'template digest mismatch')
    return {'request_id':request.request_id,'visible_prompt_sha256':sha(request.visible_prompt.encode()),
            'rendered_prompt_sha256':sha(prepared.rendered_text.encode()),'template_sha256':prepared.template_sha256,
            'input_tokens':prepared.input_tokens}


def perform_readouts(requests, bindings, backend, emit, revalidate, after_load=lambda:None):
    require(len(requests)==len(bindings)==4,'exactly four readouts required')
    rows=[{'request_id':r.request_id,'status':'NOT_RUN','attempts':0,'output_sha256':None,
           'input_tokens':None,'generated_tokens':None,'prediction':None,'reason_code':'not_attempted'} for r in requests]
    life={'load_intent':0,'load_completed':False,'close':None,'error_phase':None,'error_type':None}
    phase='before_load'; active=None
    try:
        revalidate(); emit({'phase':'before_load','load_intent':1}); life['load_intent']=1
        backend.load(); life['load_completed']=True; after_load()
        for request,binding,row in zip(requests,bindings,rows):
            active=row; phase='template_recheck'; emit({'phase':phase,'request_id':request.request_id})
            prepared=backend.prepare(request)
            require(template_binding(request,prepared)==binding,'loaded tokenizer differs from preflight')
            revalidate(); phase='before_generation'
            emit({'phase':phase,'request_id':request.request_id,'attempt_intent':1})
            row['attempts']=1; row['status']='ATTEMPTED'; started=time.monotonic()
            result=backend.generate(prepared,max_new_tokens=64,seed=0,do_sample=False,num_beams=1)
            require(type(result.generated_tokens) is int and 0<=result.generated_tokens<=64
                    and type(result.input_tokens) is int and result.input_tokens==binding['input_tokens'], 'generation usage contract mismatch')
            require(type(result.text) is str and len(result.text.encode())<=16384,'generation text size mismatch')
            row.update(output_sha256=sha(result.text.encode()),input_tokens=result.input_tokens,
                       generated_tokens=result.generated_tokens,elapsed_seconds=time.monotonic()-started,
                       output_cap_reached=result.generated_tokens==64)
            phase='after_generation'; emit({'phase':phase,'observation':{k:v for k,v in row.items() if k!='prediction'}})
            from mub.vnext.runtime.answer_model_v3 import parse_answer_prediction_v3
            from mub.vnext.contracts.enums import AnswerSchema
            prediction=parse_answer_prediction_v3(query_id=request.request_id,answer_schema=AnswerSchema.NUMBER,raw_output=result.text)
            row['prediction']=prediction.model_dump(mode='json',exclude={'raw_output'})
            row['status']='COMPLETED'; row['reason_code']=None
            emit({'phase':'after_generation','completed_row':row}); active=None
        life['status']='WORKER_COMPLETE'
    except BaseException as exc:
        life.update(status='BLOCKED',error_phase=phase,error_type=type(exc).__name__)
        if active is not None: active.update(status='TECHNICAL_FAILURE',reason_code=phase+'_failed',prediction=None)
        for row in rows:
            if row['status']=='NOT_RUN':row['reason_code']='earlier_technical_failure'
    finally:
        try: emit({'phase':'before_close'})
        except BaseException: life['status']='BLOCKED'
        try: life['close']=backend.close()
        except BaseException as exc: life.update(status='BLOCKED',close_error_type=type(exc).__name__)
        if not isinstance(life['close'],dict) or type(life['close'].get('allocated_bytes_after')) is not int or life['close']['allocated_bytes_after']!=0:
            life['status']='BLOCKED'
    result={'status':life['status'],'rows':rows,'lifecycle':life,
            'backend_observations':dict(getattr(backend,'observations',{})), 'retries':0}
    emit({'phase':'after_close','readout_result':result})
    return result


def watch_phase(current, event, now):
    require(type(event['sequence']) is int and event['sequence']>=0 and event['phase'] in PHASE_BUDGETS,'invalid worker phase')
    if current is None:return (event['sequence'],event['phase'],now)
    require(event['sequence']>=current[0],'phase sequence regressed')
    if event['sequence']!=current[0]:return (event['sequence'],event['phase'],now)
    require(event['phase']==current[1],'phase changed without sequence')
    if now-current[2]>PHASE_BUDGETS[event['phase']]:raise TimeoutError('worker phase deadline')
    return current


def final_status(worker, cleanup, bindings_ok):
    return 'COMPLETE' if worker and worker['status']=='WORKER_COMPLETE' and bindings_ok and all(
        cleanup.get(k) is True for k in ('worker_exited','owned_group_empty','gpu_pid_absent','temporary_root_removed')) else 'BLOCKED'


def framework_proxy_implementation(name, module, modules):
    proxies = {'torch.ops': ('torch._ops', '_Ops', '_ops.py'),
               'torch.classes': ('torch._classes', '_Classes', '_classes.py')}
    if name not in proxies:
        return None
    implementation, class_name, placeholder = proxies[name]
    owner = modules.get(implementation)
    cls = type(module)
    require(owner is not None and getattr(owner, '__file__', None)
            and cls is getattr(owner, class_name, None)
            and cls.__module__ == implementation and cls.__name__ == class_name
            and getattr(module, '__spec__', None) is None
            and getattr(module, '__file__', None) == placeholder,
            'unexpected framework proxy origin')
    return implementation


def runtime_identity():
    import importlib.metadata as meta
    names=('torch','transformers','accelerate','tokenizers','safetensors','numpy','huggingface-hub','pydantic')
    dists={name:meta.distribution(name) for name in names}
    require(dists['torch'].version=='2.5.1+cu121' and dists['transformers'].version=='5.9.0','framework version mismatch')
    rows=[]
    for name,dist in dists.items():
        members=list(dist.files or ())
        for item in members:
            if str(item).endswith(('.dist-info/METADATA','.dist-info/RECORD')):
                p=Path(dist.locate_file(item));rows.append({'distribution':name,'member':str(item),'sha256':file_sha(p),'bytes':p.stat().st_size})
    prefixes={'torch':'torch','transformers':'transformers','accelerate':'accelerate','tokenizers':'tokenizers',
              'safetensors':'safetensors','numpy':'numpy','huggingface_hub':'huggingface-hub','pydantic':'pydantic'}
    seen={(r['distribution'],r['member']) for r in rows}
    proxies=[]
    for name,module in tuple(sys.modules.items()):
        prefix=name.split('.')[0]; filename=getattr(module,'__file__',None)
        if prefix not in prefixes or not filename:continue
        implementation=framework_proxy_implementation(name,module,sys.modules)
        if implementation is not None:
            proxies.append({'module':name,'implementation':implementation})
            filename=sys.modules[implementation].__file__
        distname=prefixes[prefix]; dist=dists[distname]; p=Path(filename).resolve()
        if p.suffix=='.pyc' and p.with_suffix('.py').is_file():p=p.with_suffix('.py')
        base=Path(dist.locate_file('')).resolve()
        require(base in p.parents,'import escaped measured distribution')
        member=p.relative_to(base).as_posix()
        if (distname,member) in seen:continue
        seen.add((distname,member));rows.append({'distribution':distname,'member':member,'sha256':file_sha(p),'bytes':p.stat().st_size})
    rows.sort(key=lambda r:(r['distribution'],r['member']))
    return {'python_version':sys.version.split()[0], 'interpreter_sha256':file_sha(Path(sys.executable).resolve()),
        'versions':{name:dist.version for name,dist in dists.items()},'files':rows,
        'framework_proxies':sorted(proxies,key=lambda r:r['module']),
        'scope':'actual_loaded_framework_files_and_distribution_metadata_not_complete_OS_dependency_closure'}


def revalidate_runtime(identity):
    import importlib.metadata as meta
    require(file_sha(Path(sys.executable).resolve())==identity['interpreter_sha256'],'interpreter changed')
    for name,version in identity['versions'].items():require(meta.version(name)==version,'package version changed')
    for row in identity['files']:
        path=Path(meta.distribution(row['distribution']).locate_file(row['member']))
        require(path.is_file() and path.stat().st_size==row['bytes'] and file_sha(path)==row['sha256'],'qualified runtime code changed')


class Reporter:
    def __init__(self,root):
        self.root=root;self.sequence=0
        self.journal=(root/'worker_journal.jsonl').open('xb')
    def emit(self,event):
        record={'sequence':self.sequence,**event};self.sequence+=1
        self.journal.write(canonical(record));self.journal.flush();os.fsync(self.journal.fileno())
        temporary=self.root/'phase.tmp'
        with temporary.open('wb') as stream:stream.write(canonical({'sequence':record['sequence'],'phase':event['phase']}));stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,self.root/'phase.json')
    def close(self):self.journal.close()


def worker(args):
    source=Path(__file__).resolve().parents[1]
    require(Path(args.run_root)==RUN_ROOT and Path(args.temp_root)==TEMP_ROOT
            and source==TEMP_ROOT/'source','worker roots are not the authorized trial')
    verify_source(source,args.source_sha)
    sys.path.insert(0,str(source))
    from scripts import vnext_prepare_cross_domain_answer_replay as prep
    from scripts.vnext_run_cross_domain_answer_replay import PublicRequest
    from mub.vnext.runtime.cross_domain_qwen_backend import QwenReadoutBackend
    run_root=Path(args.run_root);temp=Path(args.temp_root)
    config_raw=(run_root/'profile.json').read_bytes();require(sha(config_raw)==args.profile_sha,'profile digest mismatch')
    config=json.loads(config_raw);validate_profile(config)
    require(config['source_manifest_sha256']==args.source_sha,'profile source mismatch')
    bundle=prep.load_inputs(prep.DEFAULT_EVIDENCE,prep.DEFAULT_BEA,prep.DEFAULT_NOAA)
    prep.read_preparation(source/'results/vnext/cross_domain_answer_replay_20261002_v2',PREPARATION_SHA,bundle)
    plans=prep.build_plans(bundle)
    requests=[PublicRequest(p['request_id'],prep.render_plan(bundle,p)) for p in plans]
    report=Reporter(temp)
    backend=None
    try:
        report.emit({'phase':'qualification'})
        require(file_sha(HISTORY)==HISTORY_SHA,'historical runtime receipt drift')
        snapshot=verify_snapshot()
        require(file_sha(SNAPSHOT/'chat_template.jinja')==TEMPLATE_SHA,'snapshot chat template drift')
        if args.mode=='qualify':
            require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','qualification must be CPU-only')
            backend=QwenReadoutBackend(snapshot=SNAPSHOT,expected_gpu_uuid=GPU_UUID,
                expected_template_sha256=TEMPLATE_SHA,require_device=lambda:None)
            report.emit({'phase':'template_preflight'})
            bindings=[template_binding(r,backend.prepare(r)) for r in requests]
            from transformers import AutoConfig, AutoModelForCausalLM
            model_config=AutoConfig.from_pretrained(str(SNAPSHOT),local_files_only=True,trust_remote_code=False,revision=REVISION)
            model_class=AutoModelForCausalLM._model_mapping[type(model_config)]
            runtime=runtime_identity()
            backend.close();backend=None
            qualification={'status':'QUALIFIED_NO_MODEL_LOAD','profile_sha256':args.profile_sha,
                'source_manifest_sha256':args.source_sha,'snapshot':snapshot,'runtime':runtime,
                'bindings':bindings,'model_class':model_class.__module__+'.'+model_class.__name__,
                'model_loads':0,'tokenizer_loads':1,'generations':0,'provider_calls':0,
                'claim_boundary':'qualification_only_not_answer_accuracy_or_independent_signature'}
            save_new(temp/'qualification.json',qualification)
            report.emit({'phase':'worker_complete'});return 0
        raw=(temp/'qualification.json').read_bytes()
        require(sha(raw)==os.environ.get('MUB_QUALIFICATION_SHA'),'qualification expected digest mismatch')
        qualification=json.loads(raw)
        require(qualification['profile_sha256']==args.profile_sha and qualification['source_manifest_sha256']==args.source_sha
                and qualification['snapshot']==snapshot,'qualification identity drift')
        revalidate_runtime(qualification['runtime'])
        observe_device()
        loaded=[False]
        backend=QwenReadoutBackend(snapshot=SNAPSHOT,expected_gpu_uuid=GPU_UUID,
            expected_template_sha256=TEMPLATE_SHA,
            require_device=lambda:observe_device(os.getpid(),1024 if loaded[0] else 30720))
        def revalidate():
            bundle.assert_unchanged();verify_source(source,args.source_sha)
            require(sha((run_root/'profile.json').read_bytes())==args.profile_sha,'authorization profile changed')
        result=perform_readouts(requests,qualification['bindings'],backend,report.emit,revalidate,
                                after_load=lambda:loaded.__setitem__(0,True))
        backend=None
        save_new(temp/'readouts.json',result)
        report.emit({'phase':'post_validation'})
        require(verify_snapshot()==snapshot,'model snapshot changed after readout')
        revalidate_runtime(qualification['runtime']);revalidate()
        for name,module in tuple(sys.modules.items()):
            if (name=='mub' or name.startswith('mub.') or name.startswith('scripts.')) and getattr(module,'__file__',None):
                require(source in Path(module.__file__).resolve().parents,'project import escaped source bundle')
        scores=[]
        from mub.vnext.contracts.v3.common import typed_json_equal
        for row,plan in zip(result['rows'],plans):
            score={'request_id':row['request_id'],'release_name':plan['release_name'],'repetition_index':plan['repetition_index'],
                   'task_id':plan['task_id'],'status':row['status'],'format_valid':None,'typed_exact_match':None}
            if row['status']=='COMPLETED':
                prediction=row['prediction'];gold=bundle.tasks[plan['release_name']].gold_evidence[0].answer
                score.update(format_valid=prediction['format_valid'],typed_exact_match=bool(prediction['format_valid']
                    and prediction['disposition']=='answered' and typed_json_equal(prediction['parsed_answer'],gold)))
            scores.append(score)
        result.update(scores=scores,postvalidation=True,preparation_index_sha256=PREPARATION_SHA)
        save_new(temp/'worker_result.json',result)
        report.emit({'phase':'worker_complete'})
        return 0 if result['status']=='WORKER_COMPLETE' else 2
    except BaseException as exc:
        frames=[];tb=exc.__traceback__
        while tb is not None:
            frames.append({'file':Path(tb.tb_frame.f_code.co_filename).name,'line':tb.tb_lineno,
                           'function':tb.tb_frame.f_code.co_name});tb=tb.tb_next
        save_new(temp/'worker_error.json',{'phase':args.mode,'error_type':type(exc).__name__,'frames':frames[-10:]})
        return 2
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException:pass
        report.close()


def supervise(args):
    run=Path(args.run_root);temp=Path(args.temp_root)
    require(sys.platform=='linux' and run==RUN_ROOT and temp==TEMP_ROOT,'unauthorized live root/platform')
    require(ROOT==run/'source','supervisor source origin mismatch')
    config=json.loads((run/'profile.json').read_bytes());validate_profile(config)
    require(sha((run/'profile.json').read_bytes())==args.profile_sha and config['source_manifest_sha256']==args.source_sha,'profile pin mismatch')
    manifest=verify_source(ROOT,args.source_sha)
    output=run/'results';output.mkdir(mode=0o700,exist_ok=False)
    spec=importlib.util.spec_from_file_location('owned_runtime_tools',ROOT/'scripts/vnext_run_cross_domain_qdrant_state.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    owned=None;process=None;processes=[];files=[];error=None;worker_result=None;qualification=None;bindings_ok=False
    cleanup={k:False for k in ('worker_exited','owned_group_empty','gpu_pid_absent','temporary_root_removed')}
    counters={'load_worker_launches':0,'qualification_processes':0,'paid_provider_calls':0,'qdrant_calls':0,
              'peak_temporary_bytes':0,'retries':0}
    before_device=None;started=time.monotonic()
    def alarm(signum,frame):raise TimeoutError('overall readout deadline')
    old_alarm=signal.signal(signal.SIGALRM,alarm);old_term=signal.signal(signal.SIGTERM,alarm);signal.alarm(TOTAL_SECONDS)
    def sample():
        counters['peak_temporary_bytes']=max(counters['peak_temporary_bytes'],helper._storage_sample(temp))
    def launch(mode,env,timeout):
        nonlocal process
        journal=temp/'worker_journal.jsonl';phase=temp/'phase.json'
        if mode=='execute':
            os.rename(journal,temp/'qualification_journal.jsonl');phase.unlink()
        command=[str(PYTHON),'-B','-u',str(temp/'source/scripts/vnext_run_cross_domain_qwen_readout.py'),
                 '--mode',mode,'--run-root',str(run),'--temp-root',str(temp),
                 '--source-sha',args.source_sha,'--profile-sha',args.profile_sha]
        log=(temp/(mode+'.log')).open('xb');files.append(log)
        process=subprocess.Popen(command,cwd=temp,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        deadline=time.monotonic()+timeout;current=None
        while process.poll() is None:
            sample();now=time.monotonic()
            if now>deadline:raise TimeoutError('child absolute deadline')
            if phase.is_file():current=watch_phase(current,json.loads(phase.read_bytes()),now)
            time.sleep(0.5)
        require(process.returncode==0,'owned child failed')
    try:
        before_device=observe_device()
        helper._filesystem_info(temp.parent)
        require(shutil.disk_usage(temp.parent).free>=2*1024**3,'temporary space unavailable')
        owned=helper.claim_owned_directory(temp)
        for name in ('home','tmp','cache'): (temp/name).mkdir(mode=0o700)
        shutil.copytree(ROOT,temp/'source')
        verify_source(temp/'source',args.source_sha)
        env={'PATH':'/usr/bin:/bin','HOME':str(temp/'home'),'TMPDIR':str(temp/'tmp'),
             'TMP':str(temp/'tmp'),'TEMP':str(temp/'tmp'),'XDG_CACHE_HOME':str(temp/'cache'),
             'HF_HOME':str(temp/'cache'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
             'HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','TRITON_CACHE_DIR':str(temp/'cache'),
             'TORCH_HOME':str(temp/'cache'),'TORCHINDUCTOR_CACHE_DIR':str(temp/'cache'),
             'PYTHONPATH':str(temp/'source'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
             'CUDA_VISIBLE_DEVICES':'','CUBLAS_WORKSPACE_CONFIG':':4096:8','TOKENIZERS_PARALLELISM':'false',
             'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','LC_ALL':'C.UTF-8'}
        counters['qualification_processes']=1
        launch('qualify',env,2000)
        qualification=json.loads((temp/'qualification.json').read_bytes())
        require(qualification['status']=='QUALIFIED_NO_MODEL_LOAD','model/runtime qualification incomplete')
        observe_device()
        env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,MUB_QUALIFICATION_SHA=file_sha(temp/'qualification.json'))
        counters['load_worker_launches']=1
        launch('execute',env,1600)
        worker_result=json.loads((temp/'worker_result.json').read_bytes())
        require(worker_result['postvalidation'] is True,'postvalidation missing')
        verify_source(ROOT,args.source_sha);verify_source(temp/'source',args.source_sha)
        require(sha((run/'profile.json').read_bytes())==args.profile_sha,'profile drift')
        bindings_ok=True
    except BaseException as exc:error={'error_type':type(exc).__name__,'phase':'supervision'}
    finally:
        signal.alarm(120)
        collected={}
        try:
            for p in reversed(processes):
                try:helper._stop_process(p)
                except BaseException as exc:cleanup['process_error_type']=type(exc).__name__
            cleanup['worker_exited']=all(p.poll() is not None for p in processes)
            cleanup['owned_group_empty']=all(not helper._group_members(p.pid) for p in processes)
            for stream in files:stream.close()
            current_gpu=gpu_processes()
            cleanup['gpu_pid_absent']=not any(p['pid'] in {child.pid for child in processes} for p in current_gpu)
            try:
                if owned is not None:
                    for name in ('worker_result.json','readouts.json','qualification.json','worker_error.json',
                                 'worker_journal.jsonl','qualification_journal.jsonl'):
                        path=temp/name
                        if path.is_file():
                            require(path.stat().st_size<=8*1024**2,'oversized worker artifact')
                            collected[name]=path.read_bytes()
                    if 'worker_result.json' in collected:worker_result=json.loads(collected['worker_result.json'])
                    if 'qualification.json' in collected:qualification=json.loads(collected['qualification.json'])
            finally:
                if owned is not None and all(cleanup[k] for k in ('worker_exited','owned_group_empty','gpu_pid_absent')):
                    helper.remove_owned_directory(temp,owned);cleanup['temporary_root_removed']=True
                elif owned is None:cleanup['temporary_root_removed']=not temp.exists()
        except BaseException as exc:error=error or {'error_type':type(exc).__name__,'phase':'cleanup'}
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
    status=final_status(worker_result,cleanup,bindings_ok and error is None)
    scores=worker_result.get('scores',[]) if worker_result else []
    observed=worker_result or (json.loads(collected['readouts.json']) if 'readouts.json' in collected else None)
    result={'schema':'memupdatebench.cross-domain.qwen-readout-result.v1','status':status,
        'source_manifest_sha256':args.source_sha,'profile_sha256':args.profile_sha,'preparation_index_sha256':PREPARATION_SHA,
        'model_id':MODEL_ID,'model_revision':REVISION,'model_tree_sha256':MODEL_TREE,
        'device':before_device,'cleanup':cleanup,'error':error,'bindings_unchanged':bindings_ok,
        'execution_accounting':counters,'scores':scores,
        'answer_metrics':({'typed_exact_matches':sum(r['typed_exact_match'] is True for r in scores),'denominator':4,
                           'format_valid':sum(r['format_valid'] is True for r in scores)} if status=='COMPLETE' else None),
        'model_loads_completed':(int(observed['lifecycle']['load_completed']) if observed else
                                 None if counters['load_worker_launches'] else 0),
        'generation_attempts':(sum(r['attempts'] for r in observed['rows']) if observed else
                               None if counters['load_worker_launches'] else 0),
        'cluster_unit':'source_trajectory','cluster_n':2,'prospective_requests':4,'native_answer':False,
        'scientific_release_allowed':False,'benchmark_accuracy_claimed':False,
        'claim_boundary':'separate Qwen readout of four frozen single-entry contexts from two trajectories; no native Qdrant or broad benchmark claim',
        'elapsed_seconds':time.monotonic()-started}
    payload={'summary.json':canonical(result),'authorization_profile.json':canonical(config)}
    for name,raw in collected.items():
        # The worker serializes hashes and parsed predictions, never raw generation text.
        require(b'"raw_output"' not in raw and b'Use only the retrieved memory entries' not in raw,'raw text in output')
        payload[name]=raw
    index={'schema':'memupdatebench.cross-domain.qwen-readout-index.v1','status':status,
           'artifacts':[{'path':n,'bytes':len(raw),'sha256':sha(raw)} for n,raw in sorted(payload.items())],
           'scientific_release_allowed':False}
    for name,raw in payload.items():
        with (output/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    index_sha=save_new(output/'index.json',index)
    print(json.dumps({'status':status,'index_sha256':index_sha,'answer_metrics':result['answer_metrics'],'cleanup':cleanup,'error':error}),flush=True)
    return 0 if status=='COMPLETE' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--mode',choices=('supervise','qualify','execute'),default='supervise')
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--temp-root',required=True)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--profile-sha',required=True)
    args=parser.parse_args(argv)
    if args.mode=='supervise':return supervise(args)
    try:return worker(args)
    except BaseException as exc:
        target=Path(args.temp_root)/'worker_error.json'
        if Path(args.temp_root)==TEMP_ROOT and target.parent.is_dir() and not target.exists():
            frames=[];tb=exc.__traceback__
            while tb is not None:
                frames.append({'file':Path(tb.tb_frame.f_code.co_filename).name,'line':tb.tb_lineno,
                               'function':tb.tb_frame.f_code.co_name});tb=tb.tb_next
            save_new(target,{'phase':'worker_bootstrap','error_type':type(exc).__name__,'frames':frames[-10:]})
        return 2


if __name__=='__main__':raise SystemExit(main())
