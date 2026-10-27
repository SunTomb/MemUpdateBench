"""Run only the bounded native Graphiti capability smoke, never the study matrix.

Invoke with the pinned Graphiti environment. Model serving runs in the existing
Transformers environment. All process groups and files belong to this smoke.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
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
from mub.vnext.external.providers.graphiti_native_episode import GraphitiNativeEpisode
from mub.vnext.external.visibility import visible_event_input, visible_query_input
from scripts.vnext_prepare_version_transfer_decision import read_packet

GRAPHITI_REVISION = '5c27edfa8d3e0f49b2571bcebb4261bcc89ee037'
MODEL_ID = 'Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28'
EMBEDDING_ID = 'all-MiniLM-L6-v2@c9745ed1d9f207416be6d2e6f8de32d1f16199bf'


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, default=str)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available(port):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))


def wait_port(process, port, seconds):
    deadline = time.monotonic() + seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'owned service on port {port} exited before ready')
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'owned service readiness timeout on port {port}')
            time.sleep(1)


def stop_owned(process):
    if process is None:
        return None
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return process.poll() is not None


def database_config(run, neo):
    conf = run / 'database-conf'
    conf.mkdir()
    for name in ('database-data', 'database-logs', 'database-run', 'tmp', 'home', 'cache'):
        (run / name).mkdir()
    for xml in (neo / 'conf').glob('*.xml'):
        (conf / xml.name).write_bytes(xml.read_bytes())
    settings = {
        'server.default_listen_address': '127.0.0.1',
        'server.bolt.enabled': 'true', 'server.bolt.listen_address': '127.0.0.1:17687',
        'server.bolt.advertised_address': '127.0.0.1:17687',
        'server.http.enabled': 'false', 'server.https.enabled': 'false',
        'server.memory.heap.initial_size': '256m', 'server.memory.heap.max_size': '512m',
        'server.memory.pagecache.size': '128m', 'dbms.security.auth_enabled': 'false',
        'dbms.usage_report.enabled': 'false', 'dbms.security.procedures.allowlist': 'apoc.*',
        'db.tx_log.rotation.retention_policy': 'false',
        'server.directories.data': str(run / 'database-data'),
        'server.directories.logs': str(run / 'database-logs'),
        'server.directories.run': str(run / 'database-run'),
    }
    text = '\n'.join(f'{key}={value}' for key, value in settings.items())
    text += f'\nserver.jvm.additional=-Djava.io.tmpdir={run / "tmp"}\n'
    text += 'server.jvm.additional=-XX:-UsePerfData\n'
    (conf / 'neo4j.conf').write_text(text)
    return conf


async def native_cases(run, tasks):
    from openai import AsyncOpenAI
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
    from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator
    from graphiti_core.telemetry import is_telemetry_enabled

    assert not is_telemetry_enabled()

    class CappedClient(OpenAIGenericClient):
        async def generate_response(self, messages, response_model=None, max_tokens=None, *args, **kwargs):
            return await super().generate_response(messages, response_model,
                max_tokens=min(max_tokens if max_tokens is not None else 2048, 2048), *args, **kwargs)

    class UnsupportedCrossEncoder(CrossEncoderClient):
        async def rank(self, query, passages):
            raise RuntimeError('UNSUPPORTED_RERANKER_REQUEST: only declared native RRF is qualified')

    api = AsyncOpenAI(api_key='local-only-placeholder', base_url='http://127.0.0.1:18043/v1',
                      default_headers={'X-MUB-Role': 'graphiti'}, max_retries=0, timeout=120)
    llm = CappedClient(config=LLMConfig(model=MODEL_ID, small_model=MODEL_ID,
                                      temperature=0, max_tokens=2048),
                       client=api, max_tokens=2048, structured_output_mode='json_object')
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(embedding_model=EMBEDDING_ID,
                                                        embedding_dim=384), client=api)
    driver = Neo4jDriver('bolt://127.0.0.1:17687', None, None)
    graph = Graphiti(graph_driver=driver, llm_client=llm, embedder=embedder,
                     cross_encoder=UnsupportedCrossEncoder(), max_coroutines=1)
    cases = []
    try:
        await graph.build_indices_and_constraints()
        for index, task in enumerate(tasks):
            group = f'version_transfer_owned_smoke_{index}'
            bridge = GraphitiNativeEpisode(graph, namespace=group, source_type=EpisodeType.text,
                filter_factory=GraphitiNativeEpisode.validity_filter_factory(SearchFilters, DateFilter, ComparisonOperator),
                search_config=EDGE_HYBRID_SEARCH_RRF)
            row = {'case': task.metadata.extra['smoke_case'], 'status': 'NOT_RUN',
                   'event_attempts': 0, 'event_completions': 0, 'retrieval_completions': 0}
            cases.append(row)
            try:
                for position, event in enumerate(task.events):
                    row['event_attempts'] += 1
                    result = await bridge.ingest(visible_event_input(event, runtime_namespace=group))
                    assert result.episode.valid_at.isoformat().replace('+00:00', 'Z') == event.timestamp
                    assert result.episode.group_id == group
                    assert all(node.group_id == group for node in result.nodes)
                    assert all(edge.group_id == group for edge in result.edges)
                    write_new(run / f'case_{index}_event_{position}.json', result.model_dump(mode='json'))
                    row['event_completions'] += 1
                for position, query in enumerate(task.queries):
                    public = visible_query_input(query, runtime_namespace=group, k=8)
                    result = await bridge.retrieve(public)
                    assert all(edge.group_id == group for edge in result.edges)
                    write_new(run / f'case_{index}_retrieval_{position}.json', result.model_dump(mode='json'))
                    row['retrieval_completions'] += 1
                row['status'] = 'OBSERVED'
            except Exception as exc:
                row.update(status='BLOCKED', error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                await driver.execute_query('MATCH (n) WHERE n.group_id = $group DETACH DELETE n', group=group)
                records, _, _ = await driver.execute_query('MATCH (n) WHERE n.group_id = $group RETURN count(n) AS n', group=group)
                row['namespace_empty_after_cleanup'] = records[0]['n'] == 0
                write_new(run / f'case_{index}_receipt.json', row)
        return cases
    finally:
        try:
            await graph.close()
        finally:
            await api.close()
            write_new(run / 'native_case_summary.json', cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--packet-root', type=Path, required=True)
    parser.add_argument('--packet-index-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    base = Path('/NAS/yesh/MemUpdateBench')
    run = args.output_root.absolute()
    if not run.is_relative_to(base / '_local') or run.exists() or run.is_symlink() or '..' in run.parts:
        raise ValueError('smoke output must be a new owned project-local directory')
    assert_no_reparse_components(run)
    args.packet_root = args.packet_root.absolute()
    if not args.packet_root.resolve().is_relative_to(base):
        raise ValueError('packet must remain within authorized project boundary')
    from importlib.metadata import version
    if version('graphiti-core') != '0.31.2':
        raise ValueError('pinned Graphiti SDK environment required')
    read_packet(args.packet_root, args.packet_index_sha256)
    tasks = [MemUpdateTaskV3.model_validate_json(line) for line in (args.packet_root / 'smoke.jsonl').read_bytes().splitlines()]
    assert len(tasks) == 4 and all(len(task.events) <= 5 for task in tasks)
    for port in (17687, 18043):
        available(port)
    free = subprocess.check_output(['nvidia-smi', '--id='+args.gpu_uuid,
                                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    if int(free.strip()) < 24576:
        raise RuntimeError('GPU admission blocked before any model loading')
    run.mkdir()
    neo_base = base / 'external/version_transfer_neo4j_5260_v1'
    neo = neo_base / 'database_runtime/neo4j-community-5.26.0'
    java = neo_base / 'java_runtime/zulu21.54.107-ca-jre21.0.13-linux_x64'
    conf = database_config(run, neo)
    env = dict(os.environ, HOME=str(run/'home'), TMPDIR=str(run/'tmp'), XDG_CACHE_HOME=str(run/'cache'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', GRAPHITI_TELEMETRY_ENABLED='false',
               EMBEDDING_DIM='384', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=args.gpu_uuid,
               HF_HUB_CACHE='/NAS/yesh/hf_cache/hub', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', NO_PROXY='127.0.0.1,localhost',
               http_proxy='', https_proxy='', all_proxy='', no_proxy='127.0.0.1,localhost')
    for key in ('GIT_DIR', 'GIT_WORK_TREE'):
        env.pop(key, None)
    os.environ.update({key: env[key] for key in ('HOME','TMPDIR','XDG_CACHE_HOME','GRAPHITI_TELEMETRY_ENABLED',
                                              'EMBEDDING_DIM','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY',
                                              'http_proxy','https_proxy','all_proxy','no_proxy')})
    report = {'status': 'BLOCKED', 'evidence_class': 'native_graphiti_capability_smoke',
              'scientific_evidence': False, 'study_validation_tasks': 0, 'paid_api_calls': 0,
              'graphiti_revision': GRAPHITI_REVISION, 'retrieval_recipe': 'EDGE_HYBRID_SEARCH_RRF',
              'packet_index_sha256': args.packet_index_sha256, 'llm_attempt_ceiling': 256,
              'embedding_model': EMBEDDING_ID, 'model': MODEL_ID,
              'runtime_source_hashes': {name: sha(ROOT/name) for name in (
                  'scripts/vnext_run_version_transfer_decision.py',
                  'scripts/vnext_serve_version_transfer_local.py',
                  'mub/vnext/runtime/version_transfer_budget.py',
                  'mub/vnext/runtime/version_transfer_embedding.py',
                  'mub/vnext/runtime/version_transfer_local.py',
                  'mub/vnext/external/providers/graphiti_native_episode.py')}}
    processes, streams = [], []
    started = time.monotonic()
    try:
        db_env = dict(env, JAVA_HOME=str(java), NEO4J_HOME=str(neo), NEO4J_CONF=str(conf),
                      JAVA_TOOL_OPTIONS=f'-Djava.io.tmpdir={run/"tmp"} -XX:-UsePerfData')
        db_log = (run/'database-console.log').open('xb'); streams.append(db_log)
        database = subprocess.Popen(['bash',str(neo/'bin/neo4j'),'console'], cwd=base, env=db_env,
                                    stdout=db_log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(database)
        wait_port(database, 17687, 120)
        model_log = (run/'model-console.log').open('xb'); streams.append(model_log)
        free_now = subprocess.check_output(['nvidia-smi', '--id='+args.gpu_uuid,
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        report['admitted_free_mib_before_model_load'] = int(free_now.strip())
        if int(free_now.strip()) < 24576:
            raise RuntimeError('GPU headroom changed before model load; no model started')
        snapshot = base.parent/'hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
        mini = base.parent/'hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
        model = subprocess.Popen([str(base.parent/'miniconda3/envs/gmsra/bin/python'),'-B','-I',
            str(ROOT/'scripts/vnext_serve_version_transfer_local.py'), '--snapshot',str(snapshot),
            '--gpu-uuid',args.gpu_uuid,'--model-id',MODEL_ID,'--ledger',str(run/'llm-calls.jsonl'),
            '--attempts','256','--input-tokens','1000000','--output-tokens','524288','--seconds','3600',
            '--embedding-snapshot',str(mini),'--embedding-model-id',EMBEDDING_ID,
            '--embedding-ledger',str(run/'embedding-calls.jsonl')], cwd=base, env=env,
            stdout=model_log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(model)
        wait_port(model,18043,600)
        with urlopen('http://127.0.0.1:18043/health',timeout=5) as response:
            assert json.load(response)['model'] == MODEL_ID
        cases = asyncio.run(asyncio.wait_for(native_cases(run,tasks),timeout=2400))
        report['cases'] = cases
        if len(cases)==4 and all(c['status']=='OBSERVED' and c['namespace_empty_after_cleanup'] for c in cases):
            report['status']='READY'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        report['owned_process_cleanup'] = [stop_owned(p) for p in reversed(processes)]
        for stream in streams:
            stream.close()
        report['owned_ports_closed'] = {}
        for port in (17687,18043):
            with socket.socket() as sock:
                report['owned_ports_closed'][str(port)] = sock.connect_ex(('127.0.0.1',port)) != 0
        for label,name in (('llm','llm-calls.jsonl'),('embedding','embedding-calls.jsonl')):
            path=run/name
            if path.exists():
                events=[json.loads(line) for line in path.read_bytes().splitlines()]
                report[label+'_ledger_sha256']=sha(path)
                report[label+'_event_counts']={kind:sum(e['event']==kind for e in events) for kind in sorted({e['event'] for e in events})}
        report['elapsed_seconds']=time.monotonic()-started
        try:
            gpu_pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).split()
            report['owned_gpu_process_absent'] = all(str(p.pid) not in gpu_pids for p in processes)
        except Exception:
            report['owned_gpu_process_absent'] = None
        llm_counts = report.get('llm_event_counts', {})
        embed_counts = report.get('embedding_event_counts', {})
        report['call_accounting_complete'] = (
            llm_counts.get('reserved',0) > 0 and llm_counts.get('reserved') == llm_counts.get('settled')
            and embed_counts.get('embedding_reserved',0) > 0
            and embed_counts.get('embedding_reserved') == embed_counts.get('embedding_settled'))
        if (not all(report['owned_process_cleanup']) or not all(report['owned_ports_closed'].values())
                or report['owned_gpu_process_absent'] is not True or not report['call_accounting_complete']):
            report['status']='BLOCKED'
        write_new(run/'qualification.json',report)
        print(json.dumps(report,sort_keys=True),flush=True)
    return 0 if report['status']=='READY' else 1


if __name__=='__main__':
    raise SystemExit(main())
