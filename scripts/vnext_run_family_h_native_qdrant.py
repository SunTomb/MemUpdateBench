"""Bounded Family-H direct-Qdrant state/retrieval fixture runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.native_multi_object import NativeMultiObjectMutationRequestV3, NativeRetrievalRequestV3
from mub.vnext.contracts.common import thaw_json
from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider, build_qdrant_native_multi_object_configuration

QDRANT_VERSION = "1.19.0"
QDRANT_REVISION = "74f3e85b9473c62560006c043e13737ce6b48412"
PORT = 16333


def canonical(value: Any) -> bytes:
    return (json.dumps(thaw_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def _load_record_index(release_root: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for line in (release_root / "normalized_records.jsonl").read_bytes().splitlines():
        item = json.loads(line)
        record = item.get("record")
        if not isinstance(record, dict):
            raise ValueError("Family-H normalized record is malformed")
        record_hash = item.get("anchor", {}).get("normalized_record_sha256")
        if not isinstance(record_hash, str) or not record_hash:
            raise ValueError("Family-H normalized record hash is missing")
        records[record["record_id"]] = {"record": record, "hash": record_hash}
    if not records:
        raise ValueError("Family-H normalized record index is empty")
    return records


def load_family_h_tasks(release_root: Path) -> list[dict[str, Any]]:
    index = json.loads((release_root / "release_index.json").read_bytes())
    if index.get("status") != "FINAL_APPROVED" or index.get("scientific_release_allowed") is not False:
        raise ValueError("Family-H release boundary mismatch")
    manifest = json.loads((release_root / "release_manifest.json").read_bytes())
    if manifest.get("counts") != {"decisions": 33, "events": 27, "records": 27, "semantic_cores": 3, "sources": 3, "tasks": 3}:
        raise ValueError("Family-H release counts mismatch")
    tasks = [json.loads(line) for line in (release_root / "tasks.jsonl").read_bytes().splitlines()]
    if len(tasks) != 3 or sum(len(task["events"]) for task in tasks) != 27:
        raise ValueError("Family-H task cardinality mismatch")
    record_index = _load_record_index(release_root)
    for task in tasks:
        task["__normalized_record_index"] = record_index
    return tasks


def public_event_request(task_id: str, event: dict[str, Any], record_index: dict[str, dict[str, Any]] | None = None) -> NativeMultiObjectMutationRequestV3:
    if not isinstance(event, dict) or not isinstance(event.get("raw_text"), str):
        raise ValueError("Family-H event raw source payload is missing")
    try:
        payload = json.loads(event["raw_text"])
    except json.JSONDecodeError as exc:
        raise ValueError("Family-H event raw source payload is invalid") from exc
    if payload.get("action") not in {item.value for item in Operation}:
        raise ValueError("Family-H event action is invalid")
    if payload.get("object_key") is None:
        raise ValueError("Family-H event identity is missing")
    if payload.get("action") == "NOOP":
        objects = ()
    else:
        item = {"object_key": payload["object_key"]}
        if payload["action"] in {"ADD", "UPDATE"}:
            if "value" not in payload:
                raise ValueError("Family-H event value is missing")
            item["value"] = payload["value"]
        objects = (item,)
    if event.get("event_id") != payload.get("event_id", event.get("event_id")):
        raise ValueError("Family-H event identity drift")
    if event.get("sequence_index") != int(event.get("timestamp", event.get("sequence_index"))):
        raise ValueError("Family-H event sequence drift")
    record_index = record_index or _load_record_index(Path(__file__).resolve().parents[1] / "data/vnext/family_h_independent/v1_published")
    record_id = payload.get("source_heading_anchor")
    indexed = record_index.get(record_id)
    record_hash = payload.get("normalized_record_sha256")
    if indexed is None or indexed.get("hash") != record_hash:
        raise ValueError("Family-H source payload record hash/anchor mismatch")
    record = indexed["record"]
    if record.get("object_key") != payload.get("object_key") or record.get("value") != payload.get("value"):
        raise ValueError("Family-H source payload event identity/value drift")
    return NativeMultiObjectMutationRequestV3(
        request_id=f"{task_id}-event-{event['event_id']}", event_id=event["event_id"],
        sequence_index=event["sequence_index"], operation=Operation(payload["action"]),
        objects=objects, source_event_ids=(event["event_id"],),
        provenance={"kind": "declared", "provider_owned": False},
    )


def request_from_action(task_id: str, action: dict[str, Any], index: int) -> NativeMultiObjectMutationRequestV3:
    objects = []
    for key in action.get("target_object_keys", []):
        item = {"object_key": key}
        if action.get("operation") in {"ADD", "UPDATE"}:
            item["value"] = action.get("value")
        objects.append(item)
    return NativeMultiObjectMutationRequestV3(
        request_id=f"{task_id}-mutation-{index}", event_id=action["event_id"], sequence_index=index,
        operation=Operation(action["operation"]), objects=tuple(objects), source_event_ids=(action["event_id"],),
        provenance={"kind": "declared", "provider_owned": False},
    )


def retrieval_request(task: dict[str, Any], namespace: str) -> NativeRetrievalRequestV3:
    query = task["queries"][0]
    return NativeRetrievalRequestV3(query_id=query["query_id"], query_text=query["text"], k=32, namespace=namespace, filters={}, options={})


def _point_field(point: Any, name: str, default=None):
    if isinstance(point, dict):
        return point.get(name, default)
    return getattr(point, name, default)


def export_namespace(provider, namespace: str) -> list[dict[str, Any]]:
    response = provider._provider_call("scroll", collection_name=provider.collection_name, scroll_filter=provider._filter_for_scope(namespace, "namespace"), limit=128, with_payload=True, with_vectors=False)
    points, cursor = provider._scroll_points(response)
    if cursor is not None:
        raise ValueError("Family-H export exceeded one bounded page")
    return [{"id": str(_point_field(point, "id")), "payload": _point_field(point, "payload")} for point in points]


def cleanup_provider(provider, client):
    if provider is None or client is None:
        return {"client_closed": True, "collection_deleted": True}
    collection_deleted = False
    client_closed = False
    try:
        deleted = bool(client.delete_collection(collection_name=provider.collection_name))
        exists = bool(client.collection_exists(collection_name=provider.collection_name))
        collection_deleted = deleted and not exists
    finally:
        provider.close()
        client_closed = True
    if not collection_deleted or not client_closed:
        raise RuntimeError("Family-H collection/client cleanup incomplete")
    return {"client_closed": True, "collection_deleted": True}


def _witnesses(provider):
    value = getattr(provider, "observation_witnesses", ())
    if callable(value): value = value()
    return [thaw_json(item.model_dump(mode='json')) if hasattr(item, 'model_dump') else item for item in value]


def run_fixture(*, release_root: Path, temp_root: Path, qdrant_binary: Path, api_key: str) -> dict[str, Any]:
    release_snapshot = snapshot(release_root)
    tasks = load_family_h_tasks(release_root)
    server = None; log = None; rows = []; error = None
    try:
        config = {"log_level":"INFO","telemetry_disabled":True,"service":{"host":"127.0.0.1","http_port":PORT,"grpc_port":None,"enable_cors":False,"max_workers":2},"cluster":{"enabled":False},"storage":{"storage_path":str(temp_root/'storage'),'snapshots_path':str(temp_root/'snapshots'),'temp_path':str(temp_root/'temp'),'performance':{'max_search_threads':2},'optimizers':{'max_optimization_threads':1},'wal':{'wal_capacity_mb':16}}}
        (temp_root/'config.yaml').write_bytes(canonical(config)); env={'PATH':'/usr/bin:/bin','HOME':str(temp_root/'home'),'TMPDIR':str(temp_root/'temp'),'QDRANT__SERVICE__API_KEY':api_key,'RUST_BACKTRACE':'0','CUDA_VISIBLE_DEVICES':''}
        log=(temp_root/'server.log').open('xb'); server=subprocess.Popen([str(qdrant_binary),'--config-path',str(temp_root/'config.yaml')],cwd=temp_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        import http.client
        ready=False
        for _ in range(240):
            if server.poll() is not None: raise RuntimeError("Qdrant server exited before readiness")
            try:
                c=http.client.HTTPConnection('127.0.0.1',PORT,timeout=2); c.request('GET','/',headers={'api-key':api_key}); r=c.getresponse(); raw=r.read(); c.close()
                if r.status==200:
                    info=json.loads(raw); ready=info.get('version')==QDRANT_VERSION and info.get('commit')==QDRANT_REVISION
                    if not ready: raise RuntimeError('Qdrant identity mismatch')
                    break
            except (OSError,TimeoutError,json.JSONDecodeError): time.sleep(.5)
        if not ready: raise TimeoutError('Qdrant readiness timeout')
        from qdrant_client import QdrantClient, models
        for task in tasks:
            namespace=f"family_h_{task['task_id']}"; client=None; provider=None; row={'task_id':task['task_id'],'namespace':namespace,'status':'BLOCKED','mutation_rows':[]}
            try:
                client=QdrantClient(url=f'http://127.0.0.1:{PORT}',api_key=api_key,prefer_grpc=False,timeout=10,trust_env=False,check_compatibility=False); client.models=models
                config=build_qdrant_native_multi_object_configuration(run_id=f"family-h-{task['task_id']}",path=temp_root/'unused')
                provider=QdrantNativeMultiObjectProvider(configuration=config,client=client,runtime_namespace=namespace); provider.reset(namespace)
                for i, event in enumerate(task['events']):
                    req=public_event_request(task['task_id'],event,task['__normalized_record_index']); res=provider.mutate(req); row['mutation_rows'].append({'request':req.model_dump(mode='json'),'result':res.model_dump(mode='json'),'export':export_namespace(provider,namespace)})
                trace=provider.retrieve(retrieval_request(task,namespace)); row['retrieval']=trace.model_dump(mode='json'); row['retrieval_raw']=[{'id':e.provider_entry_id,'score':e.score,'payload':e.provider_payload} for e in trace.entries]; row['final_export']=export_namespace(provider,namespace); provider.reset(namespace); row['final_empty']=export_namespace(provider,namespace); row['status']='OBSERVED'
            except BaseException as exc: row['error']={'type':type(exc).__name__}
            finally:
                if provider is not None:
                    row["provider_witnesses"] = _witnesses(provider)
                    try: row["cleanup"] = cleanup_provider(provider, client)
                    except BaseException as exc: row["cleanup"] = {"client_closed": False, "collection_deleted": False}; row["cleanup_error"] = {"type": type(exc).__name__}
                    provider = None
                    client = None
                elif client is not None:
                    try: client.close()
                    except BaseException: pass
                    row["cleanup"] = {"client_closed": True, "collection_deleted": False}
            rows.append(row)
            if row['status']!='OBSERVED': break
    except BaseException as exc: error={'type':type(exc).__name__}
    finally:
        if server is not None:
            try: os.killpg(server.pid,signal.SIGTERM); server.wait(timeout=20)
            except BaseException:
                try: os.killpg(server.pid,signal.SIGKILL)
                except BaseException: pass
        if log is not None: log.close()
    if snapshot(release_root)!=release_snapshot: raise RuntimeError('Family-H release changed during run')
    return {'schema':'memupdatebench.family-h.native-qdrant-observations.v1','status':'OBSERVATIONS_COMPLETE' if not error and len(rows)==3 and all(r['status']=='OBSERVED' for r in rows) else 'BLOCKED','scientific_release_allowed':False,'answer_metrics':None,'error':error,'rows':rows,'source_release_snapshot':release_snapshot,'runtime':{'qdrant_version':QDRANT_VERSION,'qdrant_revision':QDRANT_REVISION,'qdrant_binary_sha256':hashlib.sha256(qdrant_binary.read_bytes()).hexdigest()},'execution':{'model_loads':0,'generations':0,'gpu_calls':0,'provider_model_calls':0}}


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--release-root',type=Path,required=True); parser.add_argument('--temp-root',type=Path,required=True); parser.add_argument('--qdrant-binary',type=Path,required=True); parser.add_argument('--api-key',required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(argv)
    if args.temp_root.exists():
        expected_dirs = {"home", "temp", "storage", "snapshots", "config", "source", "release"}
        if not args.temp_root.is_dir() or set(item.name for item in args.temp_root.iterdir()) != expected_dirs:
            raise FileExistsError(args.temp_root)
    else:
        args.temp_root.mkdir(parents=True, exist_ok=False)
    result=run_fixture(release_root=args.release_root,temp_root=args.temp_root,qdrant_binary=args.qdrant_binary,api_key=args.api_key); args.output.write_bytes(canonical(result)); print(json.dumps({'status':result['status'],'tasks':len(result['rows']),'path':str(args.output)},sort_keys=True)); return 0 if result['status']=='OBSERVATIONS_COMPLETE' else 2

if __name__=='__main__': raise SystemExit(main())
