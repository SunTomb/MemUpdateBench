from __future__ import annotations
import json, os
from pathlib import Path
from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider, build_qdrant_native_multi_object_configuration
from scripts.vnext_native_canary_cases import audit_cases, run_cases
from qdrant_client import QdrantClient, models

ROOT=Path(__file__).resolve().parents[1]
PORT=16333

def canonical(v): return (json.dumps(v, sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False)+'\n').encode()

def main():
    temp=Path(os.environ['MUB_NATIVE_TEMP']); api_key=os.environ['MUB_NATIVE_API_KEY']
    requests=[json.loads(line) for line in (temp/'preparation'/'requests.jsonl').read_bytes().splitlines()]
    expectations=[json.loads(line) for line in (temp/'preparation'/'expectations.jsonl').read_bytes().splitlines()]
    def factory(case):
        client=QdrantClient(url=f'http://127.0.0.1:{PORT}', api_key=api_key, prefer_grpc=False, timeout=10, trust_env=False, check_compatibility=False)
        client.models=models
        config=build_qdrant_native_multi_object_configuration(run_id=f"native-{case['case_id']}-{case['repetition']}", path=temp/'unused_client_path')
        provider=QdrantNativeMultiObjectProvider(configuration=config, client=client, runtime_namespace=case['runtime_namespace'])
        def cleanup():
            deleted=bool(client.delete_collection(collection_name=provider.collection_name))
            exists=bool(client.collection_exists(collection_name=provider.collection_name))
            provider.close()
            return {'client_closed':True, 'collection_deleted':deleted and not exists}
        return provider, cleanup
    observations=run_cases(requests, factory)
    audit=audit_cases(requests, expectations, observations) if observations['status']=='OBSERVATIONS_COMPLETE' else None
    (temp/'observations.json').write_bytes(canonical(observations))
    if audit is not None: (temp/'audit.json').write_bytes(canonical(audit))
    print(json.dumps({'status':observations['status'],'audit':audit,'counts':observations['counts']},sort_keys=True))
    return 0 if audit is not None else 2
if __name__=='__main__': raise SystemExit(main())
