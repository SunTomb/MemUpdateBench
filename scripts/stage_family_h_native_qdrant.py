from __future__ import annotations
import argparse, hashlib, io, json, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUN_ID='family_h_native_qdrant_full_20261011_tang3_v8'
def canonical(value): return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def _tree_manifest(root,exclude=()):
 rows=[]
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.name not in exclude:
   raw=path.read_bytes(); rows.append({'path':path.relative_to(root).as_posix(),'bytes':len(raw),'sha256':sha(raw)})
 return {'schema':'memupdatebench.family-h.native-tree-manifest.v1','files':rows,'tree_sha256':sha(canonical(rows))}
def stage(output_root,*,source_root,project_root=ROOT):
 output_root=Path(output_root).absolute(); source_root=Path(source_root).absolute(); project_root=Path(project_root).absolute()
 if output_root.exists() or output_root.is_symlink(): raise FileExistsError(output_root)
 if not source_root.is_dir() or not (source_root/'release_index.json').is_file(): raise ValueError('Family-H release root is unavailable')
 output_root.mkdir(parents=True,exist_ok=False); source=output_root/'source'; release=output_root/'release'; source.mkdir(); release.mkdir()
 source_files=list((project_root/'mub').rglob('*.py'))+[project_root/'scripts/__init__.py',project_root/'scripts/vnext_run_family_h_native_qdrant.py']
 for path in sorted(set(source_files)):
  target=source/path.relative_to(project_root); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(path.read_bytes())
 (source/'source_manifest.json').write_bytes(canonical(_tree_manifest(source)))
 for path in sorted(source_root.rglob('*')):
  if path.is_file():
   target=release/path.relative_to(source_root); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(path.read_bytes())
 profile={'schema':'memupdatebench.family-h.native-qdrant-stage.v1','run_id':RUN_ID,'source_manifest_sha256':sha((source/'source_manifest.json').read_bytes()),'release_index_sha256':sha((release/'release_index.json').read_bytes()),'release_manifest_sha256':sha((release/'release_manifest.json').read_bytes()),'scientific_release_allowed':False,'answer_metrics':None,'execution':{'model_loads':0,'generations':0,'provider_model_calls':0,'gpu_calls':0}}
 (output_root/'profile.json').write_bytes(canonical(profile)); archive=output_root.with_suffix('.tar')
 with tarfile.open(archive,'w') as tar:
  for path in sorted(output_root.rglob('*')):
   if path.is_file():
    raw=path.read_bytes(); member=tarfile.TarInfo(path.relative_to(output_root).as_posix()); member.size=len(raw); member.mode=0o600; member.mtime=0; tar.addfile(member,io.BytesIO(raw))
 return {'run_id':RUN_ID,**{k:profile[k] for k in ('source_manifest_sha256','release_index_sha256','release_manifest_sha256')},'profile_sha256':sha((output_root/'profile.json').read_bytes()),'archive_sha256':sha(archive.read_bytes()),'archive_bytes':archive.stat().st_size}
def main(argv=None):
 parser=argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--output-root',type=Path,required=True); parser.add_argument('--source-root',type=Path,default=ROOT/'data/vnext/family_h_independent/v1_published'); args=parser.parse_args(argv); print(json.dumps(stage(args.output_root,source_root=args.source_root),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
