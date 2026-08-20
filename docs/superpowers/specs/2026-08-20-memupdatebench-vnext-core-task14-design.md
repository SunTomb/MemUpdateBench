# MemUpdateBench vNext Core Task 14 Final Review Design

**Status:** FROZEN FOR IMPLEMENTATION  
**Scope:** read-only final Core release review  
**Non-goal:** no task regeneration, model execution, new experiment, new statistic, or claim expansion

## 1. Purpose

Core Tasks 9–13 and the immutable Core task release are complete, but overall Core is not approved. Task 14 is the only gate allowed to decide overall Core `FINAL_APPROVED`. It aggregates and freshly verifies existing authenticated evidence; it never creates new scientific evidence.

The gate is fail-closed. A structural report cannot approve itself. Only an explicit verification of current source roots, followed by atomic publication and final-root reopen verification, can create a verified release object whose historical attestation records `final_approval_at_verification=true`.

## 2. Immutable boundaries

Task 14 must not modify, regenerate, overwrite, rebind, or add files beneath:

```text
data/vnext/core/v3
results/vnext/core_task13_bc82566_v1
```

It must preserve these evidence classes:

- Phase 0 and Pilot: bounded engineering/release prerequisites.
- Core task release: immutable task/schema/human-audit evidence only.
- Task 9: deterministic built-in/runtime/control engineering evidence.
- Task 10: genuine Mem0 capability/admission evidence, never accuracy.
- Task 11: answer-model provenance and qualification evidence, never inference results.
- Task 12: real offline prompted-answer matrix; fake-offline and `slot_direct` are excluded from scientific accuracy.
- Task 13: semantic-core statistics, contrasts, claims, and cases over Task 12; unsupported/null remains unsupported/null.

Task 13 platform boundary is immutable:

```text
published local final:
  results/vnext/core_task13_bc82566_v1

verified remote NFS staging evidence only:
  /NAS/yesh/MemUpdateBench/results/vnext/.mub-task13-stage-1a791f4cbfdd471aa6a8bd45ab6432d4
```

The remote staging path may be represented as evidence but can never satisfy a published-final-root check.

## 3. Output set and acyclic hashes

Task 14 publishes exactly five canonical JSON files:

```text
core_final_review_report.json
core_final_evidence_graph.json
core_final_verification_attestation.json
core_final_root_manifest.json
core_final_root_index.json
```

Hash construction is acyclic:

1. Build report and graph.
2. Build attestation over report, graph, source snapshots, trusted revision, and verification-time decision.
3. Build manifest over report, graph, and attestation.
4. Build index over report, graph, attestation, and manifest.
5. The index never binds or hashes itself.

The structural report status is one of:

```text
NOT_APPROVED
READY_FOR_VERIFICATION
```

It never stores `FINAL_APPROVED`. The persisted attestation may record `final_approval_at_verification=true` only when all structural checks pass and current roots match. After publication, a reopen verifier re-parses and re-hashes the five-file root and returns an immutable `VerifiedCoreFinalRelease`. The overall release statement requires that wrapper and its verified attestation.

## 4. Evidence graph

Required nodes:

1. immutable Core release and candidate validation receipt/root digest;
2. Core human-audit report and verification attestation;
3. Task 9 built-in/control implementation revision and frozen model provenance;
4. Task 10 Mem0 external admission report and decision;
5. Task 11 qualification and Mistral snapshot provenance;
6. Task 12 matrix manifest, summary, integrity audit, and real-run completeness;
7. Task 13 local final index, receipt, statistics, contrasts, cases, case index, ledger, independent audit, and NFS-staging exclusion;
8. Task 14 report, graph, attestation, manifest, and index.

Allowed edge types:

```text
depends_on
authenticates
derived_from
qualifies
excludes
```

Every node and edge carries resolved lowercase SHA-256 bindings. Every scientific claim must trace through Task 13 to Task 12 run/score evidence and immutable Core task identity. Engineering/admission evidence cannot be used as accuracy evidence.

## 5. Frozen authoritative anchors

Task 14 fixes at least these existing anchors:

```text
Core root manifest:
  f953283a10dd45d3f9d1de066570a9c09b9d132ed458f8dea3c948641b89e99d
Core candidate root digest:
  71a6beb3ac8a28dabc753c969e96a47a59f92031d217bebf0fa63d6061012af1
Core human-audit attestation:
  45461659ab3f65a0a559897e50340a470f27cdecf55b999a1431988567cf00c2
Task 9 built-in/control revision:
  9118d491fb3f13a2b4278f131fd2520f9c4fe809
Task 9 model provenance:
  8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e
Task 10 Mem0 report:
  2a00a350c750fc02f727af188a8f3d63f68df474e55a53a0710b5b62c6b43fae
Task 10 admission decision:
  c4355fdd1149325306eecf3242eeaf4e3e47a0d9ee616b0f9777058529e04f1c
Task 11 qualification:
  00699e0d7a027d9bb63dca52753d53fe06bcdd0f7c87535aff6f25a7cb496672
Task 11 Mistral provenance:
  0fc48730152bafa005e3f18b12861bec295db02d9ff221ff7b0871cb9bf409da
Task 12 matrix manifest:
  85145a8a460ee6cec3785926f9aaa85c8bee8cd41d4ad0582d2b0333b8cf10d2
Task 12 matrix summary:
  a1c4f89af2b9f39de9791ce9c6348c24b4c81474abf3da865f22e5dfe68f1f15
Task 12 integrity audit:
  bfc85922c36dcc87deca983ce39ff395b10da00c2ee91c8aba7a6c02c3f04f60
Task 13 artifact index:
  da02787276dd171cce716258ec071947ae99fb047a607df983f52125a20937aa
Task 13 independent audit:
  c60c49d917c582506e262534a6c48bb68668027e428ba0c06557ae8381982145
```

Full file paths are explicit CLI inputs and must resolve to regular single-link files under the declared roots.

## 6. Check matrix

Required checks include:

- immutable Core candidate/current-root receipt verification;
- human-audit historical attestation verification;
- exact Core task counts, schemas, and release anchors;
- Task 9 implementation/provenance and engineering-only scope;
- Task 10 exact report/decision hashes, 14/14 PASS, ADMITTED, fallback false, 128 `NOT_SUPPORTED`, zero failed/partial, and non-accuracy scope;
- Task 11 exact qualification/provenance hashes and frozen model slots;
- Task 12 exact manifest/summary/audit hashes, 18 runs, 1,440 task rows, 1,440 score rows, zero failed/partial, and real-vs-fake exclusion;
- Task 13 exact five-root/eight-artifact closure, 126 cells, 84 contrasts, 210 claims, 57 cases, 18 runs, 1,440 rejoined observations, frozen bootstrap hash, typed unsupported/null metrics, local final root, and remote NFS staging exclusion;
- trusted clean source revision;
- source snapshots unchanged before and immediately before publication;
- no input/output alias or overlap;
- exact five-file publication and reopen verification.

## 7. NOT_APPROVED conditions

Any of the following derives `NOT_APPROVED`:

- missing, extra, noncanonical, linked, reparse-point, or mismatched artifact;
- mismatched current root, receipt, attestation, revision, or tree hash;
- incomplete or non-pass prior gate;
- Task 12 row/run incompleteness or failed/partial result;
- Task 13 index/receipt/ledger/case/statistic splice;
- unsupported/null converted to numeric or omitted;
- fake-offline, `slot_direct`, deterministic Pilot, Mem0 admission, or API probe represented as external scientific accuracy;
- remote NFS staging represented as published final;
- caller-controlled approval or override;
- model/provider/token/API/fake execution path;
- source mutation during review/publication;
- existing output root, clobber request, unsafe filesystem object, or publication/reopen failure;
- any scientific claim without a complete evidence path.

## 8. Source snapshots and publication

All required local roots receive deterministic recursive snapshots over sorted `(relative_path, byte_count, sha256)` tuples plus filesystem identity. Remote evidence refs preserve paths and hashes but do not satisfy current-local-root checks.

Capture snapshots before validation, after validation, in `pre_publish`, and after final-root reopen where applicable. A mismatch yields stale-source failure and no approval.

Use `publish_files_atomically` for the five output files with overwrite disabled, validators for every staged file, and a `pre_publish` callback that rechecks all current roots. The output root must be an absent local directory outside every input root. No `--force`, copy/delete, or approval override exists.

## 9. CLI

Production entry point:

```text
scripts/vnext_review_core_task14.py
```

It accepts explicit Core/Task 9–13 evidence paths, expected Task 13 index hash, output root, review ID, trusted source revision, and `--execute`. It uses `allow_abbrev=False` and has no model/provider/token/API/fake/offline/slot-direct/metric-override/allow-failure/force-approve flag.

Exit codes:

```text
0   freshly verified FINAL_APPROVED
10  validated NOT_APPROVED
11  usage or contract error
12  stale/changed source snapshot
13  atomic publication or reopen failure
14  dirty/untrusted runtime
```

## 10. Verification and release statement

Task 14 requires fresh contract/source/graph/approval/atomic/CLI tests, targeted Core candidate/audit/atomic/Task 12/Task 13 regressions, independent spec and code-quality reviews, clean-tree diff checks, a real no-model final review execution, and an independent final-root verifier.

Only after all checks pass may documentation say:

```text
The bounded MemUpdateBench vNext Core release is FINAL_APPROVED.
```

That statement does not convert Mem0 admission into accuracy, broaden Task 12/13 beyond their frozen scope, or complete the separate future main-track external-validity expansion.
