# Native Multi-Object Canary Preparation

## Boundary

This is a synthetic API-contract preparation, not a task release, direct Qdrant result, model capability probe or accuracy experiment. It constructs no provider and performs no model, GPU, network, database, service or remote operation. Existing Core, Pilot, Family H, NOAA/BEA, Qdrant v13 and Qwen result roots remain unchanged.

The existing direct-Qdrant provider and native contracts already supply the protocol boundary. The missing additive unit is a reproducible small canary input packet with reference expectations kept separate from the provider inputs. Restoring archived runners or launching a runtime is intentionally outside this unit.

## Four scenarios

| Scenario | Planned mutation requests per repetition | Constraint tested by the offline reference |
| --- | ---: | --- |
| `interleaved_updates` | 7 | Interleaved objects, repeated same-slot updates, equal-value update linkage, NOOP, delete without collateral |
| `four_part_identity` | 5 | Namespace/subkey isolation; object type excluded from identity; integer versus float; explicit JSON null |
| `no_effect_batches` | 6 | Duplicate ADD, absent UPDATE/DELETE, mixed no-effect/executed batch outcomes |
| `delete_readd` | 7 | Delete then re-add, absent-update no-effect, later multi-object update and protected collateral |

Two future repetitions per scenario yield **eight planned case instances, 50 planned mutation requests and eight planned retrieval requests**. These are preparation counts, not executed calls, independent samples or a determinism result. Real-source trajectory count is zero.

## Provider input versus reference

`requests.jsonl` contains only case coordinates, distinct runtime-namespace labels, canonical `NativeMultiObjectMutationRequestV3` requests and `NativeRetrievalRequestV3` queries. Input provenance is `declared` with `provider_owned=false`; no provider-owned observations or IDs are fabricated. The retrieval query contains only public text and a namespace label, with empty filters/options and a fixed k=8 ceiling. No hidden target, answer, expected-state or gold metadata enters that query.

`expectations.jsonl` is separate evaluator-only material. Its pure reference projection uses the four-part object identity and records expected per-object effect/reason, last successful source-event linkage and state after each operation. This dictionary is an offline oracle, never an authoritative store in a Qdrant adapter. It is not the full benchmark reference runner, an extractor, provider trace, history API or scientific scoring result.

The reference assumes the current provider contract: duplicate ADD, absent UPDATE and absent DELETE are explicit no-effects; equal-value UPDATE is still a write with the new event link. Uniform batches have unknown atomicity, mixed-effect batches are partial, and NOOP is not applicable. No transactional multi-object capability is claimed.

Final-state comparisons in a future run must be order-independent and typed. Actual retrieval order, ranks, scores, point IDs and payloads must remain provider observations; this packet does not prescribe or fabricate them. A query that returns all live records in a tiny synthetic collection cannot establish semantic-embedding quality.

## Publication and reopening

The offline CLI publishes three payloads plus an index into a new no-replace directory:

- `manifest.json`: source hashes, zero execution counters and explicit not-executed gates;
- `requests.jsonl`: provider-visible synthetic API inputs;
- `expectations.jsonl`: separate offline reference projections;
- `index.json`: exact payload byte counts and SHA-256 hashes.

Publication rejects parent traversal and protected input roots, exclusively claims a new output directory, and reuses the transactional publisher with source and directory-identity revalidation. Successful publication is reopened before returning. Only an empty directory still owned by that invocation may be removed on failure. Selected Python source hashes use LF-normalized bytes, explicitly labeled in the manifest; no frozen data bytes are normalized. A selected-source snapshot is captured at module import, and post-import source drift requires restarting from unchanged code. This is not complete interpreter/dependency closure. Reopening requires an external expected index SHA-256 and rederives both requests and reference expectations from the current source. Rehashed-but-altered oracle expectations are rejected. Reopening is a source-bound reconstruction, not archival compatibility with a different generator revision, a signature or independent runtime attestation.

```bash
python -B scripts/vnext_prepare_native_multi_object_canary.py \
  --output-root results/vnext/native_multi_object_canary_preparation_20261010_v1

# Use the index hash actually returned by the first command.
python -B scripts/vnext_prepare_native_multi_object_canary.py \
  --output-root results/vnext/native_multi_object_canary_preparation_20261010_v1 \
  --verify-index-sha256 <recorded-index-sha256>
```

No flag in this CLI authorizes execution or promotes a Qdrant gate to READY. Its status is `PREPARED_NOT_EXECUTED`, `execution_authorized=false`, `native_admission=NOT_REQUESTED`, `scientific_release_allowed=false`, with null answer metrics.

## Remaining runtime work

A separately authorized run still needs pinned server/client/source/configuration identity, owned collection/storage verification, fresh namespace isolation, reset before and after each case, preserved foreign namespaces, direct request/response witnesses, actual entry exports, retrieval order/score observations, independent trace checks, a fresh second repetition and cleanup. The namespace labels here do not reserve a collection or prove reset isolation. Only validated real observations can qualify a native provider; they must live in new sibling roots, not inside this preparation.

## Recorded preparation

The finalized packet is `results/vnext/native_multi_object_canary_preparation_20261010_v1`. Its index SHA-256 is:

```text
cb50151dceba0256f1962f00e51bfbd86c0f8383fa895d434e0f73f56ba578cd
```

Pinned reopening passed. The relevant native/preparation regression gate passed 128 tests, including 13 new tests. A bounded automated review and reproduced failure tests covered protected-path traversal, output-directory competition, post-import source edits and checkout line endings before publication. These are engineering checks, not a human audit or a native runtime result.

An isolated export of staged tree `42e594685b27bb9dc85d80fd1ad19bd57fdcbc5b` reopened the packet under that same index pin and passed all 13 new tests. `PYTHONPATH` was unset; test dependencies were made available explicitly without processing user-site path hooks. Every imported `mub`/`scripts` module was inside the export. This also exercised LF source reconstruction rather than depending on the development checkout's CRLF copy of one helper. No real provider or model was executed.

## Real Tang-3 observation

With explicit user authorization, one CPU-only direct-Qdrant run was executed on Tang-3 using Qdrant 1.19.0, server revision `74f3e85b9473c62560006c043e13737ce6b48412`, the pinned binary/environment hashes and loopback port 16333. Earlier v1–v8 roots stopped before provider calls because of deployment/readiness/orchestration defects and are retained as technical diagnostics; they are not accuracy zeros.

The corrected v10 run is retained in the compact evidence package:

```text
results/vnext/native_multi_object_canary_tang3_evidence_20261010_v3
```

Its package index SHA-256 is:

```text
a1ccf19e20c53c89308dd39afab4eb2dec45f7c57c26409860b60061448809c6
```

The direct provider observation completed:

- 8/8 synthetic case instances observed;
- 50/50 planned mutation operations completed;
- 8/8 retrieval calls completed;
- 48 reset attempts (six per case, including guard/subject isolation and final cleanup resets);
- 374 direct provider observation witnesses, including 8 `query_points`, 154 `scroll`, 74 `retrieve`, 54 `upsert`, 60 `delete`, 8 collection creation, 8 schema reads and 8 existence checks;
- all case cleanup values `client_closed=true` and `collection_deleted=true`;
- owned server, worker and temporary-root cleanup all passed;
- no model loads, generations, GPU calls, paid-provider calls or answer metrics.

The same-process producer audit and a separate local `audit_cases` replay both returned `AUDITED`. The audit checks the frozen synthetic expectations, typed values, four-part identities, per-object no-effect/executed outcomes, final provider exports, guard isolation, retrieval IDs/payloads/rank/order/score and repeated semantic projections. Raw query-point observations are retained separately from normalized retrieval traces. This is direct Qdrant CRUD/hash-vector retrieval evidence over a synthetic API canary, not semantic embedding quality, prompted answering, a general native-manager release, or a scientific accuracy claim. `native_admission` remains `NOT_GRANTED` and `scientific_release_allowed=false` until the broader qualification/release gates are separately closed.

## Bounded canary attestation

The completed v10 evidence was independently reopened by a no-execution attestor:

```text
results/vnext/native_multi_object_canary_attestation_20261011_v1
```

Attestation index SHA-256:

```text
f39eab5311620ae52d156a45333d085417fdcede98e6cbe987dba274ac2ab7cf
```

The attestation binds the preparation index, evidence index, profile/runtime identity, 8 cases, 50 mutation attempts/completions, 8 retrievals, 48 resets, 374 provider witnesses, per-case client/collection cleanup, and zero model/GPU/provider-model execution. It explicitly reports `BOUNDED_CANARY_ATTESTED`, `native_admission=NOT_GRANTED`, `scientific_release_allowed=false`, and `independent_signer=false`.

This is a hash-bound same-assistant attestation of an observed synthetic canary, not a cryptographic signature, independent human audit, native benchmark release or broad scientific qualification. A later release would still need a separate qualification receipt, broader real-source/full-fixture scope and release-level review.

## Attestation hardening

The attestation was hardened to revalidate the compact evidence index itself: exact artifact membership, byte counts and SHA-256 values are checked before summary/case counts are trusted. Owned server/worker/temp cleanup flags are also mandatory. A tampered observations artifact is rejected before attestation. The hardened no-replace sibling is:

```text
results/vnext/native_multi_object_canary_attestation_20261011_v2
```

It has the same deterministic attestation payload SHA-256 `4ff350db49c519fafdcd7b83f3f9784836bafadb189e75dd25db9fa20629fa92`; the sibling index SHA-256 is recorded by its `index.json`. This remains a same-assistant bounded attestation with `independent_signer=false`, not a scientific release.
