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
