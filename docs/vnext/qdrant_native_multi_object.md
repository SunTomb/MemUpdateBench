# Optional Direct Qdrant Native Multi-Object Foundation

This document describes the original provider-contract preparation boundary. Its local dependency observations are historical, not a current cluster availability verdict. The later bounded NOAA/BEA run is documented separately in [Cross-Domain Qdrant v13 Evidence](cross_domain_qdrant_v13_evidence.md); it does not upgrade every capability described here to benchmark evidence.

This additive foundation exposes a thin direct-Qdrant provider at
`mub.vnext.external.providers.qdrant_native_multi_object`.

## Boundary

The provider uses a run-isolated Qdrant collection and storage path, a fixed
384-dimensional cosine vector schema, and deterministic UUID5 point IDs. The
configuration treats the supplied path as a base directory and derives a
run-specific child directory, so reusing a base path for different run IDs does
not share the local Qdrant store. IDs are adapter-derived rather than Qdrant-
native semantic identifiers; this is recorded in each point's
`version_metadata`. Payloads contain the runtime namespace, canonical four-part
object identity, object type, value/content, source event IDs, sequence/version
fields, and version metadata.

Mutation and retrieval calls reach Qdrant directly through `upsert`,
`retrieve`, `delete`, `query_points`, and `scroll`. The adapter keeps no local
authoritative state, latest-value cache, reranker, deduplicator, filter, sort,
hidden target selector, or native answer path. Retrieval preserves provider
point order, score, ID, and payload while rejecting malformed or identity-
mismatched payloads. Hash-only observation witnesses preserve the raw request
and response hashes for each direct provider call.

Qdrant's per-point operations are not advertised as transactional. Multi-object
mutations therefore report `unknown` atomicity for uniform outcomes and
`partial` when outcomes differ; `NOOP` is `not_applicable`. Scoped deletion
uses provider filters for object, attribute, entity, and namespace scopes.
Reset deletes only the selected runtime namespace and verifies it is empty;
`close()` only closes the provider client.

## Optional dependency and qualification

`qdrant-client` is imported lazily. Importing the module does not require the
package. If the package is absent, provider construction raises the typed
`QdrantNativeMultiObjectUnavailable` error with blocker
`qdrant_client_missing`. The qualification script emits a typed `BLOCKED`
report for a missing package or missing direct API/runtime observation. A
`READY` report requires an explicit direct-observation signal and a nonempty
evidence anchor. The script never starts Qdrant, calls a provider, calls a
model, or claims benchmark accuracy.

The current environment has no `qdrant-client`; unit tests use an in-process
fake client only for contract behavior and do not establish Qdrant runtime,
native admission, or scientific evidence.

Example no-execution check:

```text
python scripts/vnext_qualify_qdrant_native_multi_object.py \
  --run-id local-check \
  --path C:/tmp/qdrant-local-check
```

This returns a nonzero exit code and JSON with `status=BLOCKED`,
`availability=UNAVAILABLE`, and `blocker=qdrant_client_missing` in the current
environment.
