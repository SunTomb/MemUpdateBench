# Family-H Native Qdrant Full-Fixture Evidence

## Scope and boundary

This package evaluates only the existing published Family-H independent release:

- Rust, CPython and Kubernetes software trajectories;
- 3 audited tasks;
- 27 public events;
- direct Qdrant state/retrieval only;
- no model, GPU, provider-model call or prompted answer replay.

The existing release under `data/vnext/family_h_independent/v1_published` is immutable. The run is a new sibling and does not rewrite the existing Family-H state receipt, correction audit or inherited Qwen answer receipt.

## Runtime and execution

The successful sibling used Tang-3 with Qdrant 1.19.0, server revision `74f3e85b9473c62560006c043e13737ce6b48412`, pinned binary SHA-256 `abfe97e1d0225111dec2f048790428f151846c8a049eefc85328b6c9eccaf419`, and the published Family-H release index SHA-256 `c092c94312d6dc8793862ea84549ffc6f0616fc33da37e1c8ca528685db25261`. The source/runtime worker was source-bound; release bytes were copied without replacing `release_manifest.json`.

The worker executed all 27 public events in source order and performed one public current-state retrieval per task. It never read gold actions or answer fields to construct provider requests. It preserved provider-native IDs, payloads, order, ranks and scores in the observation rows and persisted direct provider witness counts. Each task used an isolated runtime namespace and collection, with cleanup after observation.

## Observed evidence

The run completed with:

- 3/3 task rows observed;
- 27/27 mutation/state steps;
- 27/27 source-event links;
- 3/3 final-state matches;
- 3/3 typed retrieval matches;
- stale same-slot count 0;
- per-task `client_closed=true` and `collection_deleted=true`;
- final exports empty after reset;
- 0 model loads, generations, GPU calls and provider-model calls.

Provider witness counts were:

```text
collection_exists 3
create_collection 3
get_collection 3
upsert 27
retrieve 27
query_points 3
scroll 39
delete 6
```

The independent attestor recomputed these counts and metrics from the public event projection, stored mutation results, exports, retrieval traces and provider witnesses. It did not trust aggregate result fields alone.

## Attested package

The compact no-replace package is:

```text
results/vnext/family_h_native_qdrant_full_20261012_attestation_v1
```

Attestation index SHA-256:

```text
670d5cb77c044d3b158f60f00177eec9d269baf75fb8fa30f4374be2ebf1ff54
```

Attestation SHA-256:

```text
15c8e210b2cebc76e949edeaa2b5220099e629eaeff4c0e2c364320b76ba75da
```

Status and claim boundary:

```text
FAMILY_H_NATIVE_STATE_RETRIEVAL_ATTESTED
native_admission = NOT_GRANTED
scientific_release_allowed = false
answer_metrics = null
```

This is real direct-Qdrant state/retrieval evidence over three published software trajectories. It is not prompted-answer accuracy, semantic embedding quality, a universal native-manager benchmark, statistical independence beyond the three trajectories, or a final scientific release. The inherited Family-H answer receipt remains a separate evidence class and is not merged into this package.

## Technical history

Several earlier no-replace siblings were retained as deployment diagnostics: v1/v2 had packaging/claim or archive admission errors; v3/v4 had deployment-root or orchestration mismatches; later runs exposed and corrected temp-layout, YAML-port, preparation-copy, canonical FrozenDict serialization, and cleanup-evidence issues. None made provider/model calls or was interpreted as accuracy zero. Only the successful final sibling is used for this attestation.
