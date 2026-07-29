# MemUpdateBench vNext Synthetic Pilot

## Current engineering scope

This directory is reserved for the **Families A–D synthetic Pilot**. Its fixed design contains **480 semantic cores** and **1,440 rendered tasks**, with **3 variants per core**. The rendered-task split is **1,008 train / 144 dev / 288 test**.

| Family | Diagnostic focus | Semantic cores | Rendered tasks |
| --- | --- | ---: | ---: |
| A | Repeated same-slot update | 120 | 360 |
| B | Interleaved multi-slot update | 120 | 360 |
| C | Entity/attribute grounding | 120 | 360 |
| D | NOOP/write discipline | 120 | 360 |

The exact memory-object identity is:

```text
(namespace, entity, attribute, subkey)
```

`object_type` is classification metadata and is excluded from identity.

## Deterministic generation

Run a smoke generation into a temporary output directory:

```bash
python scripts/vnext_generate_pilot.py \
  --config configs/vnext/pilot.yaml \
  --output-dir .tmp/vnext-pilot-smoke
```

The generator emits exactly these five release artifacts:

- `tasks.jsonl`
- `generation_config.json`
- `split_balance.json`
- `task_manifest.json`
- `validation_report.json`

Splitting is deterministic and group-first: all rendered variants of a semantic core remain in one split, preventing cross-split core leakage. JSON and JSONL use canonical serialization, while SHA-256 references and the task manifest bind the generated artifacts to their recorded contents and provenance.

Publication is transactional: artifacts are staged, re-read, validated, and promoted only as a complete bundle. Existing output is not overwritten by default. Use `--overwrite` only as a deliberate, reviewed replacement of a chosen generated-output directory; never rely on silent overwrite of `data/vnext/pilot`.

## Validation and release boundary

`validation_report.json` records automated structural validation, gold replay, and split/manifest validation only. It does **not** establish completion of human audit, release readiness, model or baseline results, external validity, or any paper claim.

Task 7 requires no APIs, models, or external systems. Task 8 semantic validation and human audit, plus Tasks 9–15, remain required before the end-to-end Pilot release gate can be satisfied.
