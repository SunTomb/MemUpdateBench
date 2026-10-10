# Cross-Domain Qwen Readout Validation

## Purpose

`scripts/vnext_validate_cross_domain_qwen_readout.py` is a reusable offline reopen validator for the redacted cross-domain Qwen readout artifacts. It does not load a model, access a GPU, call a provider, contact Qdrant, use SSH or alter an existing result root.

## Validated roots

The validator accepts the exact fixture roots for the earlier blocked diagnostics and the final Tang-1 real-load technical failure:

```text
tests/vnext/fixtures/cross_domain_qwen_blocked/v2
tests/vnext/fixtures/cross_domain_qwen_blocked/v3
tests/vnext/fixtures/cross_domain_qwen_blocked/v4
tests/vnext/fixtures/cross_domain_qwen_blocked/tang1gpu1d
```

For every root it checks:

- artifact membership, safe single-component paths, byte counts and SHA-256 values from `index.json`;
- `BLOCKED`, `scientific_release_allowed=false`, null `answer_metrics` and no accuracy claim;
- pinned Qwen identity, frozen preparation hash and one-load/four-generation/no-retry authorization limits;
- cleanup booleans and zero paid-provider/Qdrant/retry accounting;
- four-request order when a worker result exists;
- generation intent, callback-entry and observed-attempt accounting;
- technical/not-run rows contain no prediction or generated token output;
- qualification profile binding and raw prompt/output marker absence.

The Tang-1 report preserves the real boundary: one model load, one generation attempt, first row `TECHNICAL_FAILURE`, three rows `NOT_RUN`, and null answer metrics. It does not relabel the technical failure as an incorrect answer.

## Diagnostic output

A no-replace validation bundle was generated at:

```text
results/vnext/cross_domain_qwen_readout_validation_20261010_v1
```

It contains only a compact `validation.json` report and `index.json`; it is not a scientific release. The bundle status is `BLOCKED` and `scientific_release_allowed=false`.

Validation artifact SHA-256:

```text
1307e334a3a087d15805f8b9316363f65d2bb80670ed6896455833e97c2c7372
```

## Usage

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -B scripts/vnext_validate_cross_domain_qwen_readout.py \
  tests/vnext/fixtures/cross_domain_qwen_blocked/v2 \
  tests/vnext/fixtures/cross_domain_qwen_blocked/v3 \
  tests/vnext/fixtures/cross_domain_qwen_blocked/v4 \
  tests/vnext/fixtures/cross_domain_qwen_blocked/tang1gpu1d \
  --output-root results/vnext/cross_domain_qwen_readout_validation_20261010_v1
```

The output root uses exclusive creation. Reusing an existing output root fails instead of replacing it. This is artifact validation and provenance accounting, not an independent human audit, cryptographic signature, model qualification or benchmark accuracy result.
