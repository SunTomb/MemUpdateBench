# Cross-Domain Answer Replay

## Scope and scientific boundary

This is an **offline preparation gate**, not an answer experiment. It advances the next layer after the completed NOAA/BEA Qdrant v13 state/retrieval canary: a separately qualified answer model can later read the same frozen retrieval contexts without rerunning memory mutation or retrieval.

The plan retains all four recorded contexts, in `bea/repeat-0`, `bea/repeat-1`, `noaa/repeat-0`, `noaa/repeat-1` order. There are **two source trajectories/tasks**, not four independent samples. Each current context has one retrieved entry. A later successful readout would establish a bounded readout/format result, not stale-conflict robustness, semantic retrieval quality, native Qdrant answering or broad benchmark validity.

The preparation has:

- `status=PREPARED_NOT_EXECUTED`;
- `execution_authorized=false`;
- `model_runtime_status=NOT_QUALIFIED_FOR_THIS_REPLAY`;
- zero model/tokenizer loads, generations, provider/database calls, network/GPU calls and child-process launches;
- `answer_metrics=null`, `scientific_release_allowed=false`, and `benchmark_accuracy_claimed=false`.

The Qwen3.5-9B entry in the policy is a **prospective identity and decoding intent**. It is not a runtime attestation, model result or device authorization. Earlier CPU-only Qdrant authorization is not reused as GPU/model authorization.

## Authenticated inputs

Input pin | SHA-256
--- | ---
V13 result index | `2b2b9dbfd7703f0a91cc66a9267344837f4a892bfac89e5e2db25a00b836a60a`
V13 collection audit | `5188cbbb7fbb3aacbf3a25994a75273060c9a2ab2857c180f3eacf7be3326eca`
Historical v13 source manifest | `145a19b059aa3373de39e09a2581865e9910816739c530da09c20e3deb9a8c01`
BEA formal release index | `5525357436510b0c4e89efa53cdb6caf551c59234171c71bb97563e4c641790a`
NOAA formal release index | `738f46932d8e2e8c26da0a2820dc67c7b0bad9aa43659f4be266f22f0fc8ee3f`
Canonical answer policy | `30df8bdc52e5f0dd48f7c53cecbb07b46f032c6eae81b2478c0969f9371d681a`

The default evidence input is the byte-preserving historical fixture `tests/vnext/fixtures/cross_domain_v13/`. The original collected directory can also be supplied; exact pins and validation do not change on relocation. The two formal releases are consumed in place without regeneration or metadata edits.

Admission verifies complete artifact membership, exit status, cleanup, runtime/input/launch bindings and HTTP accounting. It recomputes state matching, current source-event linkage, stale count and retrieval matching directly from recorded entries and authenticated task actions/evidence. Stored success booleans are cross-checked, not treated as independent measurements. The two repetitions are compared with the established opaque-entry-ID alpha-renaming policy; canonical object identities and source IDs are retained.

This independent implementation of checks runs in the same assistant session. It is **not another human review, external signature or independent-signer approval**.

## Public prompt projection

The module reuses `render_visible_prompt_v3` without altering it. Only the public query ID/text/number schema and authenticated provider-returned retrieval entries are used. The renderer is passed a query view with no selector or target list.

For each entry:

- content, value, object-key candidate and source-event IDs come from the retrieval record;
- order is the original provider order, not sorted by target, score or gold value;
- `raw_metadata` is restricted to provider `point_id_derivation` and `version_index`;
- unknown provider metadata is rejected;
- no final-state export, task action, gold answer, correctness annotation or hidden query selector is supplied as context.

Scores/ranks remain in the normalized trace, but the canonical renderer does not serialize those trace-level fields. That omission is explicit; this profile does not silently add a new prompt format. Provider entry IDs remain in the rendered entries, so prompts for two repetitions may differ in opaque IDs even when retrieved facts and scores are equivalent.

Gold actions/evidence are used only in validation/scoring. There is no gold-based fallback for missing retrieval. The exact renderer/parser module bytes and selected preparation source files are hash-bound. This is not a complete model, tokenizer or operating-system dependency qualification.

## Number parser and scoring

The existing `memupdatebench.answer-model-parser.v3` expects a JSON object with `disposition` and, for an answered response, a schema-valid numeric `answer`. It rejects prose, code fences, quoted numbers, booleans, duplicate keys, extra envelope fields and nonfinite values. The preparation does not introduce unit stripping, regex answer extraction, rounding or tolerance.

The descriptive primary comparator is `typed_json_equal`, not a manufactured canonical `ScoreRecordV3`. In particular, NOAA's integer `75` and floating-point `75.0` are unequal under this policy. The scientific report of a future run must disclose that strict representation boundary.

- Completed malformed answers, wrong values and abstentions remain incorrect completed readouts.
- `NOT_RUN`, `TECHNICAL_FAILURE` and `UNSUPPORTED` carry null quality fields and are excluded from answer-quality denominators.
- Synthetic parser/scorer test controls are tests only; they do not become answer-model accuracy rows.

## Published preparation

The separate no-replace root is `results/vnext/cross_domain_answer_replay_20261002_v2`.

```text
index SHA-256:
1112a371f07326130634715cfc15fe656aa11a48539f2d16b06f39c400f9e687
status: PREPARED_NOT_EXECUTED
prospective requests: 4
model/tokenizer loads: 0 / 0
generations: 0
answer metrics: null
```

The index and all three payload files were reopened and rederived against the unchanged release/evidence inputs. The combined preparation, frozen-state, artifact, staging, BEA-promotion and Task 12 regression gate passed **152 tests with three Windows symlink-permission skips**. These parser/scorer controls do not represent model accuracy.

The initial v1 preparation remains a local superseded diagnostic. An isolated Git export exposed CRLF working-copy source pins that did not match the repository's LF bytes. The old source/config bytes were preserved privately; v2 binds the committed LF renderer and dependency bytes instead. No historical task, canary or release artifact was normalized or rewritten.

The portable v2 staged tree was then exported to a short private directory with `PYTHONPATH` unset. Package rederivation and import-origin checks passed from that isolated checkout, and its preparation/evidence tests reported **46 passed, two Windows symlink-permission skips**. The original retained canary root and the committed fixture both reconstruct the same v2 preparation index.

A bounded read-only code review identified in-memory input mutation and empty verification-hash mode confusion. Both were reproduced by regression tests and fixed. The final changes have test verification; this is not a second human audit or an independent full-project approval.

## Preparation and verification

```bash
python scripts/vnext_prepare_cross_domain_answer_replay.py \
  --output-root results/vnext/cross_domain_answer_replay_20261002_v2

python scripts/vnext_prepare_cross_domain_answer_replay.py \
  --output-root results/vnext/cross_domain_answer_replay_20261002_v2 \
  --verify-index-sha256 <the-printed-index-sha256>

python -m pytest -q -p no:cacheprovider tests/vnext/test_cross_domain_answer_replay.py
```

A prepared directory contains exactly `manifest.json`, `plans.jsonl`, `validation.json` and `index.json`. It stores source references, checks and prompt/trace hashes, **not complete prompts or raw model outputs**. Reopening reconstructs the exact plan from authenticated inputs; changed source, policy, input bytes or output members cause rejection. Publication uses exclusive directory creation, byte snapshots, pre-publication revalidation and no replacement, and rejects protected/input overlap or reparse paths.

The tool has no live/model-execution mode. Future execution must separately qualify the current model/runtime and device, bind tokenizer chat-template hashes and token budgets, run the four prospective requests with complete lifecycle accounting, and publish readout results under a new root. The frozen manager fixtures and source task releases remain untouched.
