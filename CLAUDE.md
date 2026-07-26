# CLAUDE.md

This file gives Claude Code the project context and operating rules needed to continue MemUpdateBench work from a new window.

## Project Overview

MemUpdateBench is the clean project extracted from G-MSRA P6.x. It is no longer a broad RL memory-manager project. The paper direction should now be a controlled diagnostic study of repeated same-slot updates in external memory systems, not a broad general-purpose memory benchmark.

A strict simulated reviewer review in `docs/critical_review.md` changed the project priorities. The current work is useful infrastructure and a plausible workshop-level diagnostic, but it is not yet strong enough for an ACL/EMNLP/NeurIPS/ICLR main-track benchmark paper. Future work should prioritize evidence that addresses external validity, diagnostic depth, related work positioning, and data diversity rather than continuing prose-only packaging.

Core question:

```text
What happens when the same (entity, attribute) memory slot is updated repeatedly?
```

The benchmark evaluates tradeoffs among:

- final-state reliability,
- stale same-slot burden,
- memory compactness,
- answer robustness under realistic slot-conditioned prompting.

The key action/state format remains:

```text
ADD <entity>.<attribute> = <value>
UPDATE <entity>.<attribute> = <value>
NOOP
```

Central invariant for the existing P6/P8 experimental line:

```text
exact legacy slot resolution by (entity, attribute)
```

The vNext canonical identity extends this to the four-part object key defined below.

Most analysis should distinguish:

- stale value retention,
- missed final updates,
- wrong entity grounding,
- wrong attribute parsing,
- answer-layer retrieval/prompt failures.

## Current Thesis and Reviewer-Risk Position

Repeated same-slot update frequency remains the main benchmark axis, but the paper claim must be narrower and sharper than "append more causes stale entries, compact more can lose updates." That framing is too close to a truism unless backed by external systems, method-family sensitivity curves, and deeper failure-mechanism analysis.

The strongest P6.3 result:

- append-only methods can keep final value recoverable under oracle `slot_direct`,
- but they accumulate stale same-slot entries,
- under `slot_prompt`, stale burden causes severe answer collapse,
- learned compact managers reduce stale burden but can miss final updates or remain incompletely compact,
- even perfect clean state does not guarantee prompted answer correctness, which exposes a distinct retrieval/answer-layer failure mode.

At k=16 on P6.3 hard:

| method | slot_direct state_acc | slot_prompt EM/F1 | stale same-slot | final memory size |
| --- | --- | --- | --- | --- |
| constrained_slot_crud | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 |
| raw_add | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 |
| heuristic_crud | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 |
| long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 |

P6.5 prompt-robustness diagnostics show that mild prompt variants do not remove the high-k ordering: Constrained CRUD remains around 0.68-0.69 EM at k=16, Raw append remains around 0.09-0.11 EM, and Long25 remains between them. Answer traces show different mechanisms: Raw append often fails because gold values are not retrieved under stale burden; Constrained CRUD still has gold-not-retrieved and gold-retrieved-wrong-answer cases despite zero stale same-slot burden; Long25 mixes state errors with stale/distractor answer-context failures.

Current honest positioning: MemUpdateBench is a promising controlled diagnostic benchmark. The project has now strengthened the mechanism story with P8.x evidence, including latest API answer-model probes, but manuscript work should still avoid broad benchmark overclaiming and should foreground definitions, method meanings, and the order/metadata-sensitive version-arbitration mechanism.

Recent mechanism lock:

- P8.3 shows the claim must be nuanced: stale same-slot conflict is an **order- and metadata-sensitive version arbitration failure**, not a universally strongest distractor. The conflict-type decomposition found `unrelated_distractors` harder than `stale_same_slot` in one surface construction, so do not claim stale same-slot is always the strongest distractor.
- P8.3 synthetic dose-response is the strongest mechanism evidence: reverse/no-label stale=8 EM 0.000 and stale=16 EM 0.031, while reverse+latest/outdated label reaches EM 1.000 at stale=8 and stale=16. Gold is in context; missing current-version signal causes stale copying.
- P8.4 latest API answer-model probes address model-recency concerns. On the clean, format-stable subset (`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gemini-3.1-flash-lite-preview`), chronological/no-label stale=16 has EM 1.000, reverse/no-label stale=1/16 has EM 0.000 with stale copied 1.000, and reverse+latest/outdated stale=16 recovers to EM 1.000. Treat other Gemini rows with empty/truncated outputs as API/prompt-format caveats, not central mechanism evidence.

## vNext Status

MemUpdateBench vNext Phase 0 is `FINAL_APPROVED`. It establishes the reusable contract, validation, scoring, provenance, legacy-compatibility, and transactional-publication foundation; it does not add Pilot examples, model results, external-validity evidence, benchmark metrics, or paper claims.

Canonical vNext memory-object identity is exactly:

```text
(namespace, entity, attribute, subkey)
```

`object_type` is classification metadata and is excluded from identity, semantic hashes, replay keys, and exact object resolution. Imported P6.x `(entity, attribute)` slots use `namespace="default"` and `subkey=null`; legacy phase and metric names remain provenance rather than canonical namespaces.

Authoritative vNext references:

```text
docs/specs/memupdatebench_vnext_benchmark_design.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-phase0-contract-legacy-bridge.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-pilot.md
docs/vnext/legacy_bridge.md
WORKFLOW.md
```

The approved next engineering milestone is the 1,440-task Families A–D Pilot, which must reuse the Phase 0 contracts rather than introduce parallel task, runtime, score, capability, or manifest dictionaries. Pilot status remains `NOT_STARTED`; beginning it requires a separate explicit instruction. Files under `tests/vnext/fixtures/legacy/` are immutable authenticated regression inputs and must not be edited during Pilot work.

## Important Local Files

Core package:

```text
mub/config.py
mub/utils.py
mub/memory/entry.py
mub/memory/store.py
mub/manager/memory_manager.py
```

Core scripts:

```text
scripts/prepare_data.py
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
scripts/analyze_action_pathology.py
scripts/summarize_update_frequency.py
scripts/summarize_prompt_robustness.py
scripts/smoke_test.py
scripts/generate_constrained_sft.py
scripts/train_constrained_sft.py
scripts/probe_api_answer_model.py
scripts/summarize_api_latest_model_probe.py
scripts/run_p84_api_latest_models_tang2.sh
```

Main data:

```text
data/evomemory_update_frequency_hard_k1_p63_dev.json
data/evomemory_update_frequency_hard_k1_p63_test.json
data/evomemory_update_frequency_hard_k2_p63_dev.json
data/evomemory_update_frequency_hard_k2_p63_test.json
data/evomemory_update_frequency_hard_k4_p63_dev.json
data/evomemory_update_frequency_hard_k4_p63_test.json
data/evomemory_update_frequency_hard_k8_p63_dev.json
data/evomemory_update_frequency_hard_k8_p63_test.json
data/evomemory_update_frequency_hard_k16_p63_dev.json
data/evomemory_update_frequency_hard_k16_p63_test.json
data/evomemory_update_frequency_hard_p63_train.json
data/evomemory_update_frequency_hard_p63_dev.json
data/evomemory_update_frequency_hard_p63_test.json
```

Main results:

```text
results/update_frequency_p63/
results/update_frequency_p63_summary/
results/p65_prompt_robustness/
results/p65_prompt_robustness_summary/
results/p65_diagnostics/
results/p65_stability/
results/p83_conflict_type_probe_summary/
results/p83_stale_conflict_dose_summary/
results/p83_stale_specific_removal_trace/
results/p84_api_latest_model_probe_summary/
```

Historical context:

```text
docs/HISTORY.md
docs/PROJECT_WORKFLOW6.0.0.md
docs/PROJECT_WORKFLOW6.0.1.md
docs/PROJECT_WORKFLOW6.1.0.md
docs/PROJECT_WORKFLOW6.2.0.md
docs/PROJECT_WORKFLOW6.3.0.md
docs/PROJECT_WORKFLOW6.4.0.md
docs/MIGRATION_PLAN.md
```

## Main Commands

Compile and smoke test:

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/analyze_action_pathology.py scripts/summarize_update_frequency.py scripts/summarize_prompt_robustness.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py
python scripts/smoke_test.py
```

Rebuild P6.3 summary artifacts:

```bash
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_p63 \
  --output_dir results/update_frequency_p63_summary
```

Run deterministic oracle on a hard split:

```bash
PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode constrained_slot_crud \
  --answer_mode slot_direct \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/sanity_oracle_k16
```

Run learned long25 if checkpoint is available:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode learned_constrained_slot \
  --answer_mode slot_prompt \
  --no_qlora \
  --lora_checkpoint checkpoints/long25/best \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/long25_slot_prompt_k16_rerun
```

## Cluster Usage

Primary cluster target:

```text
Tang-2-Wu
```

New remote project path:

```text
/NAS/yesh/MemUpdateBench
```

Activate environment:

```bash
source /NAS/yesh/MemUpdateBench/activate.sh
```

Continue using the existing `gmsra` conda environment; only `PYTHONPATH` changes.

Remote `activate.sh` should contain:

```bash
cd /NAS/yesh/MemUpdateBench
eval "$(/NAS/yesh/miniconda3/bin/conda shell.bash hook)"
conda activate gmsra
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
export PYTHONPATH=/NAS/yesh/MemUpdateBench
echo "MemUpdateBench environment ready ✅"
```

Only modify files inside:

```text
/NAS/yesh/MemUpdateBench
```

Use `tmux` for long-running cluster jobs.

Typical pattern:

```bash
tmux new-session -d -s <session_name> \
  "cd /NAS/yesh/MemUpdateBench && source activate.sh && CUDA_VISIBLE_DEVICES=<gpu> python <script.py> <args> > <log_file> 2>&1"
```

Use `--no_qlora` for learned constrained evals if bitsandbytes 4-bit loading fails.

## Checkpoint Status

The learned long25 baseline should live at:

```text
checkpoints/long25/best
```

Migration note: before deleting old remote `/NAS/yesh/G-MSRA`, move the checkpoint from:

```text
/NAS/yesh/G-MSRA/outputs/constrained_sft_curriculum_long25
```

to:

```text
/NAS/yesh/MemUpdateBench/checkpoints/long25
```

Do not delete `/NAS/yesh/G-MSRA` unless the user explicitly asks and checkpoint migration has been verified.

## Coding Rules

Prefer minimal, targeted changes. Do not restore old Phase 1-5 G-MSRA components unless explicitly needed.

Do not reintroduce these old modules into the mainline:

```text
agent.py
reward/
consolidation/
train_phase*.py
eval_locomo.py
run_ablations.py
run_baselines.py
```

For parser/data-generator changes, run at least:

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/analyze_ood_errors.py scripts/summarize_update_frequency.py scripts/smoke_test.py
python scripts/smoke_test.py
```

For any new split:

1. generate/verify data,
2. run deterministic oracle first,
3. analyze errors if oracle is imperfect,
4. only then evaluate learned managers.

Avoid adding comments unless they explain a non-obvious invariant.

Commits are allowed when they are useful for preserving a coherent completed unit of work. Before committing, make sure the staged diff is intentional, validation has been run or the reason for skipping validation is recorded, and the commit message accurately describes the change.

## Workflow Documentation

This new project uses a single `WORKFLOW.md` plus historical docs under `docs/`.

When completing a substantial phase, append to `WORKFLOW.md` rather than creating many new numbered project files, unless the user asks for versioned workflow files.

Workflow entries should include:

- motivation,
- commands run,
- files changed/generated,
- metrics,
- error analysis,
- conclusions,
- next steps.

## Recommended Next Work

After Phase 0 `FINAL_APPROVED`, the primary engineering milestone is the approved vNext Pilot. It must consume the Phase 0 contracts and preserve the exact four-part object identity; do not start it without a separate explicit instruction. Paper production and narrative clarification remain a bounded parallel track, and broad new experiments should still be added only to resolve named reviewer or benchmark-engineering gaps. The next group-meeting material should follow the paper logic rather than mechanically listing experiment tables.

Default priority order:

1. **Rewrite the manuscript story first.** Define `slot_direct`, `slot_prompt`, `raw_add`, `latest_per_slot`, context order, version labels, EM/F1, stale copied, and stale same-slot burden before presenting numbers. The paper should read as a mechanism argument, not a results dump.
2. **Frame the central claim narrowly.** State that stale same-slot conflict is an order- and metadata-sensitive version-arbitration failure. Do not claim stale same-slot is always the strongest distractor; use the P8.3 conflict-type decomposition as a boundary result.
3. **Use P8.4 latest-model evidence carefully.** The stable latest-model subset supports the mechanism across current GPT-family models and one Gemini flash-lite model. Treat empty/truncated Gemini outputs as API/prompt-format caveats.
4. **Integrate figures/tables.** Prioritize dose-response, latest/outdated repair, stale-specific removal, latest-model replication, and a clear benchmark overview figure. Every table/figure must explain what each method/condition name means.
5. **Numerical consistency audit.** Cross-check manuscript numbers against `paper/p80_canonical_main_number_ledger.md`, `paper/p83_stale_same_slot_conflict_plan_note.md`, `paper/p84_latest_api_model_probe_note.md`, and generated summaries before editing final prose.
6. **Update presentation only after paper narrative stabilizes.** The next PPT should mirror the paper: problem definition → benchmark design → failure decomposition → main evidence → mechanism → latest-model robustness → limitations.
7. **Only add new experiments if they resolve a named narrative gap.** Candidate gaps include harder real-context API probes for Gemini, or a more natural prompt for API models, but these should not distract from manuscript integration.

Experimental expansion is allowed and encouraged when it addresses reviewer-risk evidence. For any new split or method, keep exact `(entity, attribute)` slot semantics, run deterministic oracle sanity first when applicable, analyze errors before interpreting learned-manager results, and document commands/metrics/conclusions in `WORKFLOW.md`.

Avoid spending major time on prose-only polishing until the evidence gaps above are reduced. Manuscript edits should follow new results: external validity first, mechanism depth second, narrative polish third.

Commits are allowed for coherent completed units after checking the diff and recording validation status.
