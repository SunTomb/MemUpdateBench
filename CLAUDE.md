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

Central invariant:

```text
exact slot resolution by (entity, attribute)
```

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

Current honest positioning: MemUpdateBench is a promising controlled diagnostic benchmark, but it needs external baselines, larger/more diverse data, related work, and deeper mechanism experiments before being positioned as a strong Findings/main-track paper.

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

The project should now respond directly to the strict simulated reviewer concerns in `docs/critical_review.md`. Treat that review as the current threat model: without external baselines, larger/more diverse data, related work, and deeper mechanism analysis, the project is likely workshop-level rather than a strong Findings/main-track submission.

Default priority order:

1. **External baseline feasibility and evaluation.** First try Mem0 because it is likely the lowest-cost real external memory SDK. Then investigate Memory-R1 if code/checkpoints are actually obtainable and runnable. MemGPT/Letta can be probed if they expose enough state. External baselines must be isolated from the main environment and must expose enough memory state to compute stale same-slot burden, not just final answer EM.
2. **Answer-layer failure analysis.** Deepen the Constrained CRUD k=16 finding where state accuracy is 1.00 but slot-prompt EM is only about 0.70. Run oracle-retrieval, retrieval top-k/context-length sensitivity, prompt variants, and case studies. Treat clean-state answer failure as a central research question, not a footnote.
3. **Mechanism experiments for stale burden.** Add interventions such as raw_add stale-entry deletion or stale filtering to measure how EM recovers as stale same-slot burden is removed. This should distinguish correlation from mechanism and test whether there is a tipping point.
4. **Same-method-family tradeoff curves.** Run heuristic CRUD threshold sweeps, e.g. cosine thresholds 0.70/0.80/0.85/0.90/0.95, so the paper has a genuine parameterized tradeoff curve rather than only scatter points from different methods.
5. **Data scale and diversity expansion.** Add a separate opt-in split with more examples, more attributes, and more paraphrased explicit update templates. Preserve exact `(entity, attribute)` slot semantics. Do not mix implicit updates into the main split until their gold semantics are precisely defined.
6. **Related work and positioning.** Add a serious related-work section covering AMemGym, Ledger-QA/UMA, Memory-R1, Mem0, MemGPT/Letta, LoCoMo/LongMemEval, dialogue state tracking, and knowledge editing. Position MemUpdateBench narrowly as repeated same-slot update diagnostics, not as a broad memory benchmark.
7. **Long25 stability and repair.** Finish current seed stability checks. Consider repair training only if diagnostics point to a specific failure target such as operation selection or NOOP discrimination. Do not frame repair as the main contribution unless it is validated across the four benchmark axes.
8. **k=32 stress.** Add k=32 only after external baseline feasibility and core diagnostics are under control, or when it directly tests extrapolation/saturation behavior requested by reviewers/advisors.

Experimental expansion is allowed and encouraged when it addresses reviewer-risk evidence. For any new split or method, keep exact `(entity, attribute)` slot semantics, run deterministic oracle sanity first when applicable, analyze errors before interpreting learned-manager results, and document commands/metrics/conclusions in `WORKFLOW.md`.

Avoid spending major time on prose-only polishing until the evidence gaps above are reduced. Manuscript edits should follow new results: external validity first, mechanism depth second, narrative polish third.

Commits are allowed for coherent completed units after checking the diff and recording validation status.
