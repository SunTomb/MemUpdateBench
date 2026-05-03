# P7.0 Long25 Reproducibility Note

## Motivation

`docs/critical_review_v2.md` correctly flags an apparent contradiction in the Long25 k=16 numbers. The older P6.3 canonical table reports Long25 k=16 slot-prompt around EM/F1 0.48/0.53 with final memory size 9.43, while the later P6.5 seed-stability note reports EM/F1 around 0.88/0.908 with final memory size 2.04.

This note records the provenance audit so the paper does not mix incompatible Long25 result families.

## Provenance files

```text
results/p70_long25_reproducibility/long25_provenance.json
results/p70_long25_reproducibility/long25_provenance.csv
```

The provenance table records result path, data file, answer mode, checkpoint path, prompt variant, top-k setting, EM/F1, state accuracy, stale same-slot burden, and final memory size for every local Long25 `evomemory_results.json` file.

## What caused the mismatch

The two Long25 lines are not two reproductions of the same checkpoint. They are different checkpoint families evaluated on the same P6.3 hard k=16 test split.

| Result family | Result path | Checkpoint | Data file | Answer mode | EM | F1 | State acc. | Stale same-slot | Memory size |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| P6.3 original Long25 | `results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json` | `outputs/constrained_sft_curriculum_long25/best` | `data/evomemory_update_frequency_hard_k16_p63_test.json` | slot_prompt | 0.48 | 0.53 | 0.91 | 1.13 | 9.43 |
| P6.5 seed 11 | `results/p65_stability_sharded/seed11_merged/evomemory_results.json` | `outputs/p65_long25_seed11/best` | `data/evomemory_update_frequency_hard_k16_p63_test.json` | slot_prompt | 0.87 | 0.903 | 0.94 | 0.10 | 2.06 |
| P6.5 seed 22 | `results/p65_stability_sharded/seed22_merged/evomemory_results.json` | `outputs/p65_long25_seed22/best` | `data/evomemory_update_frequency_hard_k16_p63_test.json` | slot_prompt | 0.88 | 0.913 | 0.99 | 0.07 | 2.04 |
| P6.5 seed 33 | `results/p65_stability_sharded/seed33_merged/evomemory_results.json` | `outputs/p65_long25_seed33/best` | `data/evomemory_update_frequency_hard_k16_p63_test.json` | slot_prompt | 0.89 | 0.910 | 0.97 | 0.05 | 2.02 |

The P6.5 sharded merge is not the obvious source of the contradiction: the merged outputs still point to `data/evomemory_update_frequency_hard_k16_p63_test.json`, contain 100 examples, use `answer_mode=slot_prompt`, and preserve the intended seed-specific checkpoint paths.

The older pilot documents also contain 0.87/0.88-like Long25 numbers, but those are from earlier pilot splits and should not be cited as P6.3 hard results.

## Canonical reporting decision

For paper-facing tables, Long25 must be reported as two explicitly separated rows/families unless a new clean rerun supersedes both:

1. **P6.3 original Long25 checkpoint**: `outputs/constrained_sft_curriculum_long25/best`. This is the result family used by the original P6.3 summary tables and historical workflow documents.
2. **P6.5 reseeded Long25 checkpoints**: `outputs/p65_long25_seed{11,22,33}/best`. These are independently trained/reseeded checkpoints and show that the learned compact-memory point can be much stronger than the original single checkpoint.

Do not describe the P6.5 stability numbers as a reproduction of the exact P6.3 checkpoint. They are a stability check across a new checkpoint family.

## Paper implication

The safest manuscript wording is:

> The original P6.3 Long25 checkpoint produced EM/F1 0.48/0.53 at k=16 under slot-prompt. A later three-seed retraining/stability check on the same P6.3 hard k=16 test split produced substantially stronger compact-memory results, EM 0.87-0.89 and final memory size around 2.04. We therefore treat the original row as a historical single-checkpoint diagnostic and report the reseeded Long25 family separately, with explicit checkpoint provenance.

This resolves the apparent numerical contradiction without pretending the two rows are identical runs.

## Remaining verification if time allows

A final clean rerun can further strengthen the story:

```bash
PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode learned_constrained_slot \
  --answer_mode slot_prompt \
  --slot_prompt_variant v0_current \
  --save_answer_traces \
  --no_qlora \
  --lora_checkpoint outputs/p65_long25_seed22/best \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/p70_long25_reproducibility/seed22_clean_rerun_k16_prompt
```

If cluster time is limited, the existing merged seed outputs are already traceable enough for the provenance correction.
