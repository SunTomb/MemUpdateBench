#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/MemUpdateBench
source activate.sh

for variant in v0_current v1_value_only v2_ignore_distractors; do
  for k in 1 2 4 8 16; do
    PYTHONPATH=/NAS/yesh/MemUpdateBench python scripts/eval_evomemory.py \
      --mode constrained_slot_crud \
      --answer_mode slot_prompt \
      --slot_prompt_variant "${variant}" \
      --save_answer_traces \
      --no_qlora \
      --data_file "data/evomemory_update_frequency_hard_k${k}_p63_dev.json" \
      --output_dir "results/p65_prompt_robustness/constrained_${variant}_k${k}"

    PYTHONPATH=/NAS/yesh/MemUpdateBench python scripts/eval_evomemory.py \
      --mode raw_add \
      --answer_mode slot_prompt \
      --slot_prompt_variant "${variant}" \
      --save_answer_traces \
      --no_qlora \
      --data_file "data/evomemory_update_frequency_hard_k${k}_p63_dev.json" \
      --output_dir "results/p65_prompt_robustness/raw_add_${variant}_k${k}"
  done
done
