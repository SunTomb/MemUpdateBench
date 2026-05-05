#!/usr/bin/env bash
set -euo pipefail
cd /NAS/yesh/MemUpdateBench
source activate.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
OUT_ROOT=${OUT_ROOT:-results/p81_synthetic_same_slot_probe_expanded}
mkdir -p "${OUT_ROOT}"
python scripts/run_synthetic_same_slot_probe.py \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUT_ROOT}" \
  --examples_per_condition 64 \
  --condition_set selected_v4 \
  --stale_counts 1,2,4,8,16 \
  --value_policies conflict,same_as_current \
  --context_orders chronological,reverse_chronological \
  --context_annotations none,latest_outdated_label \
  --no_qlora

echo "ALL_DONE $(date)"
