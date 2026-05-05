#!/usr/bin/env bash
set -euo pipefail
cd /NAS/yesh/MemUpdateBench
source activate.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
OUT_ROOT=results/p80_expanded_latest_per_slot_allk
mkdir -p "${OUT_ROOT}"
run_k() {
  local k="$1"
  local data="data/evomemory_update_frequency_expanded_k${k}_p68_dev.json"
  local out_dir="${OUT_ROOT}/raw_add_latest_per_slot_k${k}_dev"
  echo "START k=${k} $(date)"
  python scripts/eval_evomemory.py \
    --model_name "${MODEL_NAME}" \
    --mode raw_add \
    --answer_mode slot_prompt \
    --data_file "${data}" \
    --output_dir "${out_dir}" \
    --answer_topk 5 \
    --retrieval_policy latest_per_slot \
    --save_answer_traces \
    --no_qlora \
    --lora_checkpoint ""
  echo "DONE k=${k} $(date)"
}
for k in 1 2 4 8 16; do
  run_k "${k}"
done
echo "ALL_DONE $(date)"
