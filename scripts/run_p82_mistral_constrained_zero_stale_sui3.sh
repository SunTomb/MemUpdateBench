#!/usr/bin/env bash
set -euo pipefail
cd /NAS/yesh/MemUpdateBench
source activate.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_NAME=${MODEL_NAME:-/NAS/HuggingFaceModels/Mistral-7B-Instruct-v0.1}
OUT_ROOT=${OUT_ROOT:-results/p82_mistral_constrained_zero_stale}
DATA=${DATA:-data/evomemory_update_frequency_hard_k16_p63_dev.json}
mkdir -p "${OUT_ROOT}"
run_eval() {
  local name="$1"
  shift
  echo "START ${name} $(date)"
  python scripts/eval_evomemory.py \
    --model_name "${MODEL_NAME}" \
    --mode constrained_slot_crud \
    --data_file "${DATA}" \
    --output_dir "${OUT_ROOT}/${name}" \
    --answer_topk 5 \
    --save_answer_traces \
    --no_qlora \
    --lora_checkpoint "" \
    "$@"
  echo "DONE ${name} $(date)"
}
run_eval constrained_slot_crud_slot_prompt_k16_dev --answer_mode slot_prompt --retrieval_policy normal --context_order normal --context_annotation none
run_eval constrained_slot_crud_slot_direct_k16_dev --answer_mode slot_direct

echo "ALL_DONE $(date)"
