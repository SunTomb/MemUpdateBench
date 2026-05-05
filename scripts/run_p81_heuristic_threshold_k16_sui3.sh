#!/usr/bin/env bash
set -euo pipefail
cd /NAS/yesh/MemUpdateBench
source activate.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT_ROOT=${OUT_ROOT:-results/p81_heuristic_threshold_k16_rigor}
DATA=${DATA:-data/evomemory_update_frequency_hard_k16_p63_dev.json}
THRESHOLDS=${THRESHOLDS:-0.70 0.75 0.80 0.85 0.90 0.95}
mkdir -p "${OUT_ROOT}"
for threshold in ${THRESHOLDS}; do
  for answer_mode in slot_prompt slot_direct; do
    if [[ "${answer_mode}" == "slot_prompt" ]]; then
      prefix="prompt"
    else
      prefix="direct"
    fi
    name="${prefix}_t${threshold}_k16"
    echo "START ${name} $(date)"
    python scripts/eval_evomemory.py \
      --mode heuristic_crud \
      --answer_mode "${answer_mode}" \
      --update_threshold "${threshold}" \
      --data_file "${DATA}" \
      --output_dir "${OUT_ROOT}/${name}" \
      --save_answer_traces \
      --no_qlora \
      --lora_checkpoint ""
    echo "DONE ${name} $(date)"
  done
done

echo "ALL_DONE $(date)"
