#!/usr/bin/env bash
set -euo pipefail

ROOT=/NAS/yesh/MemUpdateBench
cd "$ROOT"
source activate.sh >/dev/null 2>&1
export PYTHONPATH="$ROOT"
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/NAS/yesh/miniconda3/envs/gmsra/bin/python
GPUS=(4 5 7)
SLOTS_PER_GPU=1
TOTAL_WORKERS=$((${#GPUS[@]} * SLOTS_PER_GPU))
SHARD_SIZE=10
RESULT_ROOT=results/p69_expanded_slot_prompt
DATA=data/evomemory_update_frequency_expanded_k16_p68_dev.json
mkdir -p "$RESULT_ROOT/logs"

run_shard() {
  local gpu=$1
  local mode=$2
  local start=$3
  local end=$4
  local out_dir="$RESULT_ROOT/${mode}_prompt_k16_dev/shard_${start}_${end}"
  local log_file="$RESULT_ROOT/logs/${mode}_shard_${start}_${end}_gpu${gpu}.log"
  if [[ -f "$out_dir/evomemory_results.json" ]]; then
    echo "$(date '+%F %T') skip mode=${mode} range=[${start},${end}) gpu=${gpu}"
    return 0
  fi
  mkdir -p "$out_dir"
  echo "$(date '+%F %T') start mode=${mode} range=[${start},${end}) gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_evomemory.py \
    --mode "$mode" \
    --answer_mode slot_prompt \
    --slot_prompt_variant v0_current \
    --save_answer_traces \
    --no_qlora \
    --data_file "$DATA" \
    --start_idx "$start" \
    --end_idx "$end" \
    --output_dir "$out_dir" \
    > "$log_file" 2>&1
  echo "$(date '+%F %T') done mode=${mode} range=[${start},${end}) gpu=${gpu}"
}

run_worker() {
  local worker_id=$1
  local gpu=$2
  local task_id=0
  for mode in constrained_slot_crud raw_add; do
    for start in $(seq 0 "$SHARD_SIZE" 190); do
      local end=$((start + SHARD_SIZE))
      if (( task_id % TOTAL_WORKERS == worker_id )); then
        run_shard "$gpu" "$mode" "$start" "$end"
      fi
      task_id=$((task_id + 1))
    done
  done
  echo "$(date '+%F %T') worker ${worker_id} finished on gpu=${gpu}"
}

worker_id=0
for gpu in "${GPUS[@]}"; do
  for slot in $(seq 1 "$SLOTS_PER_GPU"); do
    run_worker "$worker_id" "$gpu" > "$RESULT_ROOT/logs/worker_${worker_id}_gpu${gpu}.log" 2>&1 &
    echo "launched worker=${worker_id} gpu=${gpu} slot=${slot}"
    worker_id=$((worker_id + 1))
  done
done

wait
"$PY" scripts/merge_evomemory_shards.py --inputs "$RESULT_ROOT/constrained_slot_crud_prompt_k16_dev" --output_dir "$RESULT_ROOT/constrained_slot_crud_prompt_k16_dev_merged"
"$PY" scripts/merge_evomemory_shards.py --inputs "$RESULT_ROOT/raw_add_prompt_k16_dev" --output_dir "$RESULT_ROOT/raw_add_prompt_k16_dev_merged"
"$PY" scripts/summarize_update_frequency.py --result_root "$RESULT_ROOT" --output_dir results/p69_expanded_slot_prompt_summary
echo "$(date '+%F %T') expanded slot_prompt k16 jobs finished"
