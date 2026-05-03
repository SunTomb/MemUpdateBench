#!/usr/bin/env bash
set -euo pipefail

ROOT=/NAS/yesh/MemUpdateBench
cd "$ROOT"
source activate.sh >/dev/null 2>&1
export PYTHONPATH="$ROOT"

GPUS=(0 2 3)
SLOTS_PER_GPU=2
TOTAL_WORKERS=$((${#GPUS[@]} * SLOTS_PER_GPU))
SHARD_SIZE=5
RESULT_ROOT=results/p65_stability_sharded
mkdir -p "$RESULT_ROOT/logs"

run_shard() {
  local gpu=$1
  local seed=$2
  local start=$3
  local end=$4
  local out_dir="$RESULT_ROOT/seed${seed}/shard_${start}_${end}"
  local log_file="$RESULT_ROOT/logs/seed${seed}_shard_${start}_${end}_gpu${gpu}.log"
  if [[ -f "$out_dir/evomemory_results.json" ]]; then
    echo "$(date '+%F %T') skip seed=${seed} range=[${start},${end}) gpu=${gpu}"
    return 0
  fi
  mkdir -p "$out_dir"
  echo "$(date '+%F %T') start seed=${seed} range=[${start},${end}) gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_evomemory.py \
    --mode learned_constrained_slot \
    --answer_mode slot_prompt \
    --slot_prompt_variant v0_current \
    --save_answer_traces \
    --no_qlora \
    --lora_checkpoint "outputs/p65_long25_seed${seed}/best" \
    --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
    --start_idx "$start" \
    --end_idx "$end" \
    --output_dir "$out_dir" \
    > "$log_file" 2>&1
  echo "$(date '+%F %T') done seed=${seed} range=[${start},${end}) gpu=${gpu}"
}

run_worker() {
  local worker_id=$1
  local gpu=$2
  local task_id=0
  for seed in 11 22 33; do
    for start in $(seq 0 "$SHARD_SIZE" 95); do
      local end=$((start + SHARD_SIZE))
      if (( task_id % TOTAL_WORKERS == worker_id )); then
        run_shard "$gpu" "$seed" "$start" "$end"
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
python scripts/merge_evomemory_shards.py --inputs "$RESULT_ROOT/seed11" --output_dir "$RESULT_ROOT/seed11_merged"
python scripts/merge_evomemory_shards.py --inputs "$RESULT_ROOT/seed22" --output_dir "$RESULT_ROOT/seed22_merged"
python scripts/merge_evomemory_shards.py --inputs "$RESULT_ROOT/seed33" --output_dir "$RESULT_ROOT/seed33_merged"
echo "$(date '+%F %T') all sharded Long25 stability jobs finished"
