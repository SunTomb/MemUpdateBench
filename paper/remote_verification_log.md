# P6.7 Remote Verification Log

## Cluster snapshot

See `paper/cluster_resource_snapshot.md`.

Selected nodes:

- Oracle sanity: `Tang-2-Wu`
- Learned Long25 spot-check: first attempted `Song-1-Wu`, then rerouted to `Sui-3-Wu`

## Environment notes

`Tang-2-Wu` uses the expected NAS environment:

```text
/NAS/yesh/miniconda3/envs/gmsra/bin/python
Python 3.10.20
transformers 4.46.3
```

`Song-1-Wu` cannot use `/NAS/yesh/miniconda3` cleanly because of GLIBC mismatch and falls back to another Python:

```text
/data3/wujcan/sumuze/miniconda3/bin/python
Python 3.12.2
transformers 4.51.3
```

The Song-1 long25 attempt failed with:

```text
dlopen: cannot load any more object with static TLS
```

Therefore, learned spot-checks were rerouted to `Sui-3-Wu`, which uses the expected NAS `gmsra` environment.

## Remote oracle sanity: constrained CRUD slot-direct k=16

Node:

```text
Tang-2-Wu
```

Command:

```bash
cd /NAS/yesh/MemUpdateBench && source activate.sh && \
PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode constrained_slot_crud \
  --answer_mode slot_direct \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/remote_sanity_oracle_k16
```

Result:

```text
F1: 1.0000
EM: 1.0000
State: resolve=1.0000 acc=1.0000 gold_present=1.0000 stale_present=0.0000
Slot burden: same_slot=1.00 stale_same_slot=0.00 gold_same_slot=1.00 write_amp=0.442 target_writes=16.00
Results saved to results/remote_sanity_oracle_k16/evomemory_results.json
```

Status: pass. This confirms the remote deterministic oracle sanity anchor.

## Remote Long25 spot-check status

### Song-1-Wu attempt

Status: failed before model load due to Python/transformers import environment issue.

### Sui-3-Wu rerun

Status: completed.

Commands:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode learned_constrained_slot \
  --answer_mode slot_direct \
  --no_qlora \
  --lora_checkpoint checkpoints/long25/best \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/remote_long25_slot_direct_k16

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode learned_constrained_slot \
  --answer_mode slot_prompt \
  --no_qlora \
  --lora_checkpoint checkpoints/long25/best \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/remote_long25_slot_prompt_k16
```

Expected comparison from canonical ledger:

```text
Long25 k=16 slot_direct state_acc = 0.91
Long25 k=16 slot_prompt EM/F1 = 0.48 / 0.53
stale_same_slot = 1.13
final_memory_size = 9.43
```

Observed Sui-3-Wu rerun:

```text
slot_direct: EM/F1 = 0.9200 / 0.9200, state_acc = 0.9200, stale_same_slot = 1.13, final_memory_size = 9.28
slot_prompt: EM/F1 = 0.4900 / 0.5467, state_acc = 0.9200, stale_same_slot = 1.13, final_memory_size = 9.28
```

Status: pass as a close remote confirmation. The rerun differs slightly from the canonical ledger (`0.92` vs `0.91` state accuracy, `0.49 / 0.5467` vs `0.48 / 0.53` slot-prompt EM/F1), but preserves the paper-level conclusion: Long25 is compact and low-stale, yet not perfectly reliable under k=16 hard distractors.

Remote error analysis was written to:

```text
paper/error_analysis_k16_remote_long25.json
```

Remote state-error pattern:

```text
slot_direct: 92 correct, 8 wrong_value
slot_prompt: 92 correct, 8 wrong_value
wrong_value errors are concentrated in company slots.
```
