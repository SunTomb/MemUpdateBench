# P6.9 External Baseline Result Note

## Motivation

The strict reviewer-risk plan requires a real external memory baseline if possible. P6.8 showed that Mem0 could be installed and imported in isolation, but did not produce a runnable baseline row. P6.9 re-opens the feasibility path with a cached local embedding model and alternate vector stores to determine whether the blocker is still environmental or whether a small Mem0 row can be produced.

## Attempts

Environment:

```text
/NAS/yesh/MemUpdateBench
PYTHONPATH=/NAS/yesh/MemUpdateBench/external/mem0_vendor:/NAS/yesh/MemUpdateBench
/NAS/yesh/miniconda3/envs/gmsra/bin/python
```

Data smoke target:

```text
data/evomemory_update_frequency_hard_k16_p63_dev.json
limit=2, then limit=20
```

### Local HuggingFace embedder + FAISS

Config:

```text
configs/mem0_local_minilm_faiss.json
embedder model: /NAS/yesh/hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf
vector_store: faiss
```

Status: unavailable.

Reason:

```text
Could not import faiss python package.
```

### Local HuggingFace embedder + Qdrant + default OpenAI provider

Config:

```text
configs/mem0_local_minilm_qdrant.json
```

Status: unavailable.

Reason:

```text
The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable
```

### Local HuggingFace embedder + Qdrant + discovered local vLLM endpoint

A useful cluster discovery changed the feasibility picture:

- Tang-2 hosts a live OpenAI-compatible vLLM server at `http://127.0.0.1:8011/v1`;
- the served model list includes `Qwen2.5-VL`;
- the cluster also has the local MiniLM embedder snapshot above.

Config used:

```text
configs/mem0_vllm_qdrant_minilm384.json
llm.provider = vllm
llm.config.vllm_base_url = http://127.0.0.1:8011/v1
llm.config.model = Qwen2.5-VL
embedder.provider = huggingface
vector_store.provider = qdrant
vector_store.config.embedding_model_dims = 384
```

Adapter changes required:

- support Mem0 v2 `search(filters=...)` and `get_all(filters=...)` instead of only top-level `user_id=`;
- reuse one Mem0 instance across examples while isolating examples by `user_id`, to avoid repeated local Qdrant lock conflicts;
- use a 384-dim Qdrant collection to match the local MiniLM embedder.

Outputs:

```text
results/p69_external_baselines/mem0_vllm_qdrant384_k16_dev2/
results/p69_external_baselines/mem0_vllm_qdrant384_k16_dev20/
```

Observed dev20 smoke summary:

```text
avg_em = 0.00
avg_f1 = 0.00
memory_inspectable_rate = 1.00
stale_same_slot_entry_count_avg = 1.15
same_slot_entry_count_avg = 1.20
gold_same_slot_entry_count_avg = 0.05
final_memory_size_avg = 20.0
```

Example predictions show that Mem0 is returning natural-language memory statements, but not the correct final slot value for this benchmark. In many cases the extracted memory text does not align cleanly with the benchmark's exact `(entity, attribute)` target slot.

## Interpretation

P6.9 improves the external-baseline status substantially:

1. Mem0 is no longer only an installation/import feasibility result.
2. A real end-to-end Mem0 run is now possible on the cluster using:
   - a discovered local OpenAI-compatible vLLM endpoint,
   - a local MiniLM embedder snapshot,
   - a local Qdrant vector store.
3. Memory inspection is available, so stale same-slot burden can be estimated.

However, the current Mem0 row is still weak as benchmark evidence. The problem is no longer environment setup; it is semantic misalignment between Mem0's extraction/retrieval behavior and the benchmark's exact final-slot target. The dev20 smoke produces 0 EM/F1 while still storing inspectable memories. That is a real external-baseline result, but only a preliminary one.

The most likely reasons are:

- Mem0's extraction pipeline compresses or rewrites facts into memory texts that do not preserve exact slot semantics;
- retrieval returns plausible but non-final statements;
- the benchmark adapter currently chooses the first returned memory text as the prediction, which is faithful as a baseline probe but not yet optimized for this benchmark.

## Paper-facing conclusion

Mem0 can now be described as a real but preliminary external baseline attempt, not merely a blocked feasibility probe. The current dev20 row is poor (`0.00` EM/F1) even though memory inspection is available, which suggests that off-the-shelf external memory extraction/retrieval does not naturally align with repeated exact-slot update tracking.

For the current paper phase, this is already useful evidence: the external-validity gap has narrowed from “cannot run” to “can run, but current behavior is badly mismatched to exact-slot update tracking.” A stronger Mem0 row would require additional adapter work or a better-aligned local LLM/extraction prompt before it should be placed next to the main benchmark rows.

## If the user wants to continue configuration

The current runnable configuration depends on three ingredients:

1. local vLLM endpoint on Tang-2:
   - `http://127.0.0.1:8011/v1`
   - served model: `Qwen2.5-VL`
2. local MiniLM embedder snapshot:
   - `/NAS/yesh/hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
3. project-local Mem0 vendor path:
   - `/NAS/yesh/MemUpdateBench/external/mem0_vendor`

A follow-up adapter-improvement pass was run after the initial vLLM/Qdrant smoke:

- `scripts/eval_mem0_baseline.py` now stores `search_payload` and `memory_texts` for diagnosis.
- The adapter supports Mem0 v2 `filters` and `--reuse_memory_instance`.
- The prediction rule was changed from returning the first memory text to selecting a benchmark-relevant candidate and extracting a slot-like value.

Output:

```text
results/p69_external_baselines/mem0_vllm_qdrant384_k16_dev20_v2/
```

Improved-extraction dev20 summary:

```text
avg_em = 0.00
avg_f1 = 0.05
value_em = 0.00
answer_value_present = 0.00
memory_inspectable_rate = 1.00
stale_same_slot_entry_count_avg = 1.00
same_slot_entry_count_avg = 1.05
gold_same_slot_entry_count_avg = 0.05
final_memory_size_avg = 20.0
```

The improved extraction fixes output format but not correctness. Example predictions change from full memory sentences to extracted values such as `Kotlin`, `black coffee`, `NetEase`, and `Jinan`, but these are still stale or wrong values relative to the final gold answers. Therefore the remaining issue is not answer formatting; it is Mem0's off-the-shelf extraction/retrieval behavior under repeated same-slot updates.

This is enough evidence to stop before dev100 for the current paper phase. A larger run would likely spend compute confirming the same near-zero row unless Mem0's memory extraction prompt, update policy, or retrieval/reranking behavior is changed.

## Current recommendation

Use Mem0 as a real, runnable but badly misaligned external baseline probe:

- It can be installed and run in isolation.
- It can use local vLLM, local MiniLM, and local Qdrant without external API keys.
- Its memories are inspectable.
- Under the current off-the-shelf configuration, it fails exact repeated same-slot final-value tracking on k=16 dev20.

Do not invest in dev100 until the goal changes from benchmarking off-the-shelf Mem0 to engineering a Mem0-specific adapter or prompt.
