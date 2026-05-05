# P7.0 Text-Backend External Baseline Probe

## Motivation

`docs/critical_review_v2.md` correctly argues that the current Mem0 probe is not a fair main-table external baseline because it uses Qwen2.5-VL and only runs dev20. A fairer Mem0 row would require a text-only backend such as Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct.

## Cluster probe

A cluster resource probe found that text-only model weights are available on NAS:

```text
/NAS/HuggingFaceModels/Qwen2.5-7B-Instruct
/NAS/HuggingFaceModels/Llama3.1-8B-Instruct
/NAS/yesh/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct
/NAS/yesh/hf_cache/hub/models--meta-llama--Llama-3.1-8B
```

However, the available `gmsra` conda environment on checked nodes does not contain the serving/client stack needed by Mem0's `vllm` provider:

```text
Sui-3 gmsra: vllm=False, openai=False, fastapi=False, uvicorn=False, transformers=True
Tang-1 gmsra: vllm=False, openai=False, fastapi=False, uvicorn=False, transformers=True
```

Tang-2 still has only the existing live OpenAI-compatible endpoint:

```text
http://127.0.0.1:8011/v1 -> Qwen2.5-VL
```

No text-only OpenAI-compatible endpoint was discovered during this probe.

## Follow-up: project-local text endpoint

A minimal project-local OpenAI-compatible text endpoint was added after this probe:

```text
scripts/serve_openai_compatible_transformers.py
configs/mem0_qwen25_text_qdrant_minilm384_cpu.json
endpoint: http://127.0.0.1:8013/v1
model: /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct
embedder: MiniLM on CPU
vector store: local Qdrant, 384-dim collection
```

The endpoint can answer `/v1/models` and `/v1/chat/completions`, so the original blocker is no longer simply absence of a text-only endpoint. However, Mem0's internal extraction prompts remain brittle under this lightweight transformers server: k=16 dev20 did not complete the first example, repeatedly logging JSON extraction parse errors such as:

```text
Error parsing extraction response: Expecting ',' delimiter
Error parsing extraction response: Extra data
```

Reducing the configured generation cap from 512 to 128 tokens avoids the earlier CUDA embedding failure but does not make the run complete. Both k=16 dev20 and k=16 dev3 stopped before the first completed example, with zero `Progress:` lines and repeated Mem0 extraction JSON parse errors.

## Decision

Do not report a fair Mem0 main-table row yet.

The honest status is now:

- Mem0 is runnable as a qualitative probe with Qwen2.5-VL + MiniLM + Qdrant.
- A project-local Qwen2.5-7B-Instruct text endpoint can be launched and queried.
- CPU embedding avoids the previous CUDA CUBLAS failure.
- The current text-backend Mem0 path is still not stable enough for a main-table row because Mem0 extraction JSON parsing stalls before completing even k=16 dev3.

Only after the text-backend smoke reliably completes should we run:

```text
k=16 dev100 -> k=16 test200 -> optional all-k test
```

Until then, the paper should keep Mem0 in qualitative/appendix feasibility and state the text-backend adapter instability explicitly.
