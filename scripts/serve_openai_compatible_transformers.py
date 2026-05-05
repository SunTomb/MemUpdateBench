from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = None
TOKENIZER = None
MODEL_ID = ""
MAX_CONTEXT_CHARS = 12000


def chat_prompt(messages: list[dict[str, Any]]) -> str:
    if hasattr(TOKENIZER, "apply_chat_template"):
        try:
            return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def generate(messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
    prompt = chat_prompt(messages)
    if len(prompt) > MAX_CONTEXT_CHARS:
        prompt = prompt[-MAX_CONTEXT_CHARS:]
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(MODEL.device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=TOKENIZER.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return TOKENIZER.decode(generated, skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
            })
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = request.get("messages", [])
            max_tokens = int(request.get("max_tokens", 512) or 512)
            temperature = float(request.get("temperature", 0.0) or 0.0)
            content = generate(messages, max_tokens=max_tokens, temperature=temperature)
            self._send(200, {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "model": request.get("model", MODEL_ID),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible chat server backed by transformers")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", default="local-text-model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = parser.parse_args()

    global MODEL, TOKENIZER, MODEL_ID
    MODEL_ID = args.model_id
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    MODEL.eval()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {MODEL_ID} at http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
