"""Bounded loopback model service for version-transfer qualification and readout."""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.runtime.version_transfer_budget import BudgetBlocked, StudyBudget
from mub.vnext.runtime.version_transfer_local import LocalCompletionBoundary


class TransformersBackend:
    def __init__(self, snapshot, gpu_uuid):
        if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
            raise ValueError('offline model loading flags required')
        if not gpu_uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
            raise ValueError('explicit admitted GPU UUID must match visibility')
        snapshot = Path(snapshot).resolve(strict=True)
        if Path('/NAS/yesh') not in snapshot.parents:
            raise ValueError('model input must stay inside authorized NAS boundary')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map='cuda:0')
        self.model.eval()
        self.context_tokens = min(32768, int(self.model.config.max_position_embeddings))

    def prepare(self, messages):
        # No fallback template and no truncation: template errors are technical failures.
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=False, add_special_tokens=False)
        return {'input_tokens': int(inputs.input_ids.shape[1]), 'inputs': inputs}

    def generate(self, prepared, max_tokens):
        inputs = prepared['inputs'].to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, prepared['input_tokens']:]
        tokens = int(generated.shape[0])
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if type(eos) is int else (eos or [])
        ended = tokens > 0 and int(generated[-1]) in eos
        return {'text': self.tokenizer.decode(generated, skip_special_tokens=True),
                'completion_tokens': tokens, 'finish_reason': 'stop' if ended else 'length'}

    def close(self):
        self.model = self.tokenizer = None
        gc.collect()
        self.torch.cuda.empty_cache()


def serve(boundary, *, port, seconds, on_ready=None):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(30)
            super().setup()

        def respond(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.respond(200, {'ready': True, 'model': boundary.model_id})
            elif self.path == '/v1/models':
                self.respond(200, {'object': 'list', 'data': [{'id': boundary.model_id, 'object': 'model'}]})
            else:
                self.respond(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.respond(404, {'error': 'not_found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2 * 1024 * 1024:
                    raise ValueError('invalid request byte limit')
                request = json.loads(self.rfile.read(length))
                role = self.headers.get('X-MUB-Role', 'unclassified')
                result = boundary.complete(request, role=role)
                self.respond(200, result)
            except BudgetBlocked:
                self.respond(429, {'error': 'STUDY_BUDGET_BLOCKED'})
            except (ValueError, TypeError, KeyError):
                self.respond(400, {'error': 'UNSUPPORTED_OR_INVALID_REQUEST'})
            except Exception:
                self.respond(500, {'error': 'LOCAL_GENERATION_TECHNICAL_FAILURE'})

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', port), Handler) as server:
        server.timeout = 1
        deadline = time.monotonic() + seconds
        if on_ready is not None:
            on_ready(server.server_address)
        while time.monotonic() < deadline:
            server.handle_request()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18043)
    parser.add_argument('--attempts', type=int, default=10000)
    parser.add_argument('--input-tokens', type=int, default=24000000)
    parser.add_argument('--output-tokens', type=int, default=2000000)
    parser.add_argument('--seconds', type=int, default=43200)
    args = parser.parse_args()
    if not (0 < args.attempts <= 10000 and 0 < args.input_tokens <= 24000000
            and 0 < args.output_tokens <= 2000000 and 0 < args.seconds <= 43200):
        parser.error('requested limits exceed study ceiling')
    if Path('/NAS/yesh') not in args.ledger.absolute().parents:
        parser.error('ledger must remain inside authorized NAS boundary')
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    budget = StudyBudget(args.ledger, attempts=args.attempts, input_tokens=args.input_tokens,
                         output_tokens=args.output_tokens, seconds=args.seconds)
    backend = None
    try:
        backend = TransformersBackend(args.snapshot, args.gpu_uuid)
        boundary = LocalCompletionBoundary(backend, budget, model_id=args.model_id)
        serve(boundary, port=args.port, seconds=max(1, args.seconds - (time.monotonic() - budget._started)),
              on_ready=lambda address: print(json.dumps({'ready': True, 'model': args.model_id,
                                                          'listen': list(address)}), flush=True))
    finally:
        if backend is not None:
            backend.close()
        budget.close()
        print(json.dumps({'terminal': True, 'usage': budget.totals}), flush=True)


if __name__ == '__main__':
    main()
