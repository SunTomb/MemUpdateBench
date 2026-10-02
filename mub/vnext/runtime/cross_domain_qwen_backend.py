"""Bounded offline Qwen readout; authorization and source qualification are external."""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import importlib
import os
from pathlib import Path
from typing import Callable

from scripts.vnext_run_cross_domain_answer_replay import GenerationOutput, PreparedInput, PublicRequest

REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
TEMPLATE_SHA256 = 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'


def _sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class QwenReadoutBackend:
    def __init__(self, snapshot: Path, expected_gpu_uuid: str, expected_template_sha256: str,
                 require_device: Callable[[], dict], modules_loader=None):
        self._torch = self._transformers = self._tokenizer = self._model = None
        self._closed = self._failed = self._loaded = False
        self._bindings, self._attempted = {}, set()
        self._observations = {'load_attempts': 0, 'tokenizer_loads': 0, 'generation_attempts': 0,
                              'close': None}
        if expected_template_sha256 != TEMPLATE_SHA256:
            raise ValueError('unexpected pinned template hash')
        if not isinstance(snapshot, Path) or not snapshot.is_dir():
            raise ValueError('snapshot must be an existing local directory')
        if type(expected_gpu_uuid) is not str or not expected_gpu_uuid.startswith('GPU-'):
            raise ValueError('explicit GPU UUID required')
        if not callable(require_device):
            raise TypeError('current device check required')
        self.snapshot, self.expected_gpu_uuid = snapshot, expected_gpu_uuid
        self.expected_template_sha256 = expected_template_sha256
        self._require_device = require_device
        self._modules_loader = modules_loader or importlib.import_module

    @property
    def observations(self):
        return deepcopy(self._observations)

    def _open(self):
        if self._closed or self._failed:
            raise RuntimeError('backend is closed or failed; no retries allowed')

    def _offline(self):
        if any(os.environ.get(k) != '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
            raise RuntimeError('offline environment flags required')

    def _device(self):
        self._offline()
        if os.environ.get('CUDA_VISIBLE_DEVICES') != self.expected_gpu_uuid:
            raise RuntimeError('visible device identity changed')
        observation = self._require_device()
        if type(observation) is not dict or observation.get('gpu_uuid', observation.get('uuid')) != self.expected_gpu_uuid:
            raise RuntimeError('current device identity changed')

    def _template(self):
        getter = getattr(self._tokenizer, 'get_chat_template', None)
        template = (getter() if callable(getter) else
                    (self.snapshot / 'chat_template.jinja').read_bytes().decode('utf-8'))
        if type(template) is not str or _sha256(template) != self.expected_template_sha256:
            raise ValueError('pinned chat template changed')
        return template

    def _render(self, request):
        template = self._template()
        rendered = self._tokenizer.apply_chat_template(
            [{'role': 'user', 'content': request.visible_prompt}], chat_template=template, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self._template()
        if type(rendered) is not str or not rendered:
            raise ValueError('invalid rendered prompt')
        ids = self._tokenizer(rendered, add_special_tokens=False)['input_ids']
        if not 0 < len(ids) <= 4032:
            raise ValueError('input token budget exceeded')
        prepared = PreparedInput(request.request_id, rendered, self.expected_template_sha256, len(ids))
        return prepared, _sha256(rendered), tuple(ids)

    def prepare(self, request: PublicRequest) -> PreparedInput:
        self._open()
        if type(request) is not PublicRequest or type(request.request_id) is not str or not request.request_id:
            raise TypeError('public request required')
        if type(request.visible_prompt) is not str or not request.visible_prompt:
            raise ValueError('visible prompt required')
        if request.request_id not in self._bindings and len(self._bindings) >= 4:
            raise RuntimeError('four-request budget exceeded')
        self._offline()
        if self._tokenizer is None:
            if self._observations['tokenizer_loads']:
                raise RuntimeError('tokenizer load cannot be retried')
            if self._transformers is None:
                self._transformers = self._modules_loader('transformers')
            self._observations['tokenizer_loads'] += 1
            self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                str(self.snapshot), revision=REVISION, local_files_only=True, trust_remote_code=False)
        prepared, rendered_sha, ids = self._render(request)
        binding = (request, prepared, rendered_sha, ids)
        if request.request_id in self._bindings and self._bindings[request.request_id] != binding:
            raise ValueError('prepared render/token binding changed')
        self._bindings[request.request_id] = binding
        return prepared

    def load(self) -> None:
        self._open()
        if self._observations['load_attempts']:
            raise RuntimeError('one load attempt only')
        self._observations['load_attempts'] += 1
        try:
            self._device()
            self._torch = self._modules_loader('torch')
            if self._transformers is None:
                self._transformers = self._modules_loader('transformers')
            torch = self._torch
            if (torch.__version__ != '2.5.1+cu121' or torch.version.cuda != '12.1'
                    or self._transformers.__version__ != '5.9.0'):
                raise RuntimeError('unqualified runtime version')
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 CUDA device required')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self._device()
            self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
                str(self.snapshot), revision=REVISION, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 0})
            if self._model.dtype != torch.bfloat16:
                raise RuntimeError('loaded model dtype is not BF16')
            if str(self._model.device) != 'cuda:0':
                raise RuntimeError('loaded model device mismatch')
            self._model.eval()
            self._loaded = True
        except BaseException:
            self._failed = True
            raise

    def generate(self, prepared: PreparedInput, *, max_new_tokens=64, seed=0,
                 do_sample=False, num_beams=1) -> GenerationOutput:
        self._open()
        if not (type(max_new_tokens) is int and max_new_tokens == 64 and type(seed) is int
                and seed == 0 and type(do_sample) is bool and do_sample is False
                and type(num_beams) is int and num_beams == 1):
            raise ValueError('exact deterministic decoding settings required')
        if not self._loaded or self._tokenizer is None:
            raise RuntimeError('model and prepared tokenizer required')
        if type(prepared) is not PreparedInput or prepared.request_id not in self._bindings:
            raise ValueError('unknown prepared input')
        if prepared.request_id in self._attempted or self._observations['generation_attempts'] >= 4:
            raise RuntimeError('no generation retries; four-attempt budget')
        encoded = row = output = suffix = prefix = None
        try:
            self._device()
            request, original, rendered_sha, ids = self._bindings[prepared.request_id]
            rechecked, current_sha, current_ids = self._render(request)
            if (prepared != original or rechecked != original or current_sha != rendered_sha
                    or current_ids != ids or _sha256(prepared.rendered_text) != rendered_sha):
                raise ValueError('render/token binding changed')
            encoded = self._tokenizer(prepared.rendered_text, add_special_tokens=False, return_tensors='pt')
            row = encoded['input_ids'][0]
            actual = tuple(row.tolist()) if hasattr(row, 'tolist') else tuple(row)
            if actual != ids or encoded['input_ids'].shape != (1, prepared.input_tokens):
                raise ValueError('input token binding changed')
            encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
            self._torch.manual_seed(0)
            self._torch.cuda.manual_seed_all(0)
            self._attempted.add(prepared.request_id)
            self._observations['generation_attempts'] += 1
            with self._torch.inference_mode():
                output = self._model.generate(**encoded, max_new_tokens=64, do_sample=False,
                    num_beams=1, return_dict_in_generate=False)
            if output.shape[0] != 1 or output.shape[1] < prepared.input_tokens:
                raise RuntimeError('invalid output token shape')
            prefix = output[0][:prepared.input_tokens]
            prefix = tuple(prefix.tolist()) if hasattr(prefix, 'tolist') else tuple(prefix)
            if prefix != ids:
                raise RuntimeError('output input-token prefix changed')
            suffix = output[0][prepared.input_tokens:]
            if len(suffix) > 64:
                raise RuntimeError('generated token budget exceeded')
            text = self._tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return GenerationOutput(text, prepared.input_tokens, len(suffix))
        except BaseException:
            self._failed = True
            raise
        finally:
            encoded = row = output = suffix = prefix = None

    def close(self) -> dict:
        if self._closed:
            if self._observations['close']['status'] != 'CLOSED':
                raise RuntimeError('previous CUDA cleanup failed')
            return self.observations
        self._closed = True
        torch = self._torch
        info = {'status': 'ATTEMPTED', 'cuda_cleanup_performed': False,
                'private_workspace_clear_used': False, 'allocated_before': None,
                'reserved_before': None, 'allocated_after': None, 'reserved_after': None}
        self._observations['close'] = info
        try:
            try:
                if torch is not None and torch.cuda.is_available():
                    info['cuda_cleanup_performed'] = True
                    info['allocated_before'] = torch.cuda.memory_allocated(0)
                    info['reserved_before'] = torch.cuda.memory_reserved(0)
            finally:
                self._model = self._tokenizer = self._transformers = self._torch = None
                self._bindings.clear()
                self._loaded = False
                gc.collect()
            if info['cuda_cleanup_performed']:
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
                clear = getattr(getattr(torch, '_C', None), '_cuda_clearCublasWorkspaces', None)
                if (torch.cuda.memory_allocated(0) and torch.__version__ == '2.5.1+cu121'
                        and torch.version.cuda == '12.1' and callable(clear)):
                    info['private_workspace_clear_used'] = True
                    if clear() is not None:
                        raise RuntimeError('unexpected cuBLAS workspace clear return')
                    torch.cuda.synchronize(0)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(0)
                info['allocated_after'] = torch.cuda.memory_allocated(0)
                info['reserved_after'] = torch.cuda.memory_reserved(0)
                if info['allocated_after'] != 0:
                    raise RuntimeError('CUDA allocated memory remains after cleanup')
            info['status'] = 'CLOSED'
        except BaseException:
            info['status'] = 'FAILED'
            raise
        finally:
            if info['cuda_cleanup_performed'] and info['allocated_after'] is None:
                try:
                    info['allocated_after'] = torch.cuda.memory_allocated(0)
                    info['reserved_after'] = torch.cuda.memory_reserved(0)
                except BaseException:
                    info['measurement_failed'] = True
            for phase in ('before', 'after'):
                for kind in ('allocated', 'reserved'):
                    self._observations[f'{kind}_bytes_{phase}'] = info[f'{kind}_{phase}']
        return self.observations
