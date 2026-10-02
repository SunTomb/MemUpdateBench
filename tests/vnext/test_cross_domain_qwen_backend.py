from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from mub.vnext.runtime import cross_domain_qwen_backend as backend
from scripts.vnext_run_cross_domain_answer_replay import GenerationOutput, PreparedInput, PublicRequest

PIN = 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'
GPU = 'GPU-test-uuid'


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        assert device == 'cuda:0'
        return self

    def __getitem__(self, item):
        return self.rows[item]


@pytest.fixture
def rig(monkeypatch, tmp_path):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', GPU)
    real_sha = backend._sha256
    monkeypatch.setattr(backend, '_sha256', lambda text: PIN if text == 'PINNED' else real_sha(text))
    r = SimpleNamespace(events=[], uuid=GPU, tokens=7, suffix=3, template='PINNED',
                        render_drift=False, token_drift=False, fail_load=False,
                        fail_eval=False, fail_generate=False, allocated=0, reserved=0,
                        release_on_empty=True, release_on_private=True, dtype='bf16',
                        tokenizer_failure=False, output_prefix_drift=False)

    class Tokenizer:
        def get_chat_template(self):
            return r.template

        def apply_chat_template(self, messages, **kwargs):
            r.events.append(('render', messages, kwargs))
            assert kwargs == {'chat_template': 'PINNED', 'tokenize': False,
                              'add_generation_prompt': True, 'enable_thinking': False}
            return '\nCHAT ' + messages[0]['content'] + (' drift' if r.render_drift else '') + '\n'

        def __call__(self, text, *, add_special_tokens, return_tensors=None):
            assert add_special_tokens is False
            ids = list(range(r.tokens))
            if r.token_drift:
                ids[-1] += 1
            if return_tensors:
                assert return_tensors == 'pt'
                return {'input_ids': Tensor([ids]), 'attention_mask': Tensor([[1] * len(ids)])}
            return {'input_ids': ids}

        def decode(self, ids, **kwargs):
            assert kwargs == {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}
            return '  RAW\n'

    class Model:
        dtype = 'bf16'
        device = 'cuda:0'

        def eval(self):
            if r.fail_eval:
                raise RuntimeError('eval failed')
            return self

        def generate(self, **kwargs):
            r.events.append(('generate', kwargs))
            if r.fail_generate:
                raise RuntimeError('generation failed')
            prefix = list(kwargs['input_ids'].rows[0])
            if r.output_prefix_drift:
                prefix[0] += 1
            return Tensor([prefix + [99] * r.suffix])

    def tokenizer_load(path, **kwargs):
        r.events.append(('tokenizer', path, kwargs))
        if r.tokenizer_failure:
            raise OSError('tokenizer failure')
        return Tokenizer()

    def model_load(path, **kwargs):
        r.events.append(('model', path, kwargs))
        r.allocated, r.reserved = 100, 200
        if r.fail_load:
            raise MemoryError('partial allocation')
        m = Model()
        m.dtype = r.dtype
        return m

    def empty_cache():
        r.events.append(('empty_cache',))
        r.reserved = 0
        if r.release_on_empty:
            r.allocated = 0

    def clear_workspace():
        r.events.append(('private_clear',))
        if r.release_on_private:
            r.allocated = 0

    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
        memory_allocated=lambda device: r.allocated, memory_reserved=lambda device: r.reserved,
        synchronize=lambda device: r.events.append(('sync',)), empty_cache=empty_cache,
        manual_seed_all=lambda seed: r.events.append(('cuda_seed', seed)))
    r.torch = SimpleNamespace(__version__='2.5.1+cu121', version=SimpleNamespace(cuda='12.1'),
        bfloat16='bf16', cuda=cuda, inference_mode=nullcontext,
        manual_seed=lambda seed: r.events.append(('seed', seed)),
        use_deterministic_algorithms=lambda enabled: r.events.append(('deterministic', enabled)),
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, deterministic=False)),
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear_workspace))
    r.transformers = SimpleNamespace(__version__='5.9.0',
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load))
    r.imports = []

    def modules(name):
        r.imports.append(name)
        return getattr(r, name)

    r.backend = backend.QwenReadoutBackend(tmp_path, GPU, PIN,
        lambda: {'gpu_uuid': r.uuid}, modules_loader=modules)
    r.request = PublicRequest('request-1', 'visible only')
    return r


def prepared_loaded(r):
    p = r.backend.prepare(r.request)
    r.backend.load()
    return p


def test_prepare_is_lazy_single_offline_tokenizer_and_public_dataclass(rig):
    r = rig
    assert r.imports == []
    p = r.backend.prepare(r.request)
    assert type(p) is PreparedInput and p.template_sha256 == PIN and p.input_tokens == 7
    assert p.rendered_text == '\nCHAT visible only\n'
    assert r.backend.prepare(r.request) == p
    assert r.imports == ['transformers']
    loads = [e for e in r.events if e[0] == 'tokenizer']
    assert len(loads) == 1
    assert loads[0][2] == {'revision': backend.REVISION, 'local_files_only': True, 'trust_remote_code': False}
    assert r.backend.observations['tokenizer_loads'] == 1


def test_offline_bf16_eager_single_load_and_raw_decode(rig):
    r = rig
    p = prepared_loaded(r)
    call = next(e for e in r.events if e[0] == 'model')
    assert call[2] == {'revision': backend.REVISION, 'local_files_only': True,
        'trust_remote_code': False, 'torch_dtype': 'bf16', 'attn_implementation': 'eager', 'device_map': {'': 0}}
    assert ('deterministic', True) in r.events and ('seed', 0) in r.events
    output = r.backend.generate(p)
    assert type(output) is GenerationOutput
    assert output == GenerationOutput('  RAW\n', 7, 3)
    with pytest.raises(RuntimeError):
        r.backend.load()
    assert len([e for e in r.events if e[0] == 'model']) == 1
    obs = r.backend.close()
    assert obs['load_attempts'] == obs['generation_attempts'] == obs['tokenizer_loads'] == 1
    assert obs['close']['allocated_before'] == 100 and obs['close']['allocated_after'] == 0
    assert obs['close']['reserved_before'] == 200 and obs['close']['reserved_after'] == 0
    assert 'visible only' not in repr(obs) and 'RAW' not in repr(obs)


@pytest.mark.parametrize('flag', ['fail_load', 'fail_eval'])
def test_partial_load_can_always_be_closed_without_retry(rig, flag):
    r = rig
    r.backend.prepare(r.request)
    setattr(r, flag, True)
    with pytest.raises((MemoryError, RuntimeError)):
        r.backend.load()
    with pytest.raises(RuntimeError):
        r.backend.load()
    obs = r.backend.close()
    assert obs['load_attempts'] == 1 and obs['generation_attempts'] == 0
    assert obs['close']['allocated_after'] == 0
    assert r.backend._model is r.backend._tokenizer is r.backend._torch is None


@pytest.mark.parametrize('phase', ['load', 'generate'])
def test_changed_device_blocks_model_or_generation(rig, phase):
    r = rig
    p = r.backend.prepare(r.request)
    if phase == 'generate':
        r.backend.load()
    r.uuid = 'GPU-other'
    with pytest.raises(RuntimeError, match='device'):
        r.backend.load() if phase == 'load' else r.backend.generate(p)
    assert not any(e[0] == ('model' if phase == 'load' else 'generate') for e in r.events)
    r.backend.close()


@pytest.mark.parametrize('name,value', [('HF_HUB_OFFLINE', '0'), ('TRANSFORMERS_OFFLINE', '0'),
    ('CUDA_VISIBLE_DEVICES', '0'), ('CUDA_VISIBLE_DEVICES', GPU + ',GPU-other')])
def test_environment_gate_before_model(rig, monkeypatch, name, value):
    r = rig
    r.backend.prepare(r.request)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError):
        r.backend.load()
    assert not any(e[0] == 'model' for e in r.events)
    r.backend.close()


def test_bad_expected_template_rejected_without_import(rig):
    with pytest.raises(ValueError):
        backend.QwenReadoutBackend(rig.backend.snapshot, GPU, '0' * 64, lambda: {'gpu_uuid': GPU})


def test_bad_actual_template_blocks_preparation(rig):
    rig.template = 'changed'
    with pytest.raises(ValueError, match='template'):
        rig.backend.prepare(rig.request)
    assert not any(e[0] == 'model' for e in rig.events)


@pytest.mark.parametrize('tokens', [0, 4033])
def test_input_token_budget(rig, tokens):
    rig.tokens = tokens
    with pytest.raises(ValueError, match='token'):
        rig.backend.prepare(rig.request)


def test_context_boundary_4032_plus_64(rig):
    rig.tokens, rig.suffix = 4032, 64
    p = prepared_loaded(rig)
    out = rig.backend.generate(p)
    assert out.input_tokens + out.generated_tokens == 4096
    rig.backend.close()


@pytest.mark.parametrize('dtype', ['fp16', 'fp32'])
def test_non_bf16_model_rejected_and_cleaned(rig, dtype):
    rig.dtype = dtype
    rig.backend.prepare(rig.request)
    with pytest.raises(RuntimeError, match='dtype'):
        rig.backend.load()
    assert rig.backend.close()['close']['allocated_after'] == 0


@pytest.mark.parametrize('settings', [{'max_new_tokens': 63}, {'max_new_tokens': 65}, {'seed': 1},
    {'seed': False}, {'do_sample': True}, {'do_sample': 0}, {'num_beams': 2}, {'num_beams': True}])
def test_exact_decoding_settings_required(rig, settings):
    p = prepared_loaded(rig)
    with pytest.raises(ValueError, match='settings'):
        rig.backend.generate(p, **settings)
    assert not any(e[0] == 'generate' for e in rig.events)
    rig.backend.close()


@pytest.mark.parametrize('change', ['render', 'tokens', 'template', 'tamper', 'count'])
def test_render_template_and_token_bindings_revalidated(rig, change):
    p = prepared_loaded(rig)
    if change == 'render':
        rig.render_drift = True
    elif change == 'tokens':
        rig.token_drift = True
    elif change == 'template':
        rig.template = 'changed'
    elif change == 'tamper':
        p = replace(p, rendered_text=p.rendered_text + ' tampered')
    else:
        p = replace(p, input_tokens=p.input_tokens + 1)
    with pytest.raises((ValueError, RuntimeError)):
        rig.backend.generate(p)
    assert not any(e[0] == 'generate' for e in rig.events)
    rig.backend.close()


def test_failed_generation_cannot_be_retried(rig):
    p = prepared_loaded(rig)
    rig.fail_generate = True
    with pytest.raises(RuntimeError):
        rig.backend.generate(p)
    rig.fail_generate = False
    with pytest.raises(RuntimeError):
        rig.backend.generate(p)
    assert rig.backend.observations['generation_attempts'] == 1
    assert len([e for e in rig.events if e[0] == 'generate']) == 1
    rig.backend.close()


def test_successful_request_cannot_be_replayed_and_max_four(rig):
    p = prepared_loaded(rig)
    rig.backend.generate(p)
    with pytest.raises(RuntimeError):
        rig.backend.generate(p)
    for i in range(2, 5):
        p = rig.backend.prepare(PublicRequest(f'request-{i}', 'visible only'))
        rig.backend.generate(p)
    with pytest.raises(RuntimeError):
        rig.backend.prepare(PublicRequest('request-5', 'visible only'))
    assert rig.backend.observations['generation_attempts'] == 4
    rig.backend.close()


def test_generated_suffix_overflow_is_failure_not_truncation(rig):
    p = prepared_loaded(rig)
    rig.suffix = 65
    with pytest.raises(RuntimeError, match='token'):
        rig.backend.generate(p)
    with pytest.raises(RuntimeError):
        rig.backend.generate(p)
    rig.backend.close()


def test_known_private_workspace_cleanup_is_process_local(rig):
    prepared_loaded(rig)
    rig.release_on_empty = False
    obs = rig.backend.close()
    assert obs['close']['private_workspace_clear_used'] is True
    assert obs['close']['allocated_after'] == 0
    assert sum(e[0] == 'private_clear' for e in rig.events) == 1
    assert sum(e[0] == 'sync' for e in rig.events) >= 3


@pytest.mark.parametrize('drift', ['torch', 'cuda', 'missing'])
def test_private_cleanup_never_called_on_unverified_version(rig, drift):
    prepared_loaded(rig)
    rig.release_on_empty = False
    if drift == 'torch':
        rig.torch.__version__ = '2.6.0'
    elif drift == 'cuda':
        rig.torch.version.cuda = '12.4'
    else:
        del rig.torch._C._cuda_clearCublasWorkspaces
    with pytest.raises(RuntimeError, match='allocated'):
        rig.backend.close()
    assert not any(e[0] == 'private_clear' for e in rig.events)
    assert rig.backend.observations['close']['allocated_after'] == 100
    assert rig.backend._model is rig.backend._tokenizer is rig.backend._torch is None


def test_remaining_allocated_memory_fails_with_observations(rig):
    prepared_loaded(rig)
    rig.release_on_empty = rig.release_on_private = False
    with pytest.raises(RuntimeError, match='allocated'):
        rig.backend.close()
    assert rig.backend.observations['close']['allocated_after'] == 100


def test_close_before_load_and_idempotent_close(rig):
    obs = rig.backend.close()
    assert rig.imports == []
    assert obs['close']['cuda_cleanup_performed'] is False
    assert rig.backend.close() == obs
    with pytest.raises(RuntimeError):
        rig.backend.prepare(rig.request)


@pytest.mark.parametrize('module,version', [('torch', '2.6.0'), ('transformers', '5.8.0')])
def test_runtime_version_gate(rig, module, version):
    rig.backend.prepare(rig.request)
    getattr(rig, module).__version__ = version
    with pytest.raises(RuntimeError, match='version'):
        rig.backend.load()
    assert not any(e[0] == 'model' for e in rig.events)
    rig.backend.close()


def test_prepare_only_qualification_does_not_import_torch_or_touch_cuda(rig, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    rig.torch.cuda.is_available = lambda: pytest.fail('prepare-only must not inspect CUDA')
    rig.backend.prepare(rig.request)
    obs = rig.backend.close()
    assert rig.imports == ['transformers']
    assert obs['load_attempts'] == obs['generation_attempts'] == 0
    assert obs['close']['cuda_cleanup_performed'] is False
    assert obs['allocated_bytes_after'] is None


def test_tokenizer_failure_never_retries_load(rig):
    rig.tokenizer_failure = True
    with pytest.raises(OSError):
        rig.backend.prepare(rig.request)
    rig.tokenizer_failure = False
    with pytest.raises(RuntimeError):
        rig.backend.prepare(rig.request)
    obs = rig.backend.close()
    assert obs['tokenizer_loads'] == 1
    assert sum(e[0] == 'tokenizer' for e in rig.events) == 1


def test_template_file_fallback_is_used_without_newline_normalization(rig):
    rig.backend.prepare(rig.request)
    rig.backend._tokenizer.get_chat_template = None
    (rig.backend.snapshot / 'chat_template.jinja').write_bytes(b'PINNED')
    assert rig.backend.prepare(rig.request).template_sha256 == PIN
    (rig.backend.snapshot / 'chat_template.jinja').write_bytes(b'PINNED\r\n')
    with pytest.raises(ValueError, match='template'):
        rig.backend.prepare(rig.request)
    rig.backend.close()


def test_output_input_prefix_drift_is_a_failure(rig):
    p = prepared_loaded(rig)
    rig.output_prefix_drift = True
    with pytest.raises(RuntimeError, match='prefix'):
        rig.backend.generate(p)
    rig.backend.close()


def test_cleanup_failure_still_clears_refs_and_measures_memory(rig):
    prepared_loaded(rig)

    def fail_sync(device):
        raise RuntimeError('synchronization failure')

    rig.torch.cuda.synchronize = fail_sync
    with pytest.raises(RuntimeError, match='synchronization'):
        rig.backend.close()
    assert rig.backend._model is rig.backend._tokenizer is rig.backend._torch is None
    assert rig.backend.observations['close']['status'] == 'FAILED'
    assert rig.backend.observations['allocated_bytes_after'] == 100
    with pytest.raises(RuntimeError):
        rig.backend.close()


@pytest.mark.parametrize('unexpected', [False, 0, 'unexpected'])
def test_private_workspace_clear_requires_none_return(rig, unexpected):
    prepared_loaded(rig)
    rig.release_on_empty = False

    def unexpected_clear():
        rig.events.append(('private_clear',))
        rig.allocated = 0
        return unexpected

    rig.torch._C._cuda_clearCublasWorkspaces = unexpected_clear
    with pytest.raises(RuntimeError, match='workspace clear return'):
        rig.backend.close()
    assert rig.backend.observations['close']['status'] == 'FAILED'
    assert rig.backend.observations['close']['private_workspace_clear_used'] is True
    assert rig.backend.observations['allocated_bytes_after'] == 0
    assert sum(e[0] == 'private_clear' for e in rig.events) == 1
    assert rig.backend._model is rig.backend._tokenizer is rig.backend._torch is None
