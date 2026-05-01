"""
MemUpdateBench Shared Utilities
"""

import random
import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_kendall_tau(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute Kendall's τ rank correlation between two score lists."""
    from scipy.stats import kendalltau
    tau, _ = kendalltau(scores_a, scores_b)
    return tau if not np.isnan(tau) else 0.0


def load_model_and_tokenizer(
    model_name: str,
    use_qlora: bool = True,
    load_in_4bit: bool = False,
    torch_dtype: str = "bfloat16",
    use_accelerate: bool = False,
):
    """Load a causal LM model and tokenizer with optional QLoRA.

    Args:
        model_name: HuggingFace model name or local path.
        use_qlora: Whether to use QLoRA 4-bit quantization.
        load_in_4bit: Whether to load in 4-bit (alias for use_qlora).
        torch_dtype: Torch dtype string ("bfloat16", "float16", "float32").
        use_accelerate: If True, skip device_map so that accelerate/DDP
            can handle device placement. Required for multi-GPU training
            via `accelerate launch`.
    """
    # Workaround for PEFT/Transformers blindly checking bitsandbytes version
    # even when qlora is disabled, which crashes if it's uninstalled or broken.
    import importlib.metadata
    _orig_version = importlib.metadata.version
    def _mocked_version(pkg):
        if pkg == "bitsandbytes":
            return "0.41.1"  # Fake version
        return _orig_version(pkg)
    importlib.metadata.version = _mocked_version

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map.get(torch_dtype, torch.bfloat16),
    }

    if use_qlora or load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb_config
    elif not use_accelerate:
        # Only set device_map="auto" for single-GPU / non-DDP usage.
        # When using accelerate launch (multi-GPU DDP), accelerate handles
        # device placement and device_map="auto" would conflict.
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    logger.info(f"Loaded model: {model_name} (QLoRA={use_qlora}, accelerate={use_accelerate})")

    return model, tokenizer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Generate text from a prompt using the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=model.config.max_position_embeddings - max_new_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
