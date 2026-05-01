"""
MemUpdateBench Centralized Configuration
All hyperparameters and paths in one place.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Base model configuration."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    judge_model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # Can be separate
    use_qlora: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"
    max_seq_length: int = 4096


@dataclass
class MemoryConfig:
    """Memory store configuration."""
    max_entries: int = 500
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    retrieval_topk: int = 5
    confidence_topk: int = 10       # Top-K confident memories for Judge
    confidence_weights: dict = field(default_factory=lambda: {
        "env_reward_write": 0.4,    # w1: R_env at write time
        "hit_success_ratio": 0.4,   # w2: success rate after retrieval
        "log_age": 0.2,             # w3: log(1 + age)
    })


@dataclass
class RewardConfig:
    """Grounded self-reward configuration."""
    lambda_mem: float = 0.3         # Weight for R_mem (< 1, anchor is R_env)
    judge_max_tokens: int = 256
    reward_scale: float = 1.0       # Scale factor for total reward

    # Phase 2 annealing
    anneal_start_alpha: float = 1.0   # Start: 100% external reward
    anneal_end_alpha: float = 0.0     # End: 0% external reward
    anneal_steps: int = 3000
    tau_threshold: float = 0.5        # Kendall τ threshold to pause annealing


@dataclass
class RLConfig:
    """RL training configuration for Memory Manager."""
    algorithm: str = "grpo"         # "ppo" or "grpo"
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    kl_penalty: float = 0.02
    gamma: float = 0.99
    num_episodes: int = 5000


@dataclass
class LoRAConfig:
    """LoRA configuration for consolidation."""
    # Long-term LoRA (high-rank, preserved knowledge)
    longterm_rank: int = 32
    longterm_alpha: int = 64
    # Short-term LoRA (low-rank, new adaptation)
    shortterm_rank: int = 8
    shortterm_alpha: int = 16
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    # EWC regularization
    ewc_lambda: float = 100.0
    # Training
    consolidation_lr: float = 1e-4
    consolidation_epochs: int = 3


@dataclass
class TriggerConfig:
    """Adaptive consolidation trigger configuration."""
    alpha: float = 0.4              # Weight for conflict index
    beta: float = 0.3               # Weight for reward variance
    gamma: float = 0.3              # Weight for memory growth rate
    theta: float = 0.6              # Trigger threshold
    window_size: int = 20           # K: window for variance computation
    min_interval: int = 50          # Minimum episodes between consolidations


@dataclass
class CompactionConfig:
    """LLM-based memory compaction configuration (v2: replaces LoRA distillation).

    Instead of distilling memories into LoRA parameters, v2 compacts
    the memory store itself by clustering similar entries and merging
    them via LLM summarization. This preserves model weights intact.
    """
    similarity_threshold: float = 0.80   # Cosine sim threshold for clustering
    min_cluster_size: int = 2            # Minimum entries to form a cluster
    max_cluster_size: int = 10           # Max entries per compaction batch
    summary_max_tokens: int = 128        # Max tokens for LLM summary output
    trigger_interval: int = 50           # Compact every N events
    trigger_memory_threshold: int = 100  # Only compact when store > N entries


@dataclass
class MUBConfig:
    """Top-level MemUpdateBench configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)

    # Paths
    output_dir: str = "outputs"
    log_dir: str = "logs"
    data_dir: str = "data"

    # General
    seed: int = 42
    num_gpus: int = 4
    wandb_project: str = "mem_update_bench"

    @classmethod
    def from_yaml(cls, path: str) -> "MUBConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Recursively construct nested dataclasses
        return cls(
            model=ModelConfig(**data.get("model", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            reward=RewardConfig(**data.get("reward", {})),
            rl=RLConfig(**data.get("rl", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            trigger=TriggerConfig(**data.get("trigger", {})),
            compaction=CompactionConfig(**data.get("compaction", {})),
            **{k: v for k, v in data.items()
               if k not in ("model", "memory", "reward", "rl", "lora", "trigger", "compaction")},
        )
