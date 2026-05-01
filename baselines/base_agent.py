"""
Abstract Base Agent for Baseline Comparisons.
All baseline agents implement this interface so they can be evaluated
by the unified evaluation harness (`eval_baselines.py`).
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for all baseline agents.

    Every baseline must implement:
      - process_event(): handle a new event (dialogue turn / environment obs)
      - answer_question(): produce a prediction for a QA query
      - reset(): clear state between episodes

    Optionally:
      - train_step(): perform one training step (for RL-based baselines)
      - save / load: checkpoint management
    """

    name: str = "base"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_memories: int = 200, fast_mode: bool = False, **kwargs):
        self.model_name = model_name
        self.max_memories = max_memories
        self.fast_mode = fast_mode
        self.model = None
        self.tokenizer = None
        self._initialized = False

        # Counters
        self.total_events_processed = 0
        self.total_tokens_used = 0

    def initialize(self, model=None, tokenizer=None):
        """Initialize model and tokenizer.

        If model/tokenizer are provided, use them directly (allows sharing
        across baselines for fair comparison). Otherwise, load from model_name.
        """
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            from mub.utils import load_model_and_tokenizer
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.model_name, use_qlora=False
            )

        self._initialized = True
        logger.info(f"[{self.name}] Initialized with model: {self.model_name}")

    @abstractmethod
    def process_event(self, event: str, context: str = "") -> dict:
        """Process an incoming event (dialogue turn, observation, etc.).

        Args:
            event: The new event text.
            context: Optional task context or question.

        Returns:
            dict with at least:
              - "operation": str (what memory operation was performed)
              - "details": str (operation details)
        """
        ...

    @abstractmethod
    def answer_question(self, question: str) -> str:
        """Answer a question using the agent's current memory state.

        Args:
            question: The question to answer.

        Returns:
            Predicted answer string.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset agent state for a new episode.

        Subclasses should clear memory stores, episodic buffers, etc.
        but keep model weights intact.
        """
        ...

    def train_step(self, reward: float, **kwargs) -> dict:
        """Perform one training step (for RL-based baselines).

        Default: no-op. Override in RL-based baselines.

        Returns:
            dict with training stats (loss, grad_norm, etc.)
        """
        return {"trained": False}

    def get_memory_contents(self) -> list[str]:
        """Return current memory contents as list of strings.

        Used for diagnostics and Judge evaluation.
        """
        return []

    def get_stats(self) -> dict:
        """Return agent statistics for logging."""
        return {
            "name": self.name,
            "total_events": self.total_events_processed,
            "total_tokens": self.total_tokens_used,
            "memory_size": len(self.get_memory_contents()),
        }

    def save(self, path: str):
        """Save agent state to disk."""
        os.makedirs(path, exist_ok=True)
        stats = self.get_stats()
        with open(os.path.join(path, "agent_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str):
        """Load agent state from disk."""
        pass

    def _generate(self, prompt: str, max_new_tokens: int = 256,
                  temperature: float = 0.7) -> str:
        """Generate text using the loaded model.

        Shared utility for all baselines.
        """
        if not self._initialized:
            raise RuntimeError(f"[{self.name}] Must call initialize() first")

        from mub.utils import generate_text
        result = generate_text(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        # Track token usage (approximate)
        self.total_tokens_used += len(self.tokenizer.encode(prompt)) + max_new_tokens
        return result

    def _heuristic_crud(self, event: str, context: str = "") -> dict:
        """Shared rule-based CRUD for fast_mode evaluation.

        Avoids LLM calls entirely. Uses keyword matching to decide
        ADD/UPDATE/NOOP — same logic as Self-Consolidation's heuristic.
        """
        event_lower = event.lower()
        memories = self.get_memory_contents()

        # Check for contradictions → UPDATE
        for i, mem in enumerate(memories):
            mem_lower = mem.lower()
            event_words = set(event_lower.split())
            mem_words = set(mem_lower.split())
            overlap = event_words & mem_words
            if len(overlap) > 3 and event_lower != mem_lower:
                return {"type": "UPDATE", "content": event, "target_idx": i}

        # Check for factual content → ADD
        factual_indicators = [
            "is", "was", "are", "were", "has", "have", "had",
            "lives", "works", "moved", "started", "stopped",
            "prefers", "likes", "dislikes", "wants", "needs",
            "bought", "sold", "married", "born", "died",
            "favorite", "allergic", "speaks", "studies",
        ]
        if any(ind in event_lower for ind in factual_indicators):
            return {"type": "ADD", "content": event}

        return {"type": "NOOP", "content": ""}

    def _execute_heuristic_operation(self, operation: dict, event: str,
                                      memories: list) -> list:
        """Execute a heuristic CRUD operation on a simple list-based store.

        Returns the updated memories list.
        """
        op_type = operation["type"]
        content = operation.get("content", event)

        if op_type == "ADD" and content:
            memories.append(content)
            if len(memories) > self.max_memories:
                memories.pop(0)
        elif op_type == "UPDATE" and content:
            idx = operation.get("target_idx", -1)
            if 0 <= idx < len(memories):
                memories[idx] = content
            elif "->" in content:
                parts = content.split("->")
                old_part = parts[0].strip().lower()
                new_part = parts[1].strip()
                for i, m in enumerate(memories):
                    if old_part in m.lower():
                        memories[i] = new_part
                        break
                else:
                    memories.append(new_part)
            else:
                memories.append(content)
        elif op_type == "DELETE" and content:
            target = content.lower()
            memories[:] = [m for m in memories if target not in m.lower()]

        return memories

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 between prediction and ground truth."""
        from mub.utils import compute_f1
        return compute_f1(prediction, ground_truth)
