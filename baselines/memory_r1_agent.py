"""
Baseline 2: Memory-R1 Agent.
RL-trained memory management with external QA F1 reward.

Reference: Chen et al., "Memory-R1: Enhancing Large Language Model Agents
to Manage and Utilize Memories via Reinforcement Learning", 2025.
Related: syr-cn/ReMemR1 (ICLR 2026)

Core idea:
  - RL (GRPO/PPO) trains Memory Manager on CRUD decisions
  - Reward = external QA F1 score ONLY
  - No self-reward, no environment grounding, no consolidation
  - Essentially: our framework MINUS self-reward + consolidation

This is the most direct "upper-bound" baseline because it uses
external annotation (QA F1) — which MemUpdateBench aims to eliminate.
"""

import os
import json

from baselines.base_agent import BaseAgent
from loguru import logger


class MemoryR1Agent(BaseAgent):
    """Memory-R1: RL-trained CRUD with external QA F1 reward."""

    name = "memory_r1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_store = None
        self.memory_manager = None

    def initialize(self, model=None, tokenizer=None, **kwargs):
        """Initialize using our existing MemoryStore + MemoryManager."""
        super().initialize(model, tokenizer, **kwargs)

        from mub.memory.store import MemoryStore
        from mub.manager.memory_manager import MemoryManager

        self.memory_store = MemoryStore()
        
        self.memory_manager = MemoryManager(
            model=self.model, 
            tokenizer=self.tokenizer, 
            memory_store=self.memory_store
        )

    def process_event(self, event: str, context: str = "") -> dict:
        """Process event using RL-trained Memory Manager."""
        self.total_events_processed += 1

        # Get CRUD decision from Memory Manager
        operation_str, prompt = self.memory_manager.decide(event, context)

        # Parse and execute
        op_type, target_id, content = self.memory_manager._parse_operation(operation_str, event)

        self._execute_crud(op_type, content, event)

        return {
            "operation": op_type,
            "details": content,
        }

    def answer_question(self, question: str) -> str:
        """Answer question using Memory-R1's retrieval approach.

        Memory-R1 uses a separate Answer Agent that retrieves
        relevant memories and generates an answer.
        """
        # Retrieve relevant memories
        relevant = self.memory_store.retrieve(question, topk=5)
        memory_context = "\n".join(
            f"- {entry.content}" for entry, score in relevant
        )

        prompt = (
            "You are an Answer Agent with access to a memory bank.\n\n"
        )
        if memory_context:
            prompt += f"### Retrieved Memories:\n{memory_context}\n\n"
        prompt += (
            f"### Question:\n{question}\n\n"
            "### Answer (be concise, use only information from memories):\n"
        )

        return self._generate(prompt, max_new_tokens=100, temperature=0.1).strip()

    def train_step(self, reward: float, event: str = "",
                   context: str = "", **kwargs) -> dict:
        """Train Memory Manager with QA F1 reward (REINFORCE).

        In full Memory-R1, this uses PPO/GRPO via TRL.
        Here we use REINFORCE for simplicity (same as our Phase 1 fallback).
        """
        import torch

        if not event:
            return {"trained": False}

        # Re-generate decision to get gradients
        operation_str, prompt = self.memory_manager.decide(event, context)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            return {"trained": False}

        inputs = self.tokenizer(
            prompt + operation_str,
            return_tensors="pt", truncation=True,
            max_length=1024, padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        prompt_len = len(self.tokenizer.encode(prompt, truncation=True, max_length=1024))
        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        outputs = self.model(**inputs, labels=labels)
        loss = -reward * outputs.loss  # REINFORCE

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

        return {
            "trained": True,
            "loss": loss.item(),
            "reward": reward,
        }

    def reset(self):
        """Reset memory store for new episode."""
        if self.memory_store:
            from mub.memory.store import MemoryStore
            self.memory_store = MemoryStore()
            if hasattr(self, 'memory_manager') and self.memory_manager:
                self.memory_manager.store = self.memory_store

    def get_memory_contents(self) -> list[str]:
        if self.memory_store:
            return [e.content for e in self.memory_store.entries.values()]
        return []

    def _execute_crud(self, op_type: str, content: str, event: str):
        """Execute CRUD operation on MemoryStore."""
        if op_type == "ADD" and content:
            self.memory_store.add(
                content=content,
                keywords=content.split()[:5],
                tags=["memory_r1"],
                source=event
            )

        elif op_type == "UPDATE" and content:
            # Find most relevant entry and update
            results = self.memory_store.retrieve(content, topk=1)
            if results:
                entry, _ = results[0]
                self.memory_store.update(entry.id, content)

        elif op_type == "DELETE" and content:
            results = self.memory_store.retrieve(content, topk=1)
            if results:
                entry, _ = results[0]
                self.memory_store.delete(entry.id)

    def save(self, path: str):
        super().save(path)
        if self.memory_store:
            self.memory_store.save(os.path.join(path, "memory_store.json"))

    def load(self, path: str):
        store_path = os.path.join(path, "memory_store.json")
        if os.path.exists(store_path) and self.memory_store:
            self.memory_store.load(store_path)
