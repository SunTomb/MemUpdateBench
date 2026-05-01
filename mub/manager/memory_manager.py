"""
RL-Based Memory Manager.
Based on Memory-R1 architecture (Chen et al., 2025) with key modification:
reward signal replaced from external QA F1 to MemUpdateBench's grounded composite reward.

Action Space: ADD / UPDATE <id> / DELETE <id> / NOOP
Training: PPO or GRPO via TRL library.
"""

from __future__ import annotations
import re
from typing import Optional, Literal

from loguru import logger

from mub.config import RLConfig, MemoryConfig
from mub.memory.store import MemoryStore
from mub.memory.entry import MemoryEntry


MemoryOp = Literal["ADD", "UPDATE", "DELETE", "NOOP"]


class MemoryManager:
    """RL-based Memory Manager Agent.

    Decides how to manage memories in response to new events.
    The RL policy is trained to optimize the grounded composite reward.

    Architecture mirrors Memory-R1's Memory Manager but replaces
    the reward source from QA F1 to MemUpdateBench's R_total.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        memory_store: Optional[MemoryStore] = None,
        rl_config: Optional[RLConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.store = memory_store or MemoryStore(memory_config)
        self.rl_config = rl_config or RLConfig()
        self.operation_history: list[dict] = []

    # ---- Core Decision Making ----

    def decide(self, new_event: str, task_context: str = "",
               max_new_tokens: int = 80,
               temperature: float = 0.7) -> tuple[str, str]:
        """Given a new event, decide the memory operation.

        Args:
            new_event: New information/experience to process.
            task_context: Current task or conversation context.
            max_new_tokens: Max tokens to generate (reduced from 256 to 80
                for speed; CRUD operations rarely exceed 50 tokens).
            temperature: Generation temperature; eval should use near-deterministic values.

        Returns:
            (operation_str, formatted_prompt) — e.g., ("ADD new fact...", prompt)
            operation_str is the raw model output describing the operation.
        """
        # Retrieve relevant existing memories for context
        relevant = self.store.retrieve(new_event, topk=self.store.config.retrieval_topk)
        relevant_entries = [entry for entry, score in relevant]

        prompt = self._build_manager_prompt(new_event, relevant_entries, task_context)

        from mub.utils import generate_text
        output = generate_text(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature
        )

        return output.strip(), prompt

    def execute_operation(self, operation_str: str,
                          new_event: str,
                          env_reward: float = 0.0,
                          slot_meta: dict = None) -> dict:
        """Parse and execute the model's memory operation decision.

        Args:
            operation_str: Model output, e.g., "ADD: User prefers tea over coffee"
            new_event: The original event that triggered this operation.
            env_reward: Environment reward at this timestep.

        Returns:
            Dict with operation details for logging.
        """
        op, target_id, content = self._parse_operation(operation_str, new_event)

        result = {"op": op, "target_id": target_id, "content": content,
                  "success": False}

        if op == "ADD":
            entry = self.store.add(
                content=content, env_reward=env_reward,
                source=new_event[:100], slot_meta=slot_meta
            )
            result["success"] = True
            result["entry_id"] = entry.id

        elif op == "UPDATE" and target_id:
            resolved_id, resolved_via = self._resolve_target_id(target_id, new_event, slot_meta)
            result["resolved_via"] = resolved_via
            if resolved_id:
                entry = self.store.update(resolved_id, content, env_reward, slot_meta=slot_meta)
                result["success"] = entry is not None
                result["entry_id"] = resolved_id
                result["requested_id"] = target_id
            elif len(self.store.entries) == 0:
                entry = self.store.add(
                    content=content or new_event, env_reward=env_reward,
                    source=new_event[:100]
                )
                result["op"] = "ADD"
                result["success"] = True
                result["entry_id"] = entry.id
                result["requested_id"] = target_id

        elif op == "DELETE" and target_id:
            resolved_id, resolved_via = self._resolve_target_id(target_id, new_event, slot_meta)
            result["resolved_via"] = resolved_via
            if resolved_id:
                result["success"] = self.store.delete(resolved_id)
                result["entry_id"] = resolved_id
                result["requested_id"] = target_id

        elif op == "NOOP":
            result["success"] = True

        self.operation_history.append(result)
        logger.debug(f"Executed: {op} → success={result['success']}")
        return result

    # ---- Prompt Construction ----

    def _build_manager_prompt(self, new_event: str,
                              history: list[MemoryEntry],
                              task_context: str = "") -> str:
        """Build the Memory Manager's decision prompt."""
        memory_str = "\n".join([
            f"  [{e.id}] (conf={e.confidence:.2f}) {e.content}"
            for e in history
        ]) if history else "(empty memory)"

        prompt = (
            "You are a Memory Manager for an AI agent. "
            "Given the current memory entries and a new event, "
            "decide the best memory operation.\n\n"
            "### Available Operations\n"
            "- ADD: <content> — Store new important information\n"
            "- UPDATE <id>: <new_content> — Update existing memory\n"
            "- DELETE <id> — Remove outdated/wrong memory\n"
            "- NOOP — No action needed\n\n"
        )

        if task_context:
            prompt += f"### Current Task Context\n{task_context[:300]}\n\n"

        prompt += (
            f"### Current Memory Entries\n{memory_str}\n\n"
            f"### New Event\n{new_event}\n\n"
            "### Decision\n"
            "Think about whether this event contains new information worth "
            "storing, updates existing knowledge, or contradicts stored facts. "
            "Output your operation:\n"
        )
        return prompt

    def _parse_operation(self, output: str,
                         fallback_content: str) -> tuple[MemoryOp, str, str]:
        """Parse model output into (op, target_id, content).

        Handles formats:
        - "ADD: some content"
        - "UPDATE abc123: new content"
        - "DELETE abc123"
        - "NOOP" or anything else
        """
        output = output.strip()
        upper = output.upper()

        if upper.startswith("ADD"):
            content = output.split(":", 1)[1].strip() if ":" in output else fallback_content
            return "ADD", "", self._clean_operation_content(content, fallback_content)

        elif upper.startswith("UPDATE"):
            parts = output.split(":", 1)
            if len(parts) == 2:
                # "UPDATE abc123: new content"
                id_part = self._normalize_target_id(parts[0].replace("UPDATE", ""))
                content = self._clean_operation_content(parts[1], fallback_content)
                return "UPDATE", id_part, content
            return "NOOP", "", ""

        elif upper.startswith("DELETE"):
            target_id = self._normalize_target_id(output.replace("DELETE", ""))
            return "DELETE", target_id, ""

        else:
            return "NOOP", "", ""

    @staticmethod
    def parse_constrained_slot_operation(output: str) -> dict:
        text = output.strip().split("\n", 1)[0].strip()
        upper = text.upper()
        if upper == "NOOP":
            return {"operation": "NOOP", "entity": "", "attribute": "", "value": ""}

        match = re.match(
            r"^(ADD|UPDATE)\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return {"operation": "INVALID", "entity": "", "attribute": "", "value": ""}

        operation, entity, attribute, value = match.groups()
        value = value.strip().rstrip(".")
        value = re.sub(r"\s+(?:NOOP|ADD|UPDATE|DELETE)\s*$", "", value, flags=re.IGNORECASE).rstrip(".").strip()
        return {
            "operation": operation.upper(),
            "entity": entity.lower(),
            "attribute": attribute.lower(),
            "value": value,
        }

    def _normalize_target_id(self, raw_id: str) -> str:
        target_id = raw_id.strip().rstrip(".").strip()
        if target_id.lower().startswith("id_"):
            target_id = target_id[3:]
        elif target_id.lower().startswith("id "):
            target_id = target_id[3:].strip()
        return target_id

    def _resolve_target_id(self, target_id: str, new_event: str,
                           slot_meta: dict = None) -> tuple[str, str]:
        if target_id in self.store.entries:
            return target_id, "direct_id"
        if slot_meta:
            slot_entry = self.store.get_latest_by_slot(
                slot_meta.get("entity", ""), slot_meta.get("attribute", "")
            )
            if slot_entry:
                return slot_entry.id, "slot"
        matches = self.store.retrieve(new_event, topk=1)
        if matches:
            return matches[0][0].id, "semantic"
        return "", "none"

    def _clean_operation_content(self, content: str, fallback_content: str) -> str:
        cleaned = content.strip()
        stop_markers = [
            "\nADD", "\nUPDATE", "\nDELETE", "\nNOOP",
            "\n###", "\nExplanation:", "\nOperations:", "\nDecision:",
        ]
        for marker in stop_markers:
            idx = cleaned.find(marker)
            if idx >= 0:
                cleaned = cleaned[:idx].strip()
        return cleaned or fallback_content

    # ---- SFT Data Generation ----

    @staticmethod
    def generate_sft_examples(
        events: list[str],
        operations: list[str],
    ) -> list[dict]:
        """Generate SFT training examples for Phase 0 warmup.

        Args:
            events: List of event texts.
            operations: Corresponding correct operations.

        Returns:
            List of {"prompt": ..., "completion": ...} dicts.
        """
        examples = []
        for event, op in zip(events, operations):
            prompt = (
                "You are a Memory Manager. Given the new event, "
                "decide the operation.\n\n"
                f"New Event: {event}\n\n"
                "Decision:\n"
            )
            examples.append({"prompt": prompt, "completion": op})
        return examples

    # ---- Exploration ----

    def decide_with_exploration(
        self, new_event: str, task_context: str = "", epsilon: float = 0.0
    ) -> tuple[str, str, bool]:
        """Decide with ε-greedy exploration.

        With probability ε, returns a random operation (biased toward
        active ops to break NOOP dominance). Otherwise, runs the model.

        Returns:
            (operation_str, prompt, was_exploration)
        """
        import random

        prompt = ""
        explore_roll = random.random()

        if explore_roll < epsilon and self.store.size() > 0:
            # Random exploration — pick a non-NOOP action
            ops = ["ADD", "UPDATE", "DELETE", "NOOP"]
            # Weight active operations higher during exploration
            weights = [0.35, 0.30, 0.15, 0.20]
            chosen_op = random.choices(ops, weights=weights, k=1)[0]

            if chosen_op == "ADD":
                # ADD the event content directly
                operation_str = f"ADD: {new_event[:200]}"
            elif chosen_op == "UPDATE":
                # Pick a random existing memory to update
                mem_ids = list(self.store.entries.keys())
                if mem_ids:
                    target_id = random.choice(mem_ids)
                    operation_str = f"UPDATE {target_id}: {new_event[:200]}"
                else:
                    operation_str = f"ADD: {new_event[:200]}"
            elif chosen_op == "DELETE":
                # Pick a random existing memory to delete
                mem_ids = list(self.store.entries.keys())
                if mem_ids:
                    target_id = random.choice(mem_ids)
                    operation_str = f"DELETE {target_id}"
                else:
                    operation_str = "NOOP"
            else:
                operation_str = "NOOP"

            # Still build the prompt for log_prob computation
            relevant = self.store.retrieve(new_event, topk=self.store.config.retrieval_topk)
            relevant_entries = [entry for entry, score in relevant]
            prompt = self._build_manager_prompt(new_event, relevant_entries, task_context)
            return operation_str, prompt, True

        elif explore_roll < epsilon:
            # Memory is empty — force ADD during exploration
            relevant = self.store.retrieve(new_event, topk=self.store.config.retrieval_topk)
            relevant_entries = [entry for entry, score in relevant]
            prompt = self._build_manager_prompt(new_event, relevant_entries, task_context)
            operation_str = f"ADD: {new_event[:200]}"
            return operation_str, prompt, True

        # Normal model-based decision
        operation_str, prompt = self.decide(new_event, task_context)
        return operation_str, prompt, False


    def compute_action_log_prob(
        self, prompt: str, action_str: str
    ) -> "torch.Tensor":
        """Compute log π(action | prompt) for REINFORCE.

        This runs a forward pass through the model and computes the
        log probability of the action tokens given the prompt.

        Returns:
            Scalar tensor: sum of log probs of action tokens.
        """
        import torch
        import torch.nn.functional as F

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        action_ids = self.tokenizer.encode(action_str, add_special_tokens=False)

        if len(action_ids) == 0:
            action_ids = self.tokenizer.encode("NOOP", add_special_tokens=False)

        # Truncate prompt to keep total under 512 tokens (F10)
        max_prompt_len = 512 - len(action_ids)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        full_ids = torch.tensor(
            prompt_ids + action_ids, dtype=torch.long
        ).unsqueeze(0).to(self.model.device)

        with torch.set_grad_enabled(True):
            outputs = self.model(full_ids)
            logits = outputs.logits[0]

            prompt_len = len(prompt_ids)
            # Logits at positions [prompt_len-1 : prompt_len-1+len(action_ids)]
            # predict tokens at positions [prompt_len : prompt_len+len(action_ids)]
            response_logits = logits[prompt_len - 1: prompt_len - 1 + len(action_ids)]
            response_targets = torch.tensor(
                action_ids, dtype=torch.long
            ).to(self.model.device)

            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs.gather(
                1, response_targets.unsqueeze(1)
            ).squeeze(1)

        return token_log_probs.sum()

    # ---- Statistics ----

    def get_operation_stats(self) -> dict:
        """Get counts of each operation type."""
        stats = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0, "total": 0}
        for record in self.operation_history:
            op = record.get("op", "NOOP")
            stats[op] = stats.get(op, 0) + 1
            stats["total"] += 1
        return stats
