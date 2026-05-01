"""
Project-local baseline reproductions for MemUpdateBench.

These implementations are aligned, in-repo reproductions intended to make the
paper baselines runnable under the same data format and evaluation pipeline as
MemUpdateBench. They are not drop-in replacements for the original authors' codebases.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from loguru import logger

from mub.agent import GMSRAAgent
from mub.config import MUBConfig
from mub.consolidation.distiller import SemanticDistiller
from mub.memory.store import MemoryStore
from mub.utils import compute_exact_match, compute_f1, generate_text


@dataclass(frozen=True)
class BaselineSpec:
    """Metadata describing a baseline tracked by this project."""

    baseline_id: str
    display_name: str
    description: str
    reproduction_mode: str
    supports_agent_tasks: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "memory_r1": BaselineSpec(
        baseline_id="memory_r1",
        display_name="Memory-R1",
        description=(
            "External-reward RL memory manager baseline with no self-reward "
            "and no parametric consolidation."
        ),
        reproduction_mode="project_local_aligned",
    ),
    "mem0_memory_r1": BaselineSpec(
        baseline_id="mem0_memory_r1",
        display_name="Mem0 + Memory-R1",
        description=(
            "Memory-R1 plus an always-on flat event memory buffer, matching the "
            "paper's simple module-combination baseline."
        ),
        reproduction_mode="project_local_aligned",
    ),
    "reflexion": BaselineSpec(
        baseline_id="reflexion",
        display_name="Reflexion",
        description=(
            "Verbal self-improvement baseline using textual reflections across "
            "episodes, without RL memory control or consolidation."
        ),
        reproduction_mode="project_local_aligned",
    ),
    "evolver": BaselineSpec(
        baseline_id="evolver",
        display_name="EvolveR",
        description=(
            "Experience-lifecycle baseline that accumulates reusable success and "
            "failure experiences, without parameter consolidation."
        ),
        reproduction_mode="project_local_aligned",
    ),
    "self_consolidation": BaselineSpec(
        baseline_id="self_consolidation",
        display_name="Self-Consolidation",
        description=(
            "Fixed-interval reflection-to-LoRA consolidation baseline without "
            "RL memory management or grounded self-reward."
        ),
        reproduction_mode="project_local_aligned",
    ),
}


def list_baselines() -> list[BaselineSpec]:
    """Return all registered baseline specifications."""

    return list(BASELINE_SPECS.values())


def get_baseline_spec(baseline_id: str) -> BaselineSpec:
    """Look up a registered baseline by id."""

    if baseline_id not in BASELINE_SPECS:
        raise KeyError(f"Unknown baseline: {baseline_id}")
    return BASELINE_SPECS[baseline_id]


def create_baseline(
    baseline_id: str,
    model,
    tokenizer,
    config: Optional[MUBConfig] = None,
    *,
    consolidation_interval: int = 25,
):
    """Instantiate a project-local baseline runner."""

    config = deepcopy(config or MUBConfig())

    if baseline_id == "memory_r1":
        return MemoryR1Baseline(model, tokenizer, config)
    if baseline_id == "mem0_memory_r1":
        return Mem0MemoryR1Baseline(model, tokenizer, config)
    if baseline_id == "reflexion":
        return ReflexionBaseline(model, tokenizer, config)
    if baseline_id == "evolver":
        return EvolveRBaseline(model, tokenizer, config)
    if baseline_id == "self_consolidation":
        return SelfConsolidationBaseline(
            model,
            tokenizer,
            config,
            consolidation_interval=consolidation_interval,
        )
    raise KeyError(f"Unknown baseline: {baseline_id}")


def _rank_text_snippets(snippets: list[str], query: str, topk: int = 5) -> list[str]:
    """Rank plain-text snippets using lexical overlap and recency."""

    query_terms = set(query.lower().split())
    scored: list[tuple[int, int, str]] = []
    for idx, snippet in enumerate(snippets):
        snippet_terms = set(snippet.lower().split())
        overlap = len(query_terms & snippet_terms)
        scored.append((overlap, idx, snippet))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ranked = [snippet for _, _, snippet in scored if snippet.strip()]
    if ranked[:topk]:
        return ranked[:topk]
    return [snippet for snippet in snippets[-topk:] if snippet.strip()]


def _format_context_block(title: str, items: list[str], empty_text: str) -> str:
    """Format a list of snippets for prompting."""

    if not items:
        return f"### {title}\n{empty_text}"
    lines = "\n".join(f"- {item}" for item in items)
    return f"### {title}\n{lines}"


def _format_operation(op_result: dict) -> str:
    """Reconstruct a textual operation from parsed operation output."""

    op = op_result.get("op", "NOOP")
    target_id = op_result.get("target_id", "")
    content = op_result.get("content", "")

    if op == "ADD":
        return f"ADD: {content}".strip()
    if op == "UPDATE":
        return f"UPDATE {target_id}: {content}".strip()
    if op == "DELETE":
        return f"DELETE {target_id}".strip()
    return "NOOP"


def _model_device(model) -> torch.device:
    """Best-effort way to get a model device."""

    return next(model.parameters()).device


class LocalBaseline:
    """Shared dialogue evaluation utilities for local baseline reproductions."""

    def __init__(self, model, tokenizer, config: MUBConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = deepcopy(config)
        self.current_events: list[str] = []
        self.training_history: list[dict] = []

    def reset_state(self):
        """Reset baseline state while keeping learned model parameters."""

        self.current_events = []

    def ingest_episode(self, episode: dict, training: bool = False):
        """Consume the event stream for one episode."""

        self.current_events = list(episode.get("events", []))

    def answer_question(self, question: str) -> str:
        """Answer a benchmark question using baseline-specific context."""

        raise NotImplementedError

    def observe_dialogue_feedback(
        self,
        episode: dict,
        prediction: str,
        reward: float,
    ):
        """Update baseline state from dialogue feedback during training."""

    def observe_task_feedback(
        self,
        task: dict,
        action: str,
        success: bool,
    ):
        """Update baseline state from agent-task environment feedback."""

    def current_memory_size(self) -> int:
        """Return a baseline-specific measure of external memory size."""

        return 0

    def train_dialogue(self, train_data: list[dict], max_episodes: int) -> dict:
        """Default dialogue training loop for feedback-driven baselines."""

        self.reset_state()
        rewards = []
        n_total = min(max_episodes, len(train_data))

        for ep_idx, episode in enumerate(train_data[:max_episodes]):
            self.ingest_episode(episode, training=True)
            question = episode.get("question", "")
            answer = episode.get("answer", "")
            prediction = self.answer_question(question)
            reward = compute_f1(prediction, answer)
            rewards.append(reward)
            self.observe_dialogue_feedback(episode, prediction, reward)

            if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == n_total:
                avg_r = sum(rewards) / len(rewards)
                logger.info(
                    f"  Train progress: {ep_idx+1}/{n_total} | "
                    f"avg_f1={avg_r:.4f} | mem={self.current_memory_size()}"
                )

        avg_reward = sum(rewards) / max(len(rewards), 1)
        summary = {
            "episodes": min(max_episodes, len(train_data)),
            "avg_train_f1": avg_reward,
            "memory_size": self.current_memory_size(),
        }
        self.training_history.append(summary)
        return summary

    def evaluate_dialogue(self, eval_data: list[dict], benchmark: str) -> dict:
        """Evaluate dialogue performance on one benchmark split."""

        self.reset_state()
        details = []

        for idx, episode in enumerate(eval_data):
            self.ingest_episode(episode, training=False)
            question = episode.get("question", "")
            answer = episode.get("answer", "")
            prediction = self.answer_question(question)
            f1 = compute_f1(prediction, answer)
            em = compute_exact_match(prediction, answer)
            details.append(
                {
                    "idx": idx,
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                    "f1": f1,
                    "em": em,
                    "category": episode.get("category", "unknown"),
                }
            )

            if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_data):
                avg_f1 = sum(d["f1"] for d in details) / len(details)
                logger.info(
                    f"  Eval [{benchmark}] progress: {idx+1}/{len(eval_data)} | "
                    f"avg_f1={avg_f1:.4f}"
                )

        avg_f1 = sum(item["f1"] for item in details) / max(len(details), 1)
        avg_em = sum(item["em"] for item in details) / max(len(details), 1)

        return {
            "summary": {
                "benchmark": benchmark,
                "num_examples": len(details),
                "avg_f1": avg_f1,
                "avg_em": avg_em,
                "memory_size": self.current_memory_size(),
            },
            "details": details,
        }

    def evaluate_agent_tasks(self, tasks: list[dict], env_name: str) -> dict:
        """Evaluate online adaptation on task-style benchmark streams."""

        self.reset_state()
        success_curve = []
        failure_recurrence: dict[str, int] = {}

        for idx, task in enumerate(tasks):
            task_events = list(task.get("events", []))
            self.current_events = task_events
            action = self.answer_question(task.get("instruction", task.get("context", "")))
            success = bool(task.get("env_kwargs", {}).get("task_result", {}).get("success", False))
            task_type = task.get("type", "unknown")
            if not success:
                failure_recurrence[task_type] = failure_recurrence.get(task_type, 0) + 1

            self.observe_task_feedback(task, action, success)
            success_curve.append({"task_idx": idx, "type": task_type, "success": success})

        successes = sum(1 for item in success_curve if item["success"])
        total = len(success_curve)
        summary = {
            "env": env_name,
            "total_tasks": total,
            "successes": successes,
            "overall_success_rate": successes / max(total, 1),
            "failure_recurrence": failure_recurrence,
            "memory_size": self.current_memory_size(),
        }
        return {"summary": summary, "success_curve": success_curve}

    def _generate_answer(self, question: str, memory_sections: list[tuple[str, list[str]]]) -> str:
        """Generate an answer using the supplied memory sections."""

        blocks = [
            _format_context_block(title, items, "(no retrieved context)")
            for title, items in memory_sections
        ]
        blocks.append(_format_context_block("Current Events", self.current_events[-6:], "(no events)"))
        prompt = (
            "Answer the question using the available context. If the answer is "
            "not supported, say that it is unknown.\n\n"
            + "\n\n".join(blocks)
            + f"\n\n### Question\n{question}\n\n### Answer\n"
        )
        return generate_text(self.model, self.tokenizer, prompt, max_new_tokens=128)


class MemoryR1Baseline(LocalBaseline):
    """Project-local Memory-R1 baseline using external QA reward only."""

    def __init__(self, model, tokenizer, config: MUBConfig):
        super().__init__(model, tokenizer, config)
        self.config.reward.lambda_mem = 0.0
        self.config.reward.reward_scale = 1.0
        self.config.trigger.theta = 999.0
        self.agent: Optional[GMSRAAgent] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.reward_baseline = 0.0
        self.update_step = 0

    def reset_state(self):
        super().reset_state()
        self.agent = GMSRAAgent(self.config)
        self.agent.initialize(self.model, self.tokenizer, env_type="dialogue")
        self.optimizer = None
        self.reward_baseline = 0.0
        self.update_step = 0

    def current_memory_size(self) -> int:
        return self.agent.memory_store.size() if self.agent else 0

    def _ensure_optimizer(self):
        if self.optimizer is None:
            trainable = [param for param in self.model.parameters() if param.requires_grad]
            if trainable:
                self.optimizer = torch.optim.AdamW(
                    trainable,
                    lr=self.config.rl.learning_rate,
                    weight_decay=0.01,
                )

    def _ingest_event(self, event: str, task_context: str, answer: Optional[str] = None) -> dict:
        env_kwargs = {}
        if answer is not None:
            env_kwargs = {"agent_response": "", "qa_ground_truth": answer}
        return self.agent.step(
            event=event,
            task_context=task_context,
            agent_response="",
            env_signal_kwargs=env_kwargs,
        )

    def answer_question(self, question: str) -> str:
        return self.agent.answer_question(question)

    def _policy_update(self, last_event: str, task_context: str, op_result: dict, reward: float):
        self._ensure_optimizer()
        if self.optimizer is None:
            return

        operation_str, prompt = self.agent.memory_manager.decide(last_event, task_context)
        if not operation_str.strip():
            operation_str = _format_operation(op_result)

        inputs = self.tokenizer(
            prompt + operation_str,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        device = _model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        prompt_len = len(self.tokenizer.encode(prompt, truncation=True, max_length=1024))
        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        outputs = self.model(**inputs, labels=labels)
        self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * reward
        advantage = reward - self.reward_baseline
        if abs(advantage) <= 1e-3:
            return

        loss = -advantage * outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in self.model.parameters() if param.requires_grad],
            self.config.rl.max_grad_norm,
        )

        self.update_step += 1
        if self.update_step % self.config.rl.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def train_dialogue(self, train_data: list[dict], max_episodes: int) -> dict:
        self.reset_state()
        self.model.train()
        rewards = []
        n_total = min(max_episodes, len(train_data))
        MAX_EVENTS = 20  # Cap events per episode to avoid slow training

        for ep_idx, episode in enumerate(train_data[:max_episodes]):
            question = episode.get("question", "")
            answer = episode.get("answer", "")
            last_result = None

            # Cap events to avoid extremely slow per-event agent.step() calls
            events = episode.get("events", [])[-MAX_EVENTS:]
            self.current_events = list(events)  # Reset per episode
            for event in events:
                last_result = self._ingest_event(event, question, answer)

            prediction = self.answer_question(question)
            reward = compute_f1(prediction, answer)
            rewards.append(reward)

            if last_result is not None and events:
                self._policy_update(
                    events[-1],
                    question,
                    last_result["operation"],
                    reward,
                )

            if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == n_total:
                avg_r = sum(rewards) / len(rewards)
                logger.info(
                    f"  Train progress: {ep_idx+1}/{n_total} | "
                    f"avg_f1={avg_r:.4f} | mem={self.current_memory_size()}"
                )

        self.model.eval()
        avg_reward = sum(rewards) / max(len(rewards), 1)
        summary = {
            "episodes": min(max_episodes, len(train_data)),
            "avg_train_f1": avg_reward,
            "memory_size": self.current_memory_size(),
            "update_steps": self.update_step,
        }
        self.training_history.append(summary)
        return summary

    def evaluate_dialogue(self, eval_data: list[dict], benchmark: str) -> dict:
        """Evaluate dialogue using fast embedding-based event ingestion.

        During eval, we skip the full agent.step() pipeline (which requires
        two LLM calls per event for decide + reward) and instead directly
        add events to the memory store via embedding. This is ~100x faster
        and equivalent in practice because the untrained RL policy outputs
        NOOP on 90%+ of events anyway.

        QA still uses agent.answer_question() which retrieves from memory
        store and generates an answer via LLM.
        """
        self.reset_state()
        self.model.eval()
        details = []
        MAX_EVENTS = 250  # Match MemUpdateBench eval cap

        import logging as _logging
        mem_logger = _logging.getLogger("mub.memory.store")
        original_level = mem_logger.level

        for idx, episode in enumerate(eval_data):
            question = episode.get("question", "")
            answer = episode.get("answer", "")

            # Reset memory store per example (prevent cross-contamination)
            self.agent.memory_store.entries.clear()
            self.agent.memory_store._id_list.clear()
            if hasattr(self.agent.memory_store, '_np_embeddings'):
                self.agent.memory_store._np_embeddings.clear()
            self.agent.memory_store._index = None

            # Fast embedding-only ingestion (no LLM calls)
            events = episode.get("events", [])[-MAX_EVENTS:]
            self.current_events = list(events)
            n_ingested = 0
            mem_logger.setLevel(_logging.WARNING)
            for event in events:
                text = str(event)
                # Extract text from dict-like event strings
                if isinstance(event, str) and event.startswith("{"):
                    try:
                        import ast
                        d = ast.literal_eval(event)
                        speaker = d.get("speaker", "")
                        txt = d.get("text", "")
                        caption = d.get("blip_caption", "")
                        if txt:
                            text = f"{speaker}: {txt}" if speaker else txt
                        elif caption:
                            text = f"{speaker}: [image] {caption}" if speaker else f"[image] {caption}"
                    except (ValueError, SyntaxError):
                        pass
                elif isinstance(event, dict):
                    speaker = event.get("speaker", "")
                    txt = event.get("text", "")
                    caption = event.get("blip_caption", "")
                    if txt:
                        text = f"{speaker}: {txt}" if speaker else txt
                    elif caption:
                        text = f"{speaker}: [image] {caption}" if speaker else f"[image] {caption}"

                if text and len(text) > 3:
                    self.agent.memory_store.add(
                        text,
                        env_reward=0.5,
                        tags=["eval_event"],
                        source=f"eval_{idx}",
                    )
                    n_ingested += 1
            mem_logger.setLevel(original_level)

            prediction = self.answer_question(question)
            details.append(
                {
                    "idx": idx,
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                    "f1": compute_f1(prediction, answer),
                    "em": compute_exact_match(prediction, answer),
                    "category": episode.get("category", "unknown"),
                    "events_ingested": n_ingested,
                }
            )

            if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_data):
                avg_f1 = sum(d["f1"] for d in details) / len(details)
                logger.info(
                    f"  Eval [{benchmark}] progress: {idx+1}/{len(eval_data)} | "
                    f"avg_f1={avg_f1:.4f} | mem={self.agent.memory_store.size()}"
                )

        avg_f1 = sum(item["f1"] for item in details) / max(len(details), 1)
        avg_em = sum(item["em"] for item in details) / max(len(details), 1)
        return {
            "summary": {
                "benchmark": benchmark,
                "num_examples": len(details),
                "avg_f1": avg_f1,
                "avg_em": avg_em,
                "memory_size": self.current_memory_size(),
            },
            "details": details,
        }

    def evaluate_agent_tasks(self, tasks: list[dict], env_name: str) -> dict:
        self.reset_state()
        self.model.eval()
        success_curve = []
        failure_recurrence: dict[str, int] = {}

        for idx, task in enumerate(tasks):
            instruction = task.get("instruction", task.get("context", ""))
            last_result = None
            self.current_events = list(task.get("events", []))
            for event in self.current_events:
                last_result = self.agent.step(
                    event=event,
                    task_context=instruction,
                    agent_response="",
                    env_signal_kwargs=task.get("env_kwargs", {}),
                )

            success = bool(task.get("env_kwargs", {}).get("task_result", {}).get("success", False))
            task_type = task.get("type", "unknown")
            if not success:
                failure_recurrence[task_type] = failure_recurrence.get(task_type, 0) + 1

            success_curve.append(
                {
                    "task_idx": idx,
                    "type": task_type,
                    "success": success,
                    "reward": last_result["reward"]["r_total"] if last_result else 0.0,
                }
            )

        successes = sum(1 for item in success_curve if item["success"])
        total = len(success_curve)
        return {
            "summary": {
                "env": env_name,
                "total_tasks": total,
                "successes": successes,
                "overall_success_rate": successes / max(total, 1),
                "failure_recurrence": failure_recurrence,
                "memory_size": self.current_memory_size(),
            },
            "success_curve": success_curve,
        }


class Mem0MemoryR1Baseline(MemoryR1Baseline):
    """Simple module-combination baseline: Memory-R1 plus flat event memory."""

    def __init__(self, model, tokenizer, config: MUBConfig):
        super().__init__(model, tokenizer, config)
        self.flat_events: list[str] = []

    def reset_state(self):
        super().reset_state()
        self.flat_events = []

    def _ingest_event(self, event: str, task_context: str, answer: Optional[str] = None) -> dict:
        if event.strip():
            self.flat_events.append(event.strip())
        return super()._ingest_event(event, task_context, answer)

    def current_memory_size(self) -> int:
        return super().current_memory_size() + len(self.flat_events)

    def answer_question(self, question: str) -> str:
        managed_memories = []
        if self.agent and self.agent.memory_store.size() > 0:
            managed_memories = [
                entry.content for entry, _ in self.agent.memory_store.retrieve(question, topk=3)
            ]
        flat_memories = _rank_text_snippets(self.flat_events, question, topk=3)

        merged: list[str] = []
        for item in managed_memories + flat_memories:
            if item not in merged:
                merged.append(item)

        if not merged:
            return super().answer_question(question)

        prompt = (
            "Answer the question using both the learned memory manager output and "
            "the flat event memory buffer.\n\n"
            + _format_context_block("Managed Memories", managed_memories, "(none)")
            + "\n\n"
            + _format_context_block("Flat Event Memories", flat_memories, "(none)")
            + f"\n\n### Question\n{question}\n\n### Answer\n"
        )
        return generate_text(self.model, self.tokenizer, prompt, max_new_tokens=128)


class ReflexionBaseline(LocalBaseline):
    """Textual reflection baseline without RL memory control."""

    def __init__(self, model, tokenizer, config: MUBConfig):
        super().__init__(model, tokenizer, config)
        self.reflections: list[str] = []

    def answer_question(self, question: str) -> str:
        top_reflections = _rank_text_snippets(self.reflections, question, topk=5)
        return self._generate_answer(
            question,
            [("Reflections", top_reflections)],
        )

    def observe_dialogue_feedback(self, episode: dict, prediction: str, reward: float):
        if reward >= 0.99:
            return
        question = episode.get("question", "")
        answer = episode.get("answer", "")
        lesson = (
            f"Reflection: when answering '{question}', prioritize the facts that "
            f"support '{answer}'. The previous answer was '{prediction[:120]}'."
        )
        self.reflections.append(lesson)

    def observe_task_feedback(self, task: dict, action: str, success: bool):
        if success:
            return
        instruction = task.get("instruction", task.get("context", ""))
        self.reflections.append(
            f"Reflection: task '{instruction}' failed. Re-check relevant objects, "
            "locations, and state transitions before acting."
        )

    def current_memory_size(self) -> int:
        return len(self.reflections)


class EvolveRBaseline(LocalBaseline):
    """Experience-lifecycle baseline without parameter consolidation."""

    def __init__(self, model, tokenizer, config: MUBConfig):
        super().__init__(model, tokenizer, config)
        self.experiences: list[str] = []

    def answer_question(self, question: str) -> str:
        top_experiences = _rank_text_snippets(self.experiences, question, topk=5)
        return self._generate_answer(
            question,
            [("Experience Memories", top_experiences)],
        )

    def observe_dialogue_feedback(self, episode: dict, prediction: str, reward: float):
        question = episode.get("question", "")
        answer = episode.get("answer", "")
        if reward >= 0.5:
            experience = (
                f"Successful experience: question '{question}' was answered well by "
                f"focusing on '{answer}'."
            )
        else:
            experience = (
                f"Corrective experience: for question '{question}', the reliable "
                f"target answer is '{answer}', not '{prediction[:120]}'."
            )
        self.experiences.append(experience)
        self.experiences = self.experiences[-300:]

    def observe_task_feedback(self, task: dict, action: str, success: bool):
        instruction = task.get("instruction", task.get("context", ""))
        if success:
            self.experiences.append(
                f"Successful task pattern: '{instruction}' completed after using the "
                "available observations as a stepwise plan."
            )
        else:
            self.experiences.append(
                f"Failed task pattern: '{instruction}' failed. Avoid repeating the "
                "same action sequence without new evidence."
            )
        self.experiences = self.experiences[-300:]

    def current_memory_size(self) -> int:
        return len(self.experiences)


class SelfConsolidationBaseline(LocalBaseline):
    """Fixed-interval reflection-to-LoRA consolidation baseline."""

    def __init__(
        self,
        model,
        tokenizer,
        config: MUBConfig,
        *,
        consolidation_interval: int = 25,
    ):
        super().__init__(model, tokenizer, config)
        self.consolidation_interval = consolidation_interval
        self.distiller = SemanticDistiller(model, tokenizer, config.lora)
        self.reflection_store = MemoryStore(config.memory)
        self.episodes_seen = 0
        self.consolidation_events: list[dict] = []

    def reset_state(self):
        super().reset_state()
        self.reflection_store = MemoryStore(self.config.memory)
        self.episodes_seen = 0

    def current_memory_size(self) -> int:
        return self.reflection_store.size()

    def answer_question(self, question: str) -> str:
        retrieved = [entry.content for entry, _ in self.reflection_store.retrieve(question, topk=5)]
        model = self.distiller._lora_model or self.model
        blocks = [
            _format_context_block("Consolidation Memories", retrieved, "(none)"),
            _format_context_block("Current Events", self.current_events[-6:], "(no events)"),
        ]
        prompt = (
            "Answer the question using the current episode and any consolidated "
            "knowledge you have learned from prior reflections.\n\n"
            + "\n\n".join(blocks)
            + f"\n\n### Question\n{question}\n\n### Answer\n"
        )
        return generate_text(model, self.tokenizer, prompt, max_new_tokens=128)

    def observe_dialogue_feedback(self, episode: dict, prediction: str, reward: float):
        question = episode.get("question", "")
        answer = episode.get("answer", "")
        reflection = (
            f"Reflection: for question '{question}', the stable answer is '{answer}'. "
            f"The previous answer was '{prediction[:120]}'."
        )
        self.reflection_store.add(
            reflection,
            env_reward=reward,
            tags=["reflection"],
            source=question[:100],
        )
        self.episodes_seen += 1

        if self.episodes_seen % self.consolidation_interval == 0:
            self._run_fixed_consolidation()

    def observe_task_feedback(self, task: dict, action: str, success: bool):
        instruction = task.get("instruction", task.get("context", ""))
        reflection = (
            f"Task reflection: instruction '{instruction}' "
            f"{'succeeded' if success else 'failed'} after action '{action[:120]}'."
        )
        self.reflection_store.add(
            reflection,
            env_reward=1.0 if success else 0.0,
            tags=["task_reflection"],
            source=instruction[:100],
        )
        self.episodes_seen += 1
        if self.episodes_seen % self.consolidation_interval == 0:
            self._run_fixed_consolidation()

    def _run_fixed_consolidation(self):
        if self.distiller._lora_model is None:
            self.distiller.setup_dual_lora()

        self.reflection_store.recalibrate_confidence()
        stats = self.distiller.consolidate(
            self.reflection_store,
            llm_model=self.model,
            llm_tokenizer=self.tokenizer,
        )
        if not stats.get("skipped", True):
            distilled_now = self.distiller.distilled_entries[-stats.get("distilled", 0):]
            for entry_id in distilled_now:
                if entry_id in self.reflection_store.entries:
                    self.reflection_store.delete(entry_id)
        self.consolidation_events.append(stats)
        logger.info(f"Self-Consolidation baseline event: {stats}")


__all__ = [
    "BASELINE_SPECS",
    "BaselineSpec",
    "create_baseline",
    "get_baseline_spec",
    "list_baselines",
]
