"""
MemoryEntry: Zettelkasten-style memory card dataclass.
Each memory is a structured card with content, metadata, links, and confidence score.
Inspired by A-MEM (2025) and HippoRAG (2024).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import uuid


@dataclass
class MemoryEntry:
    """A single memory card in the Zettelkasten-style memory store.

    Attributes:
        id: Unique identifier for this memory.
        content: The main content/fact stored.
        keywords: Extracted keywords for indexing.
        tags: Categorical tags (e.g., "preference", "fact", "procedure").
        links: IDs of related memories (bidirectional graph edges).
        confidence: Confidence score c_i ∈ [0, 1], updated over time.
        env_reward_at_write: R_env at the time this memory was created.
        hit_success: Number of times retrieval of this memory led to task success.
        hit_total: Total number of times this memory was retrieved.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
        source: Source context (e.g., task id, conversation turn).
        slot: Optional entity-attribute-value metadata for state tracking.
    """

    # --- Core content ---
    content: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # --- Graph links ---
    links: list[str] = field(default_factory=list)  # IDs of linked memories

    # --- Confidence tracking ---
    confidence: float = 0.5  # Initial confidence
    env_reward_at_write: float = 0.0
    hit_success: int = 0
    hit_total: int = 0

    # --- Metadata ---
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str = ""
    slot: Optional[dict] = None

    # --- Embedding (set externally) ---
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def update_confidence(self, weights: dict[str, float]) -> float:
        """Recompute confidence score using the formula:
        c_i = σ(w1·R_env_write + w2·(hit_success/hit_total) + w3·log(1+age))

        Args:
            weights: Dict with keys 'env_reward_write', 'hit_success_ratio', 'log_age'.

        Returns:
            Updated confidence score.
        """
        import math

        w1 = weights.get("env_reward_write", 0.4)
        w2 = weights.get("hit_success_ratio", 0.4)
        w3 = weights.get("log_age", 0.2)

        hit_ratio = self.hit_success / max(self.hit_total, 1)
        age = time.time() - self.created_at
        log_age = math.log(1 + age / 3600)  # Normalize by hours

        raw = w1 * self.env_reward_at_write + w2 * hit_ratio + w3 * log_age
        self.confidence = 1.0 / (1.0 + math.exp(-raw))  # Sigmoid
        return self.confidence

    def record_hit(self, success: bool):
        """Record a retrieval hit and whether it led to task success."""
        self.hit_total += 1
        if success:
            self.hit_success += 1

    def add_link(self, other_id: str):
        """Add a bidirectional link to another memory."""
        if other_id not in self.links:
            self.links.append(other_id)

    def to_text(self) -> str:
        """Serialize to a readable text representation for prompts."""
        parts = [f"[{self.id}] {self.content}"]
        if self.keywords:
            parts.append(f"  Keywords: {', '.join(self.keywords)}")
        if self.tags:
            parts.append(f"  Tags: {', '.join(self.tags)}")
        if self.slot:
            entity = self.slot.get("entity", "")
            attribute = self.slot.get("attribute", "")
            value = self.slot.get("value", "")
            parts.append(f"  Slot: {entity}.{attribute}={value}")
        parts.append(f"  Confidence: {self.confidence:.3f}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "id": self.id,
            "content": self.content,
            "keywords": self.keywords,
            "tags": self.tags,
            "links": self.links,
            "confidence": self.confidence,
            "env_reward_at_write": self.env_reward_at_write,
            "hit_success": self.hit_success,
            "hit_total": self.hit_total,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "slot": self.slot,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k != "embedding"})
