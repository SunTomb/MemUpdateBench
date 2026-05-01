"""
MemoryStore: Zettelkasten-style memory database with FAISS-backed retrieval.
Supports CRUD operations, graph-based linking, confidence filtering,
and subgraph extraction for consolidation.
"""

from __future__ import annotations
import json
import os
import time
from typing import Optional

import numpy as np
from loguru import logger

from mub.memory.entry import MemoryEntry
from mub.config import MemoryConfig


class MemoryStore:
    """External memory store with FAISS indexing and graph structure.

    Key Design Choices (per MemUpdateBench proposal):
    - Zettelkasten-style cards with explicit links (inspired by A-MEM)
    - Confidence-weighted retrieval for Judge (prevents noise pollution)
    - Subgraph extraction for consolidation (graph filters WHAT to distill)
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.entries: dict[str, MemoryEntry] = {}  # id -> MemoryEntry
        self._index = None  # FAISS index (lazy init)
        self._id_list: list[str] = []  # Ordered list of IDs in FAISS index
        self._encoder = None  # Sentence encoder (lazy init)

    # ---- Lazy initialization ----

    def _init_encoder(self):
        """Lazy-load sentence-transformers encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded encoder: {self.config.embedding_model}")
            except ImportError:
                self._encoder = lambda texts, normalize_embeddings=True: np.zeros(
                    (len(texts), self.config.embedding_dim), dtype=np.float32
                )
                logger.warning(
                    "sentence_transformers not available, using zero-vector encoder fallback"
                )

    def _init_faiss_index(self):
        """Lazy-init FAISS index (or numpy fallback if faiss unavailable)."""
        if self._index is None:
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self.config.embedding_dim)
                self._use_faiss = True
                logger.info(f"Initialized FAISS index (dim={self.config.embedding_dim})")
            except ImportError:
                self._index = "numpy"  # sentinel: use numpy fallback
                self._use_faiss = False
                self._np_embeddings = []  # list of np arrays
                logger.info("FAISS not available, using numpy fallback for retrieval")

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        self._init_encoder()
        if hasattr(self._encoder, "encode"):
            emb = self._encoder.encode([text], normalize_embeddings=True)
        else:
            emb = self._encoder([text], normalize_embeddings=True)
        return emb[0].astype(np.float32)

    # ---- CRUD Operations ----

    def add(self, content: str, env_reward: float = 0.0,
            keywords: list[str] = None, tags: list[str] = None,
            source: str = "", slot_meta: dict = None) -> MemoryEntry:
        """ADD operation: Create a new memory entry."""
        entry = MemoryEntry(
            content=content,
            keywords=keywords or [],
            tags=tags or [],
            env_reward_at_write=env_reward,
            source=source,
            slot=slot_meta,
        )
        entry.embedding = self._encode(content).tolist()

        # Add to index
        self._init_faiss_index()
        emb_arr = np.array([entry.embedding], dtype=np.float32)
        if self._use_faiss:
            self._index.add(emb_arr)
        else:
            self._np_embeddings.append(emb_arr[0])
        self._id_list.append(entry.id)

        # Auto-link: find similar existing memories and link
        if len(self.entries) > 0:
            similar = self.retrieve(content, topk=3)
            for sim_entry, score in similar:
                if score > 0.7:  # High similarity threshold for auto-linking
                    entry.add_link(sim_entry.id)
                    sim_entry.add_link(entry.id)

        self.entries[entry.id] = entry

        # Check capacity
        if len(self.entries) > self.config.max_entries:
            self._evict_lowest_confidence()

        logger.debug(f"ADD memory [{entry.id}]: {content[:50]}...")
        return entry

    def update(self, entry_id: str, new_content: str,
               env_reward: float = 0.0,
               slot_meta: dict = None) -> Optional[MemoryEntry]:
        """UPDATE operation: Modify an existing memory entry."""
        if entry_id not in self.entries:
            logger.warning(f"UPDATE failed: entry {entry_id} not found")
            return None

        entry = self.entries[entry_id]
        entry.content = new_content
        entry.updated_at = time.time()
        entry.env_reward_at_write = max(entry.env_reward_at_write, env_reward)
        if slot_meta is not None:
            entry.slot = slot_meta

        # Update embedding in FAISS
        new_emb = self._encode(new_content)
        entry.embedding = new_emb.tolist()
        # v10: guard against stale _id_list (BUG-3 complete fix)
        if entry_id in self._id_list:
            self._rebuild_index()  # FAISS doesn't support in-place update
        else:
            # Entry exists in dict but not in index — re-add it
            self._id_list.append(entry_id)
            self._rebuild_index()

        logger.debug(f"UPDATE memory [{entry_id}]: {new_content[:50]}...")
        return entry


    def delete(self, entry_id: str) -> bool:
        """DELETE operation: Remove a memory entry."""
        if entry_id not in self.entries:
            logger.warning(f"DELETE failed: entry {entry_id} not found")
            return False

        entry = self.entries.pop(entry_id)

        # Remove links from other entries
        for linked_id in entry.links:
            if linked_id in self.entries:
                other = self.entries[linked_id]
                other.links = [lid for lid in other.links if lid != entry_id]

        # Rebuild FAISS index without this entry
        self._id_list = [eid for eid in self._id_list if eid != entry_id]
        self._rebuild_index()

        logger.debug(f"DELETE memory [{entry_id}]")
        return True

    # ---- Slot lookup ----

    def get_latest_by_slot(self, entity: str, attribute: str) -> Optional[MemoryEntry]:
        candidates = []
        for entry in self.entries.values():
            slot = entry.slot or {}
            if slot.get("entity") == entity and slot.get("attribute") == attribute:
                candidates.append(entry)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda entry: (
                (entry.slot or {}).get("event_idx", -1),
                entry.updated_at,
                entry.created_at,
            ),
        )

    # ---- Retrieval ----

    def retrieve(self, query: str, topk: int = None) -> list[tuple[MemoryEntry, float]]:
        """Retrieve top-K relevant memories by semantic similarity."""
        topk = topk or self.config.retrieval_topk
        if len(self.entries) == 0:
            return []

        self._init_faiss_index()
        query_emb = self._encode(query).reshape(1, -1)
        k = min(topk, len(self._id_list))

        if self._use_faiss:
            scores, indices = self._index.search(query_emb, k)
        else:
            # Numpy fallback: brute-force inner product
            if not self._np_embeddings:
                return []
            all_embs = np.array(self._np_embeddings, dtype=np.float32)
            sim = all_embs @ query_emb.T  # (N, 1)
            sim = sim.flatten()
            top_idx = np.argsort(-sim)[:k]
            scores = np.array([sim[top_idx]])
            indices = np.array([top_idx])

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            entry_id = self._id_list[idx]
            if entry_id in self.entries:
                results.append((self.entries[entry_id], float(score)))

        return results

    def retrieve_confident(self, query: str,
                           topk: int = None) -> list[tuple[MemoryEntry, float]]:
        """Retrieve top-K memories, filtered by confidence score.
        Used by the Judge to prevent memory noise pollution.
        """
        topk = topk or self.config.confidence_topk
        # Retrieve more candidates, then filter by confidence
        candidates = self.retrieve(query, topk=topk * 3)

        # Re-rank by confidence * similarity
        scored = []
        for entry, sim_score in candidates:
            combined = entry.confidence * 0.5 + sim_score * 0.5
            scored.append((entry, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    # ---- Graph Operations ----

    def get_linked_entries(self, entry_id: str,
                          depth: int = 1) -> list[MemoryEntry]:
        """Get all entries linked to the given entry up to a certain depth."""
        if entry_id not in self.entries:
            return []

        visited = set()
        queue = [(entry_id, 0)]
        result = []

        while queue:
            current_id, d = queue.pop(0)
            if current_id in visited or d > depth:
                continue
            visited.add(current_id)
            if current_id != entry_id and current_id in self.entries:
                result.append(self.entries[current_id])
            if d < depth:
                entry = self.entries.get(current_id)
                if entry:
                    for linked_id in entry.links:
                        queue.append((linked_id, d + 1))

        return result

    def extract_high_frequency_subgraph(self,
                                        min_links: int = 2,
                                        min_confidence: float = 0.5
                                        ) -> list[MemoryEntry]:
        """Extract high-frequency subgraph for consolidation.
        Returns memories with many links AND high confidence.
        These are the "crystallized knowledge" worth distilling to LoRA.
        """
        candidates = [
            entry for entry in self.entries.values()
            if len(entry.links) >= min_links and entry.confidence >= min_confidence
        ]
        # Sort by (links × confidence) — most connected & trustworthy first
        candidates.sort(key=lambda e: len(e.links) * e.confidence, reverse=True)
        return candidates

    # ---- Bulk Operations ----

    def recalibrate_confidence(self, weights: dict[str, float] = None):
        """Recalibrate all memory confidence scores."""
        weights = weights or self.config.confidence_weights
        for entry in self.entries.values():
            entry.update_confidence(weights)
        logger.info(f"Recalibrated confidence for {len(self.entries)} memories")

    def get_growth_rate(self, window_entries: int = 50) -> float:
        """Compute memory growth rate (entries added in recent window / total)."""
        if len(self.entries) == 0:
            return 0.0
        sorted_entries = sorted(self.entries.values(),
                                key=lambda e: e.created_at, reverse=True)
        recent = sorted_entries[:window_entries]
        if len(recent) < 2:
            return 0.0
        time_span = recent[0].created_at - recent[-1].created_at
        if time_span <= 0:
            return 0.0
        return len(recent) / (time_span / 3600)  # Entries per hour

    def size(self) -> int:
        """Number of memories in the store."""
        return len(self.entries)

    # ---- Persistence ----

    def save(self, path: str):
        """Save memory store to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [entry.to_dict() for entry in self.entries.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} memories to {path}")

    def load(self, path: str):
        """Load memory store from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = {}
        for item in data:
            entry = MemoryEntry.from_dict(item)
            entry.embedding = self._encode(entry.content).tolist()
            self.entries[entry.id] = entry
        self._rebuild_index()
        logger.info(f"Loaded {len(self.entries)} memories from {path}")

    # ---- Internal ----

    def _evict_lowest_confidence(self):
        """Remove the memory with the lowest confidence score.
        
        v9 fix: entries younger than 60 seconds are exempt from eviction
        to prevent the ADD→evict→ADD death loop.
        """
        if not self.entries:
            return
        import time as _time
        now = _time.time()
        # Filter out entries that are too young (grace period)
        eligible = [e for e in self.entries.values() if now - e.created_at > 60]
        if not eligible:
            # All entries are young — evict the oldest among all
            eligible = list(self.entries.values())
        worst = min(eligible, key=lambda e: e.confidence)
        self.delete(worst.id)
        logger.debug(f"Evicted lowest-confidence memory [{worst.id}]")


    def _rebuild_index(self):
        """Rebuild the index from current entries."""
        new_id_list = []
        embeddings = []
        for entry_id in self._id_list:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                if entry.embedding:
                    embeddings.append(entry.embedding)
                    new_id_list.append(entry_id)
        self._id_list = new_id_list

        if getattr(self, '_use_faiss', False):
            import faiss
            self._index = faiss.IndexFlatIP(self.config.embedding_dim)
            if embeddings:
                self._index.add(np.array(embeddings, dtype=np.float32))
        else:
            self._np_embeddings = [np.array(e, dtype=np.float32) for e in embeddings]

    def get_all_as_text(self, max_entries: int = 20) -> str:
        """Get all memories as formatted text (for prompts)."""
        sorted_entries = sorted(self.entries.values(),
                                key=lambda e: e.confidence, reverse=True)
        texts = [e.to_text() for e in sorted_entries[:max_entries]]
        return "\n\n".join(texts)
