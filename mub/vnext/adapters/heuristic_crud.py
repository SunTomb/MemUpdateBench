from __future__ import annotations

import math
from typing import Any

from mub.vnext.contracts import AdapterInfo
from mub.vnext.adapters.exact_crud import ExactCrudAdapter
from mub.vnext.adapters.reference import _configuration_hash


class HeuristicCrudAdapter(ExactCrudAdapter):
    adapter_id = "heuristic_crud"

    def __init__(
        self,
        *,
        encoder: Any | None = None,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_revision: str = "unverified",
        backend: str = "sentence_transformers",
        retrieval_policy: str = "normal_topk",
    ) -> None:
        super().__init__(retrieval_policy=retrieval_policy)
        self.encoder = encoder
        self.encoder_model = encoder_model
        self.encoder_revision = encoder_revision
        self.backend = backend
        self._ready = self._verify_encoder()
        if not self._ready:
            self._startup_error = {
                "code": "not_supported",
                "reason": "verified_encoder_required",
                "encoder_model": self.encoder_model,
                "encoder_revision": self.encoder_revision,
                "backend": self.backend,
            }

    def _verify_encoder(self) -> bool:
        if self.encoder is None:
            return False
        try:
            if hasattr(self.encoder, "encode"):
                values = self.encoder.encode(["memupdatebench capability probe"], normalize_embeddings=True)
            elif callable(self.encoder):
                values = self.encoder(["memupdatebench capability probe"], normalize_embeddings=True)
            else:
                return False
            row = values[0]
            numbers = [float(value) for value in row]
            return bool(numbers) and all(math.isfinite(value) for value in numbers) and any(value != 0.0 for value in numbers)
        except Exception:
            return False

    def _info_config(self) -> dict[str, Any]:
        return {
            "retrieval_policy": self.retrieval_policy,
            "encoder_model": self.encoder_model,
            "encoder_revision": self.encoder_revision,
            "backend": self.backend,
            "verified": self._ready,
        }

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            system_name=self.backend,
            system_version=self.encoder_model,
            sdk_version=self.encoder_revision,
            configuration_hash=_configuration_hash(self.adapter_id, self._info_config()),
            extractor_id=self.backend,
            extractor_version=self.encoder_revision,
        )


__all__ = ["HeuristicCrudAdapter"]
