from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema
from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import MemoryQueryV3

ANSWER_MODEL_PARSER_VERSION_V3 = "memupdatebench.answer-model-parser.v3"
_SNAPSHOT_TREE_AUDIT_FILE = "snapshot-tree.json"


class DeterministicDecodeConfigV3(ImmutableContractModel):
    do_sample: Literal[False] = False
    num_beams: Literal[1] = 1
    max_new_tokens: int = Field(default=64, strict=True, ge=1, le=512)
    seed: int = Field(default=0, strict=True, ge=0)


class AnswerModelSlotV3(ImmutableContractModel):
    slot_id: Literal["answer_model_a", "answer_model_b"]
    model_id: str = Field(strict=True, min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$", strict=True)
    snapshot_path: str = Field(strict=True, min_length=1)
    license_id: Literal["apache-2.0"]
    tree_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree_sha256_v3(
    *,
    snapshot_path: str | Path,
    model_id: str,
    revision: str,
) -> str:
    root = Path(snapshot_path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("model snapshot must be a real directory")
    files = []
    for path in sorted(
        root.rglob("*"),
        key=lambda candidate: candidate.relative_to(root).as_posix().lower(),
    ):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("model snapshot contains a non-file entry")
        relative_path = path.relative_to(root).as_posix()
        if relative_path == _SNAPSHOT_TREE_AUDIT_FILE:
            continue
        files.append({
            "path": relative_path,
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        })
    identity = {
        "file_count": len(files),
        "files": files,
        "model_id": model_id,
        "revision": revision,
        "total_size_bytes": sum(file["size_bytes"] for file in files),
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def verify_snapshot_tree_v3(slot: AnswerModelSlotV3) -> None:
    observed = snapshot_tree_sha256_v3(
        snapshot_path=slot.snapshot_path,
        model_id=slot.model_id,
        revision=slot.revision,
    )
    if observed != slot.tree_manifest_sha256:
        raise ValueError("model snapshot tree manifest does not match slot")


class OfflinePromptedAnswerModelV3:
    def __init__(
        self,
        *,
        slot: AnswerModelSlotV3,
        decoding: DeterministicDecodeConfigV3 | None = None,
        device: str = "cpu",
    ) -> None:
        if type(device) is not str or not device.strip():
            raise ValueError("device must be a nonblank string")
        self.slot = slot
        self.decoding = (
            decoding if decoding is not None else DeterministicDecodeConfigV3()
        )
        self.device = device
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        require_offline_environment_v3()
        verify_snapshot_tree_v3(self.slot)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(self.decoding.seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        tokenizer = AutoTokenizer.from_pretrained(
            self.slot.snapshot_path,
            revision=self.slot.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        model_kwargs = {
            "revision": self.slot.revision,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.slot.snapshot_path,
            **model_kwargs,
        )
        model.to(self.device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def answer(self, request: PromptedAnswerRequestV3) -> AnswerPredictionV3:
        if self._torch is None or self._tokenizer is None or self._model is None:
            raise RuntimeError("offline prompted answer model is not loaded")
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": request.rendered_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }
        input_length = encoded["input_ids"].shape[-1]
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.decoding.max_new_tokens,
            )
        raw_output = self._tokenizer.decode(
            generated[0][input_length:],
            skip_special_tokens=True,
        )
        return parse_answer_prediction_v3(
            query_id=request.query.query_id,
            answer_schema=request.query.answer_schema,
            raw_output=raw_output,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None


class _DuplicateJsonKeyError(ValueError):
    pass


def require_offline_environment_v3() -> None:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    if any(os.environ.get(name) != "1" for name in required):
        raise RuntimeError("offline answer model loading requires offline environment flags")


def render_visible_prompt_v3(
    *,
    query: MemoryQueryV3,
    retrieval_trace: RetrievalTraceV3,
) -> str:
    if retrieval_trace.query_id != query.query_id:
        raise ValueError("retrieval trace query_id must match prompted query")
    visible_context = {
        "query": {
            "text": query.text,
            "answer_schema": query.answer_schema.value,
        },
        "retrieved_entries": [
            entry.model_dump(mode="json")
            for entry in retrieval_trace.retrieved_entries
        ],
    }
    payload = json.dumps(
        visible_context,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        "Use only the retrieved memory entries to answer the query. "
        "Return exactly one JSON object: "
        '{"disposition":"answered","answer":...} or '
        '{"disposition":"abstained"}.\n'
        f"{payload}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(key)
        value[key] = item
    return value


def _matches_answer_schema(value: Any, answer_schema: AnswerSchema) -> bool:
    if answer_schema is AnswerSchema.STRING:
        return type(value) is str
    if answer_schema is AnswerSchema.NUMBER:
        return type(value) in (int, float) and math.isfinite(value)
    if answer_schema is AnswerSchema.BOOLEAN:
        return type(value) is bool
    if answer_schema is AnswerSchema.LIST:
        return type(value) is list
    if answer_schema is AnswerSchema.OBJECT:
        return type(value) is dict
    raise ValueError(f"unsupported answer schema: {answer_schema}")


def _format_invalid_prediction(
    *,
    query_id: str,
    raw_output: str,
    error_flag: str,
) -> AnswerPredictionV3:
    return AnswerPredictionV3(
        query_id=query_id,
        raw_output=raw_output,
        disposition=AnswerDisposition.ANSWERED,
        format_valid=False,
        error_flags=(error_flag,),
    )


def parse_answer_prediction_v3(
    *,
    query_id: str,
    answer_schema: AnswerSchema,
    raw_output: str,
) -> AnswerPredictionV3:
    try:
        payload = json.loads(raw_output, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateJsonKeyError:
        return _format_invalid_prediction(
            query_id=query_id,
            raw_output=raw_output,
            error_flag="answer_json_duplicate_key",
        )
    except (json.JSONDecodeError, TypeError):
        return _format_invalid_prediction(
            query_id=query_id,
            raw_output=raw_output,
            error_flag="answer_json_invalid",
        )

    if type(payload) is not dict:
        return _format_invalid_prediction(
            query_id=query_id,
            raw_output=raw_output,
            error_flag="answer_envelope_invalid",
        )

    disposition = payload.get("disposition")
    if disposition == AnswerDisposition.ABSTAINED.value:
        if set(payload) != {"disposition"}:
            return _format_invalid_prediction(
                query_id=query_id,
                raw_output=raw_output,
                error_flag="answer_envelope_invalid",
            )
        return AnswerPredictionV3(
            query_id=query_id,
            raw_output=raw_output,
            disposition=AnswerDisposition.ABSTAINED,
            format_valid=True,
        )

    if disposition != AnswerDisposition.ANSWERED.value:
        return _format_invalid_prediction(
            query_id=query_id,
            raw_output=raw_output,
            error_flag="answer_disposition_invalid",
        )
    if set(payload) != {"disposition", "answer"}:
        return _format_invalid_prediction(
            query_id=query_id,
            raw_output=raw_output,
            error_flag="answer_envelope_invalid",
        )
    answer = payload["answer"]
    if not _matches_answer_schema(answer, answer_schema):
        return _format_invalid_prediction(
            query_id=query_id,
            raw_output=raw_output,
            error_flag="answer_schema_mismatch",
        )
    return AnswerPredictionV3(
        query_id=query_id,
        raw_output=raw_output,
        disposition=AnswerDisposition.ANSWERED,
        parsed_answer=answer,
        format_valid=True,
    )


__all__ = [
    "ANSWER_MODEL_PARSER_VERSION_V3",
    "AnswerModelSlotV3",
    "DeterministicDecodeConfigV3",
    "OfflinePromptedAnswerModelV3",
    "parse_answer_prediction_v3",
    "render_visible_prompt_v3",
    "require_offline_environment_v3",
    "snapshot_tree_sha256_v3",
    "verify_snapshot_tree_v3",
]
