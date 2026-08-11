from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MEM0_EXTRACTION_INSTRUCTIONS",
    "Mem0DependencyUnavailable",
    "Mem0WorkerProtocolError",
    "Mem0WorkerServiceV1",
    "OfficialMem0BackendV1",
    "build_mem0_memory_config",
    "load_mem0_worker_configuration",
    "serve_mem0_worker_jsonl",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module("mub.vnext.external.workers.mem0_worker")
    return getattr(module, name)


__all__ = sorted(_EXPORTS)
