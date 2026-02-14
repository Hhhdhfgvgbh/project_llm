from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelRuntimeConfig:
    n_ctx: int
    n_gpu_layers: int
    temperature: float
    top_p: float
    max_tokens: int
    threads: int


@dataclass
class ModelHandle:
    model_name: str
    model_path: Path
    loaded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelWrapper:
    """
    Runtime wrapper abstraction.

    Current implementation is deterministic/mock-friendly and keeps API stable for
    future llama.cpp integration.
    """

    def __init__(self) -> None:
        self._handles: dict[str, ModelHandle] = {}

    def load(self, model_name: str, model_path: Path, runtime: ModelRuntimeConfig) -> ModelHandle:
        handle = self._handles.get(model_name)
        if handle and handle.loaded:
            return handle

        handle = ModelHandle(
            model_name=model_name,
            model_path=model_path,
            loaded=True,
            metadata={"runtime": runtime.__dict__},
        )
        self._handles[model_name] = handle
        return handle

    def unload(self, model_name: str) -> None:
        handle = self._handles.get(model_name)
        if handle:
            handle.loaded = False

    def generate(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str,
        runtime: ModelRuntimeConfig,
    ) -> str:
        handle = self._handles.get(model_name)
        if not handle or not handle.loaded:
            raise ValueError(f"Model '{model_name}' is not loaded")

        return (
            f"[{model_name}]\n"
            f"SYSTEM: {system_prompt}\n"
            f"PARAMS: temp={runtime.temperature}, top_p={runtime.top_p}, "
            f"max_tokens={runtime.max_tokens}, ctx={runtime.n_ctx}, gpu_layers={runtime.n_gpu_layers}\n"
            f"INPUT: {prompt}\n"
            f"OUTPUT: generated response"
        )

    @staticmethod
    def estimate_memory(model_size_gb: float, n_ctx: int, hidden_size: int, bytes_per_weight: int) -> float:
        kv_cache_gb = (n_ctx * hidden_size * 2 * bytes_per_weight) / (1024**3)
        return model_size_gb + kv_cache_gb + 0.5
