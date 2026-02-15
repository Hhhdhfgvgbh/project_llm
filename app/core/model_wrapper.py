from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from llama_cpp import Llama

@dataclass
class ModelRuntimeConfig:
    n_ctx: int = 8192               # или 4096/16384 в зависимости от модели
    n_gpu_layers: int = -1          # -1 = максимально на GPU (или 99)
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    threads: int = 0                # 0 = auto (обычно хорошо)

@dataclass
class ModelHandle:
    model_name: str
    model_path: Path
    loaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    _model: Llama | None = None

class ModelWrapper:
    def __init__(self) -> None:
        self._handles: Dict[str, ModelHandle] = {}

    def load(self, model_name: str, model_path: Path, runtime: ModelRuntimeConfig) -> ModelHandle:
        handle = self._handles.get(model_name)
        if handle and handle.loaded:
            return handle

        print(f"Загружаем модель {model_name} → {model_path.name}")
        print(f"Параметры: n_gpu_layers={runtime.n_gpu_layers}, n_ctx={runtime.n_ctx}")

        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=runtime.n_gpu_layers,
            n_ctx=runtime.n_ctx,
            temperature=runtime.temperature,
            top_p=runtime.top_p,
            max_tokens=runtime.max_tokens,
            n_threads=runtime.threads,
            verbose=False,  # Поставь True для отладки, если нужно видеть лог загрузки
        )

        handle = ModelHandle(
            model_name=model_name,
            model_path=model_path,
            loaded=True,
            metadata={"runtime": runtime.__dict__},
            _model=llm,
        )
        self._handles[model_name] = handle
        print(f"Модель {model_name} загружена (GPU layers: {runtime.n_gpu_layers})")
        return handle

    def unload(self, model_name: str) -> None:
        handle = self._handles.pop(model_name, None)
        if handle and handle._model:
            del handle._model
            handle.loaded = False
        print(f"Модель {model_name} выгружена")

    def generate(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str = "",
        runtime: ModelRuntimeConfig | None = None,
    ) -> str:
        handle = self._handles.get(model_name)
        if not handle or not handle._model:
            raise ValueError(f"Модель '{model_name}' не загружена")

        if runtime is None:
            runtime = ModelRuntimeConfig()

        messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        response = handle._model.create_chat_completion(
            messages,
            temperature=runtime.temperature,
            top_p=runtime.top_p,
            max_tokens=runtime.max_tokens,
        )

        return response["choices"][0]["message"]["content"].strip()

    @staticmethod
    def estimate_memory(model_size_gb: float, n_ctx: int, hidden_size: int = 3072, bytes_per_weight: int = 2) -> float:
        # Для Phi-3.5-mini hidden_size ≈ 3072
        kv_cache_gb = (n_ctx * hidden_size * 2 * bytes_per_weight) / (1024 ** 3)
        return model_size_gb + kv_cache_gb + 1.0  # + overhead