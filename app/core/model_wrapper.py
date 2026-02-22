from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from llama_cpp import Llama
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime
    Llama = None


@dataclass
class ModelRuntimeConfig:
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    threads: int = field(
        default_factory=lambda: (
            1 if ModelRuntimeConfig._is_full_gpu() else max(1, os.cpu_count() // 2)
        )
    )


@dataclass
class ModelHandle:
    model_name: str
    model_path: Path
    loaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    _model: Any = None


class ModelWrapper:
    def __init__(self) -> None:
        self._handles: Dict[str, ModelHandle] = {}

    def load(
        self, model_name: str, model_path: Path, runtime: ModelRuntimeConfig
    ) -> ModelHandle:
        handle = self._handles.get(model_name)
        if handle and handle.loaded:
            return handle

        if Llama is None:
            handle = ModelHandle(
                model_name=model_name,
                model_path=model_path,
                loaded=True,
                metadata={"runtime": runtime.__dict__, "backend": "mock"},
                _model=None,
            )
            self._handles[model_name] = handle
            return handle

        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=runtime.n_gpu_layers,
            n_ctx=runtime.n_ctx,
            temperature=runtime.temperature,
            top_p=runtime.top_p,
            max_tokens=runtime.max_tokens,
            n_threads=runtime.threads,
            verbose=False,
            #
            use_mlock=False,  # не блокировать RAM (полезно при нехватке памяти)
            use_mmap=True,  # memory-mapped файл — быстрее загрузка, меньше RAM
            n_batch=512,  # размер батча для prompt processing (по умолчанию 512)
            flash_attn=True,  # если llama_cpp собран с поддержкой)
        )

        handle = ModelHandle(
            model_name=model_name,
            model_path=model_path,
            loaded=True,
            metadata={"runtime": runtime.__dict__, "backend": "llama_cpp"},
            _model=llm,
        )
        self._handles[model_name] = handle
        return handle

    def unload(self, model_name: str) -> None:
        handle = self._handles.pop(model_name, None)
        if handle and handle._model is not None:
            del handle._model
            handle.loaded = False

    def generate(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str = "",
        runtime: ModelRuntimeConfig | None = None,
    ) -> str:
        handle = self._handles.get(model_name)
        if not handle:
            raise ValueError(f"Модель '{model_name}' не загружена")

        if runtime is None:
            runtime = ModelRuntimeConfig()

        if handle.metadata.get("backend") == "mock" or handle._model is None:
            short_prompt = prompt.strip().replace("\n", " ")[:160]
            return f"[MOCK:{model_name}] generated response for: {short_prompt}"

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


    def generate_translategemma(
        self,
        model_name: str,
        text: str,
        source_language: str,
        target_language: str,
        runtime: ModelRuntimeConfig | None = None,
    ) -> str:
        system_prompt = (
            "You are TranslateGemma, a specialized machine translation model. "
            "Translate only from the source language to the target language. "
            "Return translation only with no explanations and no additional text."
        )
        prompt = (
            f"Source language: {source_language.strip()}\n"
            f"Target language: {target_language.strip()}\n"
            "Task: Translate the input text exactly and preserve meaning, names, numbers and punctuation.\n\n"
            f"Input text:\n{text.strip()}"
        )
        return self.generate(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            runtime=runtime,
        )

    @staticmethod
    def estimate_memory(
        model_size_gb: float,
        n_ctx: int,
        hidden_size: int = 3072,
        bytes_per_weight: int = 2,
    ) -> float:
        kv_cache_gb = (n_ctx * hidden_size * 2 * bytes_per_weight) / (1024**3)
        return model_size_gb + kv_cache_gb + 1.0
