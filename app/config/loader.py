from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.config.schemas import ModelsConfig, PipelineConfig, raise_config_error


class ConfigLoader:
    @staticmethod
    def load_yaml(path: Path | str) -> dict[str, Any]:
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"YAML root must be a mapping: {source}")
        return payload

    @classmethod
    def load_models_config(cls, path: Path | str) -> ModelsConfig:
        payload = cls.load_yaml(path)
        try:
            return ModelsConfig.model_validate(payload)
        except ValidationError as error:
            raise raise_config_error("models.yaml", error) from error

    @classmethod
    def load_pipeline_config(cls, path: Path | str) -> PipelineConfig:
        payload = cls.load_yaml(path)
        try:
            return PipelineConfig.model_validate(payload)
        except ValidationError as error:
            raise raise_config_error("pipeline.yaml", error) from error
