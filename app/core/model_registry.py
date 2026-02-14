from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config.schemas import GenerationParams, ModelsConfig
from app.config.validation import RuntimeValidator


@dataclass
class RegisteredModel:
    name: str
    path: Path
    required_for_base: bool
    generation: GenerationParams
    description: str
    quantization: str


class ModelRegistry:
    def __init__(self, config: ModelsConfig) -> None:
        self.config = config
        self.models: dict[str, RegisteredModel] = {}
        self.warnings: list[str] = []
        self.base_mode_blocked: bool = False

    def build(self) -> "ModelRegistry":
        report = RuntimeValidator.validate_model_files(self.config)
        self.warnings.extend(report.warnings)
        self.base_mode_blocked = bool(report.errors)

        for name, model in self.config.models.items():
            candidate = self.config.models_directory / model.file
            if not candidate.exists():
                continue

            merged_generation = self.config.defaults.model_copy(update=model.overrides.as_dict())
            self.models[name] = RegisteredModel(
                name=name,
                path=candidate,
                required_for_base=model.required_for_base,
                generation=merged_generation,
                description=model.description,
                quantization=model.quantization.value,
            )

        return self

    def list_models(self) -> list[RegisteredModel]:
        return list(self.models.values())
