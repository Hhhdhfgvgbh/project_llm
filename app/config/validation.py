from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config.schemas import ModelsConfig, PipelineConfig, StageMulti, StageSingle, StageTranslate


@dataclass
class ValidationReport:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class RuntimeValidator:
    @staticmethod
    def validate_model_files(config: ModelsConfig) -> ValidationReport:
        report = ValidationReport()
        base_dir = config.models_directory

        if not base_dir.exists() or not base_dir.is_dir():
            report.errors.append(f"models_directory does not exist or is not a directory: {base_dir}")
            return report

        for model_name, model in config.models.items():
            model_path = base_dir / model.file
            if model_path.exists():
                continue

            if model.required_for_base:
                report.errors.append(
                    f"Required base model '{model_name}' missing: expected file {model_path}"
                )
            else:
                report.warnings.append(
                    f"Optional model '{model_name}' missing and will be skipped: expected file {model_path}"
                )

        return report

    @staticmethod
    def validate_pipeline_models(pipeline: PipelineConfig, models: ModelsConfig) -> ValidationReport:
        report = ValidationReport()
        registered_models = set(models.models.keys())

        for stage in pipeline.base_pipeline.stages:
            stage_models: list[str]
            if isinstance(stage, StageSingle):
                stage_models = [stage.model]
            elif isinstance(stage, StageTranslate):
                stage_models = [stage.model]
            elif isinstance(stage, StageMulti):
                stage_models = list(stage.models)
                if stage.aggregation.synthesis_model:
                    stage_models.append(stage.aggregation.synthesis_model)
            else:
                stage_models = []

            for model_name in stage_models:
                if model_name not in registered_models:
                    report.errors.append(
                        f"Pipeline references unknown model '{model_name}' in stage '{stage.id}'"
                    )

        return report
