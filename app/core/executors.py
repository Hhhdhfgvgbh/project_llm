from __future__ import annotations

from dataclasses import dataclass

from app.config.loader import ConfigLoader
from app.config.schemas import PipelineConfig
from app.config.validation import RuntimeValidator
from app.core.aggregation import AggregationEngine
from app.core.model_registry import ModelRegistry
from app.core.model_wrapper import ModelWrapper
from app.core.pipeline_engine import PipelineEngine, PipelineResult
from app.core.resource_manager import ResourceManager
from app.core.session_manager import SessionManager


@dataclass
class ExecutorContext:
    engine: PipelineEngine
    pipeline: PipelineConfig


class BasePipelineExecutor:
    def __init__(self, models_path: str, pipeline_path: str) -> None:
        self.models_path = models_path
        self.pipeline_path = pipeline_path

    def build(self) -> ExecutorContext:
        models_cfg = ConfigLoader.load_models_config(self.models_path)
        pipeline_cfg = ConfigLoader.load_pipeline_config(self.pipeline_path)

        registry = ModelRegistry(models_cfg).build()
        if registry.base_mode_blocked:
            raise ValueError("Base mode is blocked: required base models are missing")

        report = RuntimeValidator.validate_pipeline_models(pipeline_cfg, models_cfg)
        if not report.ok:
            raise ValueError("; ".join(report.errors))

        aggregation = AggregationEngine(synthesis_callback=self._synthesis_callback)
        engine = PipelineEngine(
            registry=registry,
            model_wrapper=ModelWrapper(),
            aggregation_engine=aggregation,
            resource_manager=ResourceManager(),
            session_manager=SessionManager(),
        )
        return ExecutorContext(engine=engine, pipeline=pipeline_cfg)

    def run(self, user_input: str) -> PipelineResult:
        context = self.build()
        return context.engine.run(context.pipeline, user_input=user_input)

    @staticmethod
    def _synthesis_callback(model: str, prompt: str) -> str:
        return f"[SYNTHESIS by {model}]\n{prompt}\n[END]"


class ManualPipelineExecutor:
    def __init__(self, models_path: str, pipeline: PipelineConfig) -> None:
        self.models_path = models_path
        self.pipeline = pipeline

    def run(self, user_input: str) -> PipelineResult:
        models_cfg = ConfigLoader.load_models_config(self.models_path)
        registry = ModelRegistry(models_cfg).build()

        report = RuntimeValidator.validate_pipeline_models(self.pipeline, models_cfg)
        if not report.ok:
            raise ValueError("; ".join(report.errors))

        aggregation = AggregationEngine(
            synthesis_callback=lambda model, prompt: f"[MANUAL-SYNTHESIS:{model}] {prompt}"
        )
        engine = PipelineEngine(
            registry=registry,
            model_wrapper=ModelWrapper(),
            aggregation_engine=aggregation,
            resource_manager=ResourceManager(),
            session_manager=SessionManager(),
        )
        return engine.run(self.pipeline, user_input=user_input)
