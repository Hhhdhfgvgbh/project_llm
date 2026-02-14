from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config.schemas import AggregationConfig, PipelineConfig, StageMulti, StageSingle
from app.core.aggregation import AggregationEngine
from app.core.model_registry import ModelRegistry, RegisteredModel
from app.core.model_wrapper import ModelRuntimeConfig, ModelWrapper
from app.core.resource_manager import ResourceEstimate, ResourceManager
from app.core.session_manager import SessionManager


@dataclass
class StageExecution:
    stage_id: str
    stage_type: str
    model_outputs: dict[str, str]
    aggregated_output: str


@dataclass
class PipelineResult:
    final_output: str
    steps: list[StageExecution]


class PipelineEngine:
    def __init__(
        self,
        registry: ModelRegistry,
        model_wrapper: ModelWrapper,
        aggregation_engine: AggregationEngine,
        resource_manager: ResourceManager,
        session_manager: SessionManager,
    ) -> None:
        self.registry = registry
        self.model_wrapper = model_wrapper
        self.aggregation_engine = aggregation_engine
        self.resource_manager = resource_manager
        self.session_manager = session_manager

    def run(
        self,
        pipeline: PipelineConfig,
        user_input: str,
        available_ram_gb: float = 64,
        available_vram_gb: float = 16,
        session_enabled: bool = True,
    ) -> PipelineResult:
        outputs: dict[str, str] = {}
        steps: list[StageExecution] = []

        session_path = self.session_manager.create_session() if session_enabled else None

        for index, stage in enumerate(pipeline.base_pipeline.stages, start=1):
            incoming = user_input if stage.input_from is None else outputs[stage.input_from]

            if isinstance(stage, StageSingle):
                step = self._execute_single(stage, incoming, available_ram_gb, available_vram_gb)
            elif isinstance(stage, StageMulti):
                step = self._execute_multi(stage, incoming, available_ram_gb, available_vram_gb)
            else:
                raise ValueError(f"Unsupported stage type: {stage}")

            outputs[stage.id] = step.aggregated_output
            steps.append(step)

            if session_path is not None:
                self.session_manager.write_stage_result(
                    session_path=session_path,
                    stage_id=stage.id,
                    version=1,
                    parent_version=None,
                    input_payload={"input": incoming},
                    config_payload={"stage": stage.model_dump()},
                    output_payload={
                        "model_outputs": step.model_outputs,
                        "aggregated_output": step.aggregated_output,
                    },
                )

        final_output = outputs[pipeline.base_pipeline.stages[-1].id]
        if session_path is not None:
            self.session_manager.write_final_output(session_path, final_output, mode="base")
        return PipelineResult(final_output=final_output, steps=steps)

    def _execute_single(
        self,
        stage: StageSingle,
        incoming: str,
        available_ram_gb: float,
        available_vram_gb: float,
    ) -> StageExecution:
        model = self._require_model(stage.model)
        runtime = self._runtime_config(model, stage.generation)
        self._assert_resources(available_ram_gb, available_vram_gb)

        self.model_wrapper.load(model.name, model.path, runtime)
        output = self.model_wrapper.generate(model.name, incoming, stage.system_prompt, runtime)
        self.model_wrapper.unload(model.name)

        return StageExecution(
            stage_id=stage.id,
            stage_type="single",
            model_outputs={model.name: output},
            aggregated_output=output,
        )

    def _execute_multi(
        self,
        stage: StageMulti,
        incoming: str,
        available_ram_gb: float,
        available_vram_gb: float,
    ) -> StageExecution:
        model_outputs: dict[str, str] = {}
        responses: list[str] = []

        for model_name in stage.models:
            model = self._require_model(model_name)
            runtime = self._runtime_config(model, stage.generation)
            self._assert_resources(available_ram_gb, available_vram_gb)

            self.model_wrapper.load(model.name, model.path, runtime)
            text = self.model_wrapper.generate(model.name, incoming, stage.system_prompt, runtime)
            self.model_wrapper.unload(model.name)

            model_outputs[model.name] = text
            responses.append(text)

        aggregation = self.aggregation_engine.aggregate(responses, self._aggregation_cfg(stage.aggregation))
        return StageExecution(
            stage_id=stage.id,
            stage_type="multi",
            model_outputs=model_outputs,
            aggregated_output=aggregation.output,
        )

    @staticmethod
    def _runtime_config(model: RegisteredModel, generation: dict[str, Any]) -> ModelRuntimeConfig:
        merged = model.generation.model_copy(update=generation)
        return ModelRuntimeConfig(**merged.model_dump())

    def _require_model(self, model_name: str) -> RegisteredModel:
        model = self.registry.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' is not registered or unavailable")
        return model

    def _assert_resources(self, available_ram_gb: float, available_vram_gb: float) -> None:
        estimate = ResourceEstimate(required_ram_gb=4.0, required_vram_gb=2.0)
        if not self.resource_manager.can_run(available_ram_gb, available_vram_gb, estimate):
            raise MemoryError("Insufficient resources for requested stage execution")

    @staticmethod
    def _aggregation_cfg(config: AggregationConfig) -> AggregationConfig:
        return config
