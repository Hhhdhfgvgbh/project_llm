from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config.schemas import (
    AggregationConfig,
    PipelineConfig,
    StageMulti,
    StageOutputMode,
    StageSingle,
)
from app.core.aggregation import AggregationEngine
from app.core.model_registry import ModelRegistry, RegisteredModel
from app.core.model_wrapper import ModelRuntimeConfig, ModelWrapper
from app.core.resource_manager import ResourceManager
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
        stage_outputs: dict[str, str] = {}
        forwarded_outputs: dict[str, str] = {}
        steps: list[StageExecution] = []

        session_path = self.session_manager.create_session() if session_enabled else None

        for index, stage in enumerate(pipeline.base_pipeline.stages, start=1):
            incoming = self._resolve_stage_input(
                user_input=user_input,
                stage_input_from=stage.input_from,
                outputs=forwarded_outputs,
            )

            if isinstance(stage, StageSingle):
                step = self._execute_single(stage, incoming, available_ram_gb, available_vram_gb)
            elif isinstance(stage, StageMulti):
                step = self._execute_multi(stage, incoming, available_ram_gb, available_vram_gb)
            else:
                raise ValueError(f"Unsupported stage type: {stage}")

            stage_outputs[stage.id] = step.aggregated_output
            forwarded_outputs[stage.id] = self._build_forward_payload(stage.output_mode, incoming, step.aggregated_output)
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
                        "forward_payload": forwarded_outputs[stage.id],
                    },
                )

        final_output = stage_outputs[pipeline.base_pipeline.stages[-1].id]
        if session_path is not None:
            self.session_manager.write_final_output(session_path, final_output, mode="base")
        return PipelineResult(final_output=final_output, steps=steps)


    @staticmethod
    def _resolve_stage_input(user_input: str, stage_input_from: str | list[str] | None, outputs: dict[str, str]) -> str:
        if stage_input_from is None:
            return user_input
        if isinstance(stage_input_from, str):
            return outputs[stage_input_from]

        chunks = [outputs[item] for item in stage_input_from]
        return "\n\n".join(chunks)


    @staticmethod
    def _build_model_input(instructions: str, incoming: str) -> str:
        cleaned = instructions.strip()
        if not cleaned:
            return incoming
        return f"{cleaned}\n\n{incoming}"

    @staticmethod
    def _build_forward_payload(output_mode: StageOutputMode, incoming: str, response: str) -> str:
        if output_mode == StageOutputMode.QUESTION_AND_ANSWER:
            return f"Question:\n{incoming}\n\nAnswer:\n{response}"
        return response

    def _execute_single(
        self,
        stage: StageSingle,
        incoming: str,
        available_ram_gb: float,
        available_vram_gb: float,
    ) -> StageExecution:
        model = self._require_model(stage.model)
        runtime = self._runtime_config(model, stage.generation)
        self._assert_resources(
            available_ram_gb=available_ram_gb,
            available_vram_gb=available_vram_gb,
            model_sizes_gb=[model.file_size_gb],
            n_ctx_values=[runtime.n_ctx],
        )

        self.model_wrapper.load(model.name, model.path, runtime)
        model_input = self._build_model_input(stage.instructions, incoming)
        output = self.model_wrapper.generate(model.name, model_input, stage.system_prompt, runtime)
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
            self._assert_resources(
                available_ram_gb=available_ram_gb,
                available_vram_gb=available_vram_gb,
                model_sizes_gb=[model.file_size_gb],
                n_ctx_values=[runtime.n_ctx],
            )

            self.model_wrapper.load(model.name, model.path, runtime)
            model_input = self._build_model_input(stage.instructions, incoming)
            text = self.model_wrapper.generate(model.name, model_input, stage.system_prompt, runtime)
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

    def _assert_resources(
        self,
        available_ram_gb: float,
        available_vram_gb: float,
        model_sizes_gb: list[float],
        n_ctx_values: list[int],
    ) -> None:
        estimate = self.resource_manager.estimate_for_models(model_sizes_gb, n_ctx_values)
        if not self.resource_manager.can_run(available_ram_gb, available_vram_gb, estimate):
            raise MemoryError(
                "Insufficient resources for requested stage execution "
                f"(required RAM~{estimate.required_ram_gb:.2f}GB, VRAM~{estimate.required_vram_gb:.2f}GB)"
            )

    @staticmethod
    def _aggregation_cfg(config: AggregationConfig) -> AggregationConfig:
        return config
