from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class Quantization(str, Enum):
    Q2_K = "Q2_K"
    Q3_K_M = "Q3_K_M"
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


class GenerationParams(BaseModel):
    n_ctx: int = Field(default=8192, ge=256)
    n_gpu_layers: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    max_tokens: int = Field(default=1024, ge=1)
    threads: int = Field(default=8, ge=1)


class ModelOverrides(BaseModel):
    n_ctx: int | None = Field(default=None, ge=256)
    n_gpu_layers: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    threads: int | None = Field(default=None, ge=1)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ModelDefinition(BaseModel):
    file: str
    description: str = ""
    quantization: Quantization
    required_for_base: bool = False
    strip_reasoning: bool = True
    overrides: ModelOverrides = Field(default_factory=ModelOverrides)


class ModelsConfig(BaseModel):
    version: int
    models_directory: Path
    defaults: GenerationParams = Field(default_factory=GenerationParams)
    models: dict[str, ModelDefinition]

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Only models config version=1 is supported")
        return value

    @field_validator("models")
    @classmethod
    def validate_models_not_empty(cls, value: dict[str, ModelDefinition]) -> dict[str, ModelDefinition]:
        if not value:
            raise ValueError("models dictionary cannot be empty")
        return value


class AggregationType(str, Enum):
    CONCAT = "concat"
    MAJORITY_VOTE = "majority_vote"
    SYNTHESIS = "synthesis"
    CUSTOM_TEMPLATE = "custom_template"


class AggregationConfig(BaseModel):
    type: AggregationType
    synthesis_model: str | None = None
    template: str | None = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> "AggregationConfig":
        if self.type in (AggregationType.SYNTHESIS, AggregationType.CUSTOM_TEMPLATE) and not self.synthesis_model:
            raise ValueError("synthesis_model is required for synthesis/custom_template aggregations")
        if self.type == AggregationType.CUSTOM_TEMPLATE and not self.template:
            raise ValueError("template is required for custom_template aggregation")
        return self


class StageOutputMode(str, Enum):
    ANSWER_ONLY = "answer_only"
    QUESTION_AND_ANSWER = "question_and_answer"


class StageSingle(BaseModel):
    id: str
    type: Literal["single"]
    model: str
    system_prompt: str = ""
    instructions: str = ""
    input_from: str | list[str] | None = None
    output_mode: StageOutputMode = StageOutputMode.ANSWER_ONLY
    generation: dict[str, Any] = Field(default_factory=dict)


class StageMulti(BaseModel):
    id: str
    type: Literal["multi"]
    models: list[str]
    system_prompt: str = ""
    instructions: str = ""
    input_from: str | list[str] | None = None
    output_mode: StageOutputMode = StageOutputMode.ANSWER_ONLY
    generation: dict[str, Any] = Field(default_factory=dict)
    aggregation: AggregationConfig

    @field_validator("models")
    @classmethod
    def validate_models_min_count(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("multi stage requires at least 2 models")
        return value


Stage = StageSingle | StageMulti


class BasePipeline(BaseModel):
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    stages: list[Stage]

    @field_validator("stages")
    @classmethod
    def validate_stages_not_empty(cls, value: list[Stage]) -> list[Stage]:
        if not value:
            raise ValueError("base_pipeline.stages cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_stage_links(self) -> "BasePipeline":
        ids: list[str] = [stage.id for stage in self.stages]
        if len(set(ids)) != len(ids):
            raise ValueError("Stage IDs must be unique")

        for index, stage in enumerate(self.stages):
            if index == 0 and stage.input_from is not None:
                raise ValueError("First stage must not define input_from")
            if index > 0 and stage.input_from is None:
                raise ValueError(f"Stage '{stage.id}' must define input_from")

            refs: list[str]
            if isinstance(stage.input_from, str):
                refs = [stage.input_from]
            elif isinstance(stage.input_from, list):
                refs = stage.input_from
                if not refs:
                    raise ValueError(f"Stage '{stage.id}' input_from list cannot be empty")
            else:
                refs = []

            for ref in refs:
                if ref not in ids:
                    raise ValueError(f"Stage '{stage.id}' references unknown input_from='{ref}'")

        return self


class PipelineConfig(BaseModel):
    version: int
    base_pipeline: BasePipeline
    pipelines: dict[str, BasePipeline] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Only pipeline config version=1 is supported")
        return value

    def list_pipelines(self) -> list[str]:
        return ["base_pipeline", *self.pipelines.keys()]

    def get_pipeline(self, name: str) -> BasePipeline:
        if name == "base_pipeline":
            return self.base_pipeline
        if name not in self.pipelines:
            raise KeyError(f"Unknown pipeline '{name}'")
        return self.pipelines[name]


def raise_config_error(context: str, error: ValidationError) -> ValueError:
    details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return ValueError(f"{context} validation failed: {details}")
