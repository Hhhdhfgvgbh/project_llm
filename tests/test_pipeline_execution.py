from pathlib import Path

from app.config.loader import ConfigLoader
from app.core.aggregation import AggregationEngine
from app.core.executors import BasePipelineExecutor
from app.core.model_registry import ModelRegistry
from app.core.model_wrapper import ModelWrapper
from app.core.pipeline_engine import PipelineEngine
from app.core.resource_manager import ResourceManager
from app.core.session_manager import SessionManager


def _write_models(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "llama3_q5.gguf").write_text("x", encoding="utf-8")
    (models_dir / "mistral_q4.gguf").write_text("x", encoding="utf-8")

    models_cfg = tmp_path / "models.yaml"
    models_cfg.write_text(
        f"""
version: 1
models_directory: "{models_dir}"
models:
  llama3_q5:
    file: "llama3_q5.gguf"
    quantization: "Q5_K_M"
    required_for_base: true
  mistral_q4:
    file: "mistral_q4.gguf"
    quantization: "Q4_K_M"
    required_for_base: false
""",
        encoding="utf-8",
    )
    return models_cfg


def _write_pipeline(tmp_path: Path) -> Path:
    pipeline_cfg = tmp_path / "pipeline.yaml"
    pipeline_cfg.write_text(
        """
version: 1
base_pipeline:
  stages:
    - id: stage1
      type: single
      model: llama3_q5
      system_prompt: "Analyze"
      output_mode: input_plus_answer

    - id: stage2
      type: multi
      models: [llama3_q5, mistral_q4]
      system_prompt: "Alternatives"
      aggregation:
        type: concat
      input_from: stage1

    - id: stage3
      type: single
      model: llama3_q5
      input_from: [stage1, stage2]
""",
        encoding="utf-8",
    )
    return pipeline_cfg


def test_base_pipeline_executor_runs(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = _write_pipeline(tmp_path)

    result = BasePipelineExecutor(str(models), str(pipeline)).run("hello")
    assert result.steps[0].stage_id == "stage1"
    assert result.steps[1].stage_id == "stage2"
    assert result.steps[2].stage_id == "stage3"
    assert "generated response" in result.final_output



def test_second_stage_system_prompt_is_forwarded_to_model(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = _write_pipeline(tmp_path)

    models_cfg = ConfigLoader.load_models_config(models)
    pipeline_cfg = ConfigLoader.load_pipeline_config(pipeline)
    registry = ModelRegistry(models_cfg).build()
    engine = PipelineEngine(
        registry=registry,
        model_wrapper=ModelWrapper(),
        aggregation_engine=AggregationEngine(),
        resource_manager=ResourceManager(),
        session_manager=SessionManager(root=tmp_path / "sessions"),
    )

    result = engine.run(pipeline_cfg, user_input="hello")

    stage1_output = result.steps[0].aggregated_output
    for model_output in result.steps[1].model_outputs.values():
        assert "system=<Alternatives>" in model_output
        assert stage1_output in model_output


def test_output_mode_input_plus_answer_changes_next_stage_input(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = _write_pipeline(tmp_path)

    models_cfg = ConfigLoader.load_models_config(models)
    pipeline_cfg = ConfigLoader.load_pipeline_config(pipeline)
    registry = ModelRegistry(models_cfg).build()
    engine = PipelineEngine(
        registry=registry,
        model_wrapper=ModelWrapper(),
        aggregation_engine=AggregationEngine(),
        resource_manager=ResourceManager(),
        session_manager=SessionManager(root=tmp_path / "sessions2"),
    )

    result = engine.run(pipeline_cfg, user_input="hello")

    for model_output in result.steps[1].model_outputs.values():
        assert "Input:" in model_output
        assert "Answer:" in model_output
