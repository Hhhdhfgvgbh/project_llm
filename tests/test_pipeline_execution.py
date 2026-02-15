from pathlib import Path

from app.core.executors import BasePipelineExecutor


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
      output_mode: question_and_answer

    - id: stage2
      type: multi
      models: [llama3_q5, mistral_q4]
      system_prompt: "Alternatives"
      instructions: "ONLY_JSON"
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


def test_stage_can_forward_question_and_answer(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = _write_pipeline(tmp_path)

    context = BasePipelineExecutor(str(models), str(pipeline)).build()
    result = context.engine.run(context.pipeline, user_input="hello", session_enabled=False)
    stage2_prompt_echo = result.steps[1].model_outputs["llama3_q5"]
    assert "Question:" in stage2_prompt_echo
    assert "Answer:" in stage2_prompt_echo


def test_stage_instructions_are_prepended_only_to_model_input(tmp_path: Path) -> None:
    models = _write_models(tmp_path)
    pipeline = _write_pipeline(tmp_path)

    context = BasePipelineExecutor(str(models), str(pipeline)).build()
    result = context.engine.run(context.pipeline, user_input="hello", session_enabled=False)
    stage2_prompt_echo = result.steps[1].model_outputs["llama3_q5"]
    assert "ONLY_JSON" in stage2_prompt_echo


def test_build_model_input_keeps_forwarded_data_clean() -> None:
    from app.core.pipeline_engine import PipelineEngine

    built = PipelineEngine._build_model_input("Use bullets", "Question:\nQ\n\nAnswer:\nA")
    assert built.startswith("Use bullets")
    assert "Question:\nQ" in built
