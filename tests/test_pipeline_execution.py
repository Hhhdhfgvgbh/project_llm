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
